"""Unified LLM-based screening engine for systematic reviews."""

import asyncio
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import ScreeningEngine
from .prompts import PromptLibrary
from .openai_client import GenericLLMClient
from ..models.types import Study, ScreeningConfig, ScreeningResult, StudyStatus
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


class LLMScreener(ScreeningEngine):
    """Unified LLM-powered screening engine for both abstract and full-text 
    screening."""

    def __init__(
        self,
        config: ScreeningConfig,
        inclusion_criteria: List[str] = None,
        exclusion_criteria: List[str] = None,
        output_dir: str = "test_output"
    ):
        """Initialize the unified LLM screener with configuration."""
        super().__init__(config)
        self._client = None
        self._cache_file = Path("cache/screening_cache.json")
        self._cache_file.parent.mkdir(exist_ok=True)
        self._cache = self._load_cache()
        self.inclusion_criteria = inclusion_criteria or []
        self.exclusion_criteria = exclusion_criteria or []
        self._llm_client: Optional[GenericLLMClient] = None
        self.output_dir = output_dir

    async def screen_abstracts(
        self, 
        studies: List[Study]
    ) -> List[ScreeningResult]:
        """
        Screen study abstracts for inclusion/exclusion.

        Args:
            studies: List of studies to screen

        Returns:
            List of screening results
        """
        logger.info(f"Starting abstract screening for {len(studies)} studies")

        # Initialize LLM client if not already done
        if self._llm_client is None:
            self._llm_client = GenericLLMClient()

        # Filter to only studies that have abstracts
        screenable_studies = [
            s for s in studies 
            if s.abstract and s.abstract.strip()
        ]

        if len(screenable_studies) < len(studies):
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} studies "
                f"without abstracts"
            )

        results = []
        abstract_config = self.config.abstract

        for study in screenable_studies:
            # Check cache first
            cache_key = self._get_cache_key(study, "abstract")
            if cache_key in self._cache:
                logger.info(f"Using cached result for {study.pmid} abstract")
                cached_result = self._cache[cache_key]
                results.append(ScreeningResult(
                    study_id=study.pmid,
                    decision=StudyStatus(cached_result["decision"]),
                    reason=cached_result["reason"],
                    confidence=cached_result["confidence"],
                    model_used=cached_result["model_used"],
                    screening_type="abstract"
                ))
                continue

            # Build prompt with inclusion/exclusion criteria from config
            prompt = PromptLibrary.get_abstract_screening_prompt(
                study=study,
                inclusion_criteria=self.inclusion_criteria,
                exclusion_criteria=self.exclusion_criteria
            )

            try:
                # Call LLM API
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._llm_client.screen_abstract,
                    prompt,
                    abstract_config.get("model", "gpt-4o-mini")
                )

                # Apply threshold
                threshold = abstract_config.get("threshold", 0.75)
                if response.confidence < threshold:
                    decision = StudyStatus.EXCLUDED
                    reason = (f"Confidence {response.confidence:.2f} below "
                              f"threshold {threshold}. {response.reason}")
                else:
                    decision = (StudyStatus.INCLUDED if response.decision ==
                                "INCLUDED" else StudyStatus.EXCLUDED)
                    reason = response.reason

                result = ScreeningResult(
                    study_id=study.pmid,
                    decision=decision,
                    reason=reason,
                    confidence=response.confidence,
                    model_used=abstract_config.get("model", "gpt-4o-mini"),
                    screening_type="abstract"
                )

                results.append(result)

                # Cache the result
                self._cache[cache_key] = {
                    "decision": decision.value,
                    "reason": reason,
                    "confidence": response.confidence,
                    "model_used": abstract_config.get("model", "gpt-4"),
                    "timestamp": datetime.now().isoformat()
                }
                self._save_cache()

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                log_error_with_debug(
                    logger,
                    f"Error screening abstract for {study.pmid}: {e}"
                )
                # Return failed screening result
                results.append(ScreeningResult(
                    study_id=study.pmid,
                    decision=StudyStatus.SCREENING_FAILED,
                    reason=f"Screening failed: {str(e)}",
                    confidence=0.0,
                    model_used=abstract_config.get("model", "gpt-4"),
                    screening_type="abstract"
                ))

        return results

    async def screen_fulltexts(
        self, 
        studies: List[Study]
    ) -> List[ScreeningResult]:
        """
        Screen full-text articles for inclusion/exclusion.

        Args:
            studies: List of studies with full text to screen

        Returns:
            List of screening results
        """
        logger.info(f"Starting full-text screening for {len(studies)} studies")

        # Initialize LLM client if not already done
        if self._llm_client is None:
            self._llm_client = GenericLLMClient()

        # Filter to only studies that have full text and correct status
        # Check if the study has a pmcid and the standard text file exists
        # Also check that the study status is appropriate for screening
        text_file_path = (
            Path(self.output_dir) / "retrieval" / "pubget_data" / "text.csv"
        )
        screenable_studies = [
            s for s in studies
            if s.pmcid and text_file_path.exists() and
            s.status in [StudyStatus.FULLTEXT_RETRIEVED,
                         StudyStatus.FULLTEXT_CACHED]
        ]

        if len(screenable_studies) < len(studies):
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} studies "
                f"without full text"
            )

        results = []
        fulltext_config = self.config.fulltext

        for study in screenable_studies:
            # Check cache first
            cache_key = self._get_cache_key(study, "fulltext")
            if cache_key in self._cache:
                logger.info(f"Using cached result for {study.pmid} fulltext")
                cached_result = self._cache[cache_key]
                results.append(ScreeningResult(
                    study_id=study.pmid,
                    decision=StudyStatus(cached_result["decision"]),
                    reason=cached_result["reason"],
                    confidence=cached_result["confidence"],
                    model_used=cached_result["model_used"],
                    screening_type="fulltext"
                ))
                continue

            # Build prompt with inclusion/exclusion criteria from config
            prompt = PromptLibrary.get_fulltext_screening_prompt(
                study=study,
                inclusion_criteria=self.inclusion_criteria,
                exclusion_criteria=self.exclusion_criteria,
                output_dir=self.output_dir
            )

            try:
                # Call LLM API
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._llm_client.screen_fulltext,
                    prompt,
                    fulltext_config.get("model", "gpt-4")
                )

                # Apply threshold
                threshold = fulltext_config.get("threshold", 0.8)
                if response.confidence < threshold:
                    decision = StudyStatus.EXCLUDED
                    reason = (f"Confidence {response.confidence:.2f} below "
                              f"threshold {threshold}. {response.reason}")
                else:
                    decision = (StudyStatus.INCLUDED if response.decision == 
                                "INCLUDED" else StudyStatus.EXCLUDED)
                    reason = response.reason

                result = ScreeningResult(
                    study_id=study.pmid,
                    decision=decision,
                    reason=reason,
                    confidence=response.confidence,
                    model_used=fulltext_config.get("model", "gpt-4"),
                    screening_type="fulltext"
                )

                results.append(result)

                # Cache the result
                self._cache[cache_key] = {
                    "decision": decision.value,
                    "reason": reason,
                    "confidence": response.confidence,
                    "model_used": fulltext_config.get("model", "gpt-4"),
                    "timestamp": datetime.now().isoformat()
                }
                self._save_cache()

            except Exception as e:
                log_error_with_debug(
                    logger, f"Error screening fulltext for {study.pmid}: {e}"
                )
                # Return failed screening result
                results.append(ScreeningResult(
                    study_id=study.pmid,
                    decision=StudyStatus.SCREENING_FAILED,
                    reason=f"Screening failed: {str(e)}",
                    confidence=0.0,
                    model_used=fulltext_config.get("model", "gpt-4"),
                    screening_type="fulltext"
                ))

        return results

    def get_screening_info(self) -> Dict[str, Any]:
        """Get information about the screening engine configuration."""
        return {
            "engine": "llm_screener",
            "abstract_model": self.config.abstract.get("model", "gpt-4"),
            "abstract_threshold": self.config.abstract.get("threshold", 0.75),
            "fulltext_model": self.config.fulltext.get("model", "gpt-4"),
            "fulltext_threshold": self.config.fulltext.get("threshold", 0.8),
            "cache_enabled": True,
            "cache_size": len(self._cache)
        }

    def _get_cache_key(self, study: Study, screening_type: str) -> str:
        """Generate a cache key for a study and screening type."""
        content = (f"{study.pmid}_{study.title}_{study.abstract}_"
                   f"{screening_type}")
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self) -> Dict[str, Any]:
        """Load screening cache from file."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save screening cache to file."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            log_error_with_debug(logger, f"Failed to save cache: {e}")

    def clear_cache(self):
        """Clear the screening cache."""
        self._cache = {}
        if self._cache_file.exists():
            self._cache_file.unlink()
        logger.info("Screening cache cleared")