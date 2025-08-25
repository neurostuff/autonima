"""Unified LLM-based screening engine for systematic reviews."""

import asyncio
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .base import ScreeningEngine
from ..models.types import Study, ScreeningConfig, ScreeningResult, StudyStatus

logger = logging.getLogger(__name__)


class LLMScreener(ScreeningEngine):
    """Unified LLM-powered screening engine for both abstract and full-text screening."""

    def __init__(self, config: ScreeningConfig):
        """Initialize the unified LLM screener with configuration."""
        super().__init__(config)
        self._client = None
        self._cache_file = Path("cache/screening_cache.json")
        self._cache_file.parent.mkdir(exist_ok=True)
        self._cache = self._load_cache()

    async def screen_abstracts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Screen study abstracts for inclusion/exclusion.

        Args:
            studies: List of studies to screen

        Returns:
            List of screening results
        """
        logger.info(f"Starting abstract screening for {len(studies)} studies")

        # Filter to only studies that have abstracts
        screenable_studies = [s for s in studies if s.abstract and s.abstract.strip()]

        if len(screenable_studies) < len(studies):
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} studies without abstracts"
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
            prompt = self.build_screening_prompt(
                study=study,
                inclusion_criteria=self.config.inclusion_criteria,
                exclusion_criteria=self.config.exclusion_criteria,
                screening_type="abstract"
            )

            try:
                response = await self._call_llm(
                    prompt=prompt,
                    model=abstract_config.get("model", "gpt-4"),
                    temperature=abstract_config.get("temperature", 0.1),
                    max_tokens=abstract_config.get("max_tokens", 1000)
                )

                decision, confidence, reason = self.parse_screening_response(response)

                # Apply threshold
                threshold = abstract_config.get("threshold", 0.75)
                if confidence < threshold:
                    decision = StudyStatus.EXCLUDED
                    reason = f"Confidence {confidence:.2f} below threshold {threshold}. {reason}"

                result = ScreeningResult(
                    study_id=study.pmid,
                    decision=decision,
                    reason=reason,
                    confidence=confidence,
                    model_used=abstract_config.get("model", "gpt-4"),
                    screening_type="abstract"
                )

                results.append(result)

                # Cache the result
                self._cache[cache_key] = {
                    "decision": decision.value,
                    "reason": reason,
                    "confidence": confidence,
                    "model_used": abstract_config.get("model", "gpt-4"),
                    "timestamp": datetime.now().isoformat()
                }
                self._save_cache()

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error screening abstract for {study.pmid}: {e}")
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

    async def screen_fulltexts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Screen full-text articles for inclusion/exclusion.

        Args:
            studies: List of studies with full text to screen

        Returns:
            List of screening results
        """
        logger.info(f"Starting full-text screening for {len(studies)} studies")

        # Filter to only studies that have full text
        screenable_studies = [
            s for s in studies
            if s.full_text_path and Path(s.full_text_path).exists()
        ]

        if len(screenable_studies) < len(studies):
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} studies without full text"
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
            prompt = self.build_screening_prompt(
                study=study,
                inclusion_criteria=self.config.inclusion_criteria,
                exclusion_criteria=self.config.exclusion_criteria,
                screening_type="fulltext"
            )

            try:
                response = await self._call_llm(
                    prompt=prompt,
                    model=fulltext_config.get("model", "gpt-4"),
                    temperature=fulltext_config.get("temperature", 0.1),
                    max_tokens=fulltext_config.get("max_tokens", 2000)
                )

                decision, confidence, reason = self.parse_screening_response(response)

                # Apply threshold
                threshold = fulltext_config.get("threshold", 0.8)
                if confidence < threshold:
                    decision = StudyStatus.EXCLUDED
                    reason = f"Confidence {confidence:.2f} below threshold {threshold}. {reason}"

                result = ScreeningResult(
                    study_id=study.pmid,
                    decision=decision,
                    reason=reason,
                    confidence=confidence,
                    model_used=fulltext_config.get("model", "gpt-4"),
                    screening_type="fulltext"
                )

                results.append(result)

                # Cache the result
                self._cache[cache_key] = {
                    "decision": decision.value,
                    "reason": reason,
                    "confidence": confidence,
                    "model_used": fulltext_config.get("model", "gpt-4"),
                    "timestamp": datetime.now().isoformat()
                }
                self._save_cache()

            except Exception as e:
                logger.error(f"Error screening fulltext for {study.pmid}: {e}")
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

    async def _call_llm(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> str:
        """
        Call the LLM API with the given prompt.

        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The LLM response string
        """
        # Mock LLM call for now - in real implementation, use OpenAI API
        logger.info(f"Calling {model} with prompt ({len(prompt)} chars)")

        # Simulate API call delay
        await asyncio.sleep(0.5)

        # Mock response based on prompt content
        if "schizophrenia" in prompt.lower() and "fmri" in prompt.lower():
            return """
DECISION: INCLUDED
CONFIDENCE: 0.92
REASON: This study investigates fMRI correlates of working memory in schizophrenia patients, meeting all inclusion criteria for neuroimaging research in psychiatric populations.
"""
        elif "animal" in prompt.lower() or "mice" in prompt.lower():
            return """
DECISION: EXCLUDED
CONFIDENCE: 0.98
REASON: This study uses animal subjects (mice), which violates the inclusion criterion requiring human participants.
"""
        else:
            return """
DECISION: EXCLUDED
CONFIDENCE: 0.73
REASON: This study does not meet the neuroimaging inclusion criteria as it lacks fMRI or other specified imaging modalities.
"""

    def _get_cache_key(self, study: Study, screening_type: str) -> str:
        """Generate a cache key for a study and screening type."""
        content = f"{study.pmid}_{study.title}_{study.abstract}_{screening_type}"
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
            logger.error(f"Failed to save cache: {e}")

    def clear_cache(self):
        """Clear the screening cache."""
        self._cache = {}
        if self._cache_file.exists():
            self._cache_file.unlink()
        logger.info("Screening cache cleared")