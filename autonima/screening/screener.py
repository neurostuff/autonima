"""Unified LLM-based screening engine for systematic reviews."""

import logging
import hashlib
import json
import concurrent.futures
import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

from .base import ScreeningEngine
from .prompts import PromptLibrary
from .openai_client import GenericLLMClient
from ..models.types import Study, ScreeningConfig, ScreeningResult, StudyStatus
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


def parallelize_screening(func):
    """Decorator to parallelize screening over individual studies."""
    def wrapper(self, studies: List[Study], *args, **kwargs):
        num_workers = kwargs.pop("num_workers", 1)
        
        if num_workers <= 1 or len(studies) <= 1:
            # Serial processing
            return [
                func(self, study, *args, **kwargs)
                for study in tqdm.tqdm(studies)
            ]
        
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = [
                executor.submit(func, self, study, *args, **kwargs)
                for study in studies
            ]
            results = [
                future.result()
                for future in tqdm.tqdm(futures, total=len(futures))
            ]
        
        return results
    
    return wrapper


class LLMScreener(ScreeningEngine):
    """Unified LLM-powered screening engine for both abstract and full-text 
    screening."""

    def __init__(
        self,
        config: ScreeningConfig,
        inclusion_criteria: List[str] = None,
        exclusion_criteria: List[str] = None,
        output_dir: str = "test_output",
        num_workers: int = 1
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
        self.num_workers = num_workers
        self._cache_lock = Lock()

    def _initialize_llm_client(self) -> GenericLLMClient:
        """Initialize the LLM client if not already done."""
        if self._llm_client is None:
            self._llm_client = GenericLLMClient()
        return self._llm_client

    def _filter_screenable_studies(
        self,
        studies: List[Study],
        screening_type: str
    ) -> List[Study]:
        """Filter studies based on screening type requirements."""
        if screening_type == "abstract":
            return [s for s in studies if s.abstract and s.abstract.strip()]
        else:  # fulltext
            return [
                s for s in studies
                if s.pmcid and
                s.status in [
                    StudyStatus.FULLTEXT_RETRIEVED,
                    StudyStatus.FULLTEXT_CACHED
                ]
            ]

    def _get_screening_config(self, screening_type: str):
        """Get the configuration for the specified screening type."""
        return (
            self.config.abstract
            if screening_type == "abstract"
            else self.config.fulltext
        )

    def _process_screening_response(
        self,
        response,
        config,
        screening_type: str
    ):
        """Process the LLM response and apply threshold logic."""
        threshold = config.get(
            "threshold",
            0.75 if screening_type == "abstract" else 0.8
        )
        if response.confidence < threshold:
            decision = StudyStatus.EXCLUDED
            reason = (f"Confidence {response.confidence:.2f} below "
                      f"threshold {threshold}. {response.reason}")
        else:
            decision = (
                StudyStatus.INCLUDED
                if response.decision == "INCLUDED"
                else StudyStatus.EXCLUDED
            )
            reason = response.reason
        return decision, reason

    def _create_screening_result(
        self,
        study: Study,
        decision: StudyStatus,
        reason: str,
        confidence: float,
        model: str,
        screening_type: str
    ) -> ScreeningResult:
        """Create a ScreeningResult object."""
        return ScreeningResult(
            study_id=study.pmid,
            decision=decision,
            reason=reason,
            confidence=confidence,
            model_used=model,
            screening_type=screening_type
        )

    def _screen_single_study(
        self,
        study: Study,
        screening_type: str,
        config: Dict[str, Any]
    ) -> ScreeningResult:
        """Screen a single study (synchronous for parallel execution)."""
        try:
            # Check cache first (although main screening function handles this)
            cache_key = self._get_cache_key(study, screening_type)
            # Build prompt with inclusion/exclusion criteria from config
            prompt = (
                PromptLibrary.get_abstract_screening_prompt
                if screening_type == "abstract"
                else PromptLibrary.get_fulltext_screening_prompt
            )(
                study=study,
                inclusion_criteria=self.inclusion_criteria,
                exclusion_criteria=self.exclusion_criteria,
                **(
                    dict(output_dir=self.output_dir)
                    if screening_type == "fulltext"
                    else {}
                )
            )

            model = config.get(
                "model",
                "gpt-4o-mini" if screening_type == "abstract" else "gpt-4"
            )
            
            # Call LLM API
            screen_method = (
                self._llm_client.screen_abstract
                if screening_type == "abstract"
                else self._llm_client.screen_fulltext
            )
            response = screen_method(prompt, model)
                
            # Process response
            decision, reason = self._process_screening_response(
                response, config, screening_type
            )
            result = self._create_screening_result(
                study, decision, reason, response.confidence,
                model,
                screening_type
            )
                
            # Cache the result
            self._cache[cache_key] = {
                "decision": decision.value,
                "reason": reason,
                "confidence": response.confidence,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save cache after each screening operation
            # for more frequent saving
            self._save_cache()
                
            return result
                
        except Exception as e:
            log_error_with_debug(
                logger,
                f"Error screening {screening_type} for {study.pmid}: {e}"
            )
            # Return failed screening result
            model = config.get(
                "model",
                "gpt-4o-mini" if screening_type == "abstract" else "gpt-4"
            )
            return self._create_screening_result(
                study,
                StudyStatus.SCREENING_FAILED,
                f"Screening failed: {str(e)}",
                0.0,
                model,
                screening_type
            )

    @parallelize_screening
    def _screen_single_study_wrapper(
        self,
        study: Study,
        screening_type: str,
        config: Dict[str, Any]
    ) -> ScreeningResult:
        """Wrapper for parallel screening of individual studies."""
        return self._screen_single_study(study, screening_type, config)

    async def screen_studies(
        self,
        studies: List[Study],
        screening_type: str = "abstract",
        num_workers: int = 1
    ) -> List[ScreeningResult]:
        """
        Screen studies using either abstract or fulltext screening.
        
        Args:
            studies: List of studies to screen
            screening_type: Either "abstract" or "fulltext"
            num_workers: Number of parallel workers (default: 1 for serial)
            
        Returns:
            List of screening results
        """
        logger.info(
            f"Starting {screening_type} screening for {len(studies)} studies"
        )
        
        self._initialize_llm_client()
        
        screenable_studies = self._filter_screenable_studies(
            studies, screening_type
        )
        if len(screenable_studies) < len(studies):
            text_type = (
                'abstracts' if screening_type == 'abstract' else 'full text'
            )
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} "
                f"studies without {text_type}"
            )
            
        if not screenable_studies:
            return []
            
        config = self._get_screening_config(screening_type)
        
        # Separate cached results from studies that need screening
        cached_results = []
        studies_to_screen = []
        
        for study in screenable_studies:
            cache_key = self._get_cache_key(study, screening_type)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                cached_results.append(self._create_screening_result(
                    study,
                    StudyStatus(cached_result["decision"]),
                    cached_result["reason"],
                    cached_result["confidence"],
                    cached_result["model_used"],
                    screening_type
                ))
            else:
                studies_to_screen.append(study)
        
        logger.info(
            f"Found {len(cached_results)} cached results, "
            f"{len(studies_to_screen)} studies to screen"
        )
        
        # Process studies that need screening
        new_results = []
        if studies_to_screen:
            # Use parallel processing if num_workers > 1
            logger.info(f"Using {num_workers} workers for parallel screening")
            new_results = self._screen_single_study_wrapper(
                studies_to_screen,
                screening_type,
                config,
                num_workers=num_workers
            )
        
        # Combine cached and new results
        results = cached_results + new_results
        
        return results
    
    async def screen_abstracts(
        self,
        studies: List[Study],
        num_workers: int = 1
    ) -> List[ScreeningResult]:
        """
        Screen study abstracts for inclusion/exclusion.

        Args:
            studies: List of studies to screen
            num_workers: Number of parallel workers (default: 1 for serial)

        Returns:
            List of screening results
        """
        return await self.screen_studies(studies, "abstract", num_workers)

    async def screen_fulltexts(
        self,
        studies: List[Study],
        num_workers: int = 1
    ) -> List[ScreeningResult]:
        """
        Screen full-text articles for inclusion/exclusion.

        Args:
            studies: List of studies with full text to screen
            num_workers: Number of parallel workers (default: 1 for serial)

        Returns:
            List of screening results
        """
        return await self.screen_studies(studies, "fulltext", num_workers)

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
            with self._cache_lock:
                # Create a copy of the cache to avoid "dictionary changed size
                # during iteration" errors when the cache is modified by other
                # threads during JSON serialization
                cache_copy = self._cache.copy()
                with open(self._cache_file, 'w') as f:
                    json.dump(cache_copy, f, indent=2)
        except Exception as e:
            log_error_with_debug(logger, f"Failed to save cache: {e}")

    def clear_cache(self):
        """Clear the screening cache."""
        self._cache = {}
        if self._cache_file.exists():
            self._cache_file.unlink()
        logger.info("Screening cache cleared")