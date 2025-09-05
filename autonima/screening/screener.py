"""Unified LLM-based screening engine for systematic reviews."""

import logging
import json
import concurrent.futures
import tqdm
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import ScreeningEngine
from .prompts import PromptLibrary
from .openai_client import GenericLLMClient
from ..models.types import Study, ScreeningConfig, ScreeningResult, StudyStatus
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


def load_screening_results_with_lock(
    file_path: Path,
    lock_suffix: str = "_lock"
) -> List[Dict[str, Any]]:
    """
    Load screening results from a JSON file with file locking to prevent
    concurrent modifications.
    
    Args:
        file_path: Path to the JSON file containing screening results
        lock_suffix: Suffix for the lock file
        
    Returns:
        List of screening results, or empty list if file doesn't exist
        or can't be loaded
    """
    lock_file = file_path.with_suffix(file_path.suffix + lock_suffix)
    
    # Wait for lock to be released
    while lock_file.exists():
        time.sleep(0.1)
    
    # Create lock file
    try:
        lock_file.touch()
        
        # Load results if file exists
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return data.get("screening_results", [])
            except Exception as e:
                logger.warning(
                    f"Failed to load screening results from {file_path}: {e}"
                )
                return []
        return []
    finally:
        # Remove lock file
        if lock_file.exists():
            lock_file.unlink()


def save_screening_result_with_lock(
    file_path: Path,
    new_result: Dict[str, Any],
    lock_suffix: str = "_lock"
) -> bool:
    """
    Save a screening result to a JSON file with file locking to prevent
    concurrent modifications.
    
    Args:
        file_path: Path to the JSON file containing screening results
        new_result: The new screening result to add
        lock_suffix: Suffix for the lock file
        
    Returns:
        True if successful, False otherwise
    """
    lock_file = file_path.with_suffix(file_path.suffix + lock_suffix)
    
    # Wait for lock to be released
    while lock_file.exists():
        time.sleep(0.1)
    
    # Create lock file
    try:
        lock_file.touch()
        
        # Load existing results
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load existing screening results: {e}"
                )
                data = {"screening_results": []}
        else:
            data = {
                "screening_results": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Add new result or update existing one
        existing_results = data.get("screening_results", [])
        updated = False
        for i, result in enumerate(existing_results):
            if result.get("study_id") == new_result.get("study_id"):
                existing_results[i] = new_result
                updated = True
                break
        
        if not updated:
            existing_results.append(new_result)
        
        data["screening_results"] = existing_results
        data["timestamp"] = datetime.now().isoformat()
        
        # Save results
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Failed to save screening result to {file_path}: {e}")
        return False
    finally:
        # Remove lock file
        if lock_file.exists():
            lock_file.unlink()


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
        num_workers: int = 1,
        objective: str = None
    ):
        """Initialize the unified LLM screener with configuration."""
        super().__init__(config)
        self._client = None
        self.inclusion_criteria = inclusion_criteria or []
        self.exclusion_criteria = exclusion_criteria or []
        self._llm_client: Optional[GenericLLMClient] = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.objective = objective
        # Load existing results
        self._existing_abstract_results = self._load_existing_results(
            "abstract"
        )
        self._existing_fulltext_results = self._load_existing_results(
            "fulltext"
        )

    def _load_existing_results(self, screening_type: str) -> Dict[str, Dict]:
        """
        Load existing screening results from the output directory.
        
        Args:
            screening_type: Either "abstract" or "fulltext"
            
        Returns:
            Dictionary mapping study IDs to their screening results
        """
        filename = f"{screening_type}_screening_results.json"
        results_file = self.output_dir / filename
        if not results_file.exists():
            return {}
            
        try:
            results = load_screening_results_with_lock(results_file)
            # Convert list to dictionary for faster lookup
            return {result["study_id"]: result for result in results}
        except Exception as e:
            logger.warning(
                f"Failed to load existing {screening_type} results: {e}"
            )
            return {}

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
            # Build prompt with inclusion/exclusion criteria from config
            prompt = (
                PromptLibrary.get_abstract_screening_prompt
                if screening_type == "abstract"
                else PromptLibrary.get_fulltext_screening_prompt
            )(
                study=study,
                inclusion_criteria=self.inclusion_criteria,
                exclusion_criteria=self.exclusion_criteria,
                objective=self.objective,
                **(
                    dict(output_dir=str(self.output_dir))
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
                
            # Save result to file after each screening operation
            result_dict = result.to_dict()
            results_file = (
                self.output_dir / f"{screening_type}_screening_results.json"
            )
            save_screening_result_with_lock(results_file, result_dict)
                
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
            result = self._create_screening_result(
                study,
                StudyStatus.SCREENING_FAILED,
                f"Screening failed: {str(e)}",
                0.0,
                model,
                screening_type
            )
            
            # Save failed result to file
            result_dict = result.to_dict()
            results_file = (
                self.output_dir / f"{screening_type}_screening_results.json"
            )
            save_screening_result_with_lock(results_file, result_dict)
            
            return result

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
        
        # Separate existing results from studies that need screening
        existing_results = []
        studies_to_screen = []
        
        # Get the appropriate existing results dictionary
        existing_results_dict = (
            self._existing_abstract_results
            if screening_type == "abstract"
            else self._existing_fulltext_results
        )
        
        for study in screenable_studies:
            if study.pmid in existing_results_dict:
                existing_result = existing_results_dict[study.pmid]
                existing_results.append(self._create_screening_result(
                    study,
                    StudyStatus(existing_result["decision"]),
                    existing_result["reason"],
                    existing_result["confidence"],
                    existing_result["model_used"],
                    screening_type
                ))
            else:
                studies_to_screen.append(study)
        
        logger.info(
            f"Found {len(existing_results)} existing results, "
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
        
        # Combine existing and new results
        results = existing_results + new_results
        
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
            "cache_enabled": False
        }