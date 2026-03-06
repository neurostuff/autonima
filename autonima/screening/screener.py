"""Unified LLM-based screening engine for systematic reviews."""

import logging
import json
import os
import concurrent.futures
import tqdm
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import ScreeningEngine
from .prompts import PromptLibrary
from .openai_client import ScreeningLLMClient as GenericLLMClient
from ..models.types import Study, ScreeningConfig, ScreeningResult, StudyStatus
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


def _acquire_lock(lock_file: Path, poll_seconds: float = 0.05) -> None:
    """Acquire an inter-process lock via atomic lock-file creation."""
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(poll_seconds)


def _release_lock(lock_file: Path) -> None:
    """Release lock file if present."""
    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


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
    
    # Acquire lock (atomic create to avoid races between workers/processes)
    try:
        _acquire_lock(lock_file)
        
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
        _release_lock(lock_file)


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
    
    # Acquire lock (atomic create to avoid races between workers/processes)
    try:
        _acquire_lock(lock_file)
        
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
        
        # Save results atomically to avoid partially written JSON.
        temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, file_path)
            
        return True
    except Exception as e:
        logger.error(f"Failed to save screening result to {file_path}: {e}")
        return False
    finally:
        _release_lock(lock_file)


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
        output_dir: str,
        num_workers: int = 1
    ):
        """Initialize the unified LLM screener with configuration."""
        super().__init__(config)
        self._client = None
        self._llm_client: Optional[GenericLLMClient] = None
        self.result_dir = Path(output_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
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
        results_file = self.result_dir / "outputs" / filename
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
            # Only screen studies that:
            # 1. Have full text available
            # 2. Passed abstract screening (INCLUDED_ABSTRACT status)
            return [
                s for s in studies
                if (
                    (
                        s.fulltext_available
                        or bool(s.full_text_path)
                        or bool(s.pmcid)
                    )
                    and s.status == StudyStatus.INCLUDED_ABSTRACT
                )
            ]

    def _get_screening_config(self, screening_type: str):
        """Get the configuration for the specified screening type."""
        return (
            self.config.abstract
            if screening_type == "abstract"
            else self.config.fulltext
        )

    def _get_status_for_decision(
        self,
        screening_type: str,
        decision: str
    ) -> StudyStatus:
        """Map screening type and decision to appropriate StudyStatus."""
        if screening_type == "abstract":
            return (
                StudyStatus.INCLUDED_ABSTRACT
                if decision == "INCLUDED"
                else StudyStatus.EXCLUDED_ABSTRACT
            )
        else:  # fulltext
            return (
                StudyStatus.INCLUDED_FULLTEXT
                if decision == "INCLUDED"
                else StudyStatus.EXCLUDED_FULLTEXT
            )
    
    def _process_screening_response(
        self,
        response,
        config,
        confidence_reporting: bool = False,
        screening_type: str = "abstract"
    ):
        """Process the LLM response and apply threshold logic.
        
        Returns:
            Tuple of (decision_string, reason)
        """
        # Check if threshold should be applied
        threshold = config.get("threshold")
        threshold_enabled = (
            confidence_reporting and
            threshold is not None and
            0.0 <= threshold <= 1.0
        )
        
        if threshold_enabled and response.confidence < threshold:
            decision = "EXCLUDED"
            reason = (f"Confidence {response.confidence:.2f} below "
                      f"threshold {threshold}. {response.reason}")
        else:
            decision = response.decision  # "INCLUDED" or "EXCLUDED"
            reason = response.reason
                
        return decision, reason

    def _create_screening_result(
        self,
        study: Study,
        decision: StudyStatus,
        reason: str,
        confidence: float,
        model: str,
        screening_type: str,
        inclusion_criteria_applied: List[str] = None,
        exclusion_criteria_applied: List[str] = None
    ) -> ScreeningResult:
        """Create a ScreeningResult object."""
        return ScreeningResult(
            study_id=study.pmid,
            decision=decision,
            reason=reason,
            confidence=confidence,
            model_used=model,
            screening_type=screening_type,
            inclusion_criteria_applied=inclusion_criteria_applied or [],
            exclusion_criteria_applied=exclusion_criteria_applied or []
        )

    def _screen_single_study(
        self,
        study: Study,
        screening_type: str,
        config: Dict[str, Any]
    ) -> ScreeningResult:
        """Screen a single study (synchronous for parallel execution)."""
        try:
            # Check if confidence reporting is enabled
            confidence_reporting = config.get("confidence_reporting", False)
            
            # Get stage-specific criteria and instructions
            objective = config.get("objective")
            inclusion_criteria = config.get("inclusion_criteria") or []
            exclusion_criteria = config.get("exclusion_criteria") or []
            additional_instructions = config.get("additional_instructions")
            
            # Get criteria mapping from config
            criteria_mapping = config.get('criteria_mapping')
            
            # Build prompt with inclusion/exclusion criteria from config
            if screening_type == "abstract":
                prompt = PromptLibrary.get_abstract_screening_prompt(
                    study=study,
                    inclusion_criteria=inclusion_criteria,
                    exclusion_criteria=exclusion_criteria,
                    criteria_mapping=criteria_mapping,
                    objective=objective,
                    confidence_reporting=confidence_reporting,
                    additional_instructions=additional_instructions
                )
            else:
                if not study.full_text_output_dir:
                    study.full_text_output_dir = str(self.result_dir)
                if not study.fulltext_available and (
                    study.full_text_path or study.pmcid
                ):
                    study.fulltext_available = True

                prompt = PromptLibrary.get_fulltext_screening_prompt(
                    study=study,
                    inclusion_criteria=inclusion_criteria,
                    exclusion_criteria=exclusion_criteria,
                    output_dir=str(self.result_dir),
                    criteria_mapping=criteria_mapping,
                    objective=objective,
                    confidence_reporting=confidence_reporting,
                    additional_instructions=additional_instructions
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
                
            # Process response to get decision string
            decision_str, reason = self._process_screening_response(
                response, config, confidence_reporting, screening_type
            )
            
            # Convert decision string to stage-appropriate status
            decision = self._get_status_for_decision(
                screening_type, decision_str
            )
            
            # Extract criteria IDs from response and store on study object
            if screening_type == "abstract":
                study.abstract_inclusion_criteria_applied = getattr(
                    response, 'inclusion_criteria_applied', [])
                study.abstract_exclusion_criteria_applied = getattr(
                    response, 'exclusion_criteria_applied', [])
            else:  # fulltext
                study.fulltext_inclusion_criteria_applied = getattr(
                    response, 'inclusion_criteria_applied', [])
                study.fulltext_exclusion_criteria_applied = getattr(
                    response, 'exclusion_criteria_applied', [])
            
            # Get criteria applied from response
            inclusion_criteria_applied = getattr(
                response, 'inclusion_criteria_applied', [])
            exclusion_criteria_applied = getattr(
                response, 'exclusion_criteria_applied', [])
            
            result = self._create_screening_result(
                study, decision, reason, response.confidence,
                model,
                screening_type,
                inclusion_criteria_applied,
                exclusion_criteria_applied
            )
                
            # Save result to file after each screening operation
            result_dict = result.to_dict()
            output_dir = self.result_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = (
                output_dir / f"{screening_type}_screening_results.json"
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
                self.result_dir / f"{screening_type}_screening_results.json"
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
        self._initialize_llm_client()

        total_studies = len(studies)
        screenable_studies = self._filter_screenable_studies(
            studies, screening_type
        )
        skipped_count = total_studies - len(screenable_studies)
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
                
                # Normalize cached decisions to the current screening stage.
                old_decision = str(existing_result["decision"]).strip().lower()
                if old_decision in {
                    "included",
                    StudyStatus.INCLUDED_ABSTRACT.value,
                    StudyStatus.INCLUDED_FULLTEXT.value,
                }:
                    new_decision = self._get_status_for_decision(
                        screening_type, "INCLUDED"
                    )
                elif old_decision in {
                    "excluded",
                    StudyStatus.EXCLUDED_ABSTRACT.value,
                    StudyStatus.EXCLUDED_FULLTEXT.value,
                }:
                    new_decision = self._get_status_for_decision(
                        screening_type, "EXCLUDED"
                    )
                else:
                    # Try to use the value directly for newer serialized values.
                    try:
                        new_decision = StudyStatus(existing_result["decision"])
                    except ValueError:
                        # If invalid, skip this cached result
                        logger.warning(
                            f"Invalid cached status '{old_decision}' for "
                            f"study {study.pmid}, will re-screen"
                        )
                        studies_to_screen.append(study)
                        continue
                
                existing_results.append(self._create_screening_result(
                    study,
                    new_decision,
                    existing_result["reason"],
                    existing_result["confidence"],
                    existing_result["model_used"],
                    screening_type,
                    existing_result.get("inclusion_criteria_applied", []),
                    existing_result.get("exclusion_criteria_applied", [])
                ))
            else:
                studies_to_screen.append(study)
        
        stage_label = (
            "Abstract screening"
            if screening_type == "abstract"
            else "Full-text screening"
        )
        logger.info(
            "Starting %s: %s to screen, %s cached",
            stage_label,
            len(studies_to_screen),
            len(existing_results),
        )

        # Process studies that need screening
        new_results = []
        if studies_to_screen:
            # Use parallel processing if num_workers > 1
            logger.debug(f"Using {num_workers} workers for parallel screening")
            new_results = self._screen_single_study_wrapper(
                studies_to_screen,
                screening_type,
                config,
                num_workers=num_workers
            )
        
        # Combine existing and new results
        results = existing_results + new_results

        included_count = 0
        excluded_count = 0
        failed_count = 0
        for result in results:
            decision_value = (
                result.decision.value
                if isinstance(result.decision, StudyStatus)
                else str(result.decision).strip().lower()
            )
            if decision_value in {
                StudyStatus.INCLUDED_ABSTRACT.value,
                StudyStatus.INCLUDED_FULLTEXT.value,
            }:
                included_count += 1
            elif decision_value in {
                StudyStatus.EXCLUDED_ABSTRACT.value,
                StudyStatus.EXCLUDED_FULLTEXT.value,
            }:
                excluded_count += 1
            elif decision_value == StudyStatus.SCREENING_FAILED.value:
                failed_count += 1

        skipped_suffix = (
            f", {skipped_count} skipped (missing "
            f"{'abstracts' if screening_type == 'abstract' else 'full text'})"
            if skipped_count
            else ""
        )

        logger.info(
            f"{stage_label}: {len(screenable_studies)} eligible, "
            f"{included_count} included, {excluded_count} excluded, "
            f"{failed_count} failed"
            f"{skipped_suffix}"
        )
        
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
