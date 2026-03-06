"""Annotation processor for the pipeline."""

import logging
import json
from pathlib import Path
from typing import List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .schema import AnnotationConfig, AnnotationDecision, AnalysisMetadata, AnnotationCriteriaConfig
from .client import AnnotationClient
from ..models.types import Study
from ..coordinates.schema import Analysis
from ..coordinates.nimads_models import sanitize_analysis_name
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


class AnnotationProcessor:
    """Processor for annotating analyses based on LLM decisions."""
    
    def __init__(self, config: AnnotationConfig, num_workers: int = 1, max_retries: int = 3):
        """
        Initialize the annotation processor.
        
        Args:
            config: Annotation configuration
            num_workers: Number of parallel workers for processing
            max_retries: Maximum number of retries for malformed LLM responses
        """
        self.config = config
        self.client = AnnotationClient(max_retries=max_retries)
        self.annotation_results: List[AnnotationDecision] = []
        self.num_workers = num_workers
    
    def process_studies(
        self,
        included_studies: List[Study],
        all_studies: List[Study] = None,
        output_dir: str = None
    ) -> List[AnnotationDecision]:
        """
        Process studies and annotate their analyses.
        
        Args:
            included_studies: List of INCLUDED studies with parsed analyses
            all_studies: Optional list of ALL studies (INCLUDED + EXCLUDED)
                        with parsed analyses for the all_studies annotation
            output_dir: Output directory for caching results
            
        Returns:
            List of annotation decisions
        """
        # Load existing cached results
        existing_cached_results = self._load_cached_results(output_dir) or []
        
        if not included_studies:
            logger.info(
                "No INCLUDED studies with analyses found for annotation"
            )
            return existing_cached_results
        
        logger.info(
            f"Processing {len(included_studies)} INCLUDED studies with "
            "analyses for annotation"
        )

        if output_dir:
            for study in included_studies:
                if not study.full_text_output_dir:
                    study.full_text_output_dir = output_dir
            if all_studies:
                for study in all_studies:
                    if not study.full_text_output_dir:
                        study.full_text_output_dir = output_dir
        
        # Process all analysis-annotation combinations incrementally by study
        all_decisions = []
        
        # Process the "all_analyses" annotation for INCLUDED studies
        if self.config.create_all_included_annotation:
            all_analyses_decisions = self._create_all_analyses_annotations_by_study(
                included_studies,
                annotation_name="all_analyses",
                output_dir=output_dir,
                existing_results=existing_cached_results
            )
            all_decisions.extend(all_analyses_decisions)
        
        # Process the "all_studies" annotation for ALL studies if enabled
        if self.config.create_all_from_search_annotation and all_studies:
            logger.info(
                f"Creating 'all_studies' annotation for "
                f"{len(all_studies)} studies (INCLUDED + EXCLUDED)"
            )
            all_studies_decisions = self._create_all_analyses_annotations_by_study(
                all_studies,
                annotation_name="all_studies",
                output_dir=output_dir,
                existing_results=existing_cached_results
            )
            all_decisions.extend(all_studies_decisions)
        
        # Process custom annotations on INCLUDED studies only
        if self.config.annotations:
            custom_decisions = self._process_custom_annotations_by_study(
                included_studies,
                self.config.model,
                output_dir=output_dir,
                existing_results=existing_cached_results
            )
            all_decisions.extend(custom_decisions)
        
        # Save results to cache (only new/updated decisions)
        if all_decisions:
            self._save_results_by_study(all_decisions, output_dir, existing_cached_results)
        self.annotation_results = all_decisions
        
        # Return all cached results (existing + new)
        return self._load_cached_results(output_dir) or []
    
    def _create_all_analyses_annotations(
        self,
        studies: List[Study],
        annotation_name: str = "all_analyses"
    ) -> List[AnnotationDecision]:
        """
        Create annotation decisions for a default annotation.
        
        Args:
            studies: List of studies with analyses
            annotation_name: Name of the annotation to create
            
        Returns:
            List of annotation decisions (all marked as included)
        """
        decisions = []
        
        for study in studies:
            for i, analysis in enumerate(study.analyses):
                # Create a unique analysis ID (sanitized)
                analysis_id = sanitize_analysis_name(f"{study.pmid}_analysis_{i}")
                
                # Create decision for the annotation
                decision = AnnotationDecision(
                    annotation_name=annotation_name,
                    analysis_id=analysis_id,
                    study_id=study.pmid,
                    include=True,
                    reasoning=f"All analyses included in '{annotation_name}'",
                    model_used="none"
                )
                decisions.append(decision)
        
        logger.info(
            f"Created {len(decisions)} decisions for '{annotation_name}' "
            "annotation"
        )
        return decisions
    
    def _create_all_analyses_annotations_by_study(
        self,
        studies: List[Study],
        annotation_name: str = "all_analyses",
        output_dir: str = None,
        existing_results: List[AnnotationDecision] = None
    ) -> List[AnnotationDecision]:
        """
        Create annotation decisions for a default annotation by study.
        
        Args:
            studies: List of studies with analyses
            annotation_name: Name of the annotation to create
            output_dir: Output directory for caching
            existing_results: Existing cached results
            
        Returns:
            List of annotation decisions (all marked as included)
        """
        decisions = []
        
        # Create a set of study IDs that have complete results for this annotation
        studies_with_complete_results = self._get_studies_with_complete_results(
            existing_results or [], annotation_name, studies
        )
        
        # Process studies that don't have complete results
        studies_to_process = [s for s in studies if s.pmid not in studies_with_complete_results]
        
        if studies_to_process:
            logger.info(
                f"Processing {len(studies_to_process)} studies for '{annotation_name}' "
                "annotation (missing or incomplete cache)"
            )
            
            for study in studies_to_process:
                for i, analysis in enumerate(study.analyses):
                    # Create a unique analysis ID (sanitized)
                    analysis_id = sanitize_analysis_name(f"{study.pmid}_analysis_{i}")
                    
                    # Create decision for the annotation
                    decision = AnnotationDecision(
                        annotation_name=annotation_name,
                        analysis_id=analysis_id,
                        study_id=study.pmid,
                        include=True,
                        reasoning=f"All analyses included in '{annotation_name}'",
                        model_used="none"
                    )
                    decisions.append(decision)
        else:
            logger.info(f"All studies already have complete results for '{annotation_name}' annotation")
        
        return decisions
    
    def _process_custom_annotations_by_study(self, studies: List[Study], model: str, output_dir: str = None, existing_results: List[AnnotationDecision] = None) -> List[AnnotationDecision]:
        """
        Process custom annotations using LLM decisions by study with multi-annotation prompts.
        
        Args:
            studies: List of included studies with analyses
            model: LLM model to use for decisions
            output_dir: Output directory for caching
            existing_results: Existing cached results
            
        Returns:
            List of annotation decisions
        """
        if not self.config.annotations:
            return []
        
        # For each study, check which annotations need processing
        studies_to_process = []
        for study in studies:
            # Create a set of annotation names that have complete results for this study
            study_annotations_complete = \
                self._get_annotations_with_complete_results_for_study(
                    existing_results or [], study.pmid,
                    self.config.annotations, study
                )
            
            # Get annotations that need processing for this study
            annotations_to_process = [a for a in self.config.annotations if a.name not in study_annotations_complete]
            
            if not annotations_to_process:
                continue
            
            studies_to_process.append((study, annotations_to_process))
        
        if not studies_to_process:
            return []
        
        logger.info(
            f"Processing {len(studies_to_process)} studies for custom annotations "
            "(missing or incomplete cache)"
        )
        
        # Process studies in parallel
        decisions = []
        if self.num_workers <= 1 or len(studies_to_process) <= 1:
            # Serial processing
            for study, annotations_to_process in tqdm(studies_to_process, desc="Processing studies"):
                try:
                    study_decisions = self._process_single_study_annotations(
                        study, self.config.metadata_fields, annotations_to_process, model
                    )
                    decisions.extend(study_decisions)
                except Exception as e:
                    logger.error(f"Error processing annotations for study {study.pmid}: {e}")
                    # Continue processing other studies
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all studies for processing
                future_to_study = {
                    executor.submit(self._process_single_study_annotations, study, self.config.metadata_fields, annotations_to_process, model): (study, annotations_to_process)
                    for study, annotations_to_process in studies_to_process
                }
                
                # Collect results
                for future in tqdm(as_completed(future_to_study), total=len(studies_to_process), desc="Processing studies"):
                    try:
                        study_decisions = future.result()
                        decisions.extend(study_decisions)
                    except Exception as e:
                        study, annotations_to_process = future_to_study[future]
                        log_error_with_debug(logger, 
                            f"Error processing annotations for study {study.pmid}: {e}"
                        )   
                        # Continue processing other studies
        
        return decisions
    
    def _process_single_study_annotations(self, study: Study, metadata_fields: List[str], annotations_to_process: List[AnnotationCriteriaConfig], model: str) -> List[AnnotationDecision]:
        """
        Process annotations for a single study.
        
        Args:
            study: Study to process
            metadata_fields: List of metadata fields to extract
            annotations_to_process: List of annotations to process
            model: LLM model to use
            
        Returns:
            List of annotation decisions for this study
        """
        study_decisions = []
        
        # Check if we should use multi_analysis prompt (process whole study at once)
        if self.config.prompt_type == "multi_analysis":
            # Build StudyAnalysisGroup once for the entire study
            study_group = self._build_study_analysis_group(study, metadata_fields)
            
            # Make multi-annotation decision for all analyses in the study at once
            study_decisions = self.client.make_decision(
                study_group, annotations_to_process, metadata_fields,
                model=model,
                prompt_type=self.config.prompt_type
            )
        else:
            # Process each analysis individually (single_analysis mode)
            for i, analysis in enumerate(study.analyses):
                # Create a unique analysis ID (sanitized)
                analysis_id = sanitize_analysis_name(f"{study.pmid}_analysis_{i}")
                
                # Extract metadata for the analysis
                metadata = self._extract_analysis_metadata(study, analysis, analysis_id, metadata_fields)
                
                # Make multi-annotation decision for this analysis
                analysis_decisions = self.client.make_decision(
                    metadata, annotations_to_process, metadata_fields,
                    model=model,
                    prompt_type=self.config.prompt_type
                )
                
                study_decisions.extend(analysis_decisions)
        
        return study_decisions
    
    def _get_studies_with_complete_results(
        self, 
        existing_results: List[AnnotationDecision], 
        annotation_name: str, 
        studies: List[Study]
    ) -> Set[str]:
        """
        Get set of study IDs that have complete results for a given annotation.
        
        Args:
            existing_results: List of existing annotation decisions
            annotation_name: Name of the annotation to check
            studies: List of studies to check
            
        Returns:
            Set of study IDs with complete results
        """
        # Get expected number of analyses per study
        study_analysis_count = {study.pmid: len(study.analyses) for study in studies}
        
        # Group existing results by study
        study_results = {}
        for result in existing_results:
            if result.annotation_name == annotation_name:
                if result.study_id not in study_results:
                    study_results[result.study_id] = []
                study_results[result.study_id].append(result)
        
        # Check which studies have complete results
        studies_with_complete_results = set()
        for study_id, analyses_count in study_analysis_count.items():
            if study_id in study_results and len(study_results[study_id]) == analyses_count:
                studies_with_complete_results.add(study_id)
        
        return studies_with_complete_results
    
    def _get_annotations_with_complete_results_for_study(
        self,
        existing_results: List[AnnotationDecision],
        study_id: str,
        all_annotations: List[AnnotationCriteriaConfig],
        study: Study
    ) -> Set[str]:
        """
        Get set of annotation names that have complete results for a given study.
        
        Args:
            existing_results: List of existing annotation decisions
            study_id: Study ID to check
            all_annotations: List of all annotation criteria
            
        Returns:
            Set of annotation names with complete results
        """
        # Count existing results per annotation for this study
        annotation_counts = {}
        for result in existing_results:
            if result.study_id == study_id:
                if result.annotation_name not in annotation_counts:
                    annotation_counts[result.annotation_name] = 0
                annotation_counts[result.annotation_name] += 1
        
        # Get expected number of analyses from the study itself
        expected_count = len(study.analyses)
        
        # If we don't have any analyses, we can't determine completeness
        if expected_count == 0:
            return set()
        
        # Check which annotations have complete results
        annotations_with_complete_results = set()
        for annotation in all_annotations:
            if annotation_counts.get(annotation.name, 0) == expected_count:
                annotations_with_complete_results.add(annotation.name)
        
        return annotations_with_complete_results
    
    def _process_single_decision(self, metadata: AnalysisMetadata, criteria: AnnotationCriteriaConfig, model: str) -> AnnotationDecision:
        """
        Process a single annotation decision.
        
        Args:
            metadata: Analysis metadata
            criteria: Annotation criteria
            model: LLM model to use
            
        Returns:
            Annotation decision
        """
        return self.client.make_decision(metadata, criteria, model)
    
    def _extract_analysis_metadata(
        self,
        study: Study,
        analysis: Analysis,
        analysis_id: str,
        metadata_fields: Optional[List[str]] = None
    ) -> AnalysisMetadata:
        """
        Extract metadata for an analysis from a study.

        Args:
            study: Study containing the analysis
            analysis: Analysis to extract metadata for
            analysis_id: Unique ID for the analysis
            metadata_fields: Optional list of fields to include. If None, includes all fields.

        Returns:
            Analysis metadata with only the requested fields populated
        """
        # Extract table information if available
        table_id = None
        table_caption = None
        table_footer = None
        
        # Try to find the table associated with this analysis
        if analysis.table_id:
            if not study.activation_tables:
                raise ValueError(
                    f"Analysis {analysis_id} has table_id '{analysis.table_id}' but study "
                    f"{study.pmid} has no activation_tables"
                )
            
            # Find the table with matching table_id
            matching_table = next(
                (table for table in study.activation_tables if table.table_id == analysis.table_id),
                None
            )
            if not matching_table:
                available_tables = [t.table_id for t in study.activation_tables]
                raise ValueError(
                    f"Analysis {analysis_id} references table_id '{analysis.table_id}' which "
                    f"was not found in study {study.pmid}. Available tables: {available_tables}"
                )
            
            table_id = matching_table.table_id
            table_caption = matching_table.table_caption
            table_footer = matching_table.table_foot
        else:
            # If analysis has no table_id, generate a default
            table_id = f"{study.pmid}_default_table"
        
        # Build kwargs with required fields
        kwargs = {
            "analysis_id": analysis_id,
            "study_id": study.pmid,
            "table_id": table_id,
        }
        
        # Map of field names to their getter functions (lazy evaluation)
        field_getters = {
            "analysis_name": lambda: analysis.name,
            "analysis_description": lambda: analysis.description,
            "table_caption": lambda: table_caption,
            "table_footer": lambda: table_footer,
            "study_title": lambda: study.title,
            "study_abstract": lambda: study.abstract,
            "study_authors": lambda: study.authors,
            "study_journal": lambda: study.journal,
            "study_publication_date": lambda: study.publication_date,
            "study_fulltext": lambda: self._safe_get_study_fulltext(study),
        }
        
        # If metadata_fields is None, include all fields
        if metadata_fields is None:
            for field, getter in field_getters.items():
                kwargs[field] = getter()
        else:
            # Only include requested fields
            for field in metadata_fields:
                if field in field_getters:
                    kwargs[field] = field_getters[field]()
        
        # Create the metadata object
        metadata = AnalysisMetadata(**kwargs)
        
        return metadata

    def _safe_get_study_fulltext(self, study: Study) -> Optional[str]:
        """Best-effort full-text fetch for optional prompt fields."""
        try:
            return study.full_text
        except (ValueError, FileNotFoundError):
            return None
    
    def _build_study_analysis_group(
        self,
        study: Study,
        metadata_fields: List[str]
    ) -> 'StudyAnalysisGroup':
        """
        Build a StudyAnalysisGroup from a study for multi_analysis prompts.
        
        Args:
            study: Study to convert
            metadata_fields: List of metadata fields to include
            
        Returns:
            StudyAnalysisGroup with all analyses and tables
        """
        from .schema import StudyAnalysisGroup, TableMetadata
        
        # Build study-level metadata
        study_kwargs = {
            'study_id': study.pmid
        }
        
        if 'study_title' in metadata_fields:
            study_kwargs['study_title'] = study.title
        if 'study_abstract' in metadata_fields:
            study_kwargs['study_abstract'] = study.abstract
        if 'study_authors' in metadata_fields:
            study_kwargs['study_authors'] = study.authors
        if 'study_journal' in metadata_fields:
            study_kwargs['study_journal'] = study.journal
        if 'study_publication_date' in metadata_fields:
            study_kwargs['study_publication_date'] = study.publication_date
        if 'study_fulltext' in metadata_fields:
            study_kwargs['study_fulltext'] = self._safe_get_study_fulltext(
                study
            )
        
        # Build table metadata
        tables = []
        if study.activation_tables:
            for table in study.activation_tables:
                tables.append(TableMetadata(
                    table_id=table.table_id,
                    caption=table.table_caption,
                    footer=table.table_foot
                ))
        
        # Build analysis metadata for all analyses
        analyses = []
        for i, analysis in enumerate(study.analyses):
            # Create sanitized analysis ID
            analysis_id = sanitize_analysis_name(f"{study.pmid}_analysis_{i}")
            analysis_metadata = self._extract_analysis_metadata(
                study, analysis, analysis_id, metadata_fields
            )
            analyses.append(analysis_metadata)
        
        study_kwargs['tables'] = tables
        study_kwargs['analyses'] = analyses
        
        return StudyAnalysisGroup(**study_kwargs)

    def _are_cached_results_valid(self, cached_results: List[AnnotationDecision]) -> bool:
        """
        Check if cached results are still valid based on current configuration.
        
        Args:
            cached_results: List of cached annotation decisions
            
        Returns:
            True if cached results are valid, False otherwise
        """
        # Always return False to force reprocessing since we're changing to incremental approach
        # In a production environment, you might want more sophisticated validation
        return False
    
    def _load_cached_results(self, output_dir: str) -> List[AnnotationDecision]:
        """
        Load cached annotation results from file.
        
        Args:
            output_dir: Output directory
            
        Returns:
            List of annotation decisions or empty list if not found
        """
        try:
            cache_file = Path(output_dir) / "outputs" / "annotation_results.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to AnnotationDecision objects
                decisions = []
                for item in data:
                    # Handle datetime parsing
                    if 'timestamp' in item:
                        # Remove timestamp for now as it complicates deserialization
                        item.pop('timestamp', None)
                    
                    # Remove fields that might not be in the model
                    item.pop('timestamp', None)
                    
                    decision = AnnotationDecision(**item)
                    decisions.append(decision)
                
                return decisions
        except Exception as e:
            logger.warning(f"Failed to load cached annotation results: {e}")
        
        return []
    
    def _save_results_by_study(self, new_decisions: List[AnnotationDecision], output_dir: str, existing_results: List[AnnotationDecision]):
        """
        Save annotation results to cache file, organized by study.
        Complete results for each study overwrite any existing partial results.
        
        Args:
            new_decisions: List of new annotation decisions
            output_dir: Output directory
            existing_results: Existing cached results
        """
        try:
            output_path = Path(output_dir) / "outputs"
            output_path.mkdir(parents=True, exist_ok=True)
            cache_file = output_path / "annotation_results.json"
            
            # Load existing cache
            existing_data = []
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load existing cache: {e}")
                    existing_data = []
            
            # Convert existing data to AnnotationDecision objects for easier handling
            existing_decisions = []
            for item in existing_data:
                # Handle datetime parsing
                if 'timestamp' in item:
                    # Remove timestamp for now as it complicates deserialization
                    item.pop('timestamp', None)
                item.pop('timestamp', None)
                try:
                    decision = AnnotationDecision(**item)
                    existing_decisions.append(decision)
                except Exception as e:
                    logger.warning(f"Failed to parse existing decision: {e}")
            
            # Group new decisions by study_id
            new_decisions_by_study = {}
            for decision in new_decisions:
                if decision.study_id not in new_decisions_by_study:
                    new_decisions_by_study[decision.study_id] = []
                new_decisions_by_study[decision.study_id].append(decision)
            
            # Create a lookup for existing data by (study_id, analysis_id, annotation_name)
            existing_lookup = {(d.study_id, d.analysis_id, d.annotation_name): d 
                              for d in existing_decisions}
            
            # For each study with new decisions, replace only the annotations
            # that were reprocessed for that study.
            for study_id, study_decisions in new_decisions_by_study.items():
                updated_annotation_names = {
                    decision.annotation_name for decision in study_decisions
                }
                keep_system_annotations = {"all_analyses", "all_studies"}
                has_custom_updates = any(
                    name not in keep_system_annotations
                    for name in updated_annotation_names
                )
                existing_lookup = {
                    key: decision
                    for key, decision in existing_lookup.items()
                    if not (
                        decision.study_id == study_id
                        and (
                            decision.annotation_name in updated_annotation_names
                            or (
                                has_custom_updates
                                and decision.annotation_name
                                not in keep_system_annotations
                            )
                        )
                    )
                }
                
                # Add all new decisions for this study
                for decision in study_decisions:
                    key = (decision.study_id, decision.analysis_id, decision.annotation_name)
                    existing_lookup[key] = decision
            
            # Convert back to list of dictionaries for saving
            updated_data = []
            for decision in existing_lookup.values():
                try:
                    # Convert to dictionary, excluding timestamp to avoid JSON serialization issues
                    decision_dict = decision.model_dump(exclude={'timestamp'})
                    updated_data.append(decision_dict)
                except Exception as e:
                    logger.warning(f"Failed to convert decision to dict: {e}")
            
            with open(cache_file, 'w') as f:
                json.dump(updated_data, f, indent=2)
            
            logger.info(f"Saved {len(new_decisions)} new annotation results to {cache_file} (total: {len(updated_data)})")
        except Exception as e:
            logger.error(f"Failed to save annotation results: {e}")
