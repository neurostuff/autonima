"""Annotation processor for the pipeline."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .schema import AnnotationConfig, AnnotationDecision, AnalysisMetadata
from .client import AnnotationClient
from ..models.types import Study, StudyStatus

logger = logging.getLogger(__name__)


class AnnotationProcessor:
    """Processor for annotating analyses based on LLM decisions."""
    
    def __init__(self, config: AnnotationConfig):
        """
        Initialize the annotation processor.
        
        Args:
            config: Annotation configuration
        """
        self.config = config
        self.client = AnnotationClient()
        self.annotation_results: List[AnnotationDecision] = []
    
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
        # Load cached results if available
        cached_results = self._load_cached_results(output_dir)
        if cached_results:
            # Check if cached results are still valid
            if self._are_cached_results_valid(cached_results):
                logger.info(
                    f"Loaded {len(cached_results)} cached annotation "
                    "results"
                )
                self.annotation_results = cached_results
                return cached_results
            else:
                logger.info(
                    "Cached results are outdated, processing fresh "
                    "annotations"
                )
        
        if not included_studies:
            logger.info(
                "No INCLUDED studies with analyses found for annotation"
            )
            return []
        
        logger.info(
            f"Processing {len(included_studies)} INCLUDED studies with "
            "analyses for annotation"
        )
        
        # Process all analysis-annotation combinations
        all_decisions = []
        
        # Process the "all_analyses" annotation for INCLUDED studies
        if self.config.create_all_included_annotation:
            all_analyses_decisions = self._create_all_analyses_annotations(
                included_studies,
                annotation_name="all_analyses"
            )
            all_decisions.extend(all_analyses_decisions)
        
        # Process the "all_studies" annotation for ALL studies if enabled
        if self.config.create_all_from_search_annotation and all_studies:
            logger.info(
                f"Creating 'all_studies' annotation for "
                f"{len(all_studies)} studies (INCLUDED + EXCLUDED)"
            )
            all_studies_decisions = self._create_all_analyses_annotations(
                all_studies,
                annotation_name="all_studies"
            )
            all_decisions.extend(all_studies_decisions)
        
        # Process custom annotations on INCLUDED studies only
        if self.config.annotations:
            custom_decisions = self._process_custom_annotations(
                included_studies,
                self.config.model
            )
            all_decisions.extend(custom_decisions)
        
        # Save results to cache
        self._save_results(all_decisions, output_dir)
        self.annotation_results = all_decisions
        
        return all_decisions
    
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
                # Create a unique analysis ID
                analysis_id = f"{study.pmid}_analysis_{i}"
                
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
    
    def _process_custom_annotations(self, studies: List[Study], model: str) -> List[AnnotationDecision]:
        """
        Process custom annotations using LLM decisions.
        
        Args:
            studies: List of included studies with analyses
            model: LLM model to use for decisions
            
        Returns:
            List of annotation decisions
        """
        decisions = []
        total_combinations = sum(
            len(study.analyses) * len(self.config.annotations) 
            for study in studies
        )
        
        logger.info(f"Processing {total_combinations} analysis-annotation combinations")
        
        # Create all analysis-annotation pairs
        analysis_annotation_pairs = []
        for study in studies:
            for i, analysis in enumerate(study.analyses):
                # Create a unique analysis ID
                analysis_id = f"{study.pmid}_analysis_{i}"
                
                # Extract metadata for this analysis
                metadata = self._extract_analysis_metadata(study, analysis, analysis_id)
                
                # Create pairs for each annotation criteria
                for criteria in self.config.annotations:
                    analysis_annotation_pairs.append((metadata, criteria))
        
        # Process in parallel if configured
        if len(analysis_annotation_pairs) > 1:
            decisions = self._process_parallel(analysis_annotation_pairs, model)
        else:
            # Process sequentially
            # Update each criteria with the top-level metadata_fields if not already set
            updated_pairs = []
            for metadata, criteria in analysis_annotation_pairs:
                # If criteria doesn't have metadata_fields set, use the top-level ones
                if not criteria.metadata_fields and self.config.metadata_fields:
                    # Create a copy of the criteria with the top-level metadata_fields
                    from .schema import AnnotationCriteriaConfig
                    updated_criteria = AnnotationCriteriaConfig(
    name=criteria.name,
    description=criteria.description,
    inclusion_criteria=(self.config.inclusion_criteria + criteria.inclusion_criteria),
    exclusion_criteria=(self.config.exclusion_criteria + criteria.exclusion_criteria),
    metadata_fields=self.config.metadata_fields
)
                    updated_pairs.append((metadata, updated_criteria))
                else:
                    updated_pairs.append((metadata, criteria))
            
            for metadata, criteria in tqdm(updated_pairs, desc="Processing annotations"):
                decision = self.client.make_decision(metadata, criteria, model)
                decisions.append(decision)
        
        included_count = sum(1 for d in decisions if d.include)
        logger.info(f"Processed {len(decisions)} annotation decisions ({included_count} included)")
        
        return decisions

    
    def _process_parallel(self, pairs: List[tuple], model: str, max_workers: int = 4) -> List[AnnotationDecision]:
        """
        Process annotation decisions in parallel.
        
        Args:
            pairs: List of (metadata, criteria) tuples
            model: LLM model to use
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of annotation decisions
        """
        decisions = []
        
        # Update each criteria with the top-level metadata_fields if not already set
        updated_pairs = []
        for metadata, criteria in pairs:
            # If criteria doesn't have metadata_fields set, use the top-level ones
            if not criteria.metadata_fields and self.config.metadata_fields:
                # Create a copy of the criteria with the top-level metadata_fields
                from .schema import AnnotationCriteriaConfig
                updated_criteria = AnnotationCriteriaConfig(
    name=criteria.name,
    description=criteria.description,
    inclusion_criteria=(self.config.inclusion_criteria + criteria.inclusion_criteria),
    exclusion_criteria=(self.config.exclusion_criteria + criteria.exclusion_criteria),
    metadata_fields=self.config.metadata_fields
)
                updated_pairs.append((metadata, updated_criteria))
            else:
                updated_pairs.append((metadata, criteria))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(self._process_single_decision, metadata, criteria, model): (metadata, criteria)
                for metadata, criteria in updated_pairs
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_pair), total=len(updated_pairs), desc="Processing annotations"):
                try:
                    decision = future.result()
                    decisions.append(decision)
                except Exception as e:
                    metadata, criteria = future_to_pair[future]
                    logger.error(f"Error processing annotation for {criteria.name}: {e}")
                    # Create a default decision (exclude) in case of error
                    decision = AnnotationDecision(
                        annotation_name=criteria.name,
                        analysis_id=metadata.analysis_id,
                        study_id=metadata.study_id,
                        include=False,
                        reasoning=f"Error in processing: {str(e)}",
                        model_used=model
                    )
                    decisions.append(decision)
        
        return decisions
    
    def _process_single_decision(self, metadata: AnalysisMetadata, criteria: "AnnotationCriteriaConfig", model: str) -> AnnotationDecision:
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
    
    def _extract_analysis_metadata(self, study: Study, analysis: "Analysis", analysis_id: str) -> AnalysisMetadata:
        """
        Extract metadata for an analysis from a study.
        
        Args:
            study: Study containing the analysis
            analysis: Analysis to extract metadata for
            analysis_id: Unique ID for the analysis
            
        Returns:
            Analysis metadata
        """
        # Extract table information if available
        table_caption = None
        table_footer = None
        if study.activation_tables:
            # Use the first table's information for now
            # In the future, we might want to associate analyses with specific tables
            table_caption = study.activation_tables[0].table_caption
            table_footer = study.activation_tables[0].table_foot
        
        # Create the metadata object
        metadata = AnalysisMetadata(
            analysis_id=analysis_id,
            study_id=study.pmid,
            analysis_name=analysis.name,
            analysis_description=analysis.description,
            table_caption=table_caption,
            table_footer=table_footer,
            study_title=study.title,
            study_abstract=study.abstract,
            study_authors=study.authors,
            study_journal=study.journal,
            study_publication_date=study.publication_date
        )
        
        return metadata

    def _are_cached_results_valid(self, cached_results: List[AnnotationDecision]) -> bool:
        """
        Check if cached results are still valid based on current configuration.
        
        Args:
            cached_results: List of cached annotation decisions
            
        Returns:
            True if cached results are valid, False otherwise
        """
        try:
            # Check if we have the right number of annotations
            # (all_analyses + custom annotations)
            expected_annotation_count = 0
            if self.config.create_all_included_annotation:
                expected_annotation_count += 1
            if self.config.create_all_from_search_annotation:
                expected_annotation_count += 1
            expected_annotation_count += len(self.config.annotations)
            
            # Get unique annotation names from cached results
            cached_annotation_names = set(
                result.annotation_name for result in cached_results
            )
            
            # Check if we have the expected annotations
            if (self.config.create_all_included_annotation and
                    "all_analyses" not in cached_annotation_names):
                return False
            
            if (self.config.create_all_from_search_annotation and
                    "all_studies" not in cached_annotation_names):
                return False
            
            # Check if we have all custom annotations
            for criteria in self.config.annotations:
                if criteria.name not in cached_annotation_names:
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Error validating cached results: {e}")
            return False
    
    def _load_cached_results(self, output_dir: str) -> Optional[List[AnnotationDecision]]:
        """
        Load cached annotation results from file.
        
        Args:
            output_dir: Output directory
            
        Returns:
            List of annotation decisions or None if not found
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
        
        return None
    
    def _save_results(self, decisions: List[AnnotationDecision], output_dir: str):
        """
        Save annotation results to cache file.
        
        Args:
            decisions: List of annotation decisions
            output_dir: Output directory
        """
        try:
            output_path = Path(output_dir) / "outputs"
            output_path.mkdir(parents=True, exist_ok=True)
            cache_file = output_path / "annotation_results.json"
            
            # Convert to dictionaries, excluding timestamp to avoid JSON serialization issues
            data = [decision.model_dump(exclude={'timestamp'}) for decision in decisions]
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(decisions)} annotation results to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save annotation results: {e}")