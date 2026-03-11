"""Main pipeline orchestrator for Autonima."""

import logging
import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import concurrent.futures
from tqdm import tqdm

from .config import ConfigManager
from .models.types import (
    PipelineConfig,
    PipelineResult,
    StudyStatus
)
from .search import PubMedSearch
from .screening import LLMScreener
from .retrieval import PubGetRetriever
from .retrieval.utils import (
        _apply_activation_tables_to_studies,
        _apply_analyses_to_studies,
        _map_pmids_to_text
    )
from .utils import log_error_with_debug
from .annotation.processor import AnnotationProcessor

logger = logging.getLogger(__name__)


def _atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written output files."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, file_path)


class AutonimaPipeline:
    """
    Main pipeline orchestrator for automated systematic reviews.

    This class coordinates the entire systematic review workflow:
    1. Literature search via PubMed
    2. Abstract screening with LLMs
    3. Full-text retrieval
    4. Full-text screening
    5. Output generation with PRISMA compliance
    """

    def __init__(
        self,
        config: PipelineConfig,
        num_workers: int = 1,
        force_reextract_incomplete_fulltext: bool = False
    ):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.num_workers = num_workers
        self.force_reextract_incomplete_fulltext = (
            force_reextract_incomplete_fulltext
        )
        self.results = PipelineResult(
            config=config,
            started_at=datetime.now()
        )
        self._search_engine = None
        self._abstract_screener = None
        self._fulltext_screener = None
        self._retriever = None

        # Ensure output directory exists
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Initialize pipeline components based on configuration."""
        # Initialize search engine
        if self.config.search.database.lower() == "pubmed":
            self._search_engine = PubMedSearch(
                self.config.search,
                output_dir=self.config.output.directory
            )
        else:
            raise ValueError(
                f"Unsupported database: {self.config.search.database}")

        # Initialize screening engine
        self._screener = LLMScreener(
            self.config.screening,
            output_dir=self.config.output.directory,
            num_workers=self.num_workers,
            force_reextract_incomplete_fulltext=(
                self.force_reextract_incomplete_fulltext
            )
        )
        
        # Initialize retrieval engine
        n_jobs = getattr(self.config.retrieval, 'n_jobs', 1)
        self._retriever = PubGetRetriever(n_jobs=n_jobs)

        # Save criteria mapping early in pipeline
        from .utils.criteria import save_criteria_mapping
        save_criteria_mapping(self.config, self.config.output.directory)

    async def run(self) -> PipelineResult:
        """
        Execute the complete systematic review pipeline.

        Returns:
            PipelineResult containing all results and metadata
        """
        # Get objective from screening configuration
        abstract_objective = self.config.screening.abstract.get('objective')
        fulltext_objective = self.config.screening.fulltext.get('objective')
        objective = abstract_objective or fulltext_objective
        logger.info(f"Starting Autonima pipeline: {objective}")

        try:
            # Phase 1: Literature Search
            await self._execute_search_phase()

            # Phase 2: Abstract Screening
            await self._execute_abstract_screening()

            # Phase 3: Full-text Retrieval
            await self._execute_retrieval_phase()

            # Phase 4: Full-text Screening
            await self._execute_fulltext_screening()
 
            # Phase 5: Coordinate Parsing
            await self._execute_coordinate_parsing()
 
            # Phase 6: Analysis Annotation
            await self._execute_annotation_phase()
 
            # Phase 7: Generate Outputs
            await self._execute_output_phase()

            # Complete pipeline
            self.results.completed_at = datetime.now()
            duration = self.results.completed_at - self.results.started_at
            logger.info(f"Pipeline completed in {duration}")

            return self.results

        except Exception as e:
            log_error_with_debug(logger, f"Pipeline failed: {e}")
            self.results.errors.append(str(e))
            raise

    async def _execute_search_phase(self):
        """Execute the literature search phase."""

        if not self._search_engine:
            raise RuntimeError("Search engine not initialized")

        # Execute search or load PMIDs
        studies = await self._search_engine.search(self.config.search.query)

        # Add studies to results
        self.results.studies.extend(studies)

        # Update execution stats
        self.results.execution_stats.update({
            "search_completed": datetime.now().isoformat(),
            "studies_found": len(studies),
            "search_engine": self.config.search.database,
            "search_query": self.config.search.query,
            "pmids_file": self.config.search.pmids_file,
            "pmids_list": self.config.search.pmids_list
        })

        # Save intermediary results
        output_dir = Path(self.config.output.directory) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        search_results_file = output_dir / "search_results.json"
        search_data = {
            "studies": [study.to_dict() for study in studies],
            "timestamp": datetime.now().isoformat()
        }
        with open(search_results_file, 'w') as f:
            import json
            json.dump(search_data, f, indent=2)

    async def _execute_abstract_screening(self):
        """Execute abstract screening phase."""

        # Check if abstract screening should be skipped
        abstract_skip = self.config.screening.abstract.get('skip_stage', False)
        if abstract_skip:
            logger.info("Abstract screening skipped (skip_stage=True)")
            # Mark all pending studies as included to pass to next stage
            for study in self.results.studies:
                if study.status == StudyStatus.PENDING:
                    study.status = StudyStatus.INCLUDED_ABSTRACT
            return

        # Get studies that need screening
        pending_studies = [
            s for s in self.results.studies if s.status == StudyStatus.PENDING]

        if not pending_studies:
            logger.info("No studies require abstract screening")
            return

        if not self._screener:
            raise RuntimeError("Screener not initialized")

        # Use LLM-based screening
        screening_results = await self._screener.screen_abstracts(
            pending_studies, num_workers=self.num_workers)

        # Apply screening results to studies
        for result in screening_results:
            study = next(
                (s for s in self.results.studies if s.pmid == result.study_id),
                None
                )
            if study:
                study.status = result.decision
                study.abstract_screening_reason = result.reason
                study.abstract_screening_score = result.confidence
                study.screened_at = datetime.now()

        # Add screening results to pipeline results
        self.results.abstract_screening_results.extend(screening_results)

        # Save intermediary results
        output_dir = Path(self.config.output.directory) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        screening_results_file = output_dir / "abstract_screening_results.json"
        screening_data = {
            "screening_results": [
                result.to_dict() for result in screening_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        _atomic_write_json(screening_results_file, screening_data)

        screened_count = len([
            s for s in self.results.studies
            if s.status != StudyStatus.PENDING
        ])
        logger.debug(
            f"Abstract screening completed: {screened_count} studies "
            "screened"
        )

    async def _execute_retrieval_phase(self):
        """Execute full-text retrieval phase."""

        # Determine which studies to retrieve based on load_excluded setting
        load_excluded = getattr(self.config.retrieval, 'load_excluded', False)
        
        if load_excluded:
            # Get ALL studies (both included and excluded from abstract screening)
            studies_to_process = [
                s for s in self.results.studies
                if s.status in [
                    StudyStatus.INCLUDED_ABSTRACT,
                    StudyStatus.EXCLUDED_ABSTRACT
                ]
            ]
            scope = "included + excluded"
        else:
            # Get only included studies that need full-text retrieval
            studies_to_process = [
                s for s in self.results.studies
                if s.status == StudyStatus.INCLUDED_ABSTRACT
            ]
            scope = "included only"

        for study in studies_to_process:
            study.retrieval_in_scope = True
            study.in_included_set = (
                study.status == StudyStatus.INCLUDED_ABSTRACT
            )

        pmids_set = set(
            [int(s.pmid) for s in studies_to_process if s.pmid.isdigit()]
        )
        
        logger.info(
            "Retrieval: starting for %s studies (%s)",
            len(studies_to_process),
            scope,
        )

        if not studies_to_process:
            logger.info("Retrieval: no studies require full-text retrieval")
            self.results.execution_stats["retrieval"] = {
                "scope": scope,
                "total_considered": 0,
                "local_found": 0,
                "retrieved_or_cached": 0,
                "missing_full_text": 0,
            }
            return

        # Check for existing full texts from user-provided sources
        studies_from_user_sources = []
        studies_to_retrieve = studies_to_process[:]

        full_text_sources = getattr(
            self.config.retrieval, 'full_text_sources', []
            )
        
        try:
            for i, full_text_config in enumerate(full_text_sources):
                logger.debug(
                    "Processing full text source %s/%s",
                    i + 1,
                    len(full_text_sources),
                )

                source_name = (
                    full_text_config.get("source_name")
                    or full_text_config.get("name")
                    or full_text_config.get("root_path")
                    or f"full_text_source_{i + 1}"
                )

                # Map PMIDs to text files in source
                text_paths, analyses, tables = _map_pmids_to_text(
                    **full_text_config,
                    pmids_to_include=pmids_set
                    )

                source_matches = []
                
                # Update studies with their full text paths
                for study in studies_to_retrieve[:]:
                    if int(study.pmid) in text_paths:
                        study.full_text_path = str(text_paths[int(study.pmid)])
                        study.full_text_source = str(source_name)
                        study.fulltext_available = True
                        source_matches.append(study)
                        studies_from_user_sources.append(study)
                        studies_to_retrieve.remove(study)

                # Apply analyses from coordinates to studies
                if analyses:
                    _apply_analyses_to_studies(
                        studies=source_matches,
                        id_to_analyses=analyses,
                        identifier_key="pmid",
                        identifier_type="str"
                    )

                # Apply activation tables to studies
                if tables:
                    _apply_activation_tables_to_studies(
                        studies=source_matches,
                        id_to_tables=tables,
                        identifier_key="pmid",
                        identifier_type="str"
                    )
                
                # Set full_text_output_dir for studies from user-provided sources
                # This must be set so they can load full text during screening
                output_dir = Path(self.config.output.directory)
                for study in studies_from_user_sources:
                    study.full_text_output_dir = str(output_dir)
             
        except Exception as e:
            log_error_with_debug(logger, 
                f"Failed to load from user-provided full text sources: {e}"
            )

        logger.info(
            "Retrieval: found %s studies in local full-text sources",
            len(studies_from_user_sources),
        )

        # Fetch PMCIDs for studies that will use PubGet (those without full_text_path)
        for study in studies_to_retrieve:
            if not study.full_text_source:
                study.full_text_source = "pubget"

        studies_needing_pmcid = [
            s for s in studies_to_retrieve if not s.pmcid
        ]
        
        if studies_needing_pmcid:
            logger.debug(
                "Fetching PMCIDs for %s studies", len(studies_needing_pmcid)
            )
            pmid_to_pmcid = await self._search_engine.fetch_pmcids(
                [s.pmid for s in studies_needing_pmcid]
            )
            
            # Update studies with their PMCIDs
            not_found = []
            for study in studies_needing_pmcid:
                if study.pmid in pmid_to_pmcid and pmid_to_pmcid[study.pmid]:
                    study.pmcid = pmid_to_pmcid[study.pmid]
                else:
                    not_found.append(study.pmid)

        # Use PubGet for retrieval
        if studies_to_retrieve:
            output_dir = Path(self.config.output.directory)
            retrieval_dir = output_dir / "retrieval"

            # Retrieve full-text articles
            try:
                _ = self._retriever.retrieve(
                    studies=studies_to_retrieve,
                    output_dir=retrieval_dir,
                    api_key=getattr(self.config.retrieval, 'api_key', None),
                    n_docs=getattr(self.config.retrieval, 'max_docs', None),
                    load_excluded=load_excluded
                )
                    
            except Exception as e:
                log_error_with_debug(logger, f"Full-text retrieval failed: {e}")

            # Validate retrieval
            self._retriever.validate_retrieval(studies_to_retrieve, retrieval_dir)

        # Save intermediary results
        output_dir = Path(self.config.output.directory)
        retrieval_results_file = output_dir / "outputs" / "fulltext_retrieval_results.json"
        retrieval_data = {
            "studies_with_fulltext": [
                {
                    "pmid": study.pmid,
                    "pmcid": study.pmcid,
                    "title": study.title,
                    "retrieved_at": (
                        study.retrieved_at.isoformat()
                        if study.retrieved_at else None
                    ),
                    "status": study.status.value,
                    "full_text_path": study.full_text_path,
                    "fulltext_available": study.fulltext_available
                }
                for study in self.results.studies
                if study.fulltext_available or study.pmcid
            ],
            "timestamp": datetime.now().isoformat()
        }
        with open(retrieval_results_file, 'w') as f:
            json.dump(retrieval_data, f, indent=2)

        retrieved_count = len([
            s for s in self.results.studies
            if s.fulltext_available
        ])
        unavailable_studies = [
            study for study in studies_to_process
            if not study.fulltext_available
        ]
        unavailable_count = len(unavailable_studies)

        logger.info(
            "Retrieval completed: %s texts available, %s missing full text",
            retrieved_count,
            unavailable_count,
        )
        self.results.execution_stats["retrieval"] = {
            "scope": scope,
            "total_considered": len(studies_to_process),
            "local_found": len(studies_from_user_sources),
            "retrieved_or_cached": retrieved_count,
            "missing_full_text": unavailable_count,
        }

    async def _execute_fulltext_screening(self):
        """Execute full-text screening phase."""

        # Check if fulltext screening should be skipped
        fulltext_skip = self.config.screening.fulltext.get('skip_stage', False)
        if fulltext_skip:
            logger.info("Full-text screening skipped (skip_stage=True)")
            # Mark all studies with status INCLUDED_ABSTRACT as INCLUDED_FULLTEXT to pass to next stage
            for study in self.results.studies:
                if study.status == StudyStatus.INCLUDED_ABSTRACT:
                    study.status = StudyStatus.INCLUDED_FULLTEXT
            return

        # Get studies with full text that need screening
        screenable_studies = [
            s for s in self.results.studies
            if s.fulltext_available and s.status == StudyStatus.INCLUDED_ABSTRACT
        ]

        if not screenable_studies:
            logger.info("No studies require full-text screening")
            return

        if not self._screener:
            raise RuntimeError("Screener not initialized")

        # Use LLM-based full-text screening
        screening_results = await self._screener.screen_fulltexts(
            screenable_studies, num_workers=self.num_workers
        )

        # Apply screening results to studies
        for result in screening_results:
            study = next(
                (s for s in self.results.studies if s.pmid == result.study_id),
                None
            )
            if study:
                study.status = result.decision
                study.fulltext_screening_reason = result.reason
                study.fulltext_screening_score = result.confidence
                study.screened_at = datetime.now()

        # Add screening results to pipeline results
        self.results.fulltext_screening_results.extend(screening_results)

        # Save intermediary results
        output_dir = Path(self.config.output.directory)
        fulltext_screening_results_file = (
            output_dir / "outputs" / "fulltext_screening_results.json"
        )
        fulltext_screening_data = {
            "screening_results": [
                result.to_dict() for result in screening_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        _atomic_write_json(
            fulltext_screening_results_file, fulltext_screening_data
        )

        final_count = len([
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED_FULLTEXT
        ])
        logger.debug(
            f"Full-text screening completed: {final_count} studies included"
        )
 
    async def _execute_coordinate_parsing(self):
        """Execute coordinate parsing phase."""
        # Check if coordinate parsing is enabled
        if not getattr(self.config.parsing, 'parse_coordinates', False):
            logger.debug("Coordinate parsing is disabled")
            self.results.execution_stats["coordinate_parsing"] = {
                "enabled": False,
                "studies_with_tables": 0,
                "tables_processed": 0,
            }
            return
    
        # Load cached coordinate parsing results
        await self._load_cached_coordinate_results()
 
        # Get studies with activation tables that were retrieved and don't already have parsed analyses
        studies_with_tables = [
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED_FULLTEXT and s.activation_tables and not any([a.parsed for a in s.analyses])
        ]

        if not studies_with_tables:
            logger.debug(
                "No studies with activation tables require coordinate parsing"
            )
            self.results.execution_stats["coordinate_parsing"] = {
                "enabled": True,
                "studies_with_tables": 0,
                "tables_processed": 0,
            }
            return
 
        try:
            from .coordinates import CoordinateProcessor
        except ImportError as e:
            logger.warning(f"Coordinate parsing module not available: {e}")
            return
 
        # Initialize the coordinate processor
        model = getattr(self.config.parsing, 'coordinate_model', 'gpt-4o-mini')
        processor = CoordinateProcessor(model=model)
 
        # Prepare all table processing jobs
        table_jobs = []
        for study in studies_with_tables:
            for i, table in enumerate(study.activation_tables):
                table_jobs.append((study, table, i))

        # For all studies_with_tables, clear existing analyses
        for study in studies_with_tables:
            study.analyses = []
 
        if not table_jobs:
            logger.debug("No tables to process")
            self.results.execution_stats["coordinate_parsing"] = {
                "enabled": True,
                "studies_with_tables": len(studies_with_tables),
                "tables_processed": 0,
            }
            return
 
        # Process tables with or without parallelization
        if self.num_workers <= 1 or len(table_jobs) <= 1:
            # Serial processing
            logger.debug("Using serial processing for coordinate parsing")
            processed_count = 0
            for study, table, table_index in table_jobs:
                try:
                    # Process a single table
                    table_analyses = processor.process_single_table(table)
                    
                    # Add the analyses to the study
                    study.analyses.extend(table_analyses)
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing table {table_index} for study {study.pmid}: {e}")
                    continue
        else:
            # Parallel processing over all tables
            logger.debug(
                f"Using {self.num_workers} workers for parallel "
                f"coordinate parsing of {len(table_jobs)} tables"
            )

            def process_single_table_job(job):
                study, table, table_index = job
                try:
                    # Process a single table
                    table_analyses = processor.process_single_table(table)
                    return study, table_analyses, table_index
                except Exception as e:
                    logger.warning(f"Error processing table {table_index} for study {study.pmid}: {e}")
                    return study, None, table_index
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(process_single_table_job, job)
                    for job in table_jobs
                ]
                results = [
                    future.result()
                    for future in tqdm(futures, total=len(futures))
                ]
            
            # Apply results to studies
            processed_count = 0
            for study, table_analyses, table_index in results:
                if table_analyses is not None:
                    study.analyses.extend(table_analyses)
                    processed_count += 1
 
        # Save coordinate parsing results
        await self._save_coordinate_parsing_results()
 
        logger.info(
            f"Coordinate parsing completed: {processed_count} tables processed from {len(studies_with_tables)} studies"
        )
        self.results.execution_stats["coordinate_parsing"] = {
            "enabled": True,
            "studies_with_tables": len(studies_with_tables),
            "tables_processed": processed_count,
        }
 
    async def _execute_annotation_phase(self):
        """Execute annotation phase for parsed analyses."""
        # Check if annotation is enabled
        if not getattr(self.config.annotation, 'enabled', True):
            logger.debug("Annotation phase is disabled")
            self.results.execution_stats["annotation"] = {
                "enabled": False,
                "decisions": 0,
            }
            return
        
        # Get INCLUDED studies with analyses for standard annotations
        included_studies = [
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED_FULLTEXT and s.analyses
        ]
        
        # Get ALL studies if create_all_from_search_annotation is enabled
        all_studies = None
        if getattr(
            self.config.annotation, 'create_all_from_search_annotation', False
        ):
            all_studies = [
                s for s in self.results.studies
                if s.analyses
            ]
        
        if not included_studies and not all_studies:
            logger.debug("No studies with parsed analyses found for annotation")
            self.results.execution_stats["annotation"] = {
                "enabled": True,
                "decisions": 0,
            }
            return
                
        # Load criteria mapping and inject it into annotation config
        from .utils.criteria import load_criteria_mapping
        criteria_mapping = load_criteria_mapping(self.config.output.directory)
        
        # Create a copy of the annotation config with criteria mapping injected
        from copy import deepcopy
        annotation_config = deepcopy(self.config.annotation)
        
        # Inject criteria mapping into annotation config
        if criteria_mapping and "annotation" in criteria_mapping:
            annotation_data = criteria_mapping["annotation"]
            
            # Inject global criteria mapping
            global_mapping = annotation_data.get("global", {})
            global_inclusion = global_mapping.get("inclusion", {})
            global_exclusion = global_mapping.get("exclusion", {})
            if global_mapping:
                # Update the annotation config's global criteria with mapping
                has_inclusion = (
                    hasattr(annotation_config, 'inclusion_criteria') and
                    annotation_config.inclusion_criteria
                )
                if has_inclusion:
                    annotation_config.inclusion_criteria = list(
                        global_inclusion.values()
                    )
                    annotation_config.exclusion_criteria = list(
                        global_exclusion.values()
                    )
            
            # Inject per-annotation criteria mapping
            if "annotations" in annotation_data:
                annotations_mapping = annotation_data["annotations"]
                for annotation in annotation_config.annotations:
                    if annotation.name not in annotations_mapping:
                        continue

                    local_mapping = annotations_mapping[annotation.name]
                    local_inclusion = local_mapping.get("inclusion", {})
                    local_exclusion = local_mapping.get("exclusion", {})

                    # Merge global + local IDs for validation while retaining
                    # dedicated blocks for prompt rendering.
                    annotation.criteria_mapping = {
                        "inclusion": {**global_inclusion, **local_inclusion},
                        "exclusion": {**global_exclusion, **local_exclusion},
                        "global_inclusion": global_inclusion,
                        "global_exclusion": global_exclusion,
                        "local_inclusion": local_inclusion,
                        "local_exclusion": local_exclusion,
                    }
        
        # Initialize the annotation processor with updated config and num_workers
        processor = AnnotationProcessor(annotation_config, num_workers=self.num_workers)
        
        # Process studies
        annotation_results = processor.process_studies(
            included_studies=included_studies,
            all_studies=all_studies,
            output_dir=self.config.output.directory
        )
        
        logger.info(
            f"Annotation phase completed: {len(annotation_results)} "
            "annotation decisions made"
        )
        self.results.execution_stats["annotation"] = {
            "enabled": True,
            "decisions": len(annotation_results),
        }
            
 
    async def _load_cached_coordinate_results(self):
        """Load cached coordinate parsing results."""
        try:
            output_dir = Path(self.config.output.directory)
            coordinate_cache_file = output_dir / "outputs" / "coordinate_parsing_results.json"
            
            if not coordinate_cache_file.exists():
                logger.debug("No cached coordinate parsing results found")
                return
            
            import json
            with open(coordinate_cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Apply cached results to studies
            cached_studies = {study_data['pmid']: study_data for study_data in cached_data.get('studies', [])}
            loaded_count = 0
            
            for study in self.results.studies:
                if study.pmid in cached_studies and not any(a.parsed for a in study.analyses):
                    study.analyses = []
                    cached_study = cached_studies[study.pmid]
                    # Load analyses from cached data
                    if 'analyses' in cached_study:
                        from .coordinates.schema import Analysis, CoordinatePoint, PointsValue
                        for analysis_data in cached_study['analyses']:
                            points = []
                            for point_data in analysis_data.get('points', []):
                                values = []
                                for value_data in point_data.get('values', []):
                                    values.append(PointsValue(
                                        value=value_data.get('value'),
                                        kind=value_data.get('kind')
                                    ))
                                points.append(CoordinatePoint(
                                    coordinates=point_data['coordinates'],
                                    space=point_data.get('space'),
                                    values=values or None
                                ))
                            study.analyses.append(Analysis(
                                name=analysis_data.get('name'),
                                description=analysis_data.get('description'),
                                points=points,
                                parsed=analysis_data.get('parsed', True),
                                table_id=analysis_data.get('table_id')
                            ))
                        loaded_count += 1
            
            logger.debug(
                f"Loaded cached coordinate parsing results for {loaded_count} studies"
            )
            
        except Exception as e:
            logger.warning(f"Failed to load cached coordinate parsing results: {e}")
 
    async def _save_coordinate_parsing_results(self):
        """Save coordinate parsing results to cache."""
        try:
            # Get studies with parsed analyses
            studies_with_analyses = [
                s for s in self.results.studies
                if s.status == StudyStatus.INCLUDED_FULLTEXT and s.analyses
            ]
            
            if not studies_with_analyses:
                return
            
            # Prepare data for saving
            cached_data = {
                "studies": [
                    {
                        "pmid": study.pmid,
                        "analyses": [
                            {
                                "name": analysis.name,
                                "description": analysis.description,
                                "table_id": analysis.table_id,
                                "points": [
                                    {
                                        "coordinates": point.coordinates,
                                        "space": point.space,
                                        "values": [
                                            {
                                                "value": value.value,
                                                "kind": value.kind
                                            }
                                            for value in point.values or []
                                        ]
                                    }
                                    for point in analysis.points
                                ]
                            }
                            for analysis in study.analyses
                        ]
                    }
                    for study in studies_with_analyses
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            output_dir = Path(self.config.output.directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            coordinate_cache_file = output_dir / "outputs" / "coordinate_parsing_results.json"
            
            import json
            with open(coordinate_cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
            
            logger.debug(
                f"Saved coordinate parsing results for {len(studies_with_analyses)} studies"
            )
            
        except Exception as e:
            logger.warning(f"Failed to save coordinate parsing results: {e}")
 
    async def _execute_output_phase(self):
        """Execute output generation phase."""

        # TODO: Implement comprehensive output generation
        # For now, generate basic statistics
        await self._generate_basic_outputs()
        
        # Generate NiMADS output if requested
        if getattr(self.config.output, 'nimads', False):
            await self._generate_nimads_output()

        logger.debug("Saved final results and statistics")

    def _resolve_full_text_path_for_report(
        self, study, output_dir: Path, missing_type: str
    ) -> str:
        """Resolve a full-text path reference for missing-fulltext reporting."""
        if study.full_text_path:
            return study.full_text_path

        if missing_type == "incomplete" and study.pmcid:
            return (
                str(output_dir / "retrieval" / "pubget_data" / "text.csv")
                + f" [pmcid={study.pmcid}]"
            )

        return ""

    def _build_missing_fulltext_rows(
        self, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Build consolidated rows for unavailable and incomplete full texts."""
        unavailable_fulltext = [
            s
            for s in self.results.studies
            if (
                s.retrieval_in_scope
                and not s.fulltext_available
                and s.status != StudyStatus.FULLTEXT_INCOMPLETE
            )
        ]
        incomplete_fulltext = [
            s
            for s in self.results.studies
            if s.status == StudyStatus.FULLTEXT_INCOMPLETE
        ]

        rows: List[Dict[str, Any]] = []
        for missing_type, studies in (
            ("unavailable", unavailable_fulltext),
            ("incomplete", incomplete_fulltext),
        ):
            for study in studies:
                rows.append(
                    {
                        "pmid": study.pmid,
                        "type": missing_type,
                        "source": study.full_text_source or "",
                        "full_text_path": self._resolve_full_text_path_for_report(
                            study=study,
                            output_dir=output_dir,
                            missing_type=missing_type,
                        ),
                        "in_included_set": (
                            study.in_included_set
                            if study.in_included_set is not None
                            else (
                                study.status == StudyStatus.INCLUDED_ABSTRACT
                            )
                        ),
                    }
                )

        rows.sort(key=lambda row: (row["type"], str(row["pmid"])))
        return rows

    async def _generate_basic_outputs(self):
        """Generate basic outputs and statistics."""
        # Calculate PRISMA statistics
        total_studies = len(self.results.studies)
        included_studies = [
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED_FULLTEXT
        ]
        excluded_abstract = [
            s for s in self.results.studies
            if s.status == StudyStatus.EXCLUDED_ABSTRACT
        ]
        excluded_fulltext = [
            s for s in self.results.studies
            if s.status == StudyStatus.EXCLUDED_FULLTEXT
        ]
        incomplete_fulltext = [
            s for s in self.results.studies
            if s.status == StudyStatus.FULLTEXT_INCOMPLETE
        ]
        abstract_screened = [
            s for s in self.results.studies
            if s.status != StudyStatus.PENDING
        ]
        fulltext_assessed = [
            s for s in self.results.studies
            if s.status in {
                StudyStatus.INCLUDED_FULLTEXT,
                StudyStatus.EXCLUDED_FULLTEXT,
                StudyStatus.FULLTEXT_INCOMPLETE,
            }
        ]

        prisma_stats = {
            "total_identified": total_studies,
            "abstract_screened": len(abstract_screened),
            "abstract_excluded": len(excluded_abstract),
            "fulltext_assessed": len(fulltext_assessed),
            "fulltext_excluded": len(excluded_fulltext),
            "fulltext_incomplete": len(incomplete_fulltext),
            "final_included": len(included_studies)
        }

        self.results.execution_stats.update({
            "output_completed": datetime.now().isoformat(),
            "prisma_stats": prisma_stats,
            "final_included_count": len(included_studies)
        })

        # Save final results with only included studies
        output_dir = Path(self.config.output.directory)
        outputs_dir = output_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        final_results_file = outputs_dir / "final_results.json"
        with open(final_results_file, 'w') as f:
            import json
            json.dump(self.results.to_dict(final_studies_only=True), f, indent=2)

        missing_fulltexts_file = outputs_dir / "missing_fulltexts.csv"
        missing_fulltext_rows = self._build_missing_fulltext_rows(output_dir)
        with open(missing_fulltexts_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pmid",
                    "type",
                    "source",
                    "full_text_path",
                    "in_included_set",
                ],
            )
            writer.writeheader()
            writer.writerows(missing_fulltext_rows)

        # Compatibility output for scripts that consume one-PMID-per-line text.
        missing_fulltexts_txt_file = outputs_dir / "missing_fulltexts.txt"
        with open(missing_fulltexts_txt_file, "w") as f:
            for row in missing_fulltext_rows:
                f.write(f"{row['pmid']}\n")

    async def _generate_nimads_output(self):
        """Generate NiMADS output for studies with parsed analyses."""
        # Determine which studies to export based on export_excluded_studies
        export_excluded = getattr(
            self.config.output, 'export_excluded_studies', False
        )
        
        if export_excluded:
            # Export ALL studies with analyses (INCLUDED and EXCLUDED)
            studies_with_analyses = [
                s for s in self.results.studies
                if s.analyses
            ]
            export_scope = "included and excluded"
        else:
            # Export only INCLUDED studies with analyses
            studies_with_analyses = [
                s for s in self.results.studies
                if s.status == StudyStatus.INCLUDED_FULLTEXT and s.analyses
            ]
            export_scope = "included only"
        
        if not studies_with_analyses:
            logger.info(
                "NiMADS export skipped: no studies with parsed analyses"
            )
            return
        
        try:
            # Import NiMADS models
            from .coordinates.nimads_models import (
                convert_to_nimads_studyset,
                create_default_annotation,
                create_annotations_from_results,
                sanitize_studyset_dict,
                sanitize_annotation_dict
            )
            
            # Create a studyset from the studies
            studyset_id = f"autonima_studyset_{self.results.started_at.strftime('%Y%m%d_%H%M%S')}"
            studyset = convert_to_nimads_studyset(
                studyset_id,
                studies_with_analyses,
                name="Autonima Generated Studyset"
            )
            
            # Save NiMADS studyset output using the to_dict method
            # Apply sanitization to ensure analysis names are clean
            output_dir = Path(self.config.output.directory)
            nimads_output_file = output_dir / "outputs" / "nimads_studyset.json"
            studyset_dict = sanitize_studyset_dict(studyset.to_dict())
            with open(nimads_output_file, 'w') as f:
                json.dump(studyset_dict, f, indent=2)
            
            # Create annotations based on annotation results
            # First, try to load annotation results from the annotation processor
            try:
                # Import the annotation processor to access its results
                from .annotation.processor import AnnotationProcessor
                annotation_processor = AnnotationProcessor(self.config.annotation)
                
                # Load cached annotation results
                annotation_results = annotation_processor._load_cached_results(
                    self.config.output.directory
                )
                
                if annotation_results:
                    # Create annotations from results
                    annotations = create_annotations_from_results(
                        studyset_id, studyset, annotation_results
                    )
                    
                    # Save all annotations to a single file
                    annotations_data = sanitize_annotation_dict(annotations.to_dict())
                    nimads_annotations_file = output_dir / "outputs" / "nimads_annotation.json"
                    with open(nimads_annotations_file, 'w') as f:
                        json.dump(annotations_data, f, indent=2)
                else:
                    # Create a default annotation if no results are available
                    annotation = create_default_annotation(studyset_id, studyset)
                    nimads_annotations_file = output_dir / "outputs" / "nimads_annotation.json"
                    annotation_dict = sanitize_annotation_dict(annotation.to_dict())
                    with open(nimads_annotations_file, 'w') as f:
                        json.dump([annotation_dict], f, indent=2)
            except Exception as annotation_error:
                logger.warning(f"Failed to create annotations from results: {annotation_error}")
                # Create a default annotation as fallback
                annotation = create_default_annotation(studyset_id, studyset)
                nimads_annotations_file = output_dir / "outputs" / "nimads_annotation.json"
                annotation_dict = sanitize_annotation_dict(annotation.to_dict())
                with open(nimads_annotations_file, 'w') as f:
                    json.dump([annotation_dict], f, indent=2)
            
            total_analyses = sum(len(s.analyses) for s in studies_with_analyses)
            logger.info(
                f"NiMADS export completed: {len(studies_with_analyses)} "
                f"studies, {total_analyses} analyses ({export_scope})"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate NiMADS output: {e}")
            # Don't raise the error, as this shouldn't stop the pipeline

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        # Get objective from screening configuration
        abstract_objective = self.config.screening.abstract.get('objective')
        fulltext_objective = self.config.screening.fulltext.get('objective')
        objective = abstract_objective or fulltext_objective
        
        return {
            "config": {
                "objective": objective,
                "database": self.config.search.database,
                "query": self.config.search.query,
                "max_results": self.config.search.max_results
            },
            "results": {
                "total_studies": len(self.results.studies),
                "included_studies": len([
                    s for s in self.results.studies
                    if s.status == StudyStatus.INCLUDED_FULLTEXT
                ]),
                "excluded_studies": len([
                    s for s in self.results.studies
                    if s.status in [
                        StudyStatus.EXCLUDED_ABSTRACT,
                        StudyStatus.EXCLUDED_FULLTEXT
                    ]
                ]),
                "abstract_screening_results": len(
                    self.results.abstract_screening_results),
                "fulltext_screening_results": len(
                    self.results.fulltext_screening_results),
                "total_screening_results": (
                    len(self.results.abstract_screening_results) +
                    len(self.results.fulltext_screening_results)
                )
            },
            "execution": {
                "started_at": self.results.started_at.isoformat(),
                "completed_at": (
                    self.results.completed_at.isoformat()
                    if self.results.completed_at else None
                ),
                "duration_seconds": (
                    self.results.completed_at - self.results.started_at
                ).total_seconds() if self.results.completed_at else None,
                "errors": self.results.errors
            }
        }


# Convenience function for running pipeline from config file
async def run_pipeline_from_config(
    config_path: str = None,
    config: PipelineConfig = None,
    num_workers: int = 1,
    force_reextract_incomplete_fulltext: bool = False
) -> PipelineResult:
    """
    Run pipeline from configuration file or config object.

    Args:
        config_path: Path to YAML configuration file
        config: Pipeline configuration object
        force_reextract_incomplete_fulltext: Re-run full-text screening for
            cached fulltext_incomplete studies

    Returns:
        Pipeline results
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")
        config_manager = ConfigManager()
        config = config_manager.load_from_file(config_path)

    pipeline = AutonimaPipeline(
        config,
        num_workers=num_workers,
        force_reextract_incomplete_fulltext=(
            force_reextract_incomplete_fulltext
        ),
    )
    return await pipeline.run()
