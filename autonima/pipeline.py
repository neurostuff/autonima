"""Main pipeline orchestrator for Autonima."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .config import ConfigManager
from .models.types import (
    PipelineConfig,
    PipelineResult,
    StudyStatus
)
from .search import PubMedSearch
from .screening import LLMScreener
from .retrieval import PubGetRetriever
from .utils import log_error_with_debug

logger = logging.getLogger(__name__)


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

    def __init__(self, config: PipelineConfig, num_workers: int = 1):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.num_workers = num_workers
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
            self._search_engine = PubMedSearch(self.config.search)
        else:
            raise ValueError(
                f"Unsupported database: {self.config.search.database}")

        # Initialize screening engine
        self._screener = LLMScreener(
            self.config.screening,
            inclusion_criteria=self.config.inclusion_criteria,
            exclusion_criteria=self.config.exclusion_criteria,
            output_dir=self.config.output.directory,
            num_workers=self.num_workers
        )
        
        # Initialize retrieval engine
        n_jobs = getattr(self.config.retrieval, 'n_jobs', 1)
        self._retriever = PubGetRetriever(n_jobs=n_jobs)

    async def run(self) -> PipelineResult:
        """
        Execute the complete systematic review pipeline.

        Returns:
            PipelineResult containing all results and metadata
        """
        logger.info(f"Starting Autonima pipeline: {self.config.objective}")

        try:
            # Phase 1: Literature Search
            await self._execute_search_phase()

            # Phase 2: Abstract Screening
            await self._execute_abstract_screening()

            # Phase 3: Full-text Retrieval
            await self._execute_retrieval_phase()

            # Phase 4: Full-text Screening
            await self._execute_fulltext_screening()

            # Phase 5: Generate Outputs
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
        logger.info("Starting literature search phase")

        if not self._search_engine:
            raise RuntimeError("Search engine not initialized")

        # Execute search
        studies = await self._search_engine.search(self.config.search.query)

        # Add studies to results
        self.results.studies.extend(studies)

        # Update execution stats
        self.results.execution_stats.update({
            "search_completed": datetime.now().isoformat(),
            "studies_found": len(studies),
            "search_engine": self.config.search.database,
            "search_query": self.config.search.query
        })

        # Save intermediary results
        output_dir = Path(self.config.output.directory)
        search_results_file = output_dir / "search_results.json"
        search_data = {
            "studies": [study.to_dict() for study in studies],
            "timestamp": datetime.now().isoformat()
        }
        with open(search_results_file, 'w') as f:
            import json
            json.dump(search_data, f, indent=2)

        logger.info(f"Search completed: found {len(studies)} studies")
        logger.info(f"Search results saved to {search_results_file}")

    async def _execute_abstract_screening(self):
        """Execute abstract screening phase."""
        logger.info("Starting abstract screening phase")

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
        output_dir = Path(self.config.output.directory)
        screening_results_file = output_dir / "abstract_screening_results.json"
        screening_data = {
            "screening_results": [
                result.to_dict() for result in screening_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        with open(screening_results_file, 'w') as f:
            import json
            json.dump(screening_data, f, indent=2)

        screened_count = len([
            s for s in self.results.studies
            if s.status != StudyStatus.PENDING
        ])
        logger.info(
            f"Abstract screening completed: {screened_count} studies "
            "screened"
        )
        logger.info(
            f"Abstract screening results saved to {screening_results_file}"
        )

    async def _execute_retrieval_phase(self):
        """Execute full-text retrieval phase."""
        logger.info("Starting full-text retrieval phase")

        # Get included studies that need full-text retrieval
        included_studies = [
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED and
            s.status not in [StudyStatus.FULLTEXT_RETRIEVED,
                             StudyStatus.FULLTEXT_CACHED]
        ]

        if not included_studies:
            logger.info("No studies require full-text retrieval")
            return

        if not self._retriever:
            raise RuntimeError("Retriever not initialized")

        # Fetch PMCIDs for included studies that don't have them
        studies_needing_pmcid = [
            s for s in included_studies if not s.pmcid
        ]
        
        if studies_needing_pmcid:
            logger.info(
                f"Fetching PMCIDs for {len(studies_needing_pmcid)} studies"
            )
            pmids = [s.pmid for s in studies_needing_pmcid]
            pmid_to_pmcid = await self._search_engine.fetch_pmcids(pmids)
            
            # Update studies with their PMCIDs
            not_found = []
            for study in studies_needing_pmcid:
                if study.pmid in pmid_to_pmcid and pmid_to_pmcid[study.pmid]:
                    study.pmcid = pmid_to_pmcid[study.pmid]
                else:
                    not_found.append(study.pmid)

            if not_found:
                logger.warning(
                    f"PMCIDs not found for {len(not_found)} studies."
                )

        # Use PubGet for actual retrieval
        output_dir = Path(self.config.output.directory)
        retrieval_dir = output_dir / "retrieval"

        # Retrieve full-text articles
        try:
            api_key = getattr(self.config.retrieval, 'api_key', None)
            n_docs = getattr(self.config.retrieval, 'max_docs', None)

            _ = self._retriever.retrieve(
                studies=included_studies,
                output_dir=retrieval_dir,
                api_key=api_key,
                n_docs=n_docs
            )
                   
        except Exception as e:
            log_error_with_debug(logger, f"Full-text retrieval failed: {e}")

        # Validate retrieval
        self._retriever.validate_retrieval(included_studies, retrieval_dir)

        # Save intermediary results
        retrieval_results_file = output_dir / "fulltext_retrieval_results.json"
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
                    "status": study.status.value
                }
                for study in self.results.studies
                if (study.status in [
                        StudyStatus.FULLTEXT_RETRIEVED,
                        StudyStatus.FULLTEXT_UNAVAILABLE,
                        StudyStatus.FULLTEXT_CACHED
                    ])
            ],
            "timestamp": datetime.now().isoformat()
        }
        with open(retrieval_results_file, 'w') as f:
            import json
            json.dump(retrieval_data, f, indent=2)

        retrieved_count = len([
            s for s in self.results.studies
            if s.status in [StudyStatus.FULLTEXT_RETRIEVED,
                            StudyStatus.FULLTEXT_CACHED]
        ])
        unavailable_count = len([
            s for s in self.results.studies
            if s.status == StudyStatus.FULLTEXT_UNAVAILABLE
        ])
        logger.info(
            f"Full-text retrieval completed: {retrieved_count} texts "
            f"retrieved, {unavailable_count} unavailable"
        )
        logger.info(
            f"Full-text retrieval results saved to {retrieval_results_file}"
        )

    async def _execute_fulltext_screening(self):
        """Execute full-text screening phase."""
        logger.info("Starting full-text screening phase")

        # Get studies with full text that need screening
        screenable_studies = [
            s for s in self.results.studies
            if s.status in [StudyStatus.FULLTEXT_RETRIEVED,
                            StudyStatus.FULLTEXT_CACHED]
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
            output_dir / "fulltext_screening_results.json"
        )
        fulltext_screening_data = {
            "screening_results": [
                result.to_dict() for result in screening_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        with open(fulltext_screening_results_file, 'w') as f:
            import json
            json.dump(fulltext_screening_data, f, indent=2)

        final_count = len([
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED
        ])
        logger.info(
            f"Full-text screening completed: {final_count} studies included"
        )
        logger.info(
            f"Full-text screening results saved to "
            f"{fulltext_screening_results_file}"
        )

    async def _execute_output_phase(self):
        """Execute output generation phase."""
        logger.info("Starting output generation phase")

        # TODO: Implement comprehensive output generation
        # For now, generate basic statistics
        await self._generate_basic_outputs()

        logger.info("Output generation completed")

    async def _generate_basic_outputs(self):
        """Generate basic outputs and statistics."""
        # Calculate PRISMA statistics
        total_studies = len(self.results.studies)
        included_studies = [
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED
        ]
        excluded_studies = [
            s for s in self.results.studies
            if s.status == StudyStatus.EXCLUDED
        ]

        prisma_stats = {
            "total_identified": total_studies,
            "abstract_screened": total_studies,
            "abstract_excluded": len(excluded_studies),
            "fulltext_assessed": len(included_studies),
            "fulltext_excluded": len(excluded_studies) - len([
                s for s in excluded_studies
                if "full-text" in (s.fulltext_screening_reason or "")
            ]),
            "final_included": len(included_studies)
        }

        self.results.execution_stats.update({
            "output_completed": datetime.now().isoformat(),
            "prisma_stats": prisma_stats,
            "final_included_count": len(included_studies)
        })

        # Save final results
        output_dir = Path(self.config.output.directory)
        final_results_file = output_dir / "final_results.json"
        with open(final_results_file, 'w') as f:
            import json
            json.dump(self.results.to_dict(), f, indent=2)

        logger.info(f"Final results saved to {final_results_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "config": {
                "objective": self.config.objective,
                "database": self.config.search.database,
                "query": self.config.search.query,
                "max_results": self.config.search.max_results
            },
            "results": {
                "total_studies": len(self.results.studies),
                "included_studies": len([
                    s for s in self.results.studies
                    if s.status == StudyStatus.INCLUDED
                ]),
                "excluded_studies": len([
                    s for s in self.results.studies
                    if s.status == StudyStatus.EXCLUDED
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
    num_workers: int = 1
) -> PipelineResult:
    """
    Run pipeline from configuration file or config object.

    Args:
        config_path: Path to YAML configuration file
        config: Pipeline configuration object

    Returns:
        Pipeline results
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")
        config_manager = ConfigManager()
        config = config_manager.load_from_file(config_path)

    pipeline = AutonimaPipeline(config, num_workers=num_workers)
    return await pipeline.run()