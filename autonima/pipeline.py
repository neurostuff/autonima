"""Main pipeline orchestrator for Autonima."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .config import ConfigManager
from .models.types import (
    PipelineConfig,
    PipelineResult,
    Study,
    StudyStatus
)
from .search import PubMedSearch
from .screening import LLMScreener

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

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.results = PipelineResult(
            config=config,
            started_at=datetime.now()
        )
        self._search_engine = None
        self._abstract_screener = None
        self._fulltext_screener = None

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
            exclusion_criteria=self.config.exclusion_criteria
        )

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
            logger.error(f"Pipeline failed: {e}")
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
            pending_studies)

        # Apply screening results to studies
        for result in screening_results:
            study = next(
                (s for s in self.results.studies if s.pmid == result.study_id),
                None
                )
            if study:
                study.status = result.decision
                study.screening_reason = result.reason
                study.abstract_screening_score = result.confidence
                study.screened_at = datetime.now()

        # Add screening results to pipeline results
        self.results.screening_results.extend(screening_results)

        # Save intermediary results
        output_dir = Path(self.config.output.directory)
        screening_results_file = output_dir / "abstract_screening_results.json"
        screening_data = {
            "screening_results": [result.to_dict() for result in screening_results],
            "timestamp": datetime.now().isoformat()
        }
        with open(screening_results_file, 'w') as f:
            import json
            json.dump(screening_data, f, indent=2)

        screened_count = len([s for s in self.results.studies if s.status != StudyStatus.PENDING])
        logger.info(f"Abstract screening completed: {screened_count} studies screened")
        logger.info(f"Abstract screening results saved to {screening_results_file}")

    async def _mock_abstract_screening(self, studies: List[Study]):
        """Mock abstract screening for development purposes."""
        # Simple keyword-based screening
        inclusion_keywords = ["fMRI", "functional magnetic resonance", "neuroimaging", "brain"]
        exclusion_keywords = ["animal", "mice", "rats", "review", "meta-analysis"]

        for study in studies:
            # Check for inclusion criteria
            includes_study = any(
                keyword.lower() in (study.abstract or "").lower()
                for keyword in inclusion_keywords
            )

            # Check for exclusion criteria
            excludes_study = any(
                keyword.lower() in (study.title + " " + (study.abstract or "")).lower()
                for keyword in exclusion_keywords
            )

            if excludes_study:
                study.status = StudyStatus.EXCLUDED
                study.screening_reason = "Contains exclusion keywords"
            elif includes_study:
                study.status = StudyStatus.INCLUDED
                study.screening_reason = "Contains inclusion keywords"
            else:
                study.status = StudyStatus.EXCLUDED
                study.screening_reason = "Does not meet inclusion criteria"

            study.screened_at = datetime.now()

    async def _execute_retrieval_phase(self):
        """Execute full-text retrieval phase."""
        logger.info("Starting full-text retrieval phase")

        # Get included studies that need full-text retrieval
        included_studies = [
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED and not s.full_text_path
        ]

        if not included_studies:
            logger.info("No studies require full-text retrieval")
            return

        # TODO: Implement actual full-text retrieval (PubGet, ACE)
        # For now, mock retrieval
        await self._mock_fulltext_retrieval(included_studies)

        # Save intermediary results
        output_dir = Path(self.config.output.directory)
        retrieval_results_file = output_dir / "fulltext_retrieval_results.json"
        retrieval_data = {
            "studies_with_fulltext": [
                {
                    "pmid": study.pmid,
                    "title": study.title,
                    "full_text_path": study.full_text_path,
                    "retrieved_at": study.retrieved_at.isoformat() if study.retrieved_at else None
                }
                for study in self.results.studies if study.full_text_path
            ],
            "timestamp": datetime.now().isoformat()
        }
        with open(retrieval_results_file, 'w') as f:
            import json
            json.dump(retrieval_data, f, indent=2)

        retrieved_count = len([s for s in self.results.studies if s.full_text_path])
        logger.info(f"Full-text retrieval completed: {retrieved_count} texts retrieved")
        logger.info(f"Full-text retrieval results saved to {retrieval_results_file}")

    async def _mock_fulltext_retrieval(self, studies: List[Study]):
        """Mock full-text retrieval for development purposes."""
        for study in studies:
            # Simulate successful retrieval for most studies
            study.full_text_path = f"downloads/{study.pmid}.pdf"
            study.retrieved_at = datetime.now()

    async def _execute_fulltext_screening(self):
        """Execute full-text screening phase."""
        logger.info("Starting full-text screening phase")

        # Get studies with full text that need screening
        screenable_studies = [
            s for s in self.results.studies
            if s.full_text_path and s.status == StudyStatus.INCLUDED
        ]

        if not screenable_studies:
            logger.info("No studies require full-text screening")
            return

        if not self._screener:
            raise RuntimeError("Screener not initialized")

        # Use LLM-based full-text screening
        screening_results = await self._screener.screen_fulltexts(screenable_studies)

        # Apply screening results to studies
        for result in screening_results:
            study = next((s for s in self.results.studies if s.pmid == result.study_id), None)
            if study:
                study.status = result.decision
                study.screening_reason = result.reason
                study.fulltext_screening_score = result.confidence
                study.screened_at = datetime.now()

        # Add screening results to pipeline results
        self.results.screening_results.extend(screening_results)

        # Save intermediary results
        output_dir = Path(self.config.output.directory)
        fulltext_screening_results_file = output_dir / "fulltext_screening_results.json"
        fulltext_screening_data = {
            "screening_results": [result.to_dict() for result in screening_results],
            "timestamp": datetime.now().isoformat()
        }
        with open(fulltext_screening_results_file, 'w') as f:
            import json
            json.dump(fulltext_screening_data, f, indent=2)

        final_count = len([
            s for s in self.results.studies
            if s.status == StudyStatus.INCLUDED
        ])
        logger.info(f"Full-text screening completed: {final_count} studies included")
        logger.info(f"Full-text screening results saved to {fulltext_screening_results_file}")

    async def _mock_fulltext_screening(self, studies: List[Study]):
        """Mock full-text screening for development purposes."""
        # Simulate more strict screening for full text
        for study in studies:
            # Mock confidence score
            confidence = 0.7 + (hash(study.pmid) % 100) / 100 * 0.3

            if confidence >= self.config.screening.fulltext.get('threshold', 0.8):
                study.status = StudyStatus.INCLUDED
                study.screening_reason = "Passed full-text screening"
                study.fulltext_screening_score = confidence
            else:
                study.status = StudyStatus.EXCLUDED
                study.screening_reason = "Failed full-text screening"
                study.fulltext_screening_score = confidence

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
        included_studies = [s for s in self.results.studies if s.status == StudyStatus.INCLUDED]
        excluded_studies = [s for s in self.results.studies if s.status == StudyStatus.EXCLUDED]

        prisma_stats = {
            "total_identified": total_studies,
            "abstract_screened": total_studies,
            "abstract_excluded": len(excluded_studies),
            "fulltext_assessed": len(included_studies),
            "fulltext_excluded": len(excluded_studies) - len([
                s for s in excluded_studies
                if "full-text" in (s.screening_reason or "")
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
                "screening_results": len(self.results.screening_results)
            },
            "execution": {
                "started_at": self.results.started_at.isoformat(),
                "completed_at": self.results.completed_at.isoformat() if self.results.completed_at else None,
                "duration_seconds": (
                    self.results.completed_at - self.results.started_at
                ).total_seconds() if self.results.completed_at else None,
                "errors": self.results.errors
            }
        }


# Convenience function for running pipeline from config file
async def run_pipeline_from_config(config_path: str = None, config: PipelineConfig = None) -> PipelineResult:
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

    pipeline = AutonimaPipeline(config)
    return await pipeline.run()