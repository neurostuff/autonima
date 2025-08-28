"""PubGet integration for full-text article retrieval."""

import logging
import tempfile
from pathlib import Path
from typing import List

from ..models.types import Study, StudyStatus
from .base import BaseRetriever
from ..utils import log_error_with_debug

try:
    from pubget import download_pmcids, extract_articles, extract_data_to_csv
    PUBGET_AVAILABLE = True
except ImportError:
    PUBGET_AVAILABLE = False
    logging.warning(
        "pubget not installed. Full-text retrieval will be disabled."
    )

logger = logging.getLogger(__name__)


class PubGetRetriever(BaseRetriever):
    """Retriever using PubGet to download full-text articles from
    PubMed Central."""

    def __init__(self, n_jobs: int = 1):
        """
        Initialize the PubGet retriever.
        
        Args:
            n_jobs: Number of parallel jobs for processing
        """
        if not PUBGET_AVAILABLE:
            raise ImportError(
                "pubget is not installed. Please install it with: "
                "pip install pubget"
            )
        
        self.n_jobs = n_jobs

    def retrieve(
        self,
        studies: List[Study],
        output_dir: Path,
        **kwargs
    ) -> List[Study]:
        """
        Retrieve full-text articles for studies using PubGet.
        
        Args:
            studies: List of studies that passed screening
            output_dir: Directory to store retrieved articles
            **kwargs: Additional parameters (api_key, n_docs, etc.)
            
        Returns:
            List of studies with updated full_text_path attributes
        """
        # Filter studies that have PMCID (needed for PubGet)
        studies_with_pmcid = [
            study for study in studies 
            if study.pmcid and study.status == StudyStatus.INCLUDED
        ]
        
        if not studies_with_pmcid:
            logger.info("No studies with PMCID found for retrieval")
            return studies
        
        logger.info(
            f"Retrieving full-text for {len(studies_with_pmcid)} studies"
        )
        
        # Extract PMCIDs
        pmcids = [study.pmcid for study in studies_with_pmcid if study.pmcid]
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use temporary directory for pubget processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Download articles using PMCIDs
                logger.info("Downloading articles from PubMed Central...")
                download_dir, download_exit_code = download_pmcids(
                    pmcids=pmcids,
                    data_dir=str(temp_path),
                    n_docs=kwargs.get('n_docs'),
                    api_key=kwargs.get('api_key')
                )
                
                if download_exit_code != 0:
                    logger.warning("Article download was incomplete")
                
                # Extract articles from bulk download
                logger.info("Extracting articles...")
                articles_dir, extract_exit_code = extract_articles(
                    articlesets_dir=download_dir / "articlesets",
                    output_dir=str(temp_path / "articles"),
                    n_jobs=self.n_jobs
                )
                
                if extract_exit_code != 0:
                    logger.warning("Article extraction was incomplete")
                
                # Extract data to CSV
                logger.info("Extracting article data...")
                data_dir, data_exit_code = extract_data_to_csv(
                    articles_dir=str(articles_dir),
                    output_dir=str(temp_path / "extracted_data"),
                    n_jobs=self.n_jobs
                )
                
                if data_exit_code != 0:
                    logger.warning("Data extraction was incomplete")
                
                # Move extracted data to output directory
                final_data_dir = output_dir / "pubget_data"
                if data_dir.exists():
                    import shutil
                    shutil.copytree(
                        data_dir, final_data_dir, dirs_exist_ok=True
                    )
                
                # Update studies with full-text paths
                self._update_study_paths(studies_with_pmcid, final_data_dir)
                
                logger.info(
                    f"Successfully retrieved {len(studies_with_pmcid)} "
                    "articles"
                )
                
            except Exception as e:
                log_error_with_debug(logger, f"Error during retrieval: {e}")
                raise
        
        return studies

    def _update_study_paths(self, studies: List[Study], data_dir: Path):
        """
        Update studies with paths to their full-text files.
        
        Args:
            studies: List of studies to update
            data_dir: Directory containing extracted data
        """
        # Read metadata to map PMCID to file paths
        metadata_file = data_dir / "metadata.csv"
        if not metadata_file.exists():
            logger.warning("Metadata file not found")
            return
        
        import pandas as pd
        try:
            _ = pd.read_csv(metadata_file)
        except Exception as e:
            log_error_with_debug(logger, f"Error reading metadata: {e}")
            return
        
        # Create mapping from PMCID to article directory
        pmcid_to_path = {}
        articles_dir = data_dir.parent / "articles"
        
        if articles_dir.exists():
            for subdir in articles_dir.iterdir():
                if subdir.is_dir():
                    for article_dir in subdir.iterdir():
                        if (article_dir.is_dir() and
                                article_dir.name.startswith("pmcid_")):
                            pmcid = article_dir.name.replace("pmcid_", "")
                            pmcid_to_path[pmcid] = article_dir
        
        # Update studies with full-text paths
        for study in studies:
            pmcid = study.metadata.get('pmcid')
            if pmcid and pmcid in pmcid_to_path:
                article_dir = pmcid_to_path[pmcid]
                xml_file = article_dir / "article.xml"
                if xml_file.exists():
                    study.full_text_path = str(xml_file)
                    logger.debug(
                        f"Updated full-text path for study {pmcid}"
                    )

    def validate_retrieval(
        self,
        studies: List[Study],
        output_dir: Path
    ) -> List[Study]:
        """
        Validate which studies have been successfully retrieved.
        
        Args:
            studies: List of studies to check
            output_dir: Directory where articles were retrieved
            
        Returns:
            List of studies with updated status
        """
        data_dir = Path(output_dir) / "pubget_data"
        if not data_dir.exists():
            logger.warning("PubGet data directory not found")
            return studies
        
        # Check which studies have full-text files
        for study in studies:
            if study.full_text_path and Path(study.full_text_path).exists():
                study.status = StudyStatus.FULLTEXT_RETRIEVED
            else:
                study.status = StudyStatus.FULLTEXT_UNAVAILABLE
        
        return studies