"""PubGet integration for full-text article retrieval."""

import logging
import tempfile
import pandas as pd
from datetime import datetime
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
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing metadata for already downloaded PMCIDs
        existing_pmcids = set()
        data_dir = output_dir / "pubget_data"
        metadata_file = data_dir / "metadata.csv"
        
        if metadata_file.exists():
            try:
                existing_df = pd.read_csv(metadata_file)
                if 'pmcid' in existing_df.columns:
                    existing_pmcids = set(
                        existing_df['pmcid'].dropna().astype(int).tolist()
                    )
            except Exception as e:
                logger.warning(f"Could not read existing metadata: {e}")
        
        cached_studies = []
        studies_to_download = []

        for study in studies_with_pmcid:
            if study.pmcid and study.pmcid in existing_pmcids:
                cached_studies.append(study)
                study.status = StudyStatus.FULLTEXT_CACHED
            else:
                studies_to_download.append(study)
                
        if not studies_to_download:
            logger.info("All PMCIDs already downloaded, skipping retrieval")
            return studies
        
        logger.info(
            f"Retrieving full-text for {len(studies_to_download)} studies "
            f"(skipping {len(cached_studies)} already downloaded)"
        )
        
        # Extract PMCIDs for studies that need to be downloaded
        pmcids = [study.pmcid for study in studies_to_download if study.pmcid]
        # Add PMCID prefix if missing
        pmcids = [f"PMC{pmcid}" for pmcid in pmcids]
        
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

                # Check to see how many articles were downloaded
                # If no articles were downloaded, skip the rest of the steps
                if not download_dir or not Path(download_dir).exists():
                    return studies
                downloaded_files = list(Path(download_dir).rglob('*'))
                if len(downloaded_files) <= 1:
                    return studies
                                
                # Extract articles from bulk download
                logger.info("Extracting articles...")
                articles_dir, extract_exit_code = extract_articles(
                    articlesets_dir=download_dir,
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
                
                # Merge with existing data if it exists
                final_data_dir = output_dir / "pubget_data"
                if data_dir.exists():
                    if final_data_dir.exists():
                        # Merge with existing data
                        self._merge_pubget_data(data_dir, final_data_dir)
                    else:
                        # Move extracted data to output directory
                        import shutil
                        shutil.copytree(
                            data_dir, final_data_dir, dirs_exist_ok=True
                        )
                
                # Copy the original articles directory to preserve it in output
                if articles_dir.exists():
                    import shutil
                    articles_output_dir = final_data_dir / "articles"
                    if articles_output_dir.exists():
                        # Remove existing articles directory if it exists
                        shutil.rmtree(articles_output_dir)
                    # Copy the articles directory
                    shutil.copytree(articles_dir, articles_output_dir)
                
                logger.info(
                    f"Successfully retrieved {len(studies_to_download)} "
                    "articles"
                )
                
            except Exception as e:
                log_error_with_debug(logger, f"Error during retrieval: {e}")
                raise
        
        return studies

    def _merge_pubget_data(self, new_data_dir: Path, existing_data_dir: Path):
        """
        Merge new pubget data with existing data.
        
        Args:
            new_data_dir: Directory containing new pubget data
            existing_data_dir: Directory containing existing pubget data
        """
        import shutil
        
        # List of CSV files to merge
        csv_files = [
            "metadata.csv",
            "text.csv",
            "coordinates.csv",
            "coordinate_space.csv",
            "tables.csv",
            "authors.csv",
            "links.csv",
            "neurovault_collections.csv",
            "neurovault_images.csv"
        ]
        
        for csv_file in csv_files:
            new_file = new_data_dir / csv_file
            existing_file = existing_data_dir / csv_file
            
            if new_file.exists():
                if existing_file.exists():
                    # Merge the CSV files
                    try:
                        new_df = pd.read_csv(new_file)
                        existing_df = pd.read_csv(existing_file)
                        
                        # Concatenate and remove duplicates based on PMCID
                        if ('pmcid' in new_df.columns and
                                'pmcid' in existing_df.columns):
                            # Remove rows from existing data that are in new
                            # data
                            existing_df = existing_df[
                                ~existing_df['pmcid'].isin(new_df['pmcid'])
                            ]
                            # Concatenate the dataframes
                            merged_df = pd.concat(
                                [existing_df, new_df], ignore_index=True
                            )
                        else:
                            # If no PMCID column, just concatenate
                            merged_df = pd.concat(
                                [existing_df, new_df], ignore_index=True
                            )
                        
                        # Save merged data
                        merged_df.to_csv(existing_file, index=False)
                        logger.info(f"Merged {csv_file} with existing data")
                    except Exception as e:
                        logger.warning(f"Could not merge {csv_file}: {e}")
                        # Fall back to copying new file
                        shutil.copy2(new_file, existing_file)
                else:
                    # No existing file, just copy the new one
                    shutil.copy2(new_file, existing_file)
        
        # Copy any other files (like info.json)
        for other_file in new_data_dir.iterdir():
            if other_file.is_file() and other_file.name not in csv_files:
                existing_file = existing_data_dir / other_file.name
                shutil.copy2(other_file, existing_file)

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
        
        all_texts = data_dir / "text.csv"
        if not all_texts.exists():
            logger.warning("Extracted text file not found")
            return studies
        
        # Load first column 
        df = pd.read_csv(all_texts)
        retrieved_pmcid = set(df['pmcid'].dropna().astype(int).tolist())
        
        # Check which studies have full-text files
        for study in studies:
            if study.status == StudyStatus.FULLTEXT_CACHED:
                continue
            elif study.pmcid in retrieved_pmcid:
                study.status = StudyStatus.FULLTEXT_RETRIEVED
                study.retrieved_at = datetime.now()
            else:
                study.status = StudyStatus.FULLTEXT_UNAVAILABLE
        
        return studies