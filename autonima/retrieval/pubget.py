"""PubGet integration for full-text article retrieval."""

import logging
import tempfile
import io
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List

from ..models.types import Study, StudyStatus
from .base import BaseRetriever
from ..utils import log_error_with_debug
from .utils import (
    load_activation_table_map,
    _apply_activation_tables_to_studies,
    _apply_analyses_to_studies
)
from pubget import download_pmcids, extract_articles, extract_data_to_csv

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
        self.n_jobs = n_jobs

    def _run_pubget_call(self, func, **kwargs):
        """Run pubget call while silencing noisy stdout/stderr unless verbose."""
        if logger.isEnabledFor(logging.DEBUG):
            return func(**kwargs)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = func(**kwargs)

        suppressed_output = (
            f"{stdout_buffer.getvalue()}\n{stderr_buffer.getvalue()}".strip()
        )
        if suppressed_output:
            logger.debug("Suppressed pubget output:\n%s", suppressed_output)

        return result

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
            **kwargs: Additional parameters (api_key, n_docs, load_excluded, etc.)
            
        Returns:
            List of studies with updated full_text_path attributes
        """
        # Determine which studies to retrieve based on load_excluded setting
        load_excluded = kwargs.get('load_excluded', False)
        
        if load_excluded:
            # Filter studies with PMCID (both included and excluded from abstract)
            studies_with_pmcid = [
                study for study in studies
                if study.pmcid and study.status in [
                    StudyStatus.INCLUDED_ABSTRACT,
                    StudyStatus.EXCLUDED_ABSTRACT
                ]
            ]
        else:
            # Filter studies that have PMCID and are included only
            studies_with_pmcid = [
                study for study in studies
                if study.pmcid and study.status == StudyStatus.INCLUDED_ABSTRACT
            ]
        
        if not studies_with_pmcid:
            logger.debug("No studies with PMCID found for retrieval")
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
                # Don't change status - just mark as available
                study.fulltext_available = True
                study.full_text_source = "pubget"
            else:
                studies_to_download.append(study)
                
        if not studies_to_download:
            logger.info(
                "PubGet retrieval: 0 to download (%s cached)",
                len(cached_studies),
            )
            return studies
        
        logger.info(
            f"PubGet retrieval: downloading {len(studies_to_download)} studies "
            f"({len(cached_studies)} cached)"
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
                logger.debug("Downloading articles from PubMed Central...")
                download_dir, download_exit_code = self._run_pubget_call(
                    download_pmcids,
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
                logger.debug("Extracting articles...")
                articles_dir, extract_exit_code = self._run_pubget_call(
                    extract_articles,
                    articlesets_dir=download_dir,
                    output_dir=str(temp_path / "articles"),
                    n_jobs=self.n_jobs
                )
                
                if extract_exit_code != 0:
                    logger.warning("Article extraction was incomplete")
                
                # Extract data to CSV
                logger.debug("Extracting article data...")
                data_dir, data_exit_code = self._run_pubget_call(
                    extract_data_to_csv,
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
                    # Recursively merge the new articles directory with existing one
                    if articles_output_dir.exists():
                        self._merge_directories(articles_dir, articles_output_dir)
                    else:
                        # Copy the articles directory to the pubget_data directory
                        shutil.copytree(articles_dir, articles_output_dir)
                
                logger.debug(
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
        
        # Also handle the table data files
        table_data_dir = existing_data_dir / "table_data"
        new_table_data_dir = new_data_dir / "table_data"
        
        if new_table_data_dir.exists():
            if not table_data_dir.exists():
                table_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all table data files
            import shutil
            for file in new_table_data_dir.iterdir():
                if file.is_file():
                    dest_file = table_data_dir / file.name
                    shutil.copy2(file, dest_file)
        
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
                        logger.debug(f"Merged {csv_file} with existing data")
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

    def _merge_directories(self, src: Path, dest: Path):
        """
        Recursively merge source directory into destination directory.
        
        Args:
            src: Source directory path
            dest: Destination directory path
        """
        import shutil
        
        if not src.exists():
            return
            
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
            
        for item in src.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                # Recursively merge subdirectories
                self._merge_directories(item, dest_item)
            else:
                # Copy files, overwriting if they exist
                if dest_item.exists():
                    dest_item.unlink()
                shutil.copy2(item, dest_item)

    def _process_coordinate_space(self, data_dir: Path, studies: List[Study]) -> List[Study]:
        """
        Process coordinate_space.csv and store the coordinate space for each study.
        
        Args:
            data_dir: Directory containing pubget data files
            studies: List of studies to process
            
        Returns:
            List of studies with updated coordinate_space
        """
        try:
            # Check if coordinate_space.csv exists
            coord_space_file = data_dir / "coordinate_space.csv"
            if not coord_space_file.exists():
                logger.debug(
                    "Coordinate space file not found, skipping coordinate space processing"
                )
                return studies
            
            # Load coordinate space data
            coord_space_df = pd.read_csv(coord_space_file)
            
            # Create a mapping of pmcid to coordinate space
            pmcid_to_space = {}
            for _, row in coord_space_df.iterrows():
                pmcid = row['pmcid']
                space = row['coordinate_space']
                # Only store valid space values (MNI, TAL, or null)
                if space in ['MNI', 'TAL']:
                    pmcid_to_space[pmcid] = space
                elif pd.isna(space) or space == 'null':
                    pmcid_to_space[pmcid] = None
                # For other values, we'll handle them during sanitization
            
            # Update studies with coordinate space
            for study in studies:
                if study.pmcid and study.pmcid in pmcid_to_space:
                    study.coordinate_space = pmcid_to_space[study.pmcid]
            
            logger.debug(
                f"Processed coordinate space for {len(pmcid_to_space)} studies"
            )
            
        except Exception as e:
            logger.warning(f"Error processing coordinate space: {e}")
        
        return studies

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
        
        # Set full_text_output_dir for all studies so they can load full text
        # This must be done here (in retrieval phase) rather than in screening
        # because cached screening results also need access to full text
        for study in studies:
            study.full_text_output_dir = str(output_dir)
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
        
        pmcid_to_analyses, pmcid_to_tables = load_activation_table_map(
            processed_data_path=data_dir,
            filter_by_coordinates=True,
            identifier_key="pmcid",
        )

        # Apply analyses from coordinates to studies
        if pmcid_to_analyses:
            _apply_analyses_to_studies(
                studies=studies,
                id_to_analyses=pmcid_to_analyses,
                clear_existing=False,
                identifier_key="pmcid",
            )

        # Apply activation tables to studies
        if pmcid_to_tables:
            _apply_activation_tables_to_studies(
                studies=studies,
                id_to_tables=pmcid_to_tables,
                clear_existing=False,
                identifier_key="pmcid",
            )
        
        # Check which studies have full-text files
        for study in studies:
            if study.pmcid in retrieved_pmcid:
                study.fulltext_available = True
                study.retrieved_at = datetime.now()
                study.full_text_source = "pubget"
            # If not retrieved, fulltext_available remains False (default)
        
        return studies
