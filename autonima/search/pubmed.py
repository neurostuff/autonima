"""PubMed search engine implementation using NCBI Entrez API."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

from Bio import Entrez
import requests

from .base import SearchEngine
from ..models.types import Study, StudyStatus, SearchConfig
from ..utils import log_error_with_debug


logger = logging.getLogger(__name__)


class PubMedSearch(SearchEngine):
    """PubMed search engine using NCBI Entrez API."""

    def __init__(self, config: SearchConfig, output_dir: str = "results"):
        """Initialize PubMed search engine.
        
        Args:
            config: Search configuration
            output_dir: Output directory for saving results
        """
        super().__init__(config)
        if config.email:
            Entrez.email = config.email  # Required by NCBI
        else:
            logger.warning(
                "No email address provided for NCBI API access. "
                "Please consider adding your email to the configuration "
                "to comply with NCBI usage guidelines and ensure better "
                "API access reliability."
            )
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.max_retries = 3
        self.retry_delay = 1.0
        self.result_dir = output_dir

    async def search(self, query: str) -> List[Study]:
        """
        Execute PubMed search and return list of studies.

        Args:
            query: PubMed search query (ignored if pmids_file or pmids_list is provided)

        Returns:
            List of Study objects
        """
        try:
            # Check if we are using PMIDs list or file
            if self.config.pmids_file or self.config.pmids_list:
                # Use PMIDs from file or list
                pmids = self._load_pmids()
                logger.info(f"Using {len(pmids)} PMIDs from config")
            else:
                # Build complete query
                full_query = self.build_query(query)
                pmids = await self._execute_search(full_query)
                logger.info(f"Found {len(pmids)} potential studies from search")

            if not pmids:
                return []

            # Load existing search results for caching
            cached_studies = self._load_cached_search_results()
            cached_pmids = {study.pmid for study in cached_studies}
            
            # Identify PMIDs that need to be fetched
            new_pmids = [pmid for pmid in pmids if pmid not in cached_pmids]
            logger.info(f"Found meta-data for {len(cached_studies)} cached studies, {len(new_pmids)} new studies to fetch")

            # Fetch details for new PMIDs only
            new_studies = []
            if new_pmids:
                new_studies = await self._fetch_study_details(new_pmids)
                logger.info(f"Successfully retrieved {len(new_studies)} new studies")

            # Combine cached and new studies
            studies = cached_studies + new_studies

            return studies

        except Exception as e:
            log_error_with_debug(logger, f"Error during PubMed search: {e}")
            raise

    def _load_pmids(self) -> List[str]:
        """Load PMIDs from file or list in config."""
        if self.config.pmids_list:
            return self.config.pmids_list
        
        if self.config.pmids_file:
            try:
                with open(self.config.pmids_file, 'r') as f:
                    pmids = [line.strip() for line in f if line.strip()]
                return pmids
            except Exception as e:
                logger.error(f"Failed to read PMIDs file: {e}")
                raise
        
        return []

    def _load_cached_search_results(self) -> List[Study]:
        """
        Load existing search results from search_results.json for caching.
        
        Returns:
            List of cached Study objects
        """
        try:
            import json
            from pathlib import Path
            
            # Use the provided output directory
            search_results_file = Path(self.result_dir) / "outputs" / "search_results.json"
            
            if not search_results_file.exists():
                logger.info(
                    f"No existing search results file found at "
                    f"{search_results_file} for caching"
                )
                return []
            
            with open(search_results_file, 'r') as f:
                data = json.load(f)
            
            studies_data = data.get("studies", [])
            cached_studies = []
            
            for study_data in studies_data:
                try:
                    # Convert dictionary back to Study object
                    # Handle datetime fields
                    retrieved_at = None
                    if study_data.get("retrieved_at"):
                        retrieved_at = datetime.fromisoformat(
                            study_data.get("retrieved_at")
                        )
                    
                    screened_at = None
                    if study_data.get("screened_at"):
                        screened_at = datetime.fromisoformat(
                            study_data.get("screened_at")
                        )
                    
                    study = Study(
                        pmid=study_data.get("pmid", ""),
                        title=study_data.get("title", ""),
                        abstract=study_data.get("abstract", ""),
                        authors=study_data.get("authors", []),
                        journal=study_data.get("journal", ""),
                        publication_date=study_data.get(
                            "publication_date", ""
                        ),
                        doi=study_data.get("doi"),
                        keywords=study_data.get("keywords", []),
                        status=StudyStatus(
                            study_data.get("status", "pending")
                        ),
                        abstract_screening_reason=study_data.get(
                            "abstract_screening_reason"
                        ),
                        fulltext_screening_reason=study_data.get(
                            "fulltext_screening_reason"
                        ),
                        metadata=study_data.get("metadata", {}),
                        abstract_screening_score=study_data.get(
                            "abstract_screening_score"
                        ),
                        fulltext_screening_score=study_data.get(
                            "fulltext_screening_score"
                        ),
                        retrieved_at=retrieved_at,
                        screened_at=screened_at,
                        pmcid=study_data.get("pmcid")
                    )
                    cached_studies.append(study)
                except Exception as e:
                    logger.warning(f"Failed to load cached study: {e}")
                    continue
            
            return cached_studies
            
        except Exception as e:
            logger.warning(f"Failed to load cached search results: {e}")
            return []

    async def _execute_search(self, query: str) -> List[str]:
        """
        Execute PubMed search and return list of PMIDs.

        Args:
            query: Search query

        Returns:
            List of PubMed IDs
        """
        logger.info(f"Executing search with query: {query}")
        
        try:
            # Use Entrez.esearch to search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=self.config.max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results.get("IdList", [])
            
            return pmids
            
        except Exception as e:
            log_error_with_debug(logger, f"Error executing PubMed search: {e}")
            raise

    async def _fetch_study_details(self, pmids: List[str]) -> List[Study]:
        """
        Fetch detailed information for list of PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Study objects with full details
        """
        studies = []

        # Process PMIDs in batches to avoid overwhelming the API
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            try:
                batch_studies = await self._fetch_study_batch(batch_pmids)
                studies.extend(batch_studies)
                # Rate limiting
                await asyncio.sleep(0.5)
            except Exception as e:
                msg = (f"Failed to fetch details for batch starting "
                       f"at index {i}: {e}")
                logger.warning(msg)
                continue

        return studies

    async def _fetch_study_batch(self, pmids: List[str]) -> List[Study]:
        """
        Fetch details for a batch of PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Study objects
        """
        if not pmids:
            return []

        try:
            # Use Entrez.efetch to get detailed information
            handle = Entrez.efetch(
                db="pubmed",
                id=pmids,
                rettype="xml",
                retmode="text"
            )
            xml_data = handle.read()
            handle.close()
            
            # Parse XML data
            studies = self._parse_pubmed_xml(xml_data)
            return studies
            
        except Exception as e:
            log_error_with_debug(logger, f"Error fetching study batch: {e}")
            raise

    def _parse_pubmed_xml(self, xml_data: str) -> List[Study]:
        """
        Parse PubMed XML data and extract study information.

        Args:
            xml_data: XML string from Entrez.efetch

        Returns:
            List of Study objects
        """
        studies = []
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    study = self._parse_single_article(article)
                    if study:
                        studies.append(study)
                except Exception as e:
                    logger.warning(f"Error parsing individual article: {e}")
                    continue
                    
        except Exception as e:
            log_error_with_debug(logger, f"Error parsing PubMed XML: {e}")
            raise
            
        return studies

    def _parse_single_article(self, article) -> Optional[Study]:
        """
        Parse a single PubMed article XML element.

        Args:
            article: XML element representing a PubMed article

        Returns:
            Study object or None if parsing failed
        """
        try:
            # Extract PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            
            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = (title_elem.text if title_elem is not None 
                     else "Unknown Title")
            
            # Extract abstract
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author_elem in article.findall(".//Author"):
                last_name = author_elem.find("LastName")
                first_name = author_elem.find("ForeName")
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
                elif last_name is not None:
                    authors.append(last_name.text)
            
            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = (journal_elem.text if journal_elem is not None 
                       else "Unknown Journal")
            
            # Extract publication date
            pub_date = "Unknown Date"
            pub_date_elem = article.find(".//PubDate")
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                month_elem = pub_date_elem.find("Month")
                day_elem = pub_date_elem.find("Day")
                
                year = year_elem.text if year_elem is not None else ""
                month = month_elem.text if month_elem is not None else ""
                day = day_elem.text if day_elem is not None else ""
                
                pub_date = f"{year} {month} {day}".strip()
            
            # Extract DOI
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # Extract keywords
            keywords = []
            for keyword_elem in article.findall(".//Keyword"):
                if keyword_elem.text:
                    keywords.append(keyword_elem.text)
            
            # Create Study object
            study = Study(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                keywords=keywords,
                status=StudyStatus.PENDING,
                retrieved_at=datetime.now()
            )
            
            return study
            
        except Exception as e:
            msg = (f"Error parsing article with PMID "
                   f"{pmid if 'pmid' in locals() else 'unknown'}: {e}")
            logger.warning(msg)
            return None

    async def _fetch_single_study(self, pmid: str) -> Optional[Study]:
        """
        Fetch details for a single PMID.

        Args:
            pmid: PubMed ID

        Returns:
            Study object or None if failed
        """
        logger.debug(f"Fetching details for {pmid}")
        
        try:
            # Use Entrez.efetch to get detailed information
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="xml",
                retmode="text"
            )
            xml_data = handle.read()
            handle.close()
            
            # Parse XML data
            studies = self._parse_pubmed_xml(xml_data)
            return studies[0] if studies else None
            
        except Exception as e:
            log_error_with_debug(
                logger, f"Error fetching details for {pmid}: {e}"
            )
            return None

    def get_search_info(self) -> Dict[str, Any]:
        """Get information about the PubMed search configuration."""
        return {
            "engine": "pubmed",
            "api_url": self.base_url,
            "max_results": self.config.max_results,
            "date_from": self.config.date_from,
            "date_to": self.config.date_to,
            "email": self.config.email,
            "pmids_file": self.config.pmids_file,
            "pmids_list": self.config.pmids_list
        }

    async def _make_request_with_retry(
        self,
        url: str,
        params: Dict[str, Any],
        response_type: str = "json"
    ):
        """
        Make HTTP request with retry logic.
        
        Args:
            url: Request URL
            params: Request parameters
            response_type: Type of response expected ("json" or "xml")
            
        Returns:
            Response data as dictionary or text
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                if response_type == "xml":
                    return response
                else:
                    return response.json()
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    msg = (f"Failed to fetch {url} after "
                           f"{self.max_retries} attempts: {e}")
                    log_error_with_debug(logger, msg)
                    raise
                    
                msg = f"Attempt {attempt + 1} failed, retrying: {e}"
                logger.warning(msg)
                # Exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    async def fetch_pmcids(self, pmids: List[str]) -> Dict[str, str]:
        """
        Fetch PMCIDs for a list of PMIDs using the PMC ID Converter API.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            Dictionary mapping PMIDs to PMCIDs (empty string if not found)
        """
        if not pmids:
            return {}
        
        pmid_to_pmcid = {pmid: "" for pmid in pmids}
        
        # PMC ID Converter API endpoint
        base_url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
        
        # Process PMIDs in batches of 200 (API limit)
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                # Make API request with retry logic
                pmid_string = ",".join(batch_pmids)
                params = {
                    "ids": pmid_string,
                    "idtype": "pmid",
                    "format": "xml"
                }
                
                # Add tool and email parameters if available
                if self.config.email:
                    params["email"] = self.config.email
                params["tool"] = "autonima"
                
                # Make request with retry logic
                response = await self._make_request_with_retry(
                    base_url, params, response_type="xml"
                )
                
                # Parse XML response
                root = ET.fromstring(response.text)
                
                # Extract PMCIDs from response
                for record in root.findall(".//record"):
                    requested_id = record.get("requested-id")
                    pmcid = record.get("pmcid")
                    
                    if requested_id and pmcid:
                        pmid_to_pmcid[requested_id] = pmcid
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                msg = (f"Failed to fetch PMCIDs for batch starting "
                       f"at index {i}: {e}")
                logger.warning(msg)
                continue
        
        successful_count = len([v for v in pmid_to_pmcid.values() if v])
        logger.info(
            f"Successfully fetched PMCIDs for {successful_count} "
            f"out of {len(pmids)} PMIDs using PMC ID Converter API"
        )

        # Remove prefix "PMC" from PMCIDs
        pmid_to_pmcid = {
            pmid: (pmcid[3:] if pmcid.startswith("PMC") else pmcid)
            for pmid, pmcid in pmid_to_pmcid.items()
        }

        # Remove empty entries and convert to int
        pmid_to_pmcid = {
            pmid: int(pmcid) for pmid, pmcid in pmid_to_pmcid.items() if pmcid
        }

        return pmid_to_pmcid
