"""PubMed search engine implementation using NCBI Entrez API."""

import asyncio
import email.utils
import http.client
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from Bio import Entrez

from .base import SearchEngine
from ..models.types import Study, StudyStatus, SearchConfig
from ..utils import log_error_with_debug


logger = logging.getLogger(__name__)

# The PMC ID Converter is called through urllib, not requests. pmc.ncbi.nlm.nih.gov's
# edge answers a bare 403 to the TLS ClientHello urllib3 sends when Python is linked
# against OpenSSL 3.5 (seen with Python 3.14.0 + OpenSSL 3.5.4, urllib3 2.6.3 and
# 2.8.0). urllib3's context enables post-handshake auth, disables session tickets and
# offers ALPN "http/1.1"; the edge refuses exactly that combination. Any two of the
# three get through, as does the same urllib3 on OpenSSL 3.0, and headers make no
# difference, so pinning urllib3 does not help. urllib uses the stdlib's default TLS
# context, which is also what Biopython's Entrez calls use. eutils.ncbi.nlm.nih.gov
# accepts urllib3, so pubget's requests-based Entrez calls are unaffected.
PMC_IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

# NCBI answers a throttled or blocked client with 403 as well as 429, so both are
# retried along with server errors.
RETRYABLE_HTTP_STATUSES = frozenset({403, 429, 500, 502, 503, 504})


class PMCIDLookupError(RuntimeError):
    """The PMC ID Converter gave no answer for some PMIDs.

    Raised instead of returning a mapping without them: a PMID missing from the mapping
    means "not in PMC", and retrieval would treat those studies as having no full text.
    """


def _retry_after_seconds(headers) -> float:
    """Return the wait a Retry-After header asks for (seconds or HTTP date), else 0."""
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


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
        # Attempts per request, first backoff delay (doubling each retry), the cap on any
        # single wait, and the socket timeout, all in seconds.
        self.max_retries = 5
        self.retry_delay = 2.0
        self.max_retry_delay = 120.0
        self.request_timeout = 30.0
        self.result_dir = output_dir
        self.cache_stats = {"reused": 0, "processed": 0}

    async def search(self, query: str) -> List[Study]:
        """
        Execute PubMed search and return list of studies.

        Args:
            query: PubMed search query (ignored if pmids_file or pmids_list is provided)

        Returns:
            List of Study objects
        """
        try:
            query = (query or "").strip()
            if not query and not (
                self.config.pmids_file or self.config.pmids_list
            ):
                raise ValueError(
                    "A non-empty query or pmids_file/pmids_list is required"
                )

            # Build complete query
            pmids = []
            if query:
                full_query = self.build_query(query)
                pmids += await self._execute_search(full_query)
                logger.info(f"Found {len(pmids)} potential studies from search")

            # Check if we are using PMIDs list or file
            if self.config.pmids_file or self.config.pmids_list:
                # Use PMIDs from file or list
                pmids += self._load_pmids()
                logger.info(f"Using {len(pmids)} PMIDs from config")

            logger.debug(f"Total PMIDs to process: {len(pmids)}")

            if not pmids:
                return []

            # Load existing search results for caching
            cached_studies = self._load_cached_search_results()
            cached_by_pmid = {study.pmid: study for study in cached_studies}
            cached_pmids = set(cached_by_pmid)
            
            # Identify PMIDs that need to be fetched
            new_pmids = [pmid for pmid in pmids if pmid not in cached_pmids]
            logger.debug(
                "Found meta-data for %s cached studies, %s new studies to fetch",
                len(cached_studies),
                len(new_pmids),
            )

            # Fetch details for new PMIDs only
            new_studies = []
            if new_pmids:
                new_studies = await self._fetch_study_details(new_pmids)
                logger.debug(
                    f"Successfully retrieved {len(new_studies)} new studies"
                )

            # Combine only records requested by the current search/list. This
            # preserves metadata caching without letting stale studies from an
            # older broader query flow into the current execution.
            new_by_pmid = {study.pmid: study for study in new_studies}
            studies = [
                cached_by_pmid.get(pmid) or new_by_pmid[pmid]
                for pmid in pmids
                if pmid in cached_by_pmid or pmid in new_by_pmid
            ]
            self.cache_stats = {
                "reused": sum(1 for pmid in pmids if pmid in cached_by_pmid),
                "processed": sum(1 for pmid in pmids if pmid in new_by_pmid),
            }

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
                logger.debug(
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
        logger.debug(f"Executing search with query: {query}")
        
        try:
            if self.config.max_results is None:
                self.config.max_results = 10000  # Default max results if not set
                logger.debug("Max results not set in config, defaulting to 10000")
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
    ) -> bytes:
        """
        GET a URL and return the response body, retrying transient failures.

        403, 429 and 5xx responses and network errors are retried with
        exponential backoff, waiting at least as long as a Retry-After header
        asks (capped at ``max_retry_delay``). Other HTTP errors are raised at
        once, since repeating the request cannot change the answer.

        Args:
            url: Request URL
            params: Query parameters

        Returns:
            Raw response body
        """
        request_url = f"{url}?{urlencode(params)}"
        for attempt in range(1, self.max_retries + 1):
            backoff = self.retry_delay * (2 ** (attempt - 1))
            try:
                with urlopen(request_url, timeout=self.request_timeout) as response:
                    return response.read()
            except HTTPError as e:
                e.close()
                if (
                    e.code not in RETRYABLE_HTTP_STATUSES
                    or attempt == self.max_retries
                ):
                    raise
                error = f"HTTP {e.code}"
                delay = max(backoff, _retry_after_seconds(e.headers))
            except (OSError, http.client.HTTPException) as e:
                if attempt == self.max_retries:
                    raise
                error = f"{type(e).__name__}: {e}"
                delay = backoff

            delay = min(delay, self.max_retry_delay)
            logger.warning(
                "Request to %s failed (%s); retrying in %.0fs "
                "(attempt %s of %s)",
                url, error, delay, attempt, self.max_retries,
            )
            await asyncio.sleep(delay)

    @staticmethod
    def _parse_idconv_response(body: bytes, requested: List[str]) -> Dict[str, str]:
        """Return {pmid: "PMC..."} from one converter response, checking it is complete.

        The converter returns a record for every requested id, marked with an
        error when the article is not in PMC. A response that is not "ok", or
        that leaves a requested PMID out, did not answer for it, so it raises
        rather than letting that PMID read as "not in PMC".
        """
        root = ET.fromstring(body)
        if root.get("status") != "ok":
            raise ValueError(
                f"converter returned status={root.get('status')!r}"
            )
        answered = set()
        found = {}
        for record in root.findall(".//record"):
            requested_id = record.get("requested-id")
            answered.add(requested_id)
            pmcid = record.get("pmcid")
            if requested_id and pmcid:
                found[requested_id] = pmcid
        unanswered = set(requested) - answered
        if unanswered:
            raise ValueError(
                f"response has no record for {len(unanswered)} of "
                f"{len(requested)} requested PMIDs"
            )
        return found

    async def fetch_pmcids(self, pmids: List[str]) -> Dict[str, int]:
        """
        Fetch PMCIDs for a list of PMIDs using the PMC ID Converter API.

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dictionary mapping each PMID that is in PMC to its numeric PMCID.
            A PMID is left out only when the converter says it is not in PMC,
            or when it is not a numeric PMID at all.

        Raises:
            PMCIDLookupError: A batch still failed after retries. Nothing is
                returned for any PMID, so no study is mistaken for one
                without a PMCID.
        """
        if not pmids:
            return {}

        # The converter rejects a whole batch with 400 if any id is not a
        # positive integer, and such an id cannot have a PMCID anyway.
        lookup_pmids = [pmid for pmid in pmids if pmid.isdigit()]
        if len(lookup_pmids) < len(pmids):
            logger.warning(
                "Skipping PMCID lookup for %s non-numeric PMIDs",
                len(pmids) - len(lookup_pmids),
            )

        pmid_to_pmcid: Dict[str, str] = {}

        # Process PMIDs in batches of 200 (API limit)
        batch_size = 200
        for i in range(0, len(lookup_pmids), batch_size):
            batch_pmids = lookup_pmids[i:i + batch_size]
            params = {
                "ids": ",".join(batch_pmids),
                "idtype": "pmid",
                "format": "xml",
                "tool": "autonima",
            }
            if self.config.email:
                params["email"] = self.config.email

            try:
                body = await self._make_request_with_retry(
                    PMC_IDCONV_URL, params
                )
                pmid_to_pmcid.update(
                    self._parse_idconv_response(body, batch_pmids)
                )
            except Exception as e:
                raise PMCIDLookupError(
                    f"PMC ID Converter lookup failed for {len(batch_pmids)} "
                    f"PMIDs (batch starting at index {i} of "
                    f"{len(lookup_pmids)}): {e}. Stopping rather than "
                    "treating these studies as having no PMC full text; "
                    "re-run to retry."
                ) from e

            # Rate limiting
            await asyncio.sleep(0.5)

        successful_count = len(pmid_to_pmcid)
        logger.debug(
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
