"""Search NeuroStore's index of neuroimaging studies instead of all of PubMed.

NeuroStore returns base studies. Their PMIDs then go through the PubMed metadata fetch every
other search uses, so the studies of a NeuroStore run carry the same titles, abstracts and
DOIs as those of a PubMed run, and every later stage is unchanged.

The query is the config's PubMed query, translated (``autonima.neurostore.query``), unless
``search.neurostore.query`` gives one in NeuroStore's own syntax. What the translation
dropped or approximated, the totals, and the base studies with no PMID (which a PMID-keyed
run cannot carry) are written to ``outputs/neurostore_search_log.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.types import SearchConfig, Study
from ..neurostore.client import DEFAULT_BASE_URL, NeuroStoreClient
from ..neurostore.query import translate_pubmed_query
from .pubmed import PubMedSearch

logger = logging.getLogger(__name__)

SEARCH_LOG_FILENAME = "neurostore_search_log.json"
SEARCH_OPTIONS = {"query", "search_full_text", "data_type", "base_url"}
DEFAULT_DATA_TYPE = "coordinate"


def _year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    return int(str(value).strip()[:4])


class NeuroStoreSearch(PubMedSearch):
    """NeuroStore base-study search, with study metadata from PubMed."""

    def __init__(
        self,
        config: SearchConfig,
        output_dir: str = "results",
        client: Optional[NeuroStoreClient] = None,
    ):
        super().__init__(config, output_dir=output_dir)
        self.options: Dict[str, Any] = dict(config.neurostore or {})
        self.client = client or NeuroStoreClient(self.options.get("base_url") or DEFAULT_BASE_URL)
        self.search_log: Dict[str, Any] = {}

    def build_query(self, base_query: str) -> str:
        # Dates travel as NeuroStore's year filters, not as PubMed date clauses.
        return base_query

    async def search(self, query: str) -> List[Study]:
        if not (query or "").strip() and self.options.get("query"):
            query = self.options["query"]
        return await super().search(query)

    async def _execute_search(self, query: str) -> List[str]:
        native = self.options.get("query")
        translation = None
        if native:
            ns_query = native
        else:
            translation = translate_pubmed_query(query)
            ns_query = translation.query
            for clause in translation.dropped:
                logger.warning("NeuroStore search: dropped %s (no NeuroStore equivalent)", clause)
            if translation.approximated:
                logger.info(
                    "NeuroStore search: %s clauses approximated (MeSH as phrases, truncation as "
                    "word endings); see %s",
                    len(translation.approximated),
                    SEARCH_LOG_FILENAME,
                )

        year_min, year_max = _year(self.config.date_from), _year(self.config.date_to)
        data_type = self.options.get("data_type", DEFAULT_DATA_TYPE)
        full_text = bool(self.options.get("search_full_text", False))
        def run(query: str, in_full_text: bool):
            return list(
                self.client.search_base_studies(
                    query,
                    search_full_text=in_full_text,
                    year_min=year_min,
                    year_max=year_max,
                    data_type=data_type,
                )
            )

        excluded_ids: set = set()
        if full_text and translation is not None and translation.exclude:
            # PubMed's NOT clauses were field-scoped; keep them to titles and abstracts.
            results = await asyncio.to_thread(run, translation.include, True)
            for clause in translation.exclude:
                excluded_ids |= {r.get("id") for r in await asyncio.to_thread(run, clause, False)}
            matched = len(results)
            results = [r for r in results if r.get("id") not in excluded_ids]
            logger.info(
                "NeuroStore search: %s of %s full-text matches removed by NOT clauses applied to "
                "titles and abstracts only",
                matched - len(results),
                matched,
            )
        else:
            results = await asyncio.to_thread(run, ns_query, full_text)

        pmids: List[str] = []
        seen = set()
        without_pmid = []
        for result in results:
            pmid = str(result.get("pmid") or "").strip()
            if not pmid:
                without_pmid.append({"id": result.get("id"), "doi": result.get("doi"), "name": result.get("name")})
                continue
            if pmid not in seen:
                seen.add(pmid)
                pmids.append(pmid)
        if self.config.max_results:
            pmids = pmids[: self.config.max_results]

        self.search_log = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.client.base_url,
            "pubmed_query": None if native else query,
            "neurostore_query": ns_query,
            "translation": None if translation is None else {
                "dropped": translation.dropped,
                "approximated": translation.approximated,
                "include": translation.include,
                "exclude": translation.exclude,
                "exclude_scope": "title_abstract" if full_text and translation.exclude else None,
            },
            "search_full_text": full_text,
            "data_type": data_type,
            "year_min": year_min,
            "year_max": year_max,
            "base_studies": len(results),
            "pmids": len(pmids),
            "without_pmid": without_pmid,
        }
        log_path = Path(self.result_dir) / "outputs" / SEARCH_LOG_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(self.search_log, indent=2))
        logger.info(
            "NeuroStore search: %s base studies, %s with a PMID (%s without, not carried)",
            len(results),
            len(pmids),
            len(without_pmid),
        )
        return pmids

    def get_search_info(self) -> Dict[str, Any]:
        return {
            **super().get_search_info(),
            "engine": "neurostore",
            "api_url": self.client.base_url,
            "neurostore": self.options,
        }
