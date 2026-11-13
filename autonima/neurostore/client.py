"""A small client for the public NeuroStore API (https://neurostore.org/api).

Only the read endpoints autonima needs: base-study search and lookup by PMID, and a study
version with its analyses and points. Every list is paged to completion and checked against
the total the server reports, so a truncated search fails instead of passing for a complete
one.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://neurostore.org/api"


class NeuroStoreError(RuntimeError):
    """A NeuroStore request failed, or returned something other than what it promised."""


class NeuroStoreClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 120.0,
        max_retries: int = 4,
        retry_delay: float = 2.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = session or requests.Session()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 404:
                    return None
                if response.status_code < 500 and response.status_code != 429:
                    if not response.ok:
                        raise NeuroStoreError(
                            f"GET {response.url} returned {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    return response.json()
                last_error = NeuroStoreError(f"GET {response.url} returned {response.status_code}")
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))
        raise NeuroStoreError(f"GET {url} failed after {self.max_retries} attempts: {last_error}")

    def search_base_studies(
        self,
        search: str,
        *,
        search_full_text: bool = False,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        data_type: Optional[str] = None,
        page_size: int = 1000,
    ) -> Iterator[Dict[str, Any]]:
        """Yield every base study matching ``search``, checked against the reported total."""
        params: Dict[str, Any] = {
            "search": search,
            "search_full_text": str(bool(search_full_text)).lower(),
            "flat": "true",
            "page_size": page_size,
        }
        if year_min is not None:
            params["year_min"] = year_min
        if year_max is not None:
            params["year_max"] = year_max
        if data_type:
            params["data_type"] = data_type

        page, seen, total = 1, 0, None
        while True:
            data = self.get("base-studies/", {**params, "page": page})
            if data is None:
                raise NeuroStoreError("base-studies search returned 404")
            reported = (data.get("metadata") or {}).get("total_count")
            if total is None:
                total = reported
            elif reported != total:
                # The index changed under the search. Restarting is the only way to get one
                # consistent result set.
                raise NeuroStoreError(
                    f"search total changed while paging ({total} -> {reported}); retry the search"
                )
            results = data.get("results") or []
            for result in results:
                yield result
            seen += len(results)
            if not results or seen >= (total or 0):
                break
            page += 1
        if total is not None and seen != total:
            raise NeuroStoreError(f"search retrieved {seen} base studies but reported {total}")

    def base_study_for_pmid(self, pmid: str) -> Optional[Dict[str, Any]]:
        data = self.get("base-studies/", {"pmid": str(pmid), "flat": "true"})
        results = (data or {}).get("results") or []
        matches = [r for r in results if str(r.get("pmid")) == str(pmid)]
        if len(matches) > 1:
            logger.warning("PMID %s matches %s NeuroStore base studies; using the first", pmid, len(matches))
        return matches[0] if matches else None

    def base_study_versions(self, base_study_id: str) -> List[str]:
        data = self.get(f"base-studies/{base_study_id}", {"info": "true"}) or {}
        return [v["id"] if isinstance(v, dict) else v for v in data.get("versions") or []]

    def study(self, study_id: str) -> Optional[Dict[str, Any]]:
        """A study version, with its analyses, points and tables embedded."""
        return self.get(f"studies/{study_id}", {"nested": "true"})
