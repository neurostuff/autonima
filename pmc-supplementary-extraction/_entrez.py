"""
Purpose:
    Provide a small client, EntrezClient, for the NCBI Entrez E-utilities
    (epost, esearch, efetch) needed to find and download PMC articles that
    are in the PMC Open Access subset -- with built-in rate limiting,
    retry-with-backoff on failed requests, and optional logging of failed
    request/response details for debugging.

Origin:
    Ported essentially verbatim from pubget
    (https://github.com/neuroquery/pubget), src/pubget/_entrez.py, licensed
    BSD-3-Clause. The only change from the original: the
    `pubget._typing.PathLikeOrStr` import is replaced with a local type
    alias, so this file has no dependency on the pubget package itself.

Requirements:
    - Packages: requests, lxml

How to run:
    This is a library module with no command-line entry point of its own.
    In this pipeline it's used by extraction-pipeline.py, which calls
    EntrezClient's epost/esearch/efetch methods to fetch each article's
    JATS XML. Used on its own:

        from _entrez import EntrezClient

        client = EntrezClient()
        search_result = client.epost([1234567])  # PMCIDs as plain numbers
        client.efetch("output_dir", search_result=search_result)

pubget license (BSD-3-Clause) — reproduced per its terms:

Copyright (c) pubget contributors.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions of the BSD-3-Clause
license are met. See https://github.com/neuroquery/pubget for the full text.
"""
import datetime, json, logging, math, secrets, time, requests

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin
from lxml import etree

PathLikeOrStr = Union[str, Path]

_LOG = logging.getLogger(__name__)
_EFETCH_DEFAULT_BATCH_SIZE = 500


def _check_response_status(response: requests.Response) -> Tuple[bool, str]:
    """
    Purpose:
        Check that an HTTP request came back with a 200 OK status.

    Inputs:
        response (requests.Response): the HTTP response to check.

    Outputs:
        tuple of (bool, str):
            True and "" if the status code was 200;
            False and a reason string otherwise.
    """
    if response.status_code == 200:
        return True, ""
    return False, f"Status code {response.status_code} != 200"


def _check_efetch_response(response: requests.Response) -> Tuple[bool, str]:
    """
    Purpose:
        Check that an efetch response is a 200 OK and its content looks
        like a PMC article-set XML document.

    Inputs:
        response (requests.Response): the efetch response to check.

    Outputs:
        tuple of (bool, str):
            True and "" if the response is a valid articleset;
            False and a reason string otherwise.
    """
    check, reason = _check_response_status(response)
    if not check:
        return check, reason
    try:
        root = etree.fromstring(response.content)
        assert root.tag == "pmc-articleset"
        assert root.find("article") is not None
    except Exception:
        return (
            False,
            "response content does not appear to be an XML articleset",
        )
    return True, ""


def _check_esearch_response(response: requests.Response) -> Tuple[bool, str]:
    """
    Purpose:
        Check that an esearch response is a 200 OK and its JSON body has
        the fields needed to continue the search (count, webenv, querykey).

    Inputs:
        response (requests.Response): the esearch response to check.

    Outputs:
        tuple of (bool, str):
            True and "" if the response is valid;
            False and a reason string otherwise.
    """
    check, reason = _check_response_status(response)
    if not check:
        return check, reason
    try:
        search_result: Dict[str, str] = response.json()["esearchresult"]
    except Exception:
        return False, "response does not contain JSON with esearchresult key"
    if "ERROR" in search_result:
        return False, f"esearch returned ERROR: {search_result['ERROR']}"
    needed_keys = {"count", "webenv", "querykey"}
    missing_keys = needed_keys.difference(search_result.keys())
    if missing_keys:
        return False, f"keys missing from esearch response: {missing_keys}"
    return True, ""


def _check_epost_response(response: requests.Response) -> Tuple[bool, str]:
    """
    Purpose:
        Check that an epost response is a 200 OK and its XML body has a
        WebEnv and QueryKey to use for later requests.

    Inputs:
        response (requests.Response): the epost response to check.

    Outputs:
        tuple of (bool, str):
            True and "" if the response is valid;
            False and a reason string otherwise.
    """
    check, reason = _check_response_status(response)
    if not check:
        return check, reason
    try:
        resp_xml = etree.fromstring(response.content)
        webenv = resp_xml.find("WebEnv").text
        assert webenv
        querykey = resp_xml.find("QueryKey").text
        assert querykey
    except Exception:
        return False, "epost response missing webenv or querykey"
    return True, ""


class EntrezClient:
    """Client for esearch, epost, and efetch against the PMC database."""

    _default_timeout = 27
    _delay_before_retry_failed_request = (2.0, 8.0, 32.0, 64.0, 64.0)
    _entrez_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    _esearch_base_url = urljoin(_entrez_base_url, "esearch.fcgi")
    _efetch_base_url = urljoin(_entrez_base_url, "efetch.fcgi")
    _epost_base_url = urljoin(_entrez_base_url, "epost.fcgi")


    def __init__(self, request_period: Optional[float] = None, api_key: Optional[str] = None, 
                 failed_requests_dump_dir: Optional[PathLikeOrStr] = None,) -> None:
        """
        Purpose:
            Set up a new Entrez client: how fast it's allowed to send
            requests, an optional NCBI API key, and an optional folder to
            save debug info about failed requests.

        Inputs:
            request_period (float, optional): minimum seconds between
                requests. Defaults to 0.15 if an api_key is given, else
                1.05 (NCBI's rate limits with and without a key).
            api_key (str, optional): an NCBI API key, if you have one.
            failed_requests_dump_dir (str or Path, optional): folder to
                write details of any failed request/response to, for
                debugging. If not given, failures aren't dumped to disk.

        Outputs:
            None.
        """
        self._entrez_id = {}
        if api_key is not None:
            self._entrez_id["api_key"] = api_key
        if request_period is None:
            self._request_period = (
                0.15 if "api_key" in self._entrez_id else 1.05
            )
        else:
            self._request_period = request_period
        self._failed_requests_dump_dir: Optional[Path] = None
        if failed_requests_dump_dir is not None:
            self._failed_requests_dump_dir = Path(failed_requests_dump_dir)
        self._last_request_time: Union[None, float] = None
        self._session = requests.Session()
        self.last_search_result: Optional[Mapping[str, str]] = None
        self.n_failures = 0


    def _wait_to_send_request(self) -> None:
        """
        Purpose:
            Pause if needed so requests go out no faster than
            self._request_period seconds apart, then record this call's
            time.

        Inputs:
            None.

        Outputs:
            None.
        """
        if self._last_request_time is None:
            self._last_request_time = time.time()
            return
        wait = self._request_period - (time.time() - self._last_request_time)
        if wait > 0:
            _LOG.debug(f"wait for {wait:.3f} seconds to send request")
            time.sleep(wait)
        self._last_request_time = time.time()


    def _dump_failed_request_info(self, request: requests.PreparedRequest, 
                                  response: Optional[requests.Response],) -> None:
        """
        Purpose:
            Save a failed request's URL, headers, and body (and the
            response's URL, headers, and content, if there was a response)
            to a new folder under self._failed_requests_dump_dir, for
            debugging. Does nothing if no dump directory was configured.

        Inputs:
            request (requests.PreparedRequest): the request that failed.
            response (requests.Response or None): the response received,
                if any (None if the request itself raised an error).

        Outputs:
            None.

        Notes:
            Masks the API key in the saved request body, if one was used.
        """
        if self._failed_requests_dump_dir is None:
            return
        request_dir = self._failed_requests_dump_dir.joinpath(
            f"{datetime.datetime.now().isoformat().replace(':', '_')}"
            f"-{secrets.token_hex(2)}"
        )
        request_dir.mkdir(parents=True)
        _LOG.info(f"Logging info on failed request to {request_dir}")
        try:
            request_dir.joinpath("request_url").write_text(
                request.url or "", encoding="utf-8"
            )
            request_dir.joinpath("request_headers").write_text(
                json.dumps(dict(request.headers)), encoding="utf-8"
            )
            body = str(request.body or "")
            api_key = self._entrez_id.get("api_key")
            if api_key is not None:
                body = body.replace(api_key, f"<API KEY: {api_key[:3]}...>")
            request_dir.joinpath("request_body").write_text(
                body, encoding="utf-8"
            )
            if response is None:
                return

            request_dir.joinpath("response_url").write_text(
                response.url, encoding="utf-8"
            )
            request_dir.joinpath("response_headers").write_text(
                json.dumps(dict(response.headers)), encoding="utf-8"
            )
            request_dir.joinpath("response_content").write_bytes(
                response.content
            )
        except Exception:
            _LOG.exception("Failed to log bad request or response content.")


    def _send_one_request(self, prepped: requests.PreparedRequest, 
                          response_validator: Callable[
                              [requests.Response], Tuple[bool, str]
                              ],) -> Optional[requests.Response]:
        """
        Purpose:
            Send one already-prepared request (after throttling) and
            check its response with the given validator function.

        Inputs:
            prepped (requests.PreparedRequest): the request to send.
            response_validator (callable): a function that takes a
                response and returns (True, "") if it's valid, or
                (False, reason) if not.

        Outputs:
            requests.Response or None: the response, if it was sent
                successfully and passed validation; otherwise None (and
                failure details are dumped via _dump_failed_request_info).
        """
        self._wait_to_send_request()
        try:
            resp = self._session.send(prepped, timeout=self._default_timeout)
        except Exception as error:
            _LOG.warning(f"Request failed: {prepped.url} ({error})")
            self._dump_failed_request_info(prepped, None)
            return None
        check, reason = response_validator(resp)
        if not check:
            _LOG.warning(
                f"Response failed to validate (reason: {reason}) "
                f"for url {prepped.url}"
            )
            self._dump_failed_request_info(prepped, resp)
            return None
        _LOG.debug(
            f"received response. code: {resp.status_code}; "
            f"reason: {resp.reason}; from: {resp.url}"
        )
        return resp


    def _send_request(self, url: str, params: Optional[Mapping[str, Any]] = None, 
                      data: Optional[Mapping[str, Any]] = None, 
                      response_validator: Callable[
                          [requests.Response], Tuple[bool, str]
                          ] = _check_response_status,) -> Union[None, requests.Response]:
        """
        Purpose:
            Send a request, retrying with increasing delays if it fails or
            doesn't pass validation, up to the number of attempts in
            self._delay_before_retry_failed_request.

        Inputs:
            url (str): the request URL.
            params (dict, optional): URL query parameters.
            data (dict, optional): POST body data.
            response_validator (callable, optional): function to check the
                response; defaults to just checking for a 200 status.

        Outputs:
            requests.Response or None: the response from the first
                successful, valid attempt; None if every attempt failed.
        """
        req = requests.Request("POST", url, params=params, data=data)
        prepped = self._session.prepare_request(req)
        for attempt, delay in enumerate(
            self._delay_before_retry_failed_request
        ):
            _LOG.info(
                f"sending request: {prepped.url} (attempt #{attempt + 1})"
            )
            resp = self._send_one_request(prepped, response_validator)
            if resp is not None:
                return resp
            _LOG.warning(f"request failed: sleeping {delay}s before retrying.")
            time.sleep(delay)
        _LOG.error(
            "Request failed; giving up after "
            f"{len(self._delay_before_retry_failed_request)} attempts: "
            f"{prepped.url}"
        )
        return None


    def epost(self, all_pmcids: Sequence[int]) -> Dict[str, str]:
        """
        Purpose:
            Post a list of PMCIDs to the Entrez history server, then run
            an esearch to filter them down to just the open-access ones.

        Inputs:
            all_pmcids (sequence of int): the PMCIDs (as plain numbers,
                without the "PMC" prefix) to post.

        Outputs:
            dict: the search result (keys "count", "webenv", "querykey")
                for the open-access subset of the posted PMCIDs, or {} if
                the list was empty or the request failed.

        Notes:
            IDs could be sent to efetch directly, but posting them to
            epost first lets us filter by open-access, and efetch is then
            handled the same way whether we started from a query or an id
            list.
        """
        # not 'if not all_pmcids' in case someone passes a numpy array
        if len(all_pmcids) == 0:
            _LOG.error("Empty PMCID list.")
            self.n_failures = 1
            return {}
        data = {
            "db": "pmc",
            "id": ",".join(map(str, all_pmcids)),
            **self._entrez_id,
        }
        _LOG.info(f"Posting {len(all_pmcids)} PMCIDs to Entrez.")
        resp = self._send_request(
            self._epost_base_url,
            data=data,
            response_validator=_check_epost_response,
        )
        if resp is None:
            self.n_failures = 1
            return {}
        resp_xml = etree.fromstring(resp.content)
        webenv = resp_xml.find("WebEnv").text
        query_key = resp_xml.find("QueryKey").text
        search_result = self.esearch(webenv=webenv, query_key=query_key)
        if "count" in search_result:
            _LOG.info(
                f"{int(search_result['count'])} / {len(all_pmcids)} articles "
                "are in PMC Open Access."
            )
        return search_result


    def esearch(self, query: Optional[str] = None, webenv: Optional[str] = None, 
                query_key: Optional[str] = None) -> Dict[str, str]:
        """
        Purpose:
            Run a search, always restricted to the PMC open-access subset.
            If webenv/query_key are given, the search is intersected with
            that existing result set on the Entrez history server.

        Inputs:
            query (str, optional): a search term. If omitted, this
                searches (or filters) for all open-access articles.
            webenv (str, optional): an existing Entrez history-server
                environment to search within.
            query_key (str, optional): the query key for that environment.
                Must be given together with webenv.

        Outputs:
            dict: the search result (keys of interest are "count",
                "webenv", "querykey"), or {} if the search failed.
        """
        term = "open+access[filter]"
        if query is not None:
            term = "&".join((query, term))
        data = {
            "db": "pmc",
            "term": term,
            "usehistory": "y",
            "retmode": "json",
            "retmax": 5,
            **self._entrez_id,
        }
        if webenv is not None and query_key is not None:
            data.update({"WebEnv": webenv, "query_key": query_key})
        resp = self._send_request(
            self._esearch_base_url,
            data=data,
            response_validator=_check_esearch_response,
        )
        if resp is None:
            self.n_failures = 1
            return {}
        search_result: Dict[str, str] = resp.json()["esearchresult"]
        self.last_search_result = search_result
        _LOG.info(f"Search returned {search_result['count']} results")
        return search_result


    def _get_search_result(self, search_result: Optional[Mapping[str, str]] = None) -> Optional[Mapping[str, str]]:
        """
        Purpose:
            Return the given search result, or fall back to the result of
            the last search run on this client.

        Inputs:
            search_result (dict, optional): a search result to use
                instead of the client's last one.

        Outputs:
            dict or None: the search result to use, or None if neither
                one was given/available (and logs an error in that case).
        """
        if search_result is None:
            search_result = self.last_search_result
        if search_result is None:
            self.n_failures = 1
            _LOG.error(
                "Perform a search before calling `efetch`"
                "or provide `search_result`"
            )
            return None
        return search_result


    def efetch(self, output_dir: PathLikeOrStr, search_result: Optional[Mapping[str, str]] = None, 
               n_docs: Optional[int] = None, retmax: int = _EFETCH_DEFAULT_BATCH_SIZE,) -> None:
        """
        Purpose:
            Download the articles from a search result, in batches, saving
            each batch as its own XML file in output_dir.

        Inputs:
            output_dir (str or Path): folder to save downloaded batch
                files into.
            search_result (dict, optional): the search result to download
                (must have "webenv", "querykey", "count"). If omitted,
                uses the result of the last search on this client.
            n_docs (int, optional): maximum number of articles to
                download. Defaults to all of them.
            retmax (int, optional): how many articles per batch. Defaults
                to _EFETCH_DEFAULT_BATCH_SIZE (500).

        Outputs:
            None.

        Notes:
            Assumes the caller has made sure it's safe to resume a partial
            download already in output_dir -- i.e. that the webenv,
            querykey, and retmax are the same as before.
        """
        output_dir = Path(output_dir)
        search_result = self._get_search_result(search_result)
        if search_result is None:
            return
        search_count = int(search_result["count"])
        if n_docs is None:
            n_docs = search_count
        else:
            n_docs = min(n_docs, search_count)
        retstart = 0
        params = {
            "WebEnv": search_result["webenv"],
            "query_key": search_result["querykey"],
            "retmax": retmax,
            "retstart": retstart,
            "db": "pmc",
            **self._entrez_id,
        }
        n_batches = math.ceil(n_docs / retmax)
        self.n_failures = 0
        batch_nb = 0
        _LOG.info(f"Downloading {n_docs} articles (in {n_batches} batches)")
        while retstart < n_docs:
            self._download_batch(output_dir, batch_nb, n_batches, params)
            retstart += retmax
            batch_nb += 1
            params["retstart"] = retstart

    def _download_batch(
        self,
        output_dir: Path,
        batch_nb: int,
        n_batches: int,
        params: Dict[str, Any],
    ) -> None:
        """
        Purpose:
            Download one batch of articles via efetch and save it to
            output_dir, skipping it if it was already downloaded.

        Inputs:
            output_dir (Path): folder to save the batch file into.
            batch_nb (int): this batch's number (0-based), used to name
                the file and in log messages.
            n_batches (int): total number of batches, used in log
                messages.
            params (dict): the efetch request parameters for this batch
                (WebEnv, query_key, retmax, retstart, db, etc.).

        Outputs:
            None.

        Notes:
            On failure, increments self.n_failures instead of raising, so
            the caller (efetch) can keep going with the remaining batches.
        """
        batch_file = output_dir.joinpath(f"articleset_{batch_nb:0>5}.xml")
        if batch_file.is_file():
            _LOG.info(f"batch {batch_nb + 1} already downloaded, skipping")
            return
        _LOG.info(f"getting batch {batch_nb + 1} / {n_batches}")
        resp = self._send_request(
            self._efetch_base_url,
            data=params,
            response_validator=_check_efetch_response,
        )
        if resp is None:
            self.n_failures += 1
            _LOG.error(f"{self.n_failures} batches failed to download")
        else:
            _LOG.info(f"batch {batch_nb + 1} downloaded successfully.")
            batch_file.write_bytes(resp.content)