"""A failed PMC ID Converter lookup must not pass for "no PMCID".

``fetch_pmcids`` used to log a failed batch and carry on, so every study in it read as not
in PMC and silently dropped out of full-text screening. These tests stub the HTTP layer
(``urlopen``) and ``asyncio.sleep``, so they make no network calls and do not wait.
"""

import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from autonima.config import ConfigManager
from autonima.models.types import SearchConfig, Study, StudyStatus
from autonima.pipeline import AutonimaPipeline
from autonima.search.pubmed import (
    PMC_IDCONV_URL,
    PMCIDLookupError,
    PubMedSearch,
    _retry_after_seconds,
)


def _idconv_body(found=None, not_in_pmc=()):
    """A converter response shaped like the real one: one record per requested id."""
    records = [
        f'<record pmcid="PMC{pmcid}" pmid="{pmid}" requested-id="{pmid}"></record>'
        for pmid, pmcid in (found or {}).items()
    ] + [
        f'<record errmsg="Identifier not found in PMC" pmid="{pmid}" '
        f'requested-id="{pmid}" status="error"></record>'
        for pmid in not_in_pmc
    ]
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<pmcids response-date="2026-11-03 16:08:25" status="ok">'
        '<request format="xml" idtype="pmid"></request>'
        + "".join(records)
        + "</pmcids>"
    ).encode()


class _Response(io.BytesIO):
    """What urlopen returns: a readable body usable as a context manager."""


def _http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(PMC_IDCONV_URL, code, "error", headers, io.BytesIO(b"<title>error</title>"))


def _always(code):
    """An urlopen stand-in that fails every call with a fresh HTTPError."""
    def fail(url, timeout):
        raise _http_error(code)
    return fail


def _ids_in(url):
    return parse_qs(urlparse(url).query)["ids"][0].split(",")


def _requested_ids(urlopen_call):
    return _ids_in(urlopen_call.args[0])


@pytest.fixture
def search(tmp_path):
    return PubMedSearch(
        SearchConfig(database="pubmed", query="q", email="test@example.com"),
        output_dir=str(tmp_path),
    )


@pytest.fixture
def sleep():
    with patch("autonima.search.pubmed.asyncio.sleep", new_callable=AsyncMock) as mock:
        yield mock


def _delays(sleep_mock):
    return [call.args[0] for call in sleep_mock.await_args_list]


def test_batch_that_fails_then_succeeds_is_retried(search, sleep):
    with patch("autonima.search.pubmed.urlopen") as urlopen:
        urlopen.side_effect = [
            _http_error(403),
            _http_error(503),
            _Response(_idconv_body({"111": 9001}, not_in_pmc=["222"])),
        ]
        result = asyncio.run(search.fetch_pmcids(["111", "222"]))

    assert result == {"111": 9001}
    assert urlopen.call_count == 3
    # Exponential backoff between attempts, then the usual pause between batches.
    assert _delays(sleep) == [2.0, 4.0, 0.5]


def test_retry_waits_as_long_as_retry_after_asks(search, sleep):
    search.max_retry_delay = 60.0
    with patch("autonima.search.pubmed.urlopen") as urlopen:
        urlopen.side_effect = [
            _http_error(429, retry_after="30"),
            _http_error(429, retry_after="600"),
            _Response(_idconv_body({"111": 9001})),
        ]
        result = asyncio.run(search.fetch_pmcids(["111"]))

    assert result == {"111": 9001}
    # Retry-After beats the 2s/4s backoff, but no single wait exceeds max_retry_delay.
    assert _delays(sleep) == [30.0, 60.0, 0.5]


def test_retry_after_accepts_an_http_date():
    headers = Message()
    headers["Retry-After"] = format_datetime(
        datetime.now(timezone.utc) + timedelta(seconds=90), usegmt=True
    )
    assert 85 <= _retry_after_seconds(headers) <= 90
    assert _retry_after_seconds(Message()) == 0.0


def test_batch_that_always_fails_raises(search, sleep):
    with patch("autonima.search.pubmed.urlopen", side_effect=_always(403)) as urlopen:
        with pytest.raises(PMCIDLookupError, match="HTTP Error 403") as excinfo:
            asyncio.run(search.fetch_pmcids(["111", "222"]))

    assert urlopen.call_count == search.max_retries
    assert "2 PMIDs" in str(excinfo.value)
    assert _delays(sleep) == [2.0, 4.0, 8.0, 16.0]


def test_later_batch_failing_discards_earlier_batches(search, sleep):
    pmids = [str(1000 + i) for i in range(201)]
    first_batch = _Response(_idconv_body({"1000": 5}, not_in_pmc=pmids[1:200]))

    def respond(url, timeout):
        if len(_ids_in(url)) == 200:
            return first_batch
        raise _http_error(500)

    with patch("autonima.search.pubmed.urlopen", side_effect=respond) as urlopen:
        with pytest.raises(PMCIDLookupError, match="index 200 of 201"):
            asyncio.run(search.fetch_pmcids(pmids))

    assert [len(_requested_ids(call)) for call in urlopen.call_args_list] == [200] + [1] * 5


def test_client_error_is_not_retried(search, sleep):
    with patch("autonima.search.pubmed.urlopen", side_effect=_http_error(400)) as urlopen:
        with pytest.raises(PMCIDLookupError, match="HTTP Error 400"):
            asyncio.run(search.fetch_pmcids(["111"]))

    assert urlopen.call_count == 1
    sleep.assert_not_awaited()


def test_response_that_skips_a_requested_pmid_is_a_failure(search, sleep):
    # "222" gets no record at all, which is not the converter saying it is absent from PMC.
    body = _idconv_body({"111": 9001})
    with patch("autonima.search.pubmed.urlopen", return_value=_Response(body)):
        with pytest.raises(PMCIDLookupError, match="no record for 1 of 2"):
            asyncio.run(search.fetch_pmcids(["111", "222"]))


def test_non_numeric_pmids_are_not_sent(search, sleep):
    # One non-numeric id makes the converter reject the whole batch with 400.
    with patch("autonima.search.pubmed.urlopen") as urlopen:
        urlopen.return_value = _Response(_idconv_body(not_in_pmc=["111"]))
        assert asyncio.run(search.fetch_pmcids(["111", "Unknown"])) == {}

    assert _requested_ids(urlopen.call_args) == ["111"]


def _retrieval_pipeline(output_dir):
    config = ConfigManager().create_sample_config()
    config.output.directory = str(output_dir)
    pipeline = AutonimaPipeline(config)
    pipeline._retriever = MagicMock()
    pipeline.results.studies = [
        Study(
            pmid=pmid,
            title=f"Study {pmid}",
            abstract="",
            authors=[],
            journal="",
            publication_date="2024",
            status=StudyStatus.INCLUDED_ABSTRACT,
        )
        for pmid in ("111", "222")
    ]
    return pipeline


def test_retrieval_stage_fails_on_lookup_failure_and_next_run_retries(tmp_path, sleep):
    (tmp_path / "outputs").mkdir()
    results_file = tmp_path / "outputs" / "fulltext_retrieval_results.json"

    pipeline = _retrieval_pipeline(tmp_path)
    with patch("autonima.search.pubmed.urlopen", side_effect=_always(403)):
        with pytest.raises(PMCIDLookupError):
            asyncio.run(pipeline._execute_retrieval_phase())

    # Nothing went to pubget and nothing was recorded as retrieved or missing.
    pipeline._retriever.retrieve.assert_not_called()
    assert not results_file.exists()

    pipeline = _retrieval_pipeline(tmp_path)
    body = _idconv_body({"111": 9001}, not_in_pmc=["222"])
    with patch("autonima.search.pubmed.urlopen", return_value=_Response(body)):
        asyncio.run(pipeline._execute_retrieval_phase())

    sent = pipeline._retriever.retrieve.call_args.kwargs["studies"]
    assert {study.pmid: study.pmcid for study in sent} == {"111": 9001, "222": None}
    assert json.loads(results_file.read_text())["studies_with_fulltext"][0]["pmcid"] == 9001
