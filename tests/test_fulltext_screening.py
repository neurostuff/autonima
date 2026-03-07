"""Pytest tests for the full text screening functionality."""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from autonima.models.types import ScreeningConfig, Study, StudyStatus
from autonima.screening import LLMScreener
from autonima.screening.schema import FullTextScreeningOutput


def _write_fulltext_csv(temp_dir: Path, pmcid: str, body: str) -> None:
    test_dir = temp_dir / "retrieval" / "pubget_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"pmcid": [pmcid], "body": [body]})
    df.to_csv(test_dir / "text.csv", index=False)


def _write_cached_fulltext_result(
    temp_dir: Path,
    study_id: str,
    decision: str,
) -> None:
    output_dir = temp_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "fulltext_screening_results.json"
    data = {
        "screening_results": [
            {
                "study_id": study_id,
                "decision": decision,
                "reason": "Cached result",
                "confidence": 0.5,
                "model_used": "gpt-4",
                "screening_type": "fulltext",
                "timestamp": datetime.now().isoformat(),
                "inclusion_criteria_applied": [],
                "exclusion_criteria_applied": [],
            }
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def _build_fulltext_study(pmid: str, pmcid: str = "123456") -> Study:
    return Study(
        pmid=pmid,
        title="fMRI study of working memory in schizophrenia",
        abstract="This is an abstract.",
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        pmcid=pmcid,
        status=StudyStatus.INCLUDED_ABSTRACT,
    )


def _build_fulltext_config() -> ScreeningConfig:
    config = ScreeningConfig()
    config.fulltext.update(
        {
            "objective": "Test objective for fulltext screening",
            "inclusion_criteria": ["Test inclusion criterion"],
            "exclusion_criteria": ["Test exclusion criterion"],
        }
    )
    return config


def test_fulltext_screening():
    """Test full text screening functionality."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        _write_fulltext_csv(
            temp_dir,
            "123456",
            "This is the full text content for testing full text screening.",
        )
        study = _build_fulltext_study("TEST_FT_001")
        config = _build_fulltext_config()

        with patch("autonima.screening.screener.GenericLLMClient") as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            mock_response.fulltext_incomplete = False
            mock_client.screen_fulltext.return_value = mock_response

            screener = LLMScreener(config, output_dir=str(temp_dir))
            results = asyncio.run(screener.screen_fulltexts([study]))

            assert isinstance(results, list)
            assert len(results) == 1
            result = results[0]
            assert result.study_id == "TEST_FT_001"
            assert result.decision == StudyStatus.INCLUDED_FULLTEXT
            mock_client.screen_fulltext.assert_called_once()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fulltext_schema_includes_incomplete_flag_with_default():
    """Test full-text schema exposes fulltext_incomplete with default false."""
    result = FullTextScreeningOutput(
        decision="INCLUDED",
        confidence=0.9,
        reason="Complete full text and meets criteria.",
    )
    assert result.fulltext_incomplete is False


def test_fulltext_incomplete_maps_to_status_and_json_output():
    """Test that fulltext_incomplete maps to study status and is serialized."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        _write_fulltext_csv(
            temp_dir,
            "123456",
            "Title and references only. No methods or results.",
        )
        study = _build_fulltext_study("TEST_FT_INCOMPLETE")
        config = _build_fulltext_config()

        with patch("autonima.screening.screener.GenericLLMClient") as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.74
            mock_response.reason = "Provided document is not complete."
            mock_response.fulltext_incomplete = True
            mock_response.inclusion_criteria_applied = []
            mock_response.exclusion_criteria_applied = []
            mock_client.screen_fulltext.return_value = mock_response

            screener = LLMScreener(config, output_dir=str(temp_dir))
            results = asyncio.run(screener.screen_fulltexts([study]))

            assert len(results) == 1
            assert results[0].decision == StudyStatus.FULLTEXT_INCOMPLETE

            result_file = (
                temp_dir
                / "outputs"
                / "fulltext_screening_results.json"
            )
            payload = json.loads(result_file.read_text())
            assert payload["screening_results"][0]["decision"] == (
                "fulltext_incomplete"
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_force_reextract_incomplete_fulltext_bypasses_cache():
    """Test force flag bypasses cached fulltext_incomplete decisions."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        _write_fulltext_csv(
            temp_dir,
            "123456",
            "Introduction Methods Results Discussion",
        )
        _write_cached_fulltext_result(
            temp_dir, "TEST_FT_CACHE", StudyStatus.FULLTEXT_INCOMPLETE.value
        )
        study = _build_fulltext_study("TEST_FT_CACHE")
        config = _build_fulltext_config()

        with patch("autonima.screening.screener.GenericLLMClient") as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.92
            mock_response.reason = "Now complete and included."
            mock_response.fulltext_incomplete = False
            mock_client.screen_fulltext.return_value = mock_response

            screener = LLMScreener(
                config,
                output_dir=str(temp_dir),
                force_reextract_incomplete_fulltext=True,
            )
            results = asyncio.run(screener.screen_fulltexts([study]))

            assert len(results) == 1
            assert results[0].decision == StudyStatus.INCLUDED_FULLTEXT
            mock_client.screen_fulltext.assert_called_once()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
