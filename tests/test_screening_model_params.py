"""Tests for passing model-specific parameters through to the screening API call."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from autonima.screening.openai_client import ScreeningLLMClient


def _client_with_mocked_api(monkeypatch, arguments):
    """Build a ScreeningLLMClient whose API call returns a canned function call."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    with patch("autonima.llm.client.openai.OpenAI") as mock_openai:
        client = ScreeningLLMClient()

    mock_response = MagicMock()
    mock_response.choices[0].message.function_call.arguments = arguments
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    return client


def test_screen_abstract_sends_no_extra_params_by_default(monkeypatch):
    client = _client_with_mocked_api(
        monkeypatch,
        '{"decision": "INCLUDED", "confidence": 0.9, "reason": "Fits criteria"}'
    )

    client.screen_abstract("Screen this abstract.", "gpt-5-mini-2025-08-07")

    call_kwargs = client.client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in call_kwargs


def test_screen_abstract_passes_model_params_to_api(monkeypatch):
    client = _client_with_mocked_api(
        monkeypatch,
        '{"decision": "INCLUDED", "confidence": 0.9, "reason": "Fits criteria"}'
    )

    result = client.screen_abstract(
        "Screen this abstract.",
        "gpt-5.6-luna",
        model_params={"reasoning_effort": "none"}
    )

    call_kwargs = client.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "none"
    assert call_kwargs["function_call"] == {"name": "screen_abstract"}
    assert result.decision == "INCLUDED"


def test_screen_fulltext_passes_model_params_to_api(monkeypatch):
    client = _client_with_mocked_api(
        monkeypatch,
        '{"decision": "EXCLUDED", "confidence": 0.8, "reason": "Animal study"}'
    )

    client.screen_fulltext(
        "Screen this full text.",
        "gpt-5.6-luna",
        model_params={"reasoning_effort": "none"}
    )

    call_kwargs = client.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "none"
    assert call_kwargs["function_call"] == {"name": "screen_fulltext"}
