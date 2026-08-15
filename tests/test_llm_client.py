"""Tests for GenericLLMClient environment-variable behavior."""

from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from autonima.llm.client import GenericLLMClient, resolve_model_name


def test_openai_api_gateway_sets_default_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_API_GATEWAY", "https://gateway.example.com/v1")

    with patch("autonima.llm.client.openai.OpenAI") as mock_openai:
        client = GenericLLMClient()

    assert client.base_url == "https://gateway.example.com/v1"
    assert client.api_key == "openai-test-key"
    mock_openai.assert_called_once_with(
        api_key="openai-test-key",
        base_url="https://gateway.example.com/v1",
    )


def test_gateway_uses_openai_api_key_even_for_non_openai_hosts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_API_GATEWAY", "https://gateway.proxy.example/v1")

    with patch("autonima.llm.client.openai.OpenAI") as mock_openai:
        client = GenericLLMClient()

    assert client.base_url == "https://gateway.proxy.example/v1"
    assert client.api_key == "openai-test-key"
    mock_openai.assert_called_once_with(
        api_key="openai-test-key",
        base_url="https://gateway.proxy.example/v1",
    )


def test_explicit_base_url_overrides_openai_api_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_API_GATEWAY", "https://gateway.example.com/v1")
    explicit_base_url = "https://api.custom-gateway.example/v1"

    with patch("autonima.llm.client.openai.OpenAI") as mock_openai:
        client = GenericLLMClient(base_url=explicit_base_url)

    assert client.base_url == explicit_base_url
    assert client.api_key == "openai-test-key"
    mock_openai.assert_called_once_with(
        api_key="openai-test-key",
        base_url=explicit_base_url,
    )


def test_model_name_unchanged_without_prefix_env(monkeypatch):
    monkeypatch.delenv("AUTONIMA_MODEL_PREFIX", raising=False)

    assert resolve_model_name("gpt-5-mini-2025-08-07") == "gpt-5-mini-2025-08-07"


def test_model_prefix_qualifies_bare_model_name(monkeypatch):
    monkeypatch.setenv("AUTONIMA_MODEL_PREFIX", "@my-provider")

    assert (
        resolve_model_name("gpt-5-mini-2025-08-07")
        == "@my-provider/gpt-5-mini-2025-08-07"
    )


def test_model_prefix_leaves_already_qualified_name_alone(monkeypatch):
    monkeypatch.setenv("AUTONIMA_MODEL_PREFIX", "@my-provider")

    assert (
        resolve_model_name("@other-provider/gpt-5.1") == "@other-provider/gpt-5.1"
    )


def test_model_prefix_accepts_full_model_name_and_keeps_config_model(monkeypatch):
    """A prefix written as a full model name still yields the config's own model."""
    monkeypatch.setenv("AUTONIMA_MODEL_PREFIX", "@my-provider/gpt-5-mini-2025-08-07")

    assert resolve_model_name("gpt-5.2-2025-12-11") == "@my-provider/gpt-5.2-2025-12-11"


def test_model_prefix_ignores_empty_value_and_none_model(monkeypatch):
    monkeypatch.setenv("AUTONIMA_MODEL_PREFIX", "   ")

    assert resolve_model_name("gpt-5-mini-2025-08-07") == "gpt-5-mini-2025-08-07"
    assert resolve_model_name(None) is None
