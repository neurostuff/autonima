"""Tests for GenericLLMClient environment-variable behavior."""

from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from autonima.llm.client import GenericLLMClient


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
