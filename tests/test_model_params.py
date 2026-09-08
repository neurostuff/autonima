"""Tests for deployment-time model resolution: extra kwargs (#62) and the gateway prefix.

Both functions read the environment and both fail *silently* when misconfigured -- a malformed
AUTONIMA_MODEL_PARAMS yields {} rather than raising, which at the call site is indistinguishable
from "no parameters were configured". That is the right behaviour for a deployment detail, but it
means the parsing rules need tests or a typo in an env var looks like the feature not working.
"""

import pytest

from autonima.llm.client import (
    MODEL_PARAMS_ENV,
    MODEL_PREFIX_ENV,
    resolve_model_kwargs,
    resolve_model_name,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Neither variable may leak in from the developer's own shell."""
    monkeypatch.delenv(MODEL_PARAMS_ENV, raising=False)
    monkeypatch.delenv(MODEL_PREFIX_ENV, raising=False)


# --------------------------------------------------------------------- resolve_model_kwargs

def test_no_configuration_sends_no_extra_kwargs():
    assert resolve_model_kwargs("gpt-5-mini") == {}
    assert resolve_model_kwargs(None) == {}


def test_flat_json_applies_to_every_model(monkeypatch):
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"reasoning_effort": "none"}')
    assert resolve_model_kwargs("gpt-5.6-luna") == {"reasoning_effort": "none"}
    assert resolve_model_kwargs("anything-at-all") == {"reasoning_effort": "none"}


def test_nested_json_applies_only_to_the_named_model(monkeypatch):
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"gpt-5.6-luna": {"reasoning_effort": "none"}}')
    assert resolve_model_kwargs("gpt-5.6-luna") == {"reasoning_effort": "none"}
    # This is the case #62 is about: the parameter must NOT reach a model that rejects it.
    assert resolve_model_kwargs("gpt-5-mini-2025-08-07") == {}


def test_a_short_key_covers_a_dated_snapshot(monkeypatch):
    """Configs pin dated names; nobody wants to update the env var on every model refresh."""
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"gpt-5-mini": {"temperature": 0}}')
    assert resolve_model_kwargs("gpt-5-mini-2025-08-07") == {"temperature": 0}


def test_matching_ignores_the_gateway_prefix(monkeypatch):
    """The model reaching the API is provider-qualified; the env key is written bare."""
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"gpt-5.6-luna": {"reasoning_effort": "none"}}')
    assert resolve_model_kwargs("@my-provider/gpt-5.6-luna") == {"reasoning_effort": "none"}


def test_config_params_beat_the_environment(monkeypatch):
    """A run's own config is a semantic choice; the env var is a deployment default."""
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"reasoning_effort": "none"}')
    assert resolve_model_kwargs("gpt-5.6-luna", {"reasoning_effort": "low"}) == {
        "reasoning_effort": "low"
    }


def test_config_params_merge_rather_than_replace(monkeypatch):
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"reasoning_effort": "none"}')
    assert resolve_model_kwargs("m", {"temperature": 0}) == {
        "reasoning_effort": "none", "temperature": 0,
    }


def test_config_params_apply_with_no_environment_set():
    assert resolve_model_kwargs("m", {"reasoning_effort": "none"}) == {"reasoning_effort": "none"}


@pytest.mark.parametrize("raw", [
    "not json at all",
    "{unclosed",
    '["a", "list"]',        # valid JSON, wrong shape
    '"just a string"',
    "null",
    "   ",                  # whitespace only
])
def test_malformed_environment_is_ignored_not_raised(monkeypatch, raw):
    """A typo in a deployment env var must not take a 20,000-study run down mid-flight."""
    monkeypatch.setenv(MODEL_PARAMS_ENV, raw)
    assert resolve_model_kwargs("gpt-5.6-luna") == {}


def test_one_nested_value_makes_the_whole_block_nested(monkeypatch):
    """Sharp edge worth pinning: nested and flat keys do not combine.

    As soon as any value is an object the block is read as model-keyed, and top-level scalars are
    dropped rather than applied globally. Mixing the two forms silently loses the flat keys, so
    the two styles must not be combined in one variable.
    """
    monkeypatch.setenv(
        MODEL_PARAMS_ENV,
        '{"temperature": 0, "gpt-5.6-luna": {"reasoning_effort": "none"}}',
    )
    assert resolve_model_kwargs("gpt-5.6-luna") == {"reasoning_effort": "none"}
    assert "temperature" not in resolve_model_kwargs("gpt-5.6-luna")


def test_matching_is_substring_in_both_directions(monkeypatch):
    """Documents a real risk: an over-short key matches models it was not meant for.

    The match accepts `key in model` OR `model in key`, which is what lets a short key cover a
    dated snapshot. The cost is that a key like "gpt" matches everything. Keep env keys specific.
    """
    monkeypatch.setenv(MODEL_PARAMS_ENV, '{"gpt": {"reasoning_effort": "none"}}')
    assert resolve_model_kwargs("gpt-5-mini-2025-08-07") == {"reasoning_effort": "none"}
    assert resolve_model_kwargs("claude-opus-5") == {}


def test_several_models_can_be_configured_at_once(monkeypatch):
    monkeypatch.setenv(
        MODEL_PARAMS_ENV,
        '{"gpt-5.6-luna": {"reasoning_effort": "none"}, "gpt-5-mini": {"temperature": 0}}',
    )
    assert resolve_model_kwargs("gpt-5.6-luna") == {"reasoning_effort": "none"}
    assert resolve_model_kwargs("gpt-5-mini-2025-08-07") == {"temperature": 0}


# ----------------------------------------------------------------------- resolve_model_name

def test_name_is_unchanged_without_a_prefix():
    assert resolve_model_name("gpt-5-mini") == "gpt-5-mini"


def test_prefix_is_applied_at_request_time(monkeypatch):
    monkeypatch.setenv(MODEL_PREFIX_ENV, "@my-provider")
    assert resolve_model_name("gpt-5-mini") == "@my-provider/gpt-5-mini"


def test_an_already_qualified_name_is_not_double_prefixed(monkeypatch):
    """Configs that pin a full provider-qualified name must keep working."""
    monkeypatch.setenv(MODEL_PREFIX_ENV, "@my-provider")
    assert resolve_model_name("@other/gpt-5-mini") == "@other/gpt-5-mini"


def test_only_the_leading_segment_of_the_prefix_is_used(monkeypatch):
    """Writing the whole example model into the env var is common; it must not override the config's
    own model choice."""
    monkeypatch.setenv(MODEL_PREFIX_ENV, "@my-provider/some-other-model")
    assert resolve_model_name("gpt-5-mini") == "@my-provider/gpt-5-mini"


@pytest.mark.parametrize("value", ["", "   ", "/", "///"])
def test_empty_or_degenerate_prefix_leaves_the_name_alone(monkeypatch, value):
    monkeypatch.setenv(MODEL_PREFIX_ENV, value)
    assert resolve_model_name("gpt-5-mini") == "gpt-5-mini"


@pytest.mark.parametrize("model", [None, ""])
def test_absent_model_passes_through(monkeypatch, model):
    monkeypatch.setenv(MODEL_PREFIX_ENV, "@my-provider")
    assert resolve_model_name(model) == model
