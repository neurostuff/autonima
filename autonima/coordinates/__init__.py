"""Coordinate parsing submodule for Autonima.

Imports are deferred for the same reason as in the top-level package: reading one symbol from
here should not drag in the whole parsing runtime. Eager re-exports made ``autonima.coordinates``
-- and therefore anything that touches ``autonima.coordinates.prompts``, which includes
``autonima.execution`` and so the CLI -- require ``tqdm`` and the OpenAI SDK. That broke the docs
build, which imports the CLI to generate its reference page without installing the ``llm`` extra.
"""

from importlib import import_module

__all__ = [
    "CoordinatePoint",
    "Analysis",
    "ParseAnalysesOutput",
    "parse_tables",
    "CoordinateParsingClient",
    "CoordinateProcessor",
]

_LAZY_IMPORTS = {
    "CoordinatePoint": ("autonima.coordinates.schema", "CoordinatePoint"),
    "Analysis": ("autonima.coordinates.schema", "Analysis"),
    "ParseAnalysesOutput": ("autonima.coordinates.schema", "ParseAnalysesOutput"),
    "parse_tables": ("autonima.coordinates.parser", "parse_tables"),
    "CoordinateParsingClient": (
        "autonima.coordinates.openai_client",
        "CoordinateParsingClient",
    ),
    "CoordinateProcessor": (
        "autonima.coordinates.processor",
        "CoordinateProcessor",
    ),
}


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'autonima.coordinates' has no attribute {name!r}")

    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
