"""
Autonima: Automated Neuroimaging Meta-Analysis.

This module keeps package import lightweight. Heavy optional dependencies are
loaded only when their symbols are requested.
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "Autonima Development Team"
__description__ = "LLM-powered automated systematic review and meta-analysis"

__all__ = [
    "AutonimaPipeline",
    "PipelineConfig",
    "BaseRetriever",
    "PubGetRetriever",
    "CoordinatePoint",
    "Analysis",
    "ParseAnalysesOutput",
    "parse_tables",
    "CoordinateParsingClient",
    "CoordinateProcessor",
    "GenericLLMClient",
    "run_meta_analyses",
    "HAS_META",
]

_LAZY_IMPORTS = {
    "AutonimaPipeline": ("autonima.pipeline", "AutonimaPipeline"),
    "PipelineConfig": ("autonima.config", "PipelineConfig"),
    "BaseRetriever": ("autonima.retrieval.base", "BaseRetriever"),
    "PubGetRetriever": ("autonima.retrieval.pubget", "PubGetRetriever"),
    "CoordinatePoint": ("autonima.coordinates.schema", "CoordinatePoint"),
    "Analysis": ("autonima.coordinates.schema", "Analysis"),
    "ParseAnalysesOutput": (
        "autonima.coordinates.schema",
        "ParseAnalysesOutput",
    ),
    "parse_tables": ("autonima.coordinates.parser", "parse_tables"),
    "CoordinateParsingClient": (
        "autonima.coordinates.openai_client",
        "CoordinateParsingClient",
    ),
    "CoordinateProcessor": (
        "autonima.coordinates.processor",
        "CoordinateProcessor",
    ),
    "GenericLLMClient": ("autonima.llm.client", "GenericLLMClient"),
    "run_meta_analyses": ("autonima.meta", "run_meta_analyses"),
}


def __getattr__(name):
    if name == "HAS_META":
        try:
            import_module("autonima.meta")
        except ImportError:
            return False
        return True

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'autonima' has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
