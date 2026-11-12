"""Documents that stand in for an article's full text.

See ``docs/guides/documents.md``. The contract lives in ``source``; ``folder`` holds the two
built-in kinds, and ``attach`` applies a source to a run's studies.
"""

from typing import Dict, Type

from .attach import UNAVAILABLE_FILENAME, AttachReport, attach_documents
from .folder import FolderDocumentSource, FolderRecordSource
from .neurostore import NeuroStoreRecordSource
from .source import (
    ANALYSIS_REFERENCE,
    AnalysisBundle,
    Document,
    DocumentError,
    DocumentFormatError,
    DocumentRef,
    DocumentSource,
    IdentityError,
    PairingError,
    PrepareReport,
    SourceAnalysis,
    UnavailableStudy,
)

SOURCE_KINDS: Dict[str, Type[DocumentSource]] = {
    FolderDocumentSource.kind: FolderDocumentSource,
    FolderRecordSource.kind: FolderRecordSource,
    NeuroStoreRecordSource.kind: NeuroStoreRecordSource,
}
# Kinds whose records autonima builds itself, under the run's output directory, so they
# take no ``root``.
BUILT_KINDS = frozenset({NeuroStoreRecordSource.kind})


def create_document_source(config, output_dir=None) -> DocumentSource:
    """Build the source a ``DocumentsConfig`` describes."""
    if config.kind == NeuroStoreRecordSource.kind:
        return NeuroStoreRecordSource(
            output_dir,
            options=config.neurostore,
            description=config.description,
            fallback=config.fallback,
        )
    return SOURCE_KINDS[config.kind](config.root, description=config.description)


__all__ = [
    "ANALYSIS_REFERENCE",
    "AnalysisBundle",
    "AttachReport",
    "BUILT_KINDS",
    "Document",
    "DocumentError",
    "DocumentFormatError",
    "DocumentRef",
    "DocumentSource",
    "FolderDocumentSource",
    "FolderRecordSource",
    "IdentityError",
    "NeuroStoreRecordSource",
    "PairingError",
    "PrepareReport",
    "SOURCE_KINDS",
    "SourceAnalysis",
    "UNAVAILABLE_FILENAME",
    "UnavailableStudy",
    "attach_documents",
    "create_document_source",
]
