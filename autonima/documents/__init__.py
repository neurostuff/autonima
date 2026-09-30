"""Documents that stand in for an article's full text.

See ``docs/guides/documents.md``. The contract lives in ``source``; ``folder`` holds the two
built-in kinds, and ``attach`` applies a source to a run's studies.
"""

from typing import Dict, Type

from .attach import UNAVAILABLE_FILENAME, AttachReport, attach_documents
from .folder import FolderDocumentSource, FolderRecordSource
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
}


def create_document_source(config) -> DocumentSource:
    """Build the source a ``DocumentsConfig`` describes."""
    return SOURCE_KINDS[config.kind](config.root, description=config.description)


__all__ = [
    "ANALYSIS_REFERENCE",
    "AnalysisBundle",
    "AttachReport",
    "Document",
    "DocumentError",
    "DocumentFormatError",
    "DocumentRef",
    "DocumentSource",
    "FolderDocumentSource",
    "FolderRecordSource",
    "IdentityError",
    "PairingError",
    "PrepareReport",
    "SOURCE_KINDS",
    "SourceAnalysis",
    "UNAVAILABLE_FILENAME",
    "UnavailableStudy",
    "attach_documents",
    "create_document_source",
]
