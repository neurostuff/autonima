"""The document contract: what a document source supplies, and how it fails.

A document source supplies, per study, a serialized document that the LLM reads in place of
the article's full text during full-text screening and annotation. Nothing here knows what
produced the document -- an extraction record, a summary, a compression -- only that it is
text, and that a hash pins it to the file it came from.

A source may also supply each study's analyses (``provides_analyses``). That is a declared
capability, never inferred from a file being absent: such a document refers to analyses by
the source's own keys, so the analyses must travel with it, and autonima must never re-parse
them from the article. A re-parse that splits one table differently does not lose an
analysis, it renumbers every one after the split, and attaches each to another contrast's
coordinates with nothing to detect it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class DocumentError(Exception):
    """A study's document cannot be used. Every raise names the study and the file.

    ``reason`` is the short code recorded in ``documents_unavailable.csv``.
    """

    reason = "invalid_document"


class DocumentFormatError(DocumentError):
    """A file does not follow the documented layout."""

    reason = "invalid_document"


class IdentityError(DocumentError):
    """Two documents claim one study, or analyses and their ids no longer line up."""

    reason = "identity_conflict"


class PairingError(DocumentError):
    """A document and its analyses do not belong together."""

    reason = "pairing_mismatch"


# How a document refers to one of the analyses its source supplies. Autonima replaces each
# reference with its own id for that analysis, so the model reads -- and answers in -- one
# id space. Anything the source writes outside a reference reaches the model unchanged.
ANALYSIS_REFERENCE = re.compile(r"\{\{analysis:([^{}]+)\}\}")


def analysis_references(text: str) -> List[str]:
    """Return the analysis keys a document refers to, in order of first appearance."""
    seen: Dict[str, None] = {}
    for key in ANALYSIS_REFERENCE.findall(text):
        seen.setdefault(key.strip(), None)
    return list(seen)


def substitute_analysis_ids(text: str, id_map: Mapping[str, str], *, where: str) -> str:
    """Replace every analysis reference with autonima's id for it.

    Raises PairingError when the text refers to an analysis the source did not supply: that
    reference would otherwise reach the model as an id nothing answers to.
    """
    unknown = [key for key in analysis_references(text) if key not in id_map]
    if unknown:
        raise PairingError(
            f"{where} refers to analyses its source does not supply: {', '.join(unknown)}"
        )
    return ANALYSIS_REFERENCE.sub(lambda match: id_map[match.group(1).strip()], text)


@dataclass(frozen=True)
class DocumentRef:
    """Where one study's document lives. ``study_id`` is the source's key, opaque here."""

    study_id: str
    pmid: int
    kind: str
    path: Path


@dataclass(frozen=True)
class Document:
    ref: DocumentRef
    text: str  # What the LLM reads, in place of the article.
    content_hash: str  # sha256 of the source file's bytes.
    description: str  # What the document is, for the prompt and the audit trail.


@dataclass(frozen=True)
class SourceAnalysis:
    """One analysis as the source supplies it, addressed by the source's own key."""

    key: str
    name: Optional[str]
    description: Optional[str]
    table_id: Optional[str]
    space: Optional[str]
    points: Tuple[Mapping[str, Any], ...]
    document: Optional[str]  # The analysis's own serialized record, if the source has one.


@dataclass(frozen=True)
class AnalysisBundle:
    """A study's analyses, in the order its document numbers them."""

    analyses: Tuple[SourceAnalysis, ...]
    tables: Tuple[Mapping[str, Any], ...]
    space: Optional[str]
    document_sha256: str  # Pins these analyses to the document they were produced with.
    content_hash: str  # sha256 of the analyses file's bytes.
    path: Path

    @property
    def keys(self) -> List[str]:
        return [analysis.key for analysis in self.analyses]

    def host_id_map(self, host_ids: Sequence[str]) -> Dict[str, str]:
        """Map each source key to the host id at the same position.

        The mapping is positional and involves no model: analysis ``i`` in the bundle becomes
        analysis ``i`` of the study. Raises IdentityError when the two lists differ in length,
        which is the only sign that the analyses were re-parsed or re-ordered in between.
        """
        if len(host_ids) != len(self.analyses):
            raise IdentityError(
                f"{self.path}: {len(self.analyses)} analyses supplied but "
                f"{len(host_ids)} host ids to map them to"
            )
        return dict(zip(self.keys, host_ids))


@dataclass(frozen=True)
class UnavailableStudy:
    pmid: Optional[int]
    reason: str  # "no_pmid" | "no_document" | a DocumentError.reason
    detail: str = ""
    study_id: str = ""


@dataclass
class PrepareReport:
    requested: int
    ready: int
    unavailable: List[UnavailableStudy]
    notes: List[str] = field(default_factory=list)


class DocumentSource:
    """Supplies one document per study.

    Subclasses declare ``kind`` and ``provides_analyses``. A source that provides analyses
    must define ``open_analyses``; one that does not must not, so the capability cannot be
    claimed without the method that honours it, or the method exist without the claim.
    """

    kind: ClassVar[str]
    provides_analyses: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.provides_analyses != callable(getattr(cls, "open_analyses", None)):
            raise TypeError(
                f"{cls.__name__}: provides_analyses={cls.provides_analyses} but "
                f"open_analyses is {'missing' if cls.provides_analyses else 'defined'}"
            )

    def prepare(self, pmids: Iterable[int]) -> PrepareReport:
        """Make the documents for ``pmids`` available. May be expensive; idempotent."""
        raise NotImplementedError

    def index(self) -> Mapping[int, DocumentRef]:
        """PMID -> document. Cheap and pure, valid after prepare()."""
        raise NotImplementedError

    def open(self, ref: DocumentRef) -> Document:
        raise NotImplementedError
