"""Document sources backed by one flat directory.

Layout, one file per study, named by PMID::

    <root>/12345678.md               the document (.md, .txt, .json, .yaml or .yml)
    <root>/12345678.analyses.json    records sources only: the analyses it refers to

Documents are read verbatim; a serialized schema is passed to the model as it was written.
The analyses file is described in ``docs/guides/documents.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .source import (
    AnalysisBundle,
    Document,
    DocumentFormatError,
    DocumentRef,
    DocumentSource,
    IdentityError,
    PairingError,
    PrepareReport,
    SourceAnalysis,
    UnavailableStudy,
)

DOCUMENT_SUFFIXES = (".md", ".txt", ".json", ".yaml", ".yml")
ANALYSES_SUFFIX = ".analyses.json"
DEFAULT_DESCRIPTION = (
    "a representation of the article prepared by another tool, such as a structured "
    "extraction record or a summary"
)

_PMID_STEM = re.compile(r"[0-9]+")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _optional_text(value: Any, *, field: str, where: str) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    raise DocumentFormatError(f"{where}: '{field}' must be a string, got {type(value).__name__}")


class FolderDocumentSource(DocumentSource):
    """``kind: text`` -- documents only. Coordinates still come from the article."""

    kind = "text"
    provides_analyses = False

    def __init__(self, root: str | Path, description: Optional[str] = None):
        self.root = Path(root).expanduser()
        self.description = description or DEFAULT_DESCRIPTION
        self._index: Optional[Dict[int, DocumentRef]] = None

    def prepare(self, pmids: Iterable[int]) -> PrepareReport:
        if not self.root.is_dir():
            raise DocumentFormatError(f"Document root {self.root} is not a directory")
        index: Dict[int, DocumentRef] = {}
        for path in sorted(self.root.iterdir()):
            if (
                not path.is_file()
                or path.suffix.lower() not in DOCUMENT_SUFFIXES
                or not _PMID_STEM.fullmatch(path.stem)
            ):
                # Skips <pmid>.analyses.json, whose stem is not a bare PMID.
                continue
            pmid = int(path.stem)
            if pmid in index:
                raise IdentityError(
                    f"PMID {pmid} has two documents in {self.root}: "
                    f"{index[pmid].path.name} and {path.name}"
                )
            index[pmid] = DocumentRef(
                study_id=path.stem, pmid=pmid, kind=self.kind, path=path
            )
        self._index = index

        requested = set(pmids)
        missing = sorted(requested - set(index))
        return PrepareReport(
            requested=len(requested),
            ready=len(requested) - len(missing),
            unavailable=[
                UnavailableStudy(pmid=pmid, reason="no_document") for pmid in missing
            ],
        )

    def index(self) -> Mapping[int, DocumentRef]:
        if self._index is None:
            raise RuntimeError(f"{type(self).__name__}.index() called before prepare()")
        return self._index

    def open(self, ref: DocumentRef) -> Document:
        data = ref.path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentFormatError(f"{ref.path}: not UTF-8 text ({exc})") from exc
        if not text.strip():
            raise DocumentFormatError(f"{ref.path}: document is empty")
        return Document(
            ref=ref,
            text=text,
            content_hash=_sha256(data),
            description=self.description,
        )


class FolderRecordSource(FolderDocumentSource):
    """``kind: records`` -- documents plus the analyses they refer to."""

    kind = "records"
    provides_analyses = True

    def analyses_path(self, ref: DocumentRef) -> Path:
        return self.root / f"{ref.study_id}{ANALYSES_SUFFIX}"

    def open_analyses(self, ref: DocumentRef) -> AnalysisBundle:
        path = self.analyses_path(ref)
        if not path.is_file():
            raise PairingError(f"{ref.path.name} has no analyses file ({path.name})")
        data = path.read_bytes()
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentFormatError(f"{path}: not valid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise DocumentFormatError(f"{path}: must be a JSON object")

        document_sha256 = payload.get("document_sha256")
        if not isinstance(document_sha256, str) or not document_sha256:
            raise PairingError(
                f"{path}: 'document_sha256' is required -- the sha256 of the document these "
                "analyses were produced with"
            )

        raw_analyses = payload.get("analyses")
        if not isinstance(raw_analyses, list):
            raise DocumentFormatError(f"{path}: 'analyses' must be a list")
        analyses: List[SourceAnalysis] = []
        seen_keys: Dict[str, int] = {}
        for position, raw in enumerate(raw_analyses):
            where = f"{path} analyses[{position}]"
            if not isinstance(raw, dict):
                raise DocumentFormatError(f"{where}: must be an object")
            key = raw.get("key")
            if not isinstance(key, str) or not key.strip():
                raise DocumentFormatError(f"{where}: 'key' must be a non-empty string")
            key = key.strip()
            if key in seen_keys:
                raise IdentityError(
                    f"{path}: key '{key}' is used by analyses[{seen_keys[key]}] and "
                    f"analyses[{position}]"
                )
            seen_keys[key] = position
            points = raw.get("points", [])
            if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
                raise DocumentFormatError(f"{where}: 'points' must be a list of objects")
            document = raw.get("document")
            if document is not None and not isinstance(document, str):
                # A structured per-analysis record is passed to the model serialized.
                document = json.dumps(document, indent=2, ensure_ascii=False)
            analyses.append(
                SourceAnalysis(
                    key=key,
                    name=_optional_text(raw.get("name"), field="name", where=where),
                    description=_optional_text(
                        raw.get("description"), field="description", where=where
                    ),
                    table_id=_optional_text(raw.get("table_id"), field="table_id", where=where),
                    space=_optional_text(raw.get("space"), field="space", where=where),
                    points=tuple(points),
                    document=document,
                )
            )

        tables = payload.get("tables", [])
        if not isinstance(tables, list) or not all(
            isinstance(table, dict) and isinstance(table.get("table_id"), str)
            for table in tables
        ):
            raise DocumentFormatError(
                f"{path}: 'tables' must be a list of objects with a string 'table_id'"
            )

        return AnalysisBundle(
            analyses=tuple(analyses),
            tables=tuple(tables),
            space=_optional_text(payload.get("space"), field="space", where=str(path)),
            document_sha256=document_sha256,
            content_hash=_sha256(data),
            path=path,
        )
