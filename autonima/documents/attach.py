"""Attach a document source's documents to the studies of a run."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..coordinates.nimads_models import host_analysis_id
from ..coordinates.schema import Analysis, CoordinatePoint, PointsValue
from ..models.types import ActivationTable, Study, StudyDocument
from .source import (
    AnalysisBundle,
    Document,
    DocumentError,
    DocumentFormatError,
    DocumentSource,
    PairingError,
    UnavailableStudy,
    analysis_references,
    substitute_analysis_ids,
)

logger = logging.getLogger(__name__)

UNAVAILABLE_FILENAME = "documents_unavailable.csv"
UNAVAILABLE_FIELDS = ["pmid", "study_id", "kind", "reason", "detail"]


@dataclass
class AttachReport:
    requested: int
    attached: int
    unavailable: List[UnavailableStudy] = field(default_factory=list)


def attach_documents(
    studies: Sequence[Study],
    source: DocumentSource,
    *,
    output_dir: str | Path,
) -> AttachReport:
    """Make each study's document what the LLM reads in place of its article.

    Per attached study the document is materialized to ``<output>/documents/``, and
    ``full_text_path`` points at it. ``Study.full_text`` and every full-text cache hash then
    read the document with no further change, and a change to what the source supplies
    re-screens exactly the studies it touches.

    When the source provides analyses they replace the study's analyses and tables, and the
    document's analysis references are rewritten to autonima's ids.

    A study with no usable document is treated as one whose full text was not retrieved: it
    is not screened, it is listed in ``missing_fulltexts.csv`` like any other, and
    ``documents_unavailable.csv`` records why. It never falls back to the article, which
    would mix two kinds of input in one run.
    """
    output_dir = Path(output_dir)
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    pmids = {int(study.pmid) for study in studies if str(study.pmid).isdigit()}
    report = source.prepare(pmids)
    index = source.index()
    unavailable: List[UnavailableStudy] = []
    attached = 0

    for study in studies:
        if not str(study.pmid).isdigit():
            unavailable.append(
                UnavailableStudy(pmid=None, reason="no_pmid", detail=f"study id {study.pmid!r}")
            )
            _mark_unavailable(study, source)
            continue
        ref = index.get(int(study.pmid))
        if ref is None:
            unavailable.append(UnavailableStudy(pmid=int(study.pmid), reason="no_document"))
            _mark_unavailable(study, source)
            continue
        try:
            document = source.open(ref)
            if source.provides_analyses:
                text = _attach_analyses(study, document, source.open_analyses(ref))
            else:
                if analysis_references(document.text):
                    raise PairingError(
                        f"{ref.path} refers to analyses, but a '{source.kind}' source does "
                        "not supply analyses"
                    )
                text = document.text
                study.document = _study_document(document)
        except DocumentError as exc:
            logger.warning("Document for PMID %s is unusable: %s", study.pmid, exc)
            unavailable.append(
                UnavailableStudy(
                    pmid=int(study.pmid),
                    reason=exc.reason,
                    detail=str(exc),
                    study_id=ref.study_id,
                )
            )
            _mark_unavailable(study, source)
            continue

        materialized = documents_dir / f"{ref.study_id}{ref.path.suffix.lower()}"
        _write_if_changed(materialized, text)
        study.full_text_path = str(materialized)
        study.full_text_source = f"document:{source.kind}"
        study.full_text_output_dir = str(output_dir)
        study.fulltext_available = True
        study._full_text = None
        attached += 1

    _write_unavailable(output_dir / "outputs" / UNAVAILABLE_FILENAME, unavailable, source.kind)
    logger.info(
        "Documents: attached %s of %s (%s unavailable) from '%s' source",
        attached,
        len(studies),
        len(unavailable),
        source.kind,
    )
    return AttachReport(
        requested=report.requested,
        attached=attached,
        unavailable=unavailable,
    )


def _study_document(document: Document) -> StudyDocument:
    return StudyDocument(
        study_id=document.ref.study_id,
        kind=document.ref.kind,
        description=document.description,
        source_path=str(document.ref.path),
        content_hash=document.content_hash,
    )


def _attach_analyses(study: Study, document: Document, bundle: AnalysisBundle) -> str:
    """Replace the study's analyses with the source's, and return the rewritten document."""
    if bundle.document_sha256 != document.content_hash:
        raise PairingError(
            f"{bundle.path} was produced with a document whose sha256 is "
            f"{bundle.document_sha256}, but {document.ref.path} hashes to "
            f"{document.content_hash}"
        )

    host_ids = [host_analysis_id(study.pmid, i) for i in range(len(bundle.analyses))]
    id_map = bundle.host_id_map(host_ids)
    text = substitute_analysis_ids(document.text, id_map, where=str(document.ref.path))

    analyses: List[Analysis] = []
    analysis_documents: List[Optional[str]] = []
    for position, source_analysis in enumerate(bundle.analyses):
        where = f"{bundle.path} analyses[{position}]"
        default_space = source_analysis.space or bundle.space
        try:
            points = [
                CoordinatePoint(
                    coordinates=point.get("coordinates"),
                    space=point.get("space") or default_space,
                    values=[PointsValue(**value) for value in point.get("values") or []] or None,
                )
                for point in source_analysis.points
            ]
        except (TypeError, ValueError) as exc:
            raise DocumentFormatError(f"{where}: invalid point ({exc})") from exc
        analyses.append(
            Analysis(
                name=source_analysis.name,
                description=source_analysis.description,
                points=points,
                parsed=False,
                table_id=source_analysis.table_id,
            )
        )
        analysis_documents.append(
            substitute_analysis_ids(source_analysis.document, id_map, where=where)
            if source_analysis.document
            else None
        )

    study.analyses = analyses
    study.activation_tables = _tables_for(bundle)
    if bundle.space:
        study.coordinate_space = bundle.space
    study.document = replace(
        _study_document(document),
        provides_analyses=True,
        analyses_path=str(bundle.path),
        analyses_hash=bundle.content_hash,
        analysis_keys=bundle.keys,
        analysis_ids=host_ids,
        analysis_documents=analysis_documents,
    )
    return text


def _tables_for(bundle: AnalysisBundle) -> List[ActivationTable]:
    """Tables the analyses name, declared or not.

    Annotation looks up every analysis's ``table_id`` among the study's tables and refuses an
    analysis whose table is missing, so an undeclared one gets a bare entry.
    """
    tables: Dict[str, ActivationTable] = {}
    for table in bundle.tables:
        table_id = table["table_id"]
        tables[table_id] = ActivationTable(
            table_id=table_id,
            table_label=table.get("table_label") or table_id,
            table_caption=table.get("table_caption"),
            table_foot=table.get("table_foot"),
        )
    for analysis in bundle.analyses:
        if analysis.table_id and analysis.table_id not in tables:
            tables[analysis.table_id] = ActivationTable(
                table_id=analysis.table_id,
                table_label=analysis.table_id,
            )
    return list(tables.values())


def _mark_unavailable(study: Study, source: DocumentSource) -> None:
    """Leave the study exactly as if its full text had not been retrieved."""
    study.fulltext_available = False
    study.full_text_path = None
    study.full_text_source = f"document:{source.kind}"
    study.document = None
    study._full_text = None
    # Without a document the study has no input in this run, so nothing retrieved from its
    # article -- tables or ingested coordinates -- may reach annotation or the export either.
    study.activation_tables = []
    study.analyses = []


def _write_if_changed(path: Path, text: str) -> None:
    """Rewrite only on change, so an unchanged document keeps its mtime across runs."""
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _write_unavailable(path: Path, unavailable: Sequence[UnavailableStudy], kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=UNAVAILABLE_FIELDS)
        writer.writeheader()
        for item in sorted(unavailable, key=lambda u: (u.pmid is None, u.pmid or 0, u.detail)):
            writer.writerow(
                {
                    "pmid": "" if item.pmid is None else item.pmid,
                    "study_id": item.study_id,
                    "kind": kind,
                    "reason": item.reason,
                    "detail": item.detail,
                }
            )
