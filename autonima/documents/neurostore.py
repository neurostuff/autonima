"""NeuroStore records: each study's analyses and the features NeuroStore extracted for them.

``retrieval.records.kind: neurostore`` reads no article. Per included study it takes:

- the analyses and coordinates of the study's NeuroStore version, chosen as
  ``parsing.coordinate_source: neurostore`` chooses them (the first of ``sources`` with
  coordinates, analyses filtered to ``roles``), frozen in
  ``outputs/neurostore_record_analyses.json``;
- the features of those analyses from NeuroStore's analysis query, frozen in
  ``outputs/neurostore_features.json``.

It writes each study's record -- title, abstract and one block per analysis with its
features -- and the analyses that record refers to into ``<output>/neurostore_records/``,
in the layout of ``retrieval.records.kind: analyses``, and serves them through that source.
Full-text screening and annotation then read the record instead of the article.

A study is available only when it has analyses with coordinates and NeuroStore has features
for at least one of them. Every other study is reported unavailable, with its reason, and is
never screened from anything else: a run reads one kind of input.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..neurostore.analyses import NeuroStoreOptions, NeuroStoreSnapshot
from ..neurostore.client import DEFAULT_BASE_URL, NeuroStoreClient
from ..neurostore.features import (
    FEATURES_FILENAME,
    RECORD_DESCRIPTION,
    FeatureSnapshot,
    analysis_block,
    analysis_lines,
    render_record,
)
from .folder import ANALYSES_SUFFIX, FolderRecordSource
from .source import (
    AnalysisBundle,
    Document,
    DocumentRef,
    DocumentSource,
    PrepareReport,
    UnavailableStudy,
)

logger = logging.getLogger(__name__)

RECORDS_DIRNAME = "neurostore_records"
# A snapshot of its own: the parsing stage's neurostore_analyses.json is cleared with that
# stage, and these analyses belong to retrieval.
RECORD_ANALYSES_FILENAME = "neurostore_record_analyses.json"
# Bump when the record's layout or wording changes; it enters the retrieval stage signature.
RECORD_FORMAT_VERSION = "neurostore-record-v3"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NeuroStoreRecordSource(DocumentSource):
    """``retrieval.records.kind: neurostore``: records built from NeuroStore, with their analyses."""

    kind = "neurostore"
    provides_analyses = True

    def __init__(
        self,
        output_dir: str | Path,
        options: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        client: Optional[NeuroStoreClient] = None,
        workers: int = 8,
    ):
        options = dict(options or {})
        self.output_dir = Path(output_dir)
        self.base_url = options.get("base_url") or DEFAULT_BASE_URL
        self.options = NeuroStoreOptions.from_config(options)
        self.description = description or RECORD_DESCRIPTION
        self.client = client or NeuroStoreClient(self.base_url)
        self.workers = workers
        self.records_dir = self.output_dir / RECORDS_DIRNAME
        self._folder = FolderRecordSource(self.records_dir, description=self.description)
        self._index: Optional[Dict[int, DocumentRef]] = None

    @property
    def outputs(self) -> Path:
        return self.output_dir / "outputs"

    def prepare(self, pmids: Iterable[int], studies: Optional[Mapping[int, Any]] = None) -> PrepareReport:
        requested = sorted({int(p) for p in pmids})
        studies = studies or {}
        analyses = NeuroStoreSnapshot(
            self.outputs / RECORD_ANALYSES_FILENAME, self.options, self.base_url
        )
        analysis_stats = analyses.ensure(self.client, [str(p) for p in requested], workers=self.workers)
        papers = {
            record["base_study_id"]
            for record in (analyses.get(str(p)) for p in requested)
            if record and record.get("analyses") and record.get("base_study_id")
        }
        features = FeatureSnapshot(self.outputs / FEATURES_FILENAME, self.base_url)
        feature_stats = features.ensure(self.client, papers)

        self.records_dir.mkdir(parents=True, exist_ok=True)
        unavailable: List[UnavailableStudy] = []
        ready: List[int] = []
        analyses_with_features = analyses_total = 0
        for pmid in requested:
            record = analyses.get(str(pmid))
            built = self._build(pmid, record, features, studies.get(pmid))
            if isinstance(built, UnavailableStudy):
                unavailable.append(built)
                self._remove(pmid)
                continue
            text, payload, matched = built
            self._write(pmid, text, payload)
            ready.append(pmid)
            analyses_with_features += matched
            analyses_total += len(payload["analyses"])

        folder_report = self._folder.prepare(ready)
        ready_set = set(ready)
        self._index = {
            pmid: DocumentRef(study_id=ref.study_id, pmid=pmid, kind=self.kind, path=ref.path)
            for pmid, ref in self._folder.index().items()
            if pmid in ready_set
        }
        notes = [
            f"analyses: {analysis_stats}",
            f"features: {feature_stats}",
            f"{analyses_with_features} of {analyses_total} analyses in ready records have features",
        ]
        logger.info(
            "NeuroStore records: %s of %s studies ready (%s unavailable); %s",
            len(ready), len(requested), len(unavailable), "; ".join(notes),
        )
        return PrepareReport(
            requested=len(requested),
            ready=len(ready),
            unavailable=unavailable + list(folder_report.unavailable),
            notes=notes,
        )

    def _build(self, pmid: int, record: Optional[Mapping[str, Any]], features: FeatureSnapshot, study: Any):
        """The record text, its analyses payload and how many analyses have features; or why not."""
        if record is None:
            return UnavailableStudy(pmid=pmid, reason="fetch_failed", detail="NeuroStore lookup failed; retried next run")
        if record.get("unavailable"):
            return UnavailableStudy(pmid=pmid, reason=record["unavailable"], study_id=str(record.get("base_study_id") or ""))
        base_study_id = str(record.get("base_study_id") or "")
        version_id = str(record.get("version_id") or "")
        if not features.fetched(base_study_id):
            return UnavailableStudy(pmid=pmid, reason="features_fetch_failed", study_id=base_study_id,
                                    detail="NeuroStore features lookup failed; retried next run")
        by_analysis = features.for_study(base_study_id, version_id)
        matched = sum(1 for a in record["analyses"] if str(a.get("neurostore_id")) in by_analysis)
        if not matched:
            others = features.other_studies(base_study_id, version_id)
            if others:
                return UnavailableStudy(
                    pmid=pmid, reason="features_for_other_version", study_id=base_study_id,
                    detail=f"coordinates from {record.get('source')} version {version_id}; features for {', '.join(others)}",
                )
            return UnavailableStudy(pmid=pmid, reason="no_features", study_id=base_study_id)

        rendered, payload_analyses, tables = [], [], {}
        for item in record["analyses"]:
            key = str(item.get("neurostore_id"))
            found = by_analysis.get(key)
            table_id = item.get("table_id")
            table_label = f"Table {item['table_label']}" if item.get("table_label") else None
            name = item.get("name") or (found or {}).get("name")
            lines = analysis_lines(
                (found or {}).get("features") if found else None,
                len(item.get("points") or []),
                table_label,
            )
            rendered.append(("{{analysis:%s}}" % key, name, lines))
            if table_id:
                tables.setdefault(table_id, {"table_id": table_id, "table_label": table_label or table_id})
            payload_analyses.append({
                "key": key,
                "name": item.get("name"),
                "description": item.get("description"),
                "table_id": table_id,
                "points": item.get("points") or [],
                # The analysis's own block, every line included, for annotation's
                # per-analysis document.
                "document": analysis_block("{{analysis:%s}}" % key, name, lines),
            })
        text = render_record(
            getattr(study, "title", None),
            getattr(study, "abstract", None),
            rendered,
        )
        payload = {
            "document_sha256": _sha256_text(text),
            "source": {
                "base_study_id": base_study_id,
                "version_id": version_id,
                "version_source": record.get("source"),
                "record_format": RECORD_FORMAT_VERSION,
            },
            "analyses": payload_analyses,
            "tables": list(tables.values()),
        }
        return text, payload, matched

    def _paths(self, pmid: int):
        return self.records_dir / f"{pmid}.md", self.records_dir / f"{pmid}{ANALYSES_SUFFIX}"

    def _write(self, pmid: int, text: str, payload: Mapping[str, Any]) -> None:
        document_path, analyses_path = self._paths(pmid)
        analyses_json = json.dumps(payload, indent=1, ensure_ascii=False)
        for path, content in ((document_path, text), (analyses_path, analyses_json)):
            # Bytes, so the file hashes to the document_sha256 computed from the text on any
            # platform; an unchanged file is left alone and keeps its mtime.
            data = content.encode("utf-8")
            if path.is_file() and path.read_bytes() == data:
                continue
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)

    def _remove(self, pmid: int) -> None:
        """A study no longer ready must not be served from an earlier run's files."""
        for path in self._paths(pmid):
            path.unlink(missing_ok=True)

    def index(self) -> Mapping[int, DocumentRef]:
        if self._index is None:
            raise RuntimeError(f"{type(self).__name__}.index() called before prepare()")
        return self._index

    def open(self, ref: DocumentRef) -> Document:
        return self._folder.open(ref)

    def open_analyses(self, ref: DocumentRef) -> AnalysisBundle:
        return self._folder.open_analyses(ref)
