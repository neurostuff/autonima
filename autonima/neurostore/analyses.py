"""Analyses and coordinates from NeuroStore, in place of parsing them from the article's tables.

A NeuroStore base study (one paper) has several versions, each the paper's coordinates as one
source extracted them: Neurosynth Compose's own parse (``study_schema``), earlier LLM
extractions, Neurosynth, NeuroQuery, hand curation. One version is used per study, the first
in ``sources`` order that has coordinates, so a study's analyses never mix two extractions.

Only analyses whose role is in ``roles`` are kept. NeuroStore marks the coordinates a paper
reports as its own results ``result``; seed regions (``anchor``), coordinates cited from other
work (``reference``) and region definitions (``localization``) are not findings to pool.
An analysis with no role predates roles and counts as a result.

The server's parses change as NeuroStore re-ingests, so what a run used is frozen in a
snapshot, ``outputs/neurostore_analyses.json``, and later runs read it instead of the server.
Deleting a study's entry, or the file, fetches it again.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..coordinates.schema import Analysis, CoordinatePoint, PointsValue
from ..models.types import ActivationTable
from .client import NeuroStoreClient

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "neurostore_analyses.json"
SNAPSHOT_SCHEMA = 1
# Compose's current parse first, then the earlier extractions by how closely they recovered
# NeuroMetaBench's expert-curated peaks, then the database sources.
DEFAULT_SOURCES: Tuple[str, ...] = (
    "study_schema",
    "nuextract-v21",
    "llm",
    "neurostore",
    "pubget",
    "nuextract-v19",
    "neuroquery",
    "neurosynth",
)
DEFAULT_ROLES: Tuple[str, ...] = ("result",)
TABLE_ID_PREFIX = "neurostore-"

_VALUE_KINDS = {
    "z": "z-statistic", "t": "t-statistic", "f": "f-statistic", "r": "correlation",
    "p": "p-value", "beta": "beta",
}


@dataclass(frozen=True)
class NeuroStoreOptions:
    sources: Tuple[str, ...] = DEFAULT_SOURCES
    roles: Tuple[str, ...] = DEFAULT_ROLES

    @classmethod
    def from_config(cls, options: Optional[Mapping[str, Any]]) -> "NeuroStoreOptions":
        options = options or {}
        return cls(
            sources=tuple(options.get("sources") or DEFAULT_SOURCES),
            roles=tuple(options.get("roles") or DEFAULT_ROLES),
        )


def _role_kept(role: Optional[str], roles: Sequence[str]) -> bool:
    return (role or "result") in roles


def _point_values(point: Mapping[str, Any]) -> Optional[List[PointsValue]]:
    values = []
    for value in point.get("values") or []:
        kind = _VALUE_KINDS.get(str(value.get("kind") or "").lower(), "other" if value.get("kind") else None)
        values.append(PointsValue(value=value.get("value"), kind=kind))
    return values or None


def _space(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).upper()
    return value if value in ("MNI", "TAL") else None


def record_from_version(
    pmid: str,
    base_study: Mapping[str, Any],
    version: Mapping[str, Any],
    roles: Sequence[str],
) -> Dict[str, Any]:
    """The snapshot record for one study: the version used and its kept analyses."""
    analyses = []
    for analysis in sorted(version.get("analyses") or [], key=lambda a: (a.get("order") is None, a.get("order") or 0)):
        if not _role_kept(analysis.get("role"), roles):
            continue
        points = [
            {
                "coordinates": [float(c) for c in point["coordinates"]],
                "space": _space(point.get("space")),
                "values": [
                    {"value": v.value, "kind": v.kind} for v in _point_values(point) or []
                ],
            }
            for point in analysis.get("points") or []
            if point.get("coordinates") and len(point["coordinates"]) == 3
            and all(isinstance(c, (int, float)) for c in point["coordinates"])
        ]
        if not points:
            continue
        metadata = analysis.get("metadata") or {}
        source_table = metadata.get("table_id") if isinstance(metadata, dict) else None
        analyses.append(
            {
                "neurostore_id": analysis.get("id"),
                "name": analysis.get("name"),
                "description": analysis.get("description"),
                "role": analysis.get("role"),
                "table_id": analysis.get("table_id") if analysis.get("table_id") not in (None, "None") else None,
                "table_label": str(source_table) if source_table else None,
                "points": points,
            }
        )
    return {
        "pmid": str(pmid),
        "base_study_id": base_study.get("id"),
        "version_id": version.get("id"),
        "source": version.get("source"),
        "version_updated_at": version.get("updated_at") or version.get("created_at"),
        "fetched_at": datetime.now().isoformat(),
        "analyses": analyses,
    }


def fetch_record(client: NeuroStoreClient, pmid: str, options: NeuroStoreOptions) -> Dict[str, Any]:
    """Fetch one study's analyses, or a record saying why there are none."""
    listing = client.get("base-studies/", {"pmid": str(pmid), "info": "true"}) or {}
    matches = [r for r in listing.get("results") or [] if str(r.get("pmid")) == str(pmid)]
    if not matches:
        return {"pmid": str(pmid), "unavailable": "not_in_neurostore", "fetched_at": datetime.now().isoformat()}
    base_study = matches[0]
    versions = {}
    for version in base_study.get("versions") or []:
        if isinstance(version, dict) and version.get("has_coordinates"):
            versions.setdefault(version.get("source"), []).append(version)
    for source in options.sources:
        # Several versions from one source: the most recently updated is that source's current one.
        for listed in sorted(versions.get(source, []), key=lambda v: v.get("updated_at") or "", reverse=True):
            version = client.study(listed["id"])
            if not version:
                continue
            record = record_from_version(pmid, base_study, version, options.roles)
            if record["analyses"]:
                return record
    reason = "no_coordinates" if not versions else "no_coordinates_from_configured_sources"
    return {
        "pmid": str(pmid),
        "base_study_id": base_study.get("id"),
        "unavailable": reason,
        "sources_seen": sorted(s for s in versions if s),
        "fetched_at": datetime.now().isoformat(),
    }


def to_analyses(record: Mapping[str, Any]) -> Tuple[List[Analysis], List[ActivationTable]]:
    """autonima analyses, and a bare table entry for each NeuroStore table they name."""
    analyses: List[Analysis] = []
    tables: Dict[str, ActivationTable] = {}
    for item in record.get("analyses") or []:
        table_id = None
        if item.get("table_id"):
            table_id = TABLE_ID_PREFIX + item["table_id"]
            label = item.get("table_label") or item["table_id"]
            tables.setdefault(table_id, ActivationTable(table_id=table_id, table_label=f"Table {label}"))
        analyses.append(
            Analysis(
                name=item.get("name"),
                description=item.get("description"),
                points=[
                    CoordinatePoint(
                        coordinates=p["coordinates"],
                        space=p.get("space"),
                        values=[PointsValue(**v) for v in p.get("values") or []] or None,
                    )
                    for p in item["points"]
                ],
                parsed=True,
                table_id=table_id,
            )
        )
    return analyses, list(tables.values())


class NeuroStoreSnapshot:
    """The per-run record of which NeuroStore analyses each study used."""

    def __init__(self, path: Path, options: NeuroStoreOptions, base_url: str):
        self.path = Path(path)
        self.options = options
        self.base_url = base_url
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema") == SNAPSHOT_SCHEMA and data.get("options") == self._options_dict():
                self.records = data.get("studies") or {}
            else:
                logger.info("NeuroStore snapshot %s was made with other options; fetching again", self.path)

    def _options_dict(self) -> Dict[str, Any]:
        return {"sources": list(self.options.sources), "roles": list(self.options.roles), "base_url": self.base_url}

    def ensure(self, client: NeuroStoreClient, pmids: Iterable[str], workers: int = 8) -> Dict[str, int]:
        """Fetch every PMID not yet in the snapshot. A failed request leaves the PMID out."""
        missing = [str(p) for p in dict.fromkeys(pmids) if str(p) not in self.records]
        failed = 0
        if missing:
            from concurrent.futures import ThreadPoolExecutor

            def one(pmid: str):
                try:
                    return pmid, fetch_record(client, pmid, self.options), None
                except Exception as exc:  # recorded and retried next run, never cached
                    return pmid, None, exc

            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for pmid, record, error in pool.map(one, missing):
                    if error is not None:
                        failed += 1
                        logger.warning("NeuroStore fetch failed for PMID %s: %s", pmid, error)
                        continue
                    self.records[pmid] = record
            self.save()
        return {"fetched": len(missing) - failed, "reused": len(set(map(str, pmids))) - len(missing), "failed": failed}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"schema": SNAPSHOT_SCHEMA, "options": self._options_dict(), "studies": self.records},
            indent=1,
        ))
        tmp.replace(self.path)

    def get(self, pmid: str) -> Optional[Dict[str, Any]]:
        return self.records.get(str(pmid))
