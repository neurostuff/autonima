"""The features NeuroStore extracted for each analysis, and how a record shows them.

NeuroStore resolves each paper's latest extraction record into one row of features per
analysis of the paper's current pipeline study (``study_schema``): the contrast and its two
sides, the task, whether it is whole-brain or a region of interest, the effect kind, the
groups and their size, the medical condition, the measure and the study design. The analysis
query (``POST /analysis-queries/``) returns them per paper.

Like the analyses, features change when NeuroStore re-ingests a paper, so what a run read is
frozen in ``outputs/neurostore_features.json`` and later runs read the snapshot instead of the
server. A paper that had no features when it was fetched is recorded as such; delete its entry,
or the file, to ask again.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .client import NeuroStoreClient

logger = logging.getLogger(__name__)

FEATURES_FILENAME = "neurostore_features.json"
FEATURES_SCHEMA = 1
# Papers per analysis query. The server takes any number; a bounded request keeps a failure
# to one chunk, which is retried next run while the rest is kept.
DEFAULT_CHUNK = 500


class FeatureSnapshot:
    """Per paper (base study): the features of each of its pipeline studies' analyses."""

    def __init__(self, path: Path, base_url: str):
        self.path = Path(path)
        self.base_url = base_url
        self.papers: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema") == FEATURES_SCHEMA and data.get("base_url") == base_url:
                self.papers = data.get("papers") or {}
            else:
                logger.info("NeuroStore features %s came from elsewhere; fetching again", self.path)

    def ensure(
        self, client: NeuroStoreClient, base_study_ids: Iterable[str], chunk: int = DEFAULT_CHUNK
    ) -> Dict[str, int]:
        """Fetch every paper not yet in the snapshot. A failed chunk is left out, not cached."""
        missing = sorted({str(b) for b in base_study_ids if b} - set(self.papers))
        failed = 0
        for start in range(0, len(missing), max(1, chunk)):
            batch = missing[start:start + chunk]
            try:
                results = client.analysis_features(batch)
            except Exception as exc:  # recorded and retried next run, never cached
                failed += len(batch)
                logger.warning("NeuroStore features fetch failed for %s papers: %s", len(batch), exc)
                continue
            fetched_at = datetime.now().isoformat()
            for base_study_id in batch:
                self.papers[base_study_id] = {"fetched_at": fetched_at, "studies": {}}
            for result in results:
                paper = self.papers.setdefault(
                    str(result.get("base_study")), {"fetched_at": fetched_at, "studies": {}}
                )
                paper["studies"][str(result.get("study"))] = {
                    str(analysis.get("id")): {
                        "name": analysis.get("name"),
                        "features": analysis.get("features") or {},
                    }
                    for analysis in result.get("analyses") or []
                }
        if missing:
            self.save()
        return {
            "fetched": len(missing) - failed,
            "reused": len({str(b) for b in base_study_ids if b}) - len(missing),
            "failed": failed,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"schema": FEATURES_SCHEMA, "base_url": self.base_url, "papers": self.papers},
            indent=1,
        ))
        tmp.replace(self.path)

    def fetched(self, base_study_id: Optional[str]) -> bool:
        return bool(base_study_id) and str(base_study_id) in self.papers

    def for_study(self, base_study_id: str, study_id: str) -> Dict[str, Dict[str, Any]]:
        """Analysis id -> ``{name, features}`` for one pipeline study of one paper."""
        paper = self.papers.get(str(base_study_id)) or {}
        return (paper.get("studies") or {}).get(str(study_id)) or {}

    def other_studies(self, base_study_id: str, study_id: str) -> List[str]:
        """The paper's pipeline studies that have features, other than ``study_id``."""
        paper = self.papers.get(str(base_study_id)) or {}
        return sorted(s for s in (paper.get("studies") or {}) if s != str(study_id))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

SPATIAL_SCOPES = {
    "whole_brain": "whole brain",
    "roi": "region of interest",
    "searchlight": "searchlight",
}


def _words(value: Any) -> Optional[str]:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v not in (None, "")]
        return ", ".join(items) if items else None
    return str(value).replace("_", " ") if isinstance(value, str) and " " not in value else str(value)


def _cohort(has_patients: Any, has_healthy: Any) -> Optional[str]:
    if has_patients and has_healthy:
        return "patients and healthy participants"
    if has_patients:
        return "patients"
    if has_healthy:
        return "healthy participants"
    if has_patients is False and has_healthy is False:
        return "neither patients nor healthy participants stated"
    return None


def _side(features: Mapping[str, Any], side: str) -> Optional[str]:
    parts = []
    cohort = _cohort(features.get(f"contrast_{side}_has_patients"), features.get(f"contrast_{side}_has_healthy"))
    if cohort:
        parts.append(cohort)
    facets = _words(features.get(f"contrast_{side}_facets"))
    if facets:
        parts.append(f"facets {facets}")
    return "; ".join(parts) if parts else None


def feature_lines(features: Mapping[str, Any]) -> List[str]:
    """One line per feature the record states; a feature the record is silent on is left out."""
    lines: List[tuple] = []

    def add(label: str, value: Optional[str]) -> None:
        if value:
            lines.append((label, value))

    add("Contrast", features.get("contrast") if isinstance(features.get("contrast"), str) else _words(features.get("contrast")))
    effect = _words(features.get("effect_kind"))
    groups = features.get("group_count")
    if isinstance(groups, int) and groups:
        effect = f"{effect or 'effect'}; {'within one group' if groups == 1 else f'across {groups} groups'}"
    add("Effect", effect)
    scope = features.get("spatial_scope")
    add("Spatial scope", SPATIAL_SCOPES.get(scope, _words(scope)))
    add("Task", _words(features.get("task")))
    participants = []
    if isinstance(features.get("n"), int):
        participants.append(f"n = {features['n']}")
    cohort = _cohort(features.get("has_patients"), features.get("has_healthy"))
    if cohort:
        participants.append(cohort)
    add("Participants", "; ".join(participants) or None)
    add("Medical condition", _words(features.get("medical_condition")))
    add("Positive side", _side(features, "positive"))
    add("Negative side", _side(features, "negative"))
    measure = _words(features.get("measure_type"))
    family = _words(features.get("measure_family"))
    add("Measure", f"{measure} ({family})" if measure and family else measure or family)
    design = [
        _words(features.get("assignment_structure")),
        f"allocation {_words(features['allocation'])}" if _words(features.get("allocation")) else None,
        f"blinding {_words(features['blinding'])}" if _words(features.get("blinding")) else None,
    ]
    add("Design", "; ".join(d for d in design if d) or None)
    add("Coordinate space", _words(features.get("coordinate_space")))
    return [f"{label}: {value}" for label, value in lines]


def analysis_block(
    reference: str,
    name: Optional[str],
    features: Optional[Mapping[str, Any]],
    peaks: int,
    table_label: Optional[str],
) -> str:
    """One analysis as the record shows it: a reference line, then its features."""
    head = f"Analysis {reference}: {name or 'unnamed analysis'}"
    reported = f"Reported peaks: {peaks}" + (f" ({table_label})" if table_label else "")
    if features is None:
        body = ["No features were extracted for this analysis.", reported]
    else:
        body = feature_lines(features) + [reported]
    return "\n".join([head, *(f"  {line}" for line in body)])


RECORD_DESCRIPTION = (
    "NeuroStore's record of the article: its title and abstract, and each reported analysis "
    "with the features NeuroStore extracted from the paper (the contrast and its sides, task, "
    "whole brain or region of interest, effect, participants, measure and design)"
)


def render_record(
    title: Optional[str],
    abstract: Optional[str],
    blocks: Sequence[str],
) -> str:
    """The study's record, as the model reads it in place of the full text."""
    parts = ["NEUROSTORE RECORD"]
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    parts.append(
        f"Analyses ({len(blocks)}). Each lists only what NeuroStore's extraction states; a "
        "feature it is silent on is left out."
    )
    parts.extend(blocks)
    return "\n\n".join(parts) + "\n"
