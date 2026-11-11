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
# Caps that keep one analysis's block readable: list items shown, and characters per line.
MAX_ITEMS = 6
MAX_LINE = 400


def _text(value: Any) -> Optional[str]:
    """A feature value as words: lists joined, enum underscores as spaces, never a dict's repr."""
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        items = [t for t in (_text(v) for v in value) if t]
        if not items:
            return None
        shown = items[:MAX_ITEMS]
        more = f" (+{len(items) - MAX_ITEMS} more)" if len(items) > MAX_ITEMS else ""
        return "; ".join(dict.fromkeys(shown)) + more
    if isinstance(value, dict):
        # A normalized feature carries its wording as ``text`` beside ids, which say nothing
        # to the model; anything else is shown as its labelled parts.
        if "text" in value:
            return _text(value["text"])
        parts = [f"{k.replace('_', ' ')} {t}" for k, t in ((k, _text(v)) for k, v in value.items())
                 if t and not k.endswith("ids")]
        return "; ".join(parts) or None
    text = str(value)
    if isinstance(value, str) and " " not in text:
        text = text.replace("_", " ")
    return text


def _clip(text: Optional[str]) -> Optional[str]:
    if text and len(text) > MAX_LINE:
        return text[:MAX_LINE].rstrip() + "..."
    return text


def _cohort(has_patients: Any, has_healthy: Any) -> Optional[str]:
    if has_patients and has_healthy:
        return "patients and healthy participants"
    if has_patients:
        return "patients"
    if has_healthy:
        return "healthy participants"
    return None


def _contrast(features: Mapping[str, Any]) -> Optional[str]:
    """'positive > negative', with each side's cohort, then the kind of contrast."""
    contrast = features.get("contrast")
    if isinstance(contrast, str):  # the earlier flat layout
        return contrast
    if not isinstance(contrast, dict):
        return None
    sides = []
    for side in ("positive", "negative"):
        part = contrast.get(side) or {}
        name = _text(part.get("name"))
        cohort = _cohort(part.get("has_patients"), part.get("has_healthy"))
        if name or cohort:
            sides.append(f"{name or side}" + (f" ({cohort})" if cohort else ""))
    head = " > ".join(sides)
    kind = [_text(contrast.get("kind")), (f"{contrast['groups']}-group" if contrast.get("groups") else None)]
    kind = ", ".join(k for k in kind if k)
    if head and kind:
        return f"{head} [{kind}]"
    return head or kind or None


def _condition(condition: Any) -> Optional[str]:
    """The medical condition's wording, grouped by the role NeuroStore gave it where it did."""
    if not isinstance(condition, dict):
        return _text(condition)
    ids, texts = condition.get("ids") or [], condition.get("text") or []
    roles = {}
    for entry in condition.get("role_ids") or []:
        role, _, term = str(entry).partition("|")
        roles.setdefault(term, role)
    if ids and len(ids) == len(texts) and roles:
        grouped: Dict[str, List[str]] = {}
        for term, text in zip(ids, texts):
            grouped.setdefault(roles.get(term, "other"), []).append(text)
        return "; ".join(f"{role}: {_text(items)}" for role, items in grouped.items())
    return _text(texts)


def _participants(features: Mapping[str, Any]) -> Optional[str]:
    parts = []
    if isinstance(features.get("n"), int):
        parts.append(f"n = {features['n']}")
    cohort = _cohort(features.get("has_patients"), features.get("has_healthy"))
    if cohort:
        parts.append(cohort)
    groups = features.get("groups") if isinstance(features.get("groups"), dict) else {}
    age = []
    for key, label in (("age_mean", "mean"), ("age_standard_deviation", "sd"), ("age_minimum", "min"),
                       ("age_maximum", "max"), ("age_range", "range")):
        if _text(groups.get(key)):
            age.append(f"{label} {_text(groups.get(key))}")
    if age:
        parts.append("age " + ", ".join(age))
    sex = [f"{groups[k]} {k.split('_')[0]}" for k in ("female_count", "male_count") if isinstance(groups.get(k), int)]
    if sex:
        parts.append(", ".join(sex))
    for key in ("diagnostic_system", "diagnostic_instrument", "clinical_characteristics"):
        if _text(groups.get(key)):
            parts.append(_text(groups.get(key)))
    return "; ".join(parts) or None


def _inference(features: Mapping[str, Any]) -> Optional[str]:
    settings = features.get("inference_settings")
    if not isinstance(settings, dict):
        return None
    parts = []
    if _text(settings.get("multiple_comparison_method")):
        parts.append(_text(settings["multiple_comparison_method"]))
    if settings.get("height_threshold_value") is not None:
        kind = _text(settings.get("height_threshold_type")) or "threshold"
        parts.append(f"{kind} < {settings['height_threshold_value']}")
    if settings.get("cluster_extent_threshold") is not None:
        parts.append(f"cluster extent {settings['cluster_extent_threshold']}")
    if _text(settings.get("inference_level")):
        parts.append(f"{_text(settings['inference_level'])}-level")
    return "; ".join(parts) or None


def _inference_scope(features: Mapping[str, Any]) -> Optional[tuple]:
    """Where significance was tested, as (label, words); None when the record does not say.

    NeuroStore keeps this apart from where the model was fit: an analysis can be estimated over
    the whole brain and tested only within regions of interest, as in a small-volume correction.
    """
    settings = features.get("inference_settings")
    scope = settings.get("correction_scope") if isinstance(settings, dict) else None
    if not _text(scope):
        return None
    words = SPATIAL_SCOPES.get(scope, _text(scope))
    if scope == "whole_brain":
        return "Inference over", words
    return "Inference restricted to", words


def _task(features: Mapping[str, Any]) -> Optional[str]:
    tasks = features.get("tasks") if isinstance(features.get("tasks"), dict) else {}
    parts = [_text(features.get("task"))]
    for key, label in (("design_type", "design"), ("stimuli", "stimuli"), ("instructions", "instructions")):
        if _text(tasks.get(key)):
            parts.append(f"{label}: {_text(tasks.get(key))}")
    return "; ".join(p for p in parts if p) or None


def feature_lines(features: Mapping[str, Any]) -> List[str]:
    """One line per feature the record states; a feature the record is silent on is left out.

    NeuroStore's features are nested (the contrast and its sides, groups, tasks, inference
    settings, ...). Each is shown as words, never as a dict's repr; acquisition and
    preprocessing details are left out as rarely deciding eligibility.
    """
    lines: List[tuple] = []

    def add(label: str, value: Optional[str]) -> None:
        if value:
            lines.append((label, _clip(value)))

    add("Contrast", _contrast(features))
    sets = features.get("coordinate_sets") if isinstance(features.get("coordinate_sets"), dict) else {}
    add("Result", _text(sets.get("result_kind")))
    details = features.get("details") if isinstance(features.get("details"), dict) else {}
    if details:
        add("Connectivity", _text({k: v for k, v in details.items()}))
    scope = features.get("spatial_scope")
    add("Model fit over", SPATIAL_SCOPES.get(scope, _text(scope)))
    inference_scope = _inference_scope(features)
    if inference_scope:
        add(*inference_scope)
    add("Thresholding", _inference(features))
    regions = features.get("regions") if isinstance(features.get("regions"), dict) else {}
    if regions:
        defined = _text(regions.get("definition_method"))
        add("Regions", (_text(regions.get("name")) or "") + (f" (defined by {defined})" if defined else ""))
    add("Task", _task(features))
    add("Participants", _participants(features))
    add("Medical condition", _condition(features.get("medical_condition")))
    species = features.get("species")
    if species and species != ["human"]:
        add("Species", _text(species))
    measure, family = _text(features.get("measure_type")), _text(features.get("measure_family"))
    add("Measure", f"{measure} ({family})" if measure and family else measure or family)
    design = features.get("design") if isinstance(features.get("design"), dict) else {
        k: features.get(k) for k in ("assignment_structure", "allocation", "blinding")}
    design_text = _text({k: v for k, v in design.items() if k in ("assignment_structure", "allocation", "blinding")})
    if features.get("is_meta_analysis"):
        design_text = "this analysis is itself a meta-analysis" + (f"; {design_text}" if design_text else "")
    add("Design", design_text)
    model = features.get("model_estimation") if isinstance(features.get("model_estimation"), dict) else {}
    terms = model.get("terms") if isinstance(model.get("terms"), dict) else {}
    model_text = "; ".join(t for t in (_text(model.get("model_type")),
                                       f"terms: {_text(terms.get('name'))}" if _text(terms.get("name")) else None) if t)
    add("Model", model_text or None)
    add("Coordinate space", _text(features.get("coordinate_space")))
    return [f"{label}: {value}" for label, value in lines]


def analysis_lines(
    features: Optional[Mapping[str, Any]],
    peaks: int,
    table_label: Optional[str],
) -> List[str]:
    """What the record says about one analysis: its features, then its peaks."""
    reported = f"Reported peaks: {peaks}" + (f" ({table_label})" if table_label else "")
    if features is None:
        return ["No features were extracted for this analysis.", reported]
    return feature_lines(features) + [reported]


def analysis_block(reference: str, name: Optional[str], lines: Sequence[str]) -> str:
    """One analysis as the record shows it: a reference line, then its lines."""
    head = f"Analysis {reference}: {name or 'unnamed analysis'}"
    return "\n".join([head, *(f"  {line}" for line in lines)])


RECORD_DESCRIPTION = (
    "NeuroStore's record of the article: its title and abstract, and each reported analysis "
    "with the features NeuroStore extracted from the paper (the contrast and its sides, task, "
    "whole brain or region of interest, inference, participants, measure and design)"
)


def render_record(
    title: Optional[str],
    abstract: Optional[str],
    analyses: Sequence[tuple],
) -> str:
    """The study's record, as the model reads it in place of the full text.

    ``analyses`` holds ``(reference, name, lines)`` per analysis. A line every analysis has,
    word for word, is printed once under "Shared by all analyses" instead of in each block:
    a study's participants, condition and design are usually the same for all of them.
    """
    parts = ["NEUROSTORE RECORD"]
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    shared: List[str] = []
    if len(analyses) > 1:
        first = analyses[0][2]
        shared = [line for line in first if all(line in lines for _, _, lines in analyses[1:])
                  and not line.startswith("Reported peaks")]
    parts.append(
        f"Analyses ({len(analyses)}). Each lists only what NeuroStore's extraction states; a "
        "feature it is silent on is left out. 'Model fit over' is where the model was "
        "estimated (an estimate at every voxel of the whole brain, or of a region only). "
        "'Inference over' or 'Inference restricted to' is where significance was tested, "
        "which can be a subset of where the model was fit, as in a small-volume correction."
    )
    if shared:
        parts.append("Shared by all analyses:\n" + "\n".join(f"  {line}" for line in shared))
    parts.extend(
        analysis_block(reference, name, [line for line in lines if line not in shared])
        for reference, name, lines in analyses
    )
    return "\n\n".join(parts) + "\n"
