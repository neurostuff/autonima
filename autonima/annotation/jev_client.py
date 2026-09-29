"""Analysis-level selection through Jev: one Noul per (analysis, criterion).

The screening backend gates an ARTICLE against one set of criteria. Selection gates every
parsed ANALYSIS against the criteria of every target contrast, which is a two-dimensional
version of the same problem -- and the dimension Jev is best suited to, because it answers
every question in one call and adding questions barely moves the latency.

THE SHAPE OF A CALL

State is the article: title, abstract, full text, tables. It is sent once. The questions carry
the analysis, using the documented pattern of putting data in the instructions object and
referring to it by name:

    {"analysis": {...}, "criterion": "...", "question": "Is `criterion` true of `analysis`?"}

So one call covers `n_analyses x n_annotations x n_criteria` questions against a single copy of
the article. The chat path instead sends the article once per study and asks the model to emit
a nested structure covering every analysis at once, which is where its validation retries come
from (`_make_decision_attempt` retries on hallucinated analysis_ids). Here an analysis id
cannot be hallucinated: the keys are ours and the mapping back is a lookup table.

WHY KEYS ARE OPAQUE

Question keys are `q0, q1, ...` with a side table, not `{analysis}|{criterion}` composites.
Analysis ids and annotation names are free text from a config and a parser; building keys out
of them would need escaping, and a collision would silently attach one analysis's probability
to another. An integer and a lookup cannot collide.

BATCHING AGAINST A REAL CEILING

jev-1.13 allows 64k tokens per request (state plus all questions) and 32k for the state plus
the longest single question; both return a non-retryable HTTP 400 when exceeded. A first full
run of the emotion-regulation project lost 63 of 128 studies to this, and the failures tracked
ANALYSES PER STUDY (median 7 against 3) rather than article length, because every question
repeats its analysis metadata.

So chunking is by token budget, not question count, and the article is trimmed from the middle
if the state alone will not fit. The state is re-sent per chunk; at $0.042/MTok that is far
cheaper than losing the study.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..backends.jev import (JevClient, JevError, apply_gate, criterion_text,
                            estimate_tokens, fit_state, normalise_mapping,
                            plan_chunks)
from ..llm.usage import record as record_usage
from .schema import (
    AnalysisMetadata,
    AnnotationCriteriaConfig,
    AnnotationDecision,
    StudyAnalysisGroup,
)

logger = logging.getLogger(__name__)


def mapping_for(criteria: AnnotationCriteriaConfig) -> Dict[str, Dict[str, str]]:
    """The criterion-ID mapping for one target contrast, synthesised if the config omits it.

    Screening criteria get IDs from `ConfigManager`; annotation criteria do not -- the field is
    optional and only set when the YAML supplies it. Falling back to positional IDs keeps the
    two backends behaving the same and keeps `inclusion_criteria_applied` populated either way.
    """
    mapping = normalise_mapping(criteria.criteria_mapping)
    if mapping["inclusion"] or mapping["exclusion"]:
        return mapping
    return {
        "inclusion": {f"I{i}": spec
                      for i, spec in enumerate(criteria.inclusion_criteria or [], start=1)},
        "exclusion": {f"E{i}": spec
                      for i, spec in enumerate(criteria.exclusion_criteria or [], start=1)},
    }


def build_state(
    metadata: Union[StudyAnalysisGroup, AnalysisMetadata],
    metadata_fields: Sequence[str],
) -> Dict[str, Any]:
    """The article, as structured state. Sent once per call, shared by every question."""
    get = lambda name: getattr(metadata, name, None)  # noqa: E731
    state: Dict[str, Any] = {"study_id": get("study_id") or ""}
    for field in ("study_title", "study_abstract", "study_journal", "study_publication_date"):
        value = get(field)
        if value:
            state[field.replace("study_", "")] = value
    if "study_fulltext" in metadata_fields and get("study_fulltext"):
        state["full_text"] = get("study_fulltext")
    tables = get("tables") or []
    if tables:
        state["tables"] = [
            {k: v for k, v in
             (("id", getattr(t, "table_id", None)),
              ("caption", getattr(t, "table_caption", None)),
              ("footer", getattr(t, "table_footer", None))) if v}
            for t in tables
        ]
    return state


def describe_analysis(
    analysis: AnalysisMetadata,
    metadata_fields: Sequence[str],
) -> Dict[str, Any]:
    """The per-analysis payload that rides in the question, honouring `metadata_fields`.

    `study_fulltext` is excluded here even when requested: it is in the state already, and
    repeating a whole article inside every question would multiply the input tokens by the
    number of questions.
    """
    out: Dict[str, Any] = {"analysis_id": analysis.analysis_id}
    for field in metadata_fields:
        if field == "study_fulltext":
            continue
        value = getattr(analysis, field, None)
        if value:
            out[field] = value
    if not out.get("analysis_name") and analysis.analysis_name:
        out["analysis_name"] = analysis.analysis_name
    return out


class JevAnnotationClient:
    """Drop-in alternative to `AnnotationClient` for analysis-level selection."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jev-latest",
        inclusion_threshold: float = 0.5,
        exclusion_threshold: float = 0.5,
        guidance: Optional[str] = None,
        client: Optional[JevClient] = None,
    ) -> None:
        self.client = client or JevClient(api_key=api_key, model=model)
        self.guidance = guidance
        self.inclusion_threshold = inclusion_threshold
        self.exclusion_threshold = exclusion_threshold

    def make_decision(
        self,
        metadata: Union[AnalysisMetadata, StudyAnalysisGroup],
        criteria_list: List[AnnotationCriteriaConfig],
        metadata_fields: List[str],
        model: str = "jev-latest",
        model_params: Optional[Dict[str, Any]] = None,
        prompt_type: str = "multi_analysis",
    ) -> List[AnnotationDecision]:
        """Same signature as `AnnotationClient.make_decision`, same return type."""
        if not criteria_list:
            return []

        analyses: List[AnalysisMetadata] = (
            list(getattr(metadata, "analyses", []) or [])
            if isinstance(metadata, StudyAnalysisGroup)
            else [metadata]  # single_analysis: one analysis, same machinery
        )
        if not analyses:
            return []

        state = build_state(metadata, metadata_fields)
        questions, index = self._build_questions(
            analyses, criteria_list, metadata_fields, guidance=self.guidance)
        if not questions:
            logger.warning("No criteria to evaluate for study %s; returning no decisions",
                           getattr(metadata, "study_id", "?"))
            return []

        answers = self._ask(state, questions)
        return self._assemble(analyses, criteria_list, index, answers, model)

    # -- internals ------------------------------------------------------------------------

    def _build_questions(
        self,
        analyses: Sequence[AnalysisMetadata],
        criteria_list: Sequence[AnnotationCriteriaConfig],
        metadata_fields: Sequence[str],
        guidance: Optional[str] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[str, str, str]]]:
        """Returns (questions, index) where index maps key -> (analysis_id, annotation, crit)."""
        questions: Dict[str, Dict[str, Any]] = {}
        index: Dict[str, Tuple[str, str, str]] = {}
        n = 0
        for analysis in analyses:
            described = describe_analysis(analysis, metadata_fields)
            for criteria in criteria_list:
                mapping = mapping_for(criteria)
                local = " ".join(x for x in (guidance, criteria.additional_instructions) if x)
                for kind in ("inclusion", "exclusion"):
                    for criterion_id, spec in mapping[kind].items():
                        text = criterion_text(spec)
                        t_desc = spec.get("true") if isinstance(spec, dict) else None
                        f_desc = spec.get("false") if isinstance(spec, dict) else None
                        key = f"q{n}"
                        n += 1
                        instructions: Dict[str, Any] = {
                            "analysis": described,
                            "criterion": text,
                            "question": (
                                "Considering the article in the state, is `criterion` true of "
                                "the specific analysis given in `analysis`?"
                            ),
                        }
                        if criteria.description:
                            instructions["target_contrast"] = criteria.description
                        if local:
                            instructions["guidance"] = local
                        questions[key] = {
                            "type": "noul",
                            "instructions": instructions,
                            "criteria": {
                                "true": t_desc or f"This analysis satisfies: {text}",
                                "false": f_desc or f"This analysis does not satisfy: {text}",
                            },
                        }
                        index[key] = (analysis.analysis_id, criteria.name, criterion_id)
        return questions, index

    def _ask(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Fit the state, then split questions to respect the documented token budgets.

        Chunking on a fixed question COUNT is what lost 63 of 128 studies on the first full
        run: the binding constraint is tokens, and it is dominated by the article in the
        state plus the per-question analysis metadata, not by the number of questions.
        """
        longest = max((estimate_tokens({k: q}) for k, q in questions.items()), default=0)
        fitted, truncated = fit_state(state, reserve_tokens=longest)
        if truncated:
            logger.info("Truncated state for study %s to fit Jev's 32k state budget",
                        (state or {}).get("study_id", "?"))
        answers: Dict[str, Any] = {}
        for chunk in plan_chunks(fitted, questions):
            if not chunk:
                continue
            body = self.client.evaluate(fitted, chunk)
            answers.update(body.get("answers", {}))
            self._record(body.get("usage") or {})
        return answers

    def _assemble(
        self,
        analyses: Sequence[AnalysisMetadata],
        criteria_list: Sequence[AnnotationCriteriaConfig],
        index: Mapping[str, Tuple[str, str, str]],
        answers: Mapping[str, Any],
        model: str,
    ) -> List[AnnotationDecision]:
        """Regroup the flat answer set into one decision per (analysis, target contrast)."""
        by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key, (analysis_id, annotation_name, criterion_id) in index.items():
            by_pair.setdefault((analysis_id, annotation_name), {})[criterion_id] = \
                answers.get(key, {})

        study_ids = {a.analysis_id: a.study_id for a in analyses}
        decisions: List[AnnotationDecision] = []
        for analysis in analyses:
            for criteria in criteria_list:
                pair = (analysis.analysis_id, criteria.name)
                mapping = mapping_for(criteria)
                decision = apply_gate(
                    by_pair.get(pair, {}),
                    mapping,
                    inclusion_threshold=self.inclusion_threshold,
                    exclusion_threshold=self.exclusion_threshold,
                )
                decisions.append(AnnotationDecision(
                    annotation_name=criteria.name,
                    analysis_id=analysis.analysis_id,
                    study_id=study_ids.get(analysis.analysis_id, ""),
                    include=decision.include,
                    reasoning=_explain(decision),
                    confidence=decision.confidence,
                    model_used=model,
                    timestamp=datetime.now(),
                    inclusion_criteria_applied=decision.satisfied_inclusion_ids,
                    exclusion_criteria_applied=decision.fired_exclusion_ids,
                    criterion_probabilities=decision.probabilities,
                ))
        return decisions

    @staticmethod
    def _record(usage: Mapping[str, Any]) -> None:
        """`record` reads input_tokens/output_tokens from a dict and never raises.

        The label must be "annotation": execution.py snapshots usage per stage with exactly
        the pipeline's stage names, so a private label records and is then never surfaced.
        """
        if usage:
            record_usage("annotation", "jev", dict(usage))


def _explain(decision) -> str:
    """Factual, not a rationale -- Jev generates no text. Mirrors the screening explainer."""
    if decision.include:
        body = "all criteria satisfied"
    else:
        unmet = [v.criterion_id for v in decision.verdicts
                 if v.kind == "inclusion" and not v.satisfied]
        fired = decision.fired_exclusion_ids
        parts = []
        if unmet:
            parts.append("inclusion not met: " + ", ".join(
                f"{c} p={decision.probabilities[c]:.2f}" for c in unmet))
        if fired:
            parts.append("exclusion triggered: " + ", ".join(
                f"{c} p={decision.probabilities[c]:.2f}" for c in fired))
        body = "; ".join(parts) or "no criterion satisfied"
    closest = decision.closest_call
    if closest is not None:
        body += (f" (closest call {closest.criterion_id} "
                 f"p={closest.probability:.2f}, margin {closest.margin:.2f})")
    return f"[jev] {body}"
