"""Screening through Jev: one Noul question per criterion, gate computed in code.

The rest of the pipeline is untouched. This client returns the same
`AbstractScreeningOutput` / `FullTextScreeningOutput` objects the OpenAI client returns, so
caching, reporting, the PRISMA counts and every downstream evaluation script keep working. The
difference is where the decision is made: the chat path asks a model for a verdict, this path
asks for per-criterion probabilities and applies the gate itself.

Two fields behave differently and the difference is deliberate:

  `reason`    -- Jev generates no text. Rather than fabricate prose, this records the criteria
                 that decided the outcome and the probability of the closest call, which is the
                 audit trail the probabilities actually support.
  `confidence`-- the smallest margin to a threshold across all criteria, not a self-report. A
                 chat model's confidence is a number it wrote; this one is derived from the
                 decision boundary.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from ..backends.jev import GateDecision, JevClient, JevError
from ..llm.usage import record as record_usage
from .schema import AbstractScreeningOutput, FullTextScreeningOutput

# Asked alongside the criteria, in the same call, so it costs nothing extra. The chat path has
# the model self-report this; making it a question of its own means it is judged on the same
# footing as everything else and lands as a probability we can threshold.
FULLTEXT_INCOMPLETE_QUESTION = {
    "type": "noul",
    "instructions": {
        "question": (
            "Does the state contain only title, abstract or metadata, rather than the body of "
            "the article (methods, results or a results table)?"
        )
    },
    "criteria": {
        "true": "Only title/abstract/metadata is present; the article body is missing",
        "false": "The article body is present and can be assessed",
    },
}
_INCOMPLETE_KEY = "__fulltext_incomplete"


class JevScreeningClient:
    """Drop-in alternative to `ScreeningLLMClient` for criteria-gated screening.

    Unlike the OpenAI client this takes structured state and the criteria mapping rather than a
    rendered prompt string: Jev needs the criteria as separate questions, so flattening them
    into prose first would throw away exactly the structure it exploits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jev-latest",
        inclusion_threshold: float = 0.5,
        exclusion_threshold: float = 0.5,
        incomplete_threshold: float = 0.5,
        client: Optional[JevClient] = None,
    ) -> None:
        self.client = client or JevClient(api_key=api_key, model=model)
        self.inclusion_threshold = inclusion_threshold
        self.exclusion_threshold = exclusion_threshold
        self.incomplete_threshold = incomplete_threshold

    # -- public API -----------------------------------------------------------------------

    def screen_abstract_structured(
        self,
        state: Any,
        criteria_mapping: Mapping[str, Mapping[str, str]],
        objective: Optional[str] = None,
    ) -> AbstractScreeningOutput:
        decision, _ = self._gate(state, criteria_mapping, objective)
        return AbstractScreeningOutput(
            decision="INCLUDED" if decision.include else "EXCLUDED",
            confidence=decision.confidence,
            reason=_explain(decision),
            inclusion_criteria_applied=decision.satisfied_inclusion_ids,
            exclusion_criteria_applied=decision.fired_exclusion_ids,
        )

    def screen_fulltext_structured(
        self,
        state: Any,
        criteria_mapping: Mapping[str, Mapping[str, str]],
        objective: Optional[str] = None,
    ) -> FullTextScreeningOutput:
        decision, answers = self._gate(
            state, criteria_mapping, objective,
            extra_questions={_INCOMPLETE_KEY: FULLTEXT_INCOMPLETE_QUESTION},
        )
        raw = (answers.get(_INCOMPLETE_KEY) or {}).get("noul")
        incomplete = isinstance(raw, (int, float)) and float(raw) >= self.incomplete_threshold
        return FullTextScreeningOutput(
            decision="INCLUDED" if decision.include else "EXCLUDED",
            confidence=decision.confidence,
            reason=_explain(decision),
            fulltext_incomplete=bool(incomplete),
            inclusion_criteria_applied=decision.satisfied_inclusion_ids,
            exclusion_criteria_applied=decision.fired_exclusion_ids,
        )

    # -- internals ------------------------------------------------------------------------

    def _gate(
        self,
        state: Any,
        criteria_mapping: Optional[Mapping[str, Mapping[str, str]]],
        objective: Optional[str],
        extra_questions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        if not criteria_mapping or not (
            criteria_mapping.get("inclusion") or criteria_mapping.get("exclusion")
        ):
            # Without criteria there is nothing to gate on. The chat path would still return a
            # verdict from the objective alone; this backend cannot and should not pretend to.
            raise JevError(
                "Jev screening needs a criteria_mapping with at least one criterion. "
                "Criteria IDs are assigned by ConfigManager; check the stage config."
            )
        decision, answers = self.client.gate(
            state=state,
            criteria_mapping=criteria_mapping,
            objective=objective,
            inclusion_threshold=self.inclusion_threshold,
            exclusion_threshold=self.exclusion_threshold,
            extra_questions=extra_questions,
        )
        _record(decision)
        return decision, answers


def _explain(decision: GateDecision) -> str:
    """A factual summary, not a rationale. Says what decided it and how close it was."""
    if not decision.include:
        fired = decision.fired_exclusion_ids
        unmet = [v.criterion_id for v in decision.verdicts
                 if v.kind == "inclusion" and not v.satisfied]
        parts = []
        if unmet:
            parts.append("inclusion not met: " + ", ".join(
                f"{cid} p={decision.probabilities[cid]:.2f}" for cid in unmet))
        if fired:
            parts.append("exclusion triggered: " + ", ".join(
                f"{cid} p={decision.probabilities[cid]:.2f}" for cid in fired))
        body = "; ".join(parts) or "no criterion satisfied"
    else:
        body = "all criteria satisfied"
    closest = decision.closest_call
    if closest is not None:
        body += (f" (closest call {closest.criterion_id} "
                 f"p={closest.probability:.2f}, margin {closest.margin:.2f})")
    return f"[jev] {body}"


def _record(decision: GateDecision) -> None:
    """Feed Jev's token counts into the existing usage ledger.

    `record` expects an object with `prompt_tokens`/`completion_tokens`; Jev reports
    `input_tokens`/`output_tokens`, and output is free. Reusing the ledger keeps the cost
    figures comparable across backends instead of starting a second accounting system.
    """
    usage = decision.usage or {}
    if not usage:
        return

    class _Usage:
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens

    try:
        record_usage("jev_screening", "jev", _Usage())
    except Exception:
        # Usage accounting must never take a run down.
        pass


def build_state(study: Any, screening_type: str, full_text: Optional[str] = None) -> Dict[str, Any]:
    """Structured state for one study.

    Jev accepts an object as state, so the fields stay separate instead of being flattened into
    a prose block. That matters here: a criterion about the population should be read against
    the methods, not against whatever the title happens to say, and keeping the structure lets
    a question point at a named field.
    """
    state: Dict[str, Any] = {
        "title": getattr(study, "title", None) or "",
        "abstract": getattr(study, "abstract", None) or "",
    }
    for attr, key in (("journal", "journal"), ("year", "year"),
                      ("publication_year", "year"), ("pmid", "pmid")):
        value = getattr(study, attr, None)
        if value and key not in state:
            state[key] = value
    if screening_type != "abstract":
        body = full_text if full_text is not None else (getattr(study, "full_text", None) or "")
        state["full_text"] = body or "NO FULL TEXT AVAILABLE"
    return state
