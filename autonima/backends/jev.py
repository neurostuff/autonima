"""Jev (TypeSafe "System One") as a decision backend for criteria gating.

WHY THIS EXISTS

Screening and analysis selection are not text-generation tasks. They are a conjunction of
independent yes/no judgements -- "does this study use fMRI?", "are the participants adults?" --
that the pipeline currently obtains by asking a chat model to weigh every criterion at once and
emit a single INCLUDED/EXCLUDED verdict plus prose. That works, but the decision rule lives
inside the model, so it cannot be inspected, tuned, or moved without re-running everything.

Jev evaluates each question in parallel and in isolation against the same state and returns a
calibrated probability per question, with no generated text. Mapping one Noul question onto one
criterion therefore gives us:

  * a probability per criterion rather than one verdict per paper;
  * the gate itself as ordinary Python (see `apply_gate`), auditable and changeable without a
    model call;
  * a threshold that can be swept AFTER the run, so the precision/recall trade-off becomes a
    curve instead of a single operating point.

WHAT IT GIVES UP

No rationale. Jev does not generate strings, so `reason` cannot be populated the way the OpenAI
path populates it. The per-criterion probabilities are a different and arguably better audit
trail -- they say which criterion was marginal -- but they are not an explanation, and the HTML
review reports that surface reasoning will show the probability table instead.

Criteria are also judged INDEPENDENTLY. The chat path lets the model trade one criterion off
against another and read the objective as a whole; Jev deliberately does not. Whether that costs
accuracy on this corpus is the empirical question this backend exists to answer -- do not assume
either direction.

STATUS

Written against the published API contract (docs.typesafe.ai, API reference, 2026-09-24) and
exercised only against a fake transport. Nothing here has been run against the live service.

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# Documented for jev-1.13 (docs.typesafe.ai/models):
#   64k tokens per request  -- state plus ALL questions combined
#   32k tokens              -- state plus the SINGLE LONGEST question
# Both are hard: exceeding either returns HTTP 400 max_tokens_exceeded, which is not
# retryable. A full ER project run lost 63 of 128 studies to this before the budget existed,
# and the failures tracked analyses-per-study (median 7 vs 3) rather than article length,
# because every question repeats its analysis metadata.
MAX_REQUEST_TOKENS = 64_000
MAX_STATE_PLUS_QUESTION_TOKENS = 32_000
# Chars per token. Deliberately pessimistic: 4.0 is the usual rule of thumb, and scientific
# prose with coordinates and markup runs denser than prose. Overestimating tokens costs an
# extra chunk; underestimating costs the whole study.
CHARS_PER_TOKEN = 3.2
# Leave room for the envelope (model name, keys, JSON punctuation) and for the estimate
# being wrong in the unsafe direction.
BUDGET_HEADROOM = 0.90

# Retry on the two statuses the API documents as transient. 401 and 422 are our bug, not theirs.
RETRY_STATUSES = frozenset({429, 529})


def estimate_tokens(obj: Any) -> int:
    """Rough token count for anything JSON-serialisable, biased to overestimate."""
    import json as _json

    try:
        text = obj if isinstance(obj, str) else _json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    return int(len(text) / CHARS_PER_TOKEN) + 1


def fit_state(state: Any, reserve_tokens: int = 0) -> Tuple[Any, bool]:
    """Shrink `state` until it fits the 32k state-plus-longest-question budget.

    Only the longest string field is trimmed, and it is trimmed from the MIDDLE: the head of
    an article carries title, abstract and methods, and the tail carries results and tables,
    which is where the criteria are actually decided. Cutting the tail to fit would
    systematically drop the evidence and quietly bias every decision on long articles.

    Returns (state, was_truncated).
    """
    budget = int((MAX_STATE_PLUS_QUESTION_TOKENS - reserve_tokens) * BUDGET_HEADROOM)
    if budget <= 0 or estimate_tokens(state) <= budget:
        return state, False
    if not isinstance(state, dict):
        keep = max(1, int(budget * CHARS_PER_TOKEN))
        text = state if isinstance(state, str) else str(state)
        half = keep // 2
        return text[:half] + "\n[... truncated ...]\n" + text[-half:], True

    trimmed = dict(state)
    # repeatedly halve the largest string field until the whole object fits
    for _ in range(24):
        if estimate_tokens(trimmed) <= budget:
            return trimmed, True
        longest = max((k for k, v in trimmed.items() if isinstance(v, str)),
                      key=lambda k: len(trimmed[k]), default=None)
        if longest is None or len(trimmed[longest]) < 400:
            break
        text = trimmed[longest]
        keep = len(text) // 2
        half = keep // 2
        trimmed[longest] = text[:half] + "\n[... truncated ...]\n" + text[-half:]
    return trimmed, True


def plan_chunks(
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Dict[str, Any]]]:
    """Split questions so each request fits the 64k state-plus-all-questions budget.

    The state is re-sent with every chunk, so it is charged once per chunk -- at $0.042/MTok
    that is the right trade against losing the study entirely.
    """
    state_tokens = estimate_tokens(state)
    budget = int(MAX_REQUEST_TOKENS * BUDGET_HEADROOM) - state_tokens
    chunks: List[Dict[str, Dict[str, Any]]] = []
    current: Dict[str, Dict[str, Any]] = {}
    used = 0
    for key, question in questions.items():
        cost = estimate_tokens({key: question})
        if current and used + cost > budget:
            chunks.append(current)
            current, used = {}, 0
        current[key] = question
        used += cost
    if current:
        chunks.append(current)
    return chunks or [{}]


class JevError(RuntimeError):
    """A Jev request failed in a way retrying will not fix."""


@dataclass(frozen=True)
class CriterionVerdict:
    """One criterion's probability of being true, and how it was read."""

    criterion_id: str
    text: str
    kind: str          # "inclusion" | "exclusion"
    probability: float
    satisfied: bool    # inclusion: p >= threshold. exclusion: p < threshold (i.e. did NOT fire)
    margin: float      # distance from its threshold; small margin == borderline


@dataclass(frozen=True)
class GateDecision:
    """The boolean roll-up, plus everything needed to re-derive or re-tune it."""

    include: bool
    verdicts: Tuple[CriterionVerdict, ...]
    confidence: float
    inclusion_thresholds: float
    exclusion_thresholds: float
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def satisfied_inclusion_ids(self) -> List[str]:
        return [v.criterion_id for v in self.verdicts
                if v.kind == "inclusion" and v.satisfied]

    @property
    def fired_exclusion_ids(self) -> List[str]:
        """Exclusion criteria that tripped -- the reason for a rejection."""
        return [v.criterion_id for v in self.verdicts
                if v.kind == "exclusion" and not v.satisfied]

    @property
    def probabilities(self) -> Dict[str, float]:
        return {v.criterion_id: v.probability for v in self.verdicts}

    @property
    def closest_call(self) -> Optional[CriterionVerdict]:
        """The criterion that came nearest to flipping the decision."""
        return min(self.verdicts, key=lambda v: v.margin) if self.verdicts else None


def normalise_mapping(criteria_mapping: Any) -> Dict[str, Dict[str, str]]:
    """Accept either the `CriteriaMapping` dataclass or its dict form.

    `ConfigManager` stores the dataclass on the stage config while the executed-config YAML and
    the annotation schema carry the dict. Both reach this module, and guessing wrong silently
    produces zero questions -- which `apply_gate` would then read as "no criteria satisfied"
    and reject every study. Normalising once, here, is the only place that has to know.
    """
    if criteria_mapping is None:
        return {"inclusion": {}, "exclusion": {}}
    if hasattr(criteria_mapping, "inclusion") or hasattr(criteria_mapping, "exclusion"):
        return {
            "inclusion": dict(getattr(criteria_mapping, "inclusion", {}) or {}),
            "exclusion": dict(getattr(criteria_mapping, "exclusion", {}) or {}),
        }
    return {
        "inclusion": dict((criteria_mapping.get("inclusion") or {})),
        "exclusion": dict((criteria_mapping.get("exclusion") or {})),
    }


def build_criteria_questions(
    criteria_mapping: Any,
    objective: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """One Noul question per criterion, keyed by the criterion ID autonima already assigns.

    `criteria_mapping` is the structure `CriteriaIDAssigner` produces and the config carries:
    ``{"inclusion": {"I1": text, ...}, "exclusion": {"E1": text, ...}}``. Reusing those IDs as
    question keys means the answers come back already joined to the criteria, and the existing
    `inclusion_criteria_applied` / `exclusion_criteria_applied` fields keep their meaning.

    The criterion text is asked as a STATEMENT to be judged true or false, which is what Noul
    expects, rather than as an instruction to a chat model. `criteria` pins down what true and
    false mean so a criterion phrased as a noun phrase ("adult participants") is not read as a
    question about the study's topic.
    """
    criteria_mapping = normalise_mapping(criteria_mapping)
    questions: Dict[str, Dict[str, Any]] = {}
    for kind, prefix_key in (("inclusion", "inclusion"), ("exclusion", "exclusion")):
        for criterion_id, text in (criteria_mapping.get(prefix_key) or {}).items():
            instructions: Any = {
                "criterion": text,
                "question": "Is `criterion` true of the study described in the state?",
            }
            if objective:
                instructions["review_objective"] = objective
            questions[criterion_id] = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {
                    "true": f"The study satisfies: {text}",
                    "false": f"The study does not satisfy, or does not report, {text}",
                },
            }
    return questions


def apply_gate(
    answers: Mapping[str, Mapping[str, Any]],
    criteria_mapping: Any,
    inclusion_threshold: float = 0.5,
    exclusion_threshold: float = 0.5,
    usage: Optional[Mapping[str, Any]] = None,
) -> GateDecision:
    """Turn per-criterion probabilities into an include/exclude decision.

    THE GATE, STATED ONCE: include when every inclusion criterion clears
    `inclusion_threshold` AND no exclusion criterion reaches `exclusion_threshold`. This is the
    same rule `pipeline.py` documents for the chat path ("retained when all inclusion criteria
    are satisfied and no exclusion criteria apply"); the difference is that here it is code.

    A criterion with no answer is treated as NOT satisfied rather than skipped. Silently
    dropping a question the model failed to answer would turn a transport error into a lenient
    decision, which is the failure mode worth engineering against.

    `confidence` is the smallest margin across all criteria -- the decision is exactly as strong
    as its weakest link. It is deliberately not the mean, which would let a pile of obvious
    criteria mask one coin-flip.
    """
    if not 0.0 <= inclusion_threshold <= 1.0 or not 0.0 <= exclusion_threshold <= 1.0:
        raise ValueError("thresholds must lie in [0, 1]")
    criteria_mapping = normalise_mapping(criteria_mapping)

    verdicts: List[CriterionVerdict] = []
    for kind, threshold in (("inclusion", inclusion_threshold),
                            ("exclusion", exclusion_threshold)):
        for criterion_id, text in (criteria_mapping.get(kind) or {}).items():
            answer = answers.get(criterion_id) or {}
            raw = answer.get("noul")
            probability = float(raw) if isinstance(raw, (int, float)) else 0.0
            missing = raw is None
            if kind == "inclusion":
                satisfied = (not missing) and probability >= threshold
            else:
                # An unanswered exclusion criterion must not silently pass the study through.
                satisfied = (not missing) and probability < threshold
            verdicts.append(CriterionVerdict(
                criterion_id=criterion_id,
                text=text,
                kind=kind,
                probability=probability,
                satisfied=satisfied,
                margin=0.0 if missing else abs(probability - threshold),
            ))

    include = all(v.satisfied for v in verdicts)
    confidence = min((v.margin for v in verdicts), default=0.0) * 2.0  # margin 0.5 -> 1.0
    return GateDecision(
        include=include,
        verdicts=tuple(verdicts),
        confidence=round(min(confidence, 1.0), 4),
        inclusion_thresholds=inclusion_threshold,
        exclusion_thresholds=exclusion_threshold,
        usage=dict(usage or {}),
    )


class JevClient:
    """Thin client over the System One evaluation endpoint.

    `transport` is injectable so the whole path can be tested without a network or a key; the
    default uses `requests`, already a hard dependency, rather than adding the official SDK. If
    the SDK's retry policy turns out to be worth it, swapping it in only touches `_post`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 60.0,
        max_retries: int = 4,
        transport: Optional[Callable[[str, Dict[str, str], Dict[str, Any], float], Any]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key and transport is None:
            raise JevError(
                "No TypeSafe API key. Set TYPESAFE_API_KEY, pass api_key=, or inject a "
                "transport for testing."
            )
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport

    def _post(self, payload: Dict[str, Any]) -> Any:
        if self._transport is not None:
            return self._transport(self.endpoint, self._headers(), payload, self.timeout)
        import requests  # local import: keeps module import cheap and testable without it

        return requests.post(
            self.endpoint, headers=self._headers(), json=payload, timeout=self.timeout
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """One call, every question answered in parallel. Returns the decoded response body.

        Retries only 429 and 529, with exponential backoff and jitter. A 401 or 422 is a bug in
        the key or the payload and is raised immediately -- retrying it just burns the rate
        limit and hides the real error.
        """
        if not questions:
            raise JevError("evaluate() needs at least one question")
        payload = {"state": state, "model": self.model, "questions": dict(questions)}

        last_status: Optional[int] = None
        for attempt in range(self.max_retries + 1):
            response = self._post(payload)
            status = getattr(response, "status_code", 200)
            if status == 200:
                body = response.json()
                if "answers" not in body:
                    raise JevError(f"response had no 'answers' key: {sorted(body)}")
                return body
            last_status = status
            if status not in RETRY_STATUSES or attempt == self.max_retries:
                raise JevError(f"Jev request failed with HTTP {status}: {_body_text(response)}")
            time.sleep(min(2.0 ** attempt, 16.0) * (0.5 + random.random() / 2))
        raise JevError(f"Jev request failed after retries (last status {last_status})")

    def gate(
        self,
        state: Any,
        criteria_mapping: Any,
        objective: Optional[str] = None,
        inclusion_threshold: float = 0.5,
        exclusion_threshold: float = 0.5,
        extra_questions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Criteria in, decision out. Returns (decision, raw answers) so callers can keep both.

        `extra_questions` rides along in the same call -- used for `fulltext_incomplete`, which
        the chat path asks the model to self-report and which is cleaner as its own Noul.
        """
        questions = dict(build_criteria_questions(criteria_mapping, objective=objective))
        if extra_questions:
            overlap = set(questions) & set(extra_questions)
            if overlap:
                raise JevError(f"extra_questions collide with criterion IDs: {sorted(overlap)}")
            questions.update(extra_questions)

        body = self.evaluate(state, questions)
        answers = body.get("answers", {})
        decision = apply_gate(
            answers,
            criteria_mapping,
            inclusion_threshold=inclusion_threshold,
            exclusion_threshold=exclusion_threshold,
            usage=body.get("usage", {}),
        )
        return decision, answers


def _body_text(response: Any) -> str:
    try:
        return str(response.json())[:400]
    except Exception:
        return str(getattr(response, "text", ""))[:400]
