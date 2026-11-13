"""Full-text screening and analysis selection in one judgment per study.

``screening.fulltext.mode: combined`` replaces two calls per included study -- full-text
screening, then annotation of its analyses -- with one: the model reads the study's record
once, decides whether the study is eligible against the full-text criteria and, if it is,
which of its analyses belong to each annotation.

It needs records that carry their analyses (``retrieval.records`` of kind ``analyses`` or
``neurostore``), because the analyses must exist before full-text screening runs. Everything
downstream is unchanged:

- the study's decision is a full-text screening result, saved where full-text screening saves
  them and counted in PRISMA as one;
- the analysis decisions are annotation decisions, saved where annotation saves them and
  signed with the annotation stage's own cache signature, so the annotation stage finds them
  complete and makes no call of its own. In combined mode it never calls the model: a
  decision the combined review did not make is recorded as a failure and retried by the next
  combined review, so one run never mixes the two ways of deciding.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ..annotation.client import AnnotationClient
from ..annotation.processor import AnnotationProcessor
from ..annotation.schema import AnnotationConfig, AnnotationDecision
from ..models.types import Study
from ..utils.criteria import CriteriaMapping

logger = logging.getLogger(__name__)

# Bump when the combined prompt or response changes; it enters both stages' signatures.
COMBINED_PROMPT_VERSION = "combined-review-v1"


def combined_link(annotation_config: AnnotationConfig, fulltext_config: Dict[str, Any]) -> Dict[str, Any]:
    """What ties the two stages together: each one's cache depends on the other's settings."""
    from ..cache_versions import FULLTEXT_SCREENING_PROMPT_VERSION
    from ..execution import stable_hash

    fulltext = {k: v for k, v in fulltext_config.items() if k not in ("model_params",)}
    return {
        "combined_prompt_version": COMBINED_PROMPT_VERSION,
        "fulltext": stable_hash({**fulltext, "prompt_version": FULLTEXT_SCREENING_PROMPT_VERSION}),
        "annotation": stable_hash(annotation_config.model_dump()),
    }


def _criteria_text(mapping: Optional[CriteriaMapping], criteria: List[str], kind: str) -> str:
    items = getattr(mapping, kind, None) if mapping is not None else None
    if items:
        return "\n".join(f"{cid}: {text}" for cid, text in items.items())
    return "\n".join(f"- {c}" for c in criteria) or "None specified"


def eligibility_block(fulltext_config: Dict[str, Any], document_description: Optional[str]) -> str:
    """The full-text criteria and how to judge them from a study record."""
    mapping = fulltext_config.get("criteria_mapping")
    additional = fulltext_config.get("additional_instructions")
    described = (
        f"The study is described by {document_description}, not by its full text."
        if document_description
        else "The study is described by a record of the article, not by its full text."
    )
    return f"""
FULL-TEXT ELIGIBILITY (STEP 1)
{described} Judge every criterion from what the record states. A criterion the record
says nothing about is not, by itself, a reason to exclude: exclude on what the record
states or clearly implies, and say in the reason which criteria the record left open.
Set fulltext_incomplete=true ONLY when the record itself shows that the article could not
be read; never because it has no introduction, methods, results or discussion.

META-ANALYSIS OBJECTIVE:
{fulltext_config.get("objective") or "Not provided"}

FULL-TEXT INCLUSION CRITERIA (all must be met):
{_criteria_text(mapping, fulltext_config.get("inclusion_criteria") or [], "inclusion")}

FULL-TEXT EXCLUSION CRITERIA (none may be met):
{_criteria_text(mapping, fulltext_config.get("exclusion_criteria") or [], "exclusion")}
{f"{chr(10)}{additional}{chr(10)}" if additional else ""}
In study_screening, list every full-text inclusion ID met and every exclusion ID met, and for
an excluded study name the inclusion IDs not met.
""".strip()


class CombinedReview:
    """Makes the combined call for full-text screening, and keeps the analysis decisions."""

    def __init__(
        self,
        annotation_config: AnnotationConfig,
        fulltext_config: Dict[str, Any],
        output_dir: str,
        num_workers: int = 1,
        max_retries: int = 3,
    ):
        self.annotation_config = annotation_config
        self.fulltext_config = fulltext_config
        self.output_dir = str(output_dir)
        self.link = combined_link(annotation_config, fulltext_config)
        # Signs and saves exactly as the annotation stage will read them.
        self.processor = AnnotationProcessor(
            annotation_config, num_workers=num_workers, max_retries=max_retries,
            combined_with=self.link,
        )
        self.client = AnnotationClient(max_retries=max_retries)
        self._existing = self.processor._load_cached_results(self.output_dir) or []
        self._pending: List[AnnotationDecision] = []
        self._lock = threading.Lock()

    def signature_component(self) -> Dict[str, Any]:
        """The part of a full-text result's cache signature that the annotations decide."""
        return self.link

    def decisions_complete(self, study: Study) -> bool:
        """Whether a cached inclusion still has every analysis decision it was made with."""
        if not study.analyses:
            return True
        with self._lock:
            known = self._existing + self._pending
        complete = self.processor._get_annotations_with_complete_results_for_study(
            known, study.pmid, self.annotation_config.annotations, study
        )
        return len(complete) == len(self.annotation_config.annotations)

    def review(self, study: Study, model: str, model_params: Optional[Dict[str, Any]] = None):
        """Screen ``study`` and select its analyses; returns a full-text screening response.

        The analysis decisions are kept until ``flush``. A study that is excluded or whose
        record is unreadable has none.
        """
        metadata_fields = list(self.annotation_config.metadata_fields)
        group = self.processor._build_study_analysis_group(study, metadata_fields)
        mapping = self.fulltext_config.get("criteria_mapping")
        screening, decisions = self.client.make_combined_decision(
            group,
            self.annotation_config.annotations,
            metadata_fields,
            eligibility_block(
                self.fulltext_config,
                study.document.description if study.document is not None else None,
            ),
            model=model,
            model_params=model_params,
            inclusion_ids=list(getattr(mapping, "inclusion", {}) or {}),
            exclusion_ids=list(getattr(mapping, "exclusion", {}) or {}),
        )
        if screening["decision"] == "INCLUDED" and not screening["fulltext_incomplete"]:
            signed = self.processor._with_failures(
                study, list(self.annotation_config.annotations), decisions, model
            )
            with self._lock:
                self._pending = [d for d in self._pending if d.study_id != study.pmid] + signed
        return SimpleNamespace(**screening)

    def flush(self) -> int:
        """Save the analysis decisions made so far where the annotation stage reads them."""
        with self._lock:
            pending, self._pending = self._pending, []
        if pending:
            self.processor._save_results_by_study(pending, self.output_dir, self._existing)
            self._existing = self.processor._load_cached_results(self.output_dir) or []
        logger.info("Combined review: saved %s analysis decisions", len(pending))
        return len(pending)
