#!/usr/bin/env python3
"""Smoke-test the Jev backend against the live API. Needs TYPESAFE_API_KEY.

Nothing in the test suite touches the network, so this is the first thing to run once a key
exists. It answers the two questions the unit tests cannot: does the payload we build get
accepted, and does a long full text fit in `state`?

    python examples/jev_smoke.py
    python examples/jev_smoke.py --chars 200000     # probe the state-size ceiling
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonima.backends.jev import JevClient, JevError, apply_gate  # noqa: E402

MAPPING = {
    "inclusion": {
        "I1": "the study reports a task-based fMRI experiment",
        "I2": "participants are healthy adults",
    },
    "exclusion": {
        "E1": "the article is a review or meta-analysis rather than primary data",
        "E2": "the sample is paediatric",
    },
}

STATE = {
    "title": "Amygdala response during emotional reappraisal in healthy adults",
    "abstract": (
        "Thirty healthy adults (18-35 years) completed an emotion-regulation task during "
        "functional MRI. Participants viewed negative images and either reappraised or "
        "maintained their emotional response. Whole-brain analyses revealed reduced amygdala "
        "activity during reappraisal relative to maintain."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", type=int, default=0,
                    help="pad the state with this many characters to probe the size limit")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set; nothing to test against.", file=sys.stderr)
        return 2

    state = dict(STATE)
    if args.chars:
        filler = "The participants performed the task in a 3T scanner. " * 1000
        state["full_text"] = (filler * (args.chars // len(filler) + 1))[:args.chars]
        print(f"padded state to {args.chars:,} characters")

    client = JevClient()
    start = time.perf_counter()
    try:
        decision, answers = client.gate(
            state=state,
            criteria_mapping=MAPPING,
            objective="Studies of emotion regulation using task fMRI in healthy adults",
            inclusion_threshold=args.threshold,
            exclusion_threshold=args.threshold,
        )
    except JevError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nlatency        {elapsed:.0f} ms")
    print(f"usage          {dict(decision.usage)}")
    print(f"decision       {'INCLUDE' if decision.include else 'EXCLUDE'}"
          f"   confidence {decision.confidence:.3f}\n")
    print(f"  {'id':<5}{'kind':<11}{'p':>7}   criterion")
    for v in decision.verdicts:
        flag = " " if v.satisfied else "*"
        print(f" {flag}{v.criterion_id:<5}{v.kind:<11}{v.probability:>7.3f}   {v.text}")
    closest = decision.closest_call
    if closest:
        print(f"\nclosest call   {closest.criterion_id} "
              f"(p={closest.probability:.3f}, margin {closest.margin:.3f})")

    # The payoff: re-gate the SAME probabilities at other thresholds, no further API calls.
    print("\nthreshold sweep on the stored probabilities (no extra calls):")
    for t in (0.3, 0.5, 0.7, 0.9):
        d = apply_gate(answers, MAPPING, inclusion_threshold=t, exclusion_threshold=t)
        print(f"  tau={t:.1f}  ->  {'INCLUDE' if d.include else 'EXCLUDE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
