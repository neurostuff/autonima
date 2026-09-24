# Jev (TypeSafe System One) as a screening backend

**Status: experiment, contract validated live.** Two calls have been made against the real
service (see *Live validation* below); everything else is exercised against a fake transport.

## Why

Screening and analysis selection are conjunctions of independent yes/no judgements, not
text-generation tasks. The OpenAI path asks a chat model to weigh every criterion at once and
return one verdict plus prose; the decision rule therefore lives inside the model, where it
cannot be inspected or tuned.

[Jev](https://docs.typesafe.ai) evaluates typed questions in parallel and in isolation against
the same state and returns a calibrated probability per question, with no generated text. One
Noul question per criterion gives:

- **a probability per criterion**, not one verdict per paper;
- **the gate as ordinary Python** (`autonima.backends.jev.apply_gate`), auditable and
  changeable without a model call;
- **a threshold that can be swept after the run** — so screening precision/recall becomes a
  curve rather than a single operating point.

Pricing is $0.042/MTok input with output free, against $0.25/$2.00 for `gpt-5-mini`.

## Use

```yaml
screening:
  abstract:
    backend: jev              # default: openai
    model: jev-latest
    inclusion_threshold: 0.5  # include when every inclusion criterion >= this
    exclusion_threshold: 0.5  # reject when any exclusion criterion >= this
    objective: ...
    inclusion_criteria: [...]
    exclusion_criteria: [...]

annotation:                   # analysis-level selection
  backend: jev
  model: jev-latest
  inclusion_threshold: 0.5
  exclusion_threshold: 0.5
  annotations: [...]
```

For selection, the article is the **state** and each question carries one analysis, so a study
with `n` analyses and `m` targets costs one call of `n x m x criteria` questions rather than
`n x m` separate requests. Calls are chunked at `max_questions_per_call` (default 200).

`export TYPESAFE_API_KEY=...` before running. Criterion IDs (`I1`, `E1`, …) are assigned by
`ConfigManager` as usual and are reused verbatim as Jev question keys, so
`inclusion_criteria_applied` / `exclusion_criteria_applied` keep their existing meaning.

## The gate

Include when **every** inclusion criterion clears `inclusion_threshold` **and no** exclusion
criterion reaches `exclusion_threshold` — the same rule the chat path is instructed to follow,
but executed in code.

An unanswered question counts as **not satisfied**, never as a pass. Silently dropping a
question would turn a transport error into a lenient decision.

`confidence` is the smallest margin to a threshold across all criteria: the decision is exactly
as strong as its weakest link. Deliberately not the mean, which would let a pile of obvious
criteria mask one coin-flip.

## What changes for the reader

| | OpenAI path | Jev path |
|---|---|---|
| `reason` | model-written prose | factual summary: deciding criteria, probabilities, closest call |
| `confidence` | model's self-report | derived from the decision boundary |
| criteria | weighed together | judged **independently** |
| `fulltext_incomplete` | model self-reports | its own Noul question |

## Known unknowns

These need a key and a real run to settle:

1. **Does independent evaluation cost accuracy at corpus scale?** One study agreeing 4/4 is
   encouraging, not evidence. The chat model can trade criteria off against each other and read
   the objective as a whole; Jev cannot. Running a full project against the benchmark is the
   experiment.
2. **Threshold calibration.** 0.5 is a placeholder, not a tuned value. The right way to set it
   is to sweep against the benchmark — which this backend makes possible for the first time.
3. **Criterion phrasing.** Criteria written as instructions to a chat model may need rewording
   as statements. `build_criteria_questions` wraps them, but wrapping is not rewriting. The
   reverse-direction contrast landing at p=0.54 may be a criteria-wording problem rather than a
   model limitation.
4. **Question-count ceiling.** Undocumented. Calls are chunked at `max_questions_per_call`
   (default 200); the largest live call so far was 12.

## Live validation

Two calls, 2026-09-24, against `jev-latest`.

**1. Screening contract** (`examples/jev_smoke.py`) — 4 criteria, abstract-sized state.
790 input / 72 output tokens, **513 ms**. Probabilities cleanly separated: inclusion 0.98 and
0.99, exclusion 0.02 and 0.01.

**2. Analysis-level selection** — real article (PMID 26529426), **60,781-character full text**,
4 parsed analyses x 3 reappraisal criteria = 12 questions in **one call, 720 ms**.

| analysis | Jev | current pipeline |
|---|---|---|
| Reappraisal > Maintain | include | include |
| Maintain > Reappraisal | include (conf 0.08) | include |
| Pictures vs baseline: patients > controls | exclude (I1 p=0.04) | exclude |
| Maintain vs Reappraise: patients > controls | exclude (I1 p=0.28) | exclude |

**4/4 agreement**, and the confidences are informative: the passive-viewing contrast is
rejected decisively (p=0.04), while the reverse-direction contrast sits at p=0.54 — a coin
flip the model correctly reports as one. That is the calibration the chat path cannot give.

Two open questions are now closed: a full text **fits in `state`**, and the payload we build is
accepted as specified.

## Known failure modes we are walking into

TypeSafe publish a [jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13) for
jev-1.13. Three of its nine documented failure modes describe this integration directly.

**1. Literal reading** — *"answers the question you wrote, not the one you meant… when you look
at a wrong answer and find yourself explaining what you really meant, that explanation is the
missing half of the instruction."*

This is exactly the `maintain` failure. Criteria such as `GLOBAL_I5` ("Assign every label that
applies…") are instructions to a reader, not propositions, and a literal evaluator scores them
near zero. A chat model silently absorbs them as guidance. **Criteria have to be rewritten as
statements that can be true or false of one analysis**, with boundary cases moved into the
Noul's `criteria.true` / `criteria.false`, and compound criteria split into two questions
combined in code.

**5. Large state full of irrelevant detail** — *"Accuracy falls as the state grows with content
unrelated to the decision… Jev suffers from context rot."*

We send an entire article — 60k characters, ~15k tokens — as the state for every question,
when the evidence for any one criterion is usually a table caption, an analysis description and
a paragraph of methods. This is the documented anti-pattern, and we are at the extreme of it.
The recommended fix is to filter in code first, or to use a Noul to select relevant passages.
**Untested here, and the most likely remaining explanation for Jev's conservatism** once the
instruction-shaped criteria are fixed.

**4. Indirection** — *"a question about a property of a property… costs accuracy."*

Selection questions ask whether a criterion holds of an analysis named in the instructions,
judged against an article in the state. That is a hop. Putting the analysis's own table and
caption in the state, one call per analysis, trades calls for directness and is worth measuring.

Two smaller ones worth knowing: **contradictory instructions and criteria** degrade accuracy,
so the auto-generated `true`/`false` descriptions must read coherently for the criterion they
wrap — they do not when the criterion is an instruction. And **structural invariants do not
hold**: a Noul is absolute rather than relative, base rates differ sharply per criterion
(observed means from 0.18 to 0.75 on the same corpus), so a single global threshold across all
criteria is not obviously right. Per-criterion thresholds are a cheap thing to try, since
sweeping costs nothing.

## Testing

`tests/test_jev_backend.py` covers question construction, the gate arithmetic, transport
behaviour and screener dispatch — all against a fake transport, no network and no key.

```bash
pytest tests/test_jev_backend.py -q
```
