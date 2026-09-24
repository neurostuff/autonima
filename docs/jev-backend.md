# Jev (TypeSafe System One) as a screening backend

**Status: experiment.** Written against the published API contract and exercised only against a
fake transport. Nothing here has been run against the live service.

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
```

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

1. **Does independent evaluation cost accuracy?** The chat model can trade criteria off against
   each other and read the objective as a whole. Jev cannot. This is the experiment.
2. **State size.** Full texts are long and the API documents no limit. Untested.
3. **Threshold calibration.** 0.5 is a placeholder, not a tuned value. The right way to set it
   is to sweep against the benchmark — which this backend makes possible for the first time.
4. **Criterion phrasing.** Criteria written as instructions to a chat model may need rewording
   as statements. `build_criteria_questions` wraps them, but wrapping is not rewriting.

## Testing

`tests/test_jev_backend.py` covers question construction, the gate arithmetic, transport
behaviour and screener dispatch — all against a fake transport, no network and no key.

```bash
pytest tests/test_jev_backend.py -q
```
