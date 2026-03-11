# Interpreting Outputs

This guide explains how to read Autonima run artifacts and turn them into review decisions.

For file locations and folder structure, see the [Outputs](./outputs.md) guide.

## Which File Should I Open First?

| Question | Open this file first | What to check |
|---|---|---|
| What did search return? | `outputs/search_results.json` | `studies` list and metadata completeness (PMID/title/abstract). |
| Why was a study included/excluded at abstract stage? | `outputs/abstract_screening_results.json` | `decision`, `reason`, and confidence/criteria IDs. |
| Did full-text retrieval work? | `outputs/fulltext_retrieval_results.json` | `fulltext_available`, `pmcid`, and `full_text_path`. |
| Why was a study excluded at full-text stage? | `outputs/fulltext_screening_results.json` | `decision`, `reason`, confidence, and applied criteria IDs. |
| What are the final included studies? | `outputs/final_results.json` | `studies` section (final included only). |
| Which studies had no full text? | `outputs/unavailable_fulltexts.txt` | PMIDs that still have no retrievable full text. |
| Which studies had incomplete full text? | `outputs/incomplete_fulltext.txt` + `outputs/incomplete_fulltext.csv` | PMIDs screened as `fulltext_incomplete`; CSV also includes `full_text_path` for fixing source files. |
| What did annotation produce? | `outputs/annotation_results.json` | Per-analysis `annotation_name`, `include`, and `reasoning`. |
| Are coordinates parsed and ready for NiMADS/meta? | `outputs/coordinate_parsing_results.json` + `outputs/nimads_studyset.json` | Parsed analyses/tables and generated NiMADS payloads. |

## Status Values You Will See

Screening and study records use stage-specific statuses:

- `pending`: not screened yet
- `included_abstract`: passed abstract screening
- `excluded_abstract`: excluded at abstract screening
- `included_fulltext`: passed full-text screening (final included set)
- `excluded_fulltext`: excluded at full-text screening
- `fulltext_incomplete`: full text was present but incomplete/truncated for eligibility decisions
- `retrieval_failed`: retrieval failed
- `screening_failed`: screening call failed

Interpretation rule:

- A study is final-included only when status is `included_fulltext`.

## How to Read the Pipeline End-to-End

1. Start with `outputs/search_results.json` to confirm the search scope is correct.
2. Move to `outputs/abstract_screening_results.json` to review early exclusions and reasoning quality.
3. Use `outputs/fulltext_retrieval_results.json` and `outputs/unavailable_fulltexts.txt` to identify coverage gaps before full-text screening.
4. Review `outputs/fulltext_screening_results.json` for final screening logic and criteria application.
5. Use `outputs/final_results.json` for downstream analysis inputs and reporting.

## Important Caveat About `final_results.json`

`final_results.json` is intentionally filtered to final included studies in its `studies` list.

If you need exclusion audit trails:

- use `abstract_screening_results.json` for abstract-stage exclusions
- use `fulltext_screening_results.json` for full-text-stage exclusions

## Interpreting Confidence and Criteria IDs

- `confidence` is model-reported certainty. Low confidence can be auto-excluded if a stage threshold is enabled.
- `inclusion_criteria_applied` and `exclusion_criteria_applied` capture criterion IDs used in the decision.
- If criterion mappings are configured, these IDs should match your mapping keys (for example `I1`, `E2`, or namespaced IDs).

## Meta-Analysis Readiness Checklist

Before running `autonima meta`, confirm:

- `outputs/nimads_studyset.json` exists
- `outputs/nimads_annotation.json` exists
- Included studies in `final_results.json` align with your expected selection

Then run:

```bash
autonima meta <output-folder>/outputs
```

Meta outputs are created under:

```text
<output-folder>/outputs/meta_analysis_results/
```
