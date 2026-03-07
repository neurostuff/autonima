# Outputs

Autonima writes run artifacts under the resolved runtime output folder.

If you run:

```bash
autonima run projects/cue_reactivity/default.yaml
```

the default output folder is:

```text
projects/cue_reactivity/default/
```

## Typical Layout

```text
<output-folder>/
├── outputs/
│   ├── search_results.json
│   ├── abstract_screening_results.json
│   ├── fulltext_retrieval_results.json
│   ├── fulltext_screening_results.json
│   ├── final_results.json
│   ├── incomplete_fulltext.txt
│   ├── criteria_mapping.json
│   ├── coordinate_parsing_results.json
│   ├── nimads_studyset.json
│   └── nimads_annotation.json
├── retrieval/
│   └── pubget_data/
└── meta_analysis_results/
```

The exact contents depend on which stages are enabled.

## Need Help Interpreting These Files?

Use the [Interpreting Outputs](./interpreting-outputs.md) guide for a task-oriented walkthrough of what each artifact means and which file to inspect first.

## Key Files

## `outputs/search_results.json`

- search-stage results
- study metadata from PubMed or provided PMIDs

## `outputs/abstract_screening_results.json`

- decisions and reasoning from abstract screening

## `outputs/fulltext_retrieval_results.json`

- retrieval summary and availability information

## `outputs/fulltext_screening_results.json`

- decisions and reasoning from full-text screening

## `outputs/final_results.json`

- final included/excluded status and execution outputs

## `outputs/incomplete_fulltext.txt`

- PMIDs where full-text screening returned `fulltext_incomplete`

## `outputs/criteria_mapping.json`

- generated mapping of screening criteria IDs used in downstream results

## `outputs/coordinate_parsing_results.json`

- cached coordinate parsing results when parsing is enabled

## `outputs/nimads_studyset.json`
## `outputs/nimads_annotation.json`

- NiMADS artifacts used by `autonima meta`
- pass the containing folder to `autonima meta`, usually `<output-folder>/outputs`

## `retrieval/pubget_data/`

Contains retrieved article data from PubGet, such as:

- `metadata.csv`
- `text.csv`
- `coordinates.csv`
- extracted table data

## `meta_analysis_results/`

Created by `autonima meta`. Contains one directory per annotation column and the generated NiMARE artifacts and reports.

## Practical Notes

- The CLI runtime output path may differ from the `output.directory` stored in YAML.
- If you rerun with the same output folder, Autonima may reuse cached intermediary artifacts such as search results and annotation results.
- When documenting or sharing a run, include both the config file and the resolved runtime output folder.
