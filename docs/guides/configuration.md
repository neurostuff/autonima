# Configuration Guide

Autonima uses a YAML config file for the pipeline. The canonical example lives at `examples/sample_config.yml` and is also emitted by `autonima create-sample-config`.

## Minimal Working Config

This is the smallest practical config that passes current validation:

```yaml
search:
  database: "pubmed"
  query: "schizophrenia AND working memory AND fMRI"
  max_results: 100

retrieval:
  sources:
    - pubget
  load_excluded: false

screening:
  abstract:
    model: "gpt-5-mini-2025-08-07"
    objective: "Identify fMRI studies of working memory in schizophrenia"
    inclusion_criteria:
      - Human participants
      - fMRI neuroimaging
  fulltext:
    model: "gpt-5-mini-2025-08-07"
    objective: "Identify fMRI studies of working memory in schizophrenia"
    inclusion_criteria:
      - Human participants
      - fMRI neuroimaging

parsing:
  parse_coordinates: false
  coordinate_model: "gpt-4o-mini"

output:
  directory: "results"

annotation:
  enabled: false
```

## Validation Rules to Know

- `search.query` must be non-empty unless you provide `pmids_file` or `pmids_list`.
- `screening.abstract` must define an `objective` and `inclusion_criteria`, or set `skip_stage: true`.
- `screening.fulltext` must define an `objective` and `inclusion_criteria`, or set `skip_stage: true`.
- `output.directory` must be non-empty.
- In `retrieval.full_text_sources`, `coordinates_path_templates` and `processed_data_path` are mutually exclusive for a given source.

## Top-Level Sections

## `search`

Purpose: define the search source and the studies to start from.

Common fields:

- `database`
  Type: string
  Values: `pubmed`, `pmc`
  Default: `pubmed`
- `query`
  Type: string
  Required when not using PMIDs input
- `max_results`
  Type: integer
  Must be positive
- `date_from`, `date_to`
  Type: string, `YYYY/MM/DD`
- `email`
  Type: string
  Recommended for NCBI API usage
- `pmids_file`
  Type: path string
- `pmids_list`
  Type: list of PMID strings

Use `pmids_file` or `pmids_list` when you already know the study IDs and want to skip query-based discovery.

## `retrieval`

Purpose: control full-text fetching and external source mapping.

Common fields:

- `sources`
  Type: list of strings
  Example: `["pubget"]`
- `load_excluded`
  Type: boolean
  If `true`, retrieval also includes studies excluded at abstract screening
- `full_text_sources`
  Type: list of source mappings
  Used when you already have a local full-text corpus or custom layout

## `retrieval.full_text_sources`

Each source can map PMIDs to local text files using one of three modes:

- `pmid_source: "file_name"`
- `pmid_source: "folder_name"`
- `pmid_source: "json"`

Common fields:

- `root_path`: root directory for that source
- `text_path_templates`: preferred relative text-file paths for folder/json modes
- `allowed_extensions`: valid file extensions for `file_name` mode
- `json_filename`: metadata filename for `json` mode
- `json_pmid_key`: key holding the PMID in the JSON metadata
- `processed_data_path`: path to pubget-like processed coordinate/table CSVs
- `coordinates_path_templates`: direct coordinate-file templates when not using processed CSVs

Important interaction:

- Use either `processed_data_path` or `coordinates_path_templates` for a source, not both.

## `screening.abstract`

Purpose: define the first screening pass over abstracts.

Common fields:

- `model`
  Type: string
- `objective`
  Type: string
  Required unless `skip_stage: true`
- `inclusion_criteria`
  Type: list of strings
  Required when the stage is enabled
- `exclusion_criteria`
  Type: list of strings
- `confidence_reporting`
  Type: boolean
  Default: `false`
- `threshold`
  Type: float between `0.0` and `1.0`
  Only meaningful with `confidence_reporting: true`
- `additional_instructions`
  Type: string
- `skip_stage`
  Type: boolean

## `screening.fulltext`

Purpose: define the second screening pass over retrieved full text.

Current behavior:

- This stage does not currently inherit required fields from `screening.abstract`.
- If it is enabled, it must define its own `objective` and `inclusion_criteria`.
- If you do not want to run full-text screening, set `skip_stage: true`.

Fields are the same shape as `screening.abstract`.

## `parsing`

Purpose: control coordinate parsing.

Fields:

- `parse_coordinates`
  Type: boolean
- `coordinate_model`
  Type: string

If `parse_coordinates` is `false`, later coordinate-dependent outputs will be limited.

## `output`

Purpose: define output behavior.

Fields:

- `directory`
  Type: string
  Required by config validation
- `formats`
  Type: list of strings
- `nimads`
  Type: boolean
- `export_excluded_studies`
  Type: boolean

CLI note:

- The CLI may override `output.directory` at runtime.
- `autonima run config.yaml` defaults the runtime output path to a folder derived from the config filename.
- The YAML field should still be present because config validation requires it.

## `annotation`

Purpose: label parsed analyses after screening and parsing.

Common fields:

- `enabled`
  Type: boolean
- `model`
  Type: string
- `prompt_type`
  Type: string
  Values: `single_analysis`, `multi_analysis`
- `create_all_included_annotation`
  Type: boolean
- `create_all_from_search_annotation`
  Type: boolean
- `metadata_fields`
  Type: list of strings
- `inclusion_criteria`, `exclusion_criteria`
  Type: list of strings
- `annotations`
  Type: list of custom annotation definitions

Custom annotation item fields:

- `name`
- `description`
- `inclusion_criteria`
- `exclusion_criteria`

## Annotated Full Example

Start from:

- `examples/sample_config.yml`

That file is kept aligned with:

- `autonima create-sample-config`

Use it as the authoritative starting point rather than composing a config from scratch.
