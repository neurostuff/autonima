# Autonima

Autonima is an LLM-assisted CLI for neuroimaging review workflows: search PubMed, screen studies, retrieve full text, parse coordinates, and export NiMADS artifacts for downstream meta-analysis.

Full documentation: https://adelavega.github.io/autonima/

## Install

Base install:

```bash
git clone git@github.com:adelavega/autonima.git
cd autonima
pip install -e .
```

Useful extras:

```bash
pip install -e .[llm]          # screening and other LLM-backed workflows
pip install -e .[meta]         # `autonima meta`
pip install -e .[readability]  # enhanced HTML extraction
pip install -e .[docs]         # local docs build
```

## Quickstart

Generate a starting config:

```bash
autonima create-sample-config > config.yaml
```

Validate it:

```bash
autonima validate config.yaml
```

Run the pipeline:

```bash
autonima run config.yaml
```

If you omit `OUTPUT_FOLDER`, the CLI derives it from the config filename stem. For example:

- config: `projects/cue_reactivity/default.yaml`
- default output folder: `projects/cue_reactivity/default/`

You can still pass an explicit runtime output folder:

```bash
autonima run config.yaml runs/my_review
```

Run meta-analysis on the generated NiMADS outputs:

```bash
autonima meta runs/my_review/outputs
```

## Minimal Config Example

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

For the full sample config and field-by-field guidance, see:

- https://adelavega.github.io/autonima/getting-started/quickstart/
- https://adelavega.github.io/autonima/guides/configuration/
- https://adelavega.github.io/autonima/guides/cli/
