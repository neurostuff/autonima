# CLI Usage

Autonima exposes four main commands:

- `autonima create-sample-config`
- `autonima validate CONFIG [OUTPUT_FOLDER]`
- `autonima run CONFIG [OUTPUT_FOLDER]`
- `autonima meta OUTPUT_FOLDER`

See the [CLI reference](../reference/cli.md) for generated argument and option details.

## Create a Starting Config

```bash
autonima create-sample-config > config.yaml
```

This writes the canonical sample YAML to stdout, so shell redirection produces an editable config file directly.

## Validate a Config

```bash
autonima validate config.yaml
```

With an explicit runtime output folder:

```bash
autonima validate config.yaml runs/review_a
```

What it does:

- parses and validates the YAML
- prints a short summary of key config values
- resolves the runtime output folder exactly as `run` would

## Run the Pipeline

```bash
autonima run config.yaml
```

With an explicit runtime output folder:

```bash
autonima run config.yaml runs/review_a
```

Useful options:

- `--dry-run` to validate without running
- `-v` / `--verbose` for more logging
- `--debug` for post-mortem debugging on errors
- `-j` / `--num-workers` to control parallel screening workers
- `--force-reextract-incomplete-fulltext` to re-run full-text screening for cached `fulltext_incomplete` studies using current files

## Omitted Output Folder

If you omit `OUTPUT_FOLDER`, Autonima derives it from the config path:

```bash
autonima run projects/cue_reactivity/default.yaml
```

Runtime output folder:

```text
projects/cue_reactivity/default/
```

## Run Meta-Analysis

`autonima meta` expects the folder containing `nimads_studyset.json` and `nimads_annotation.json`.

For standard pipeline output, that is usually the `outputs/` directory:

```bash
autonima meta projects/cue_reactivity/default/outputs
```

Optional parameters let you change the estimator, corrector, and include-ID filtering.

## Common Failure Modes

## Invalid Config

Typical causes:

- missing `objective` or `inclusion_criteria` in an enabled screening stage
- empty `output.directory`
- empty `search.query` without `pmids_file` or `pmids_list`

Fix by running:

```bash
autonima validate config.yaml
```

## Missing API Keys

LLM-backed workflows require API credentials:

- `OPENAI_API_KEY`
- or `OPENROUTER_API_KEY` when using an OpenRouter path in the OpenAI-compatible client

## Missing Meta Dependencies

If `autonima meta` fails with an import error, install:

```bash
pip install -e .[meta]
```

## Missing Readability Support

Enhanced HTML cleaning needs:

```bash
pip install -e .[readability]
```

and a working Node.js installation.
