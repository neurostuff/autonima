# Autonima

Autonima is a usage-first toolkit for automated neuroimaging review workflows: search PubMed, screen studies with LLMs, retrieve full text, parse coordinates, and generate outputs for downstream meta-analysis.

The documentation here is focused on how to run the CLI and how to author a working `config.yml`. It is intentionally not a full Python API reference.

## Who This Is For

- Researchers running systematic-review style neuroimaging pipelines from the command line
- Users preparing reproducible `config.yml` files for search, screening, retrieval, parsing, and annotation
- Maintainers who need a stable reference for the CLI surface and the expected output layout

## Workflow

1. Install `autonima` and any optional extras you need.
2. Generate a starting config with `autonima create-sample-config`.
3. Edit the config for your search query, screening criteria, and retrieval setup.
4. Validate it with `autonima validate config.yaml`.
5. Run the pipeline with `autonima run config.yaml`.
6. Optionally run `autonima meta OUTPUT_FOLDER` on the generated NiMADS outputs.

## Start Here

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Configuration Guide](guides/configuration.md)
- [CLI Usage Guide](guides/cli.md)
- [Outputs Guide](guides/outputs.md)
- [Interpreting Outputs](guides/interpreting-outputs.md)
- [CLI Reference](reference/cli.md)

## Important Notes

- `run`, `run-search`, `run-abstract`, and `validate` use positional arguments:
  `autonima <command> CONFIG [OUTPUT_FOLDER]`.
- If you omit `OUTPUT_FOLDER`, the CLI defaults to a sibling directory derived from the config file stem.
  Example: `projects/cue_reactivity/default.yaml` becomes `projects/cue_reactivity/default/`.
- The config file still needs a non-empty `output.directory` field because configuration validation currently requires it.
