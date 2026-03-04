# Documentation Maintenance

## Local Build

Install the docs tooling:

```bash
pip install -e .[docs]
```

Build the site locally:

```bash
mkdocs build --strict
```

Serve it locally during editing:

```bash
mkdocs serve
```

## What the Docs Depend On

- `mkdocs`
- `mkdocs-material`
- `mkdocs-click`
- `pymdown-extensions`

## Source of Truth Rules

- `autonima/templates/sample_config.yml` is the canonical sample config.
- `autonima create-sample-config` must emit that exact config text.
- `examples/sample_config.yml` should stay in sync with the canonical template.
- CLI docs should rely on the generated Click reference instead of manually restating every option.

## GitHub Pages Publishing

Docs are built and deployed from GitHub Actions.

Repository setting expected:

- GitHub Pages source: `GitHub Actions`

Workflow behavior:

- pull requests to `master` build docs only
- pushes to `master` build and deploy the site

## Before Merging Docs Changes

Run:

```bash
mkdocs build --strict
pytest tests/test_cli.py tests/test_docs.py -q
```

Check:

- README examples still match the CLI
- links resolve
- the sample config still validates
