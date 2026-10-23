# Installation

Autonima needs Python 3.10 or newer (tested on 3.10 through 3.13).

## Full Install

```bash
git clone https://github.com/neurostuff/autonima.git
cd autonima
pip install -e ".[llm,meta,ui]"
```

This covers screening, parsing and annotation, `autonima meta` and `autonima ui`. Keep the quotes around the extras: unquoted brackets fail in zsh.

Before a real run, set `OPENAI_API_KEY` in your environment and put a contact address in the config's `search.email`, which NCBI asks for.

## Base Install

```bash
pip install -e .
```

The base install gives you the package and CLI, with no LLM client: it can run `autonima run-search` but not screening.

## Optional Extras

### LLM Support

Install this for screening and other OpenAI-compatible model calls:

```bash
pip install -e ".[llm]"
```

Environment variables:

- `OPENAI_API_KEY` for authentication
- `OPENAI_API_GATEWAY` (optional) to override the OpenAI SDK gateway/base URL

### Meta-Analysis Support

Install this to use `autonima meta`:

```bash
pip install -e ".[meta]"
```

This adds the NiMARE dependency stack. `autonima meta` needs NiMARE 0.11.0 or later.

### Enhanced HTML Cleaning (Optional)

Install this for `readabilipy`-based HTML extraction:

```bash
pip install -e ".[readability]"
```

Notes:

- `readabilipy` also requires Node.js on the system.
- If it is not installed, Autonima falls back to basic HTML cleaning.

### Documentation Tooling

Install this if you want to build the docs site locally:

```bash
pip install -e ".[docs]"
```

## Recommended Dev Install

If you are actively working on the package and docs:

```bash
pip install -e ".[dev,llm,docs]"
```

Add `meta` and `readability` only if you need those workflows locally.
