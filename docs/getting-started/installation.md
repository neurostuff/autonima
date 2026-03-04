# Installation

## Base Install

```bash
git clone git@github.com:adelavega/autonima.git
cd autonima
pip install -e .
```

The base install gives you the package and CLI. For real screening runs, you will usually also want the LLM extra.

## Optional Extras

### LLM Support

Install this for screening and other OpenAI-compatible model calls:

```bash
pip install -e .[llm]
```

Environment variables:

- `OPENAI_API_KEY` for the default OpenAI-compatible path
- `OPENROUTER_API_KEY` when using an OpenRouter base URL in the LLM client path

### Meta-Analysis Support

Install this to use `autonima meta`:

```bash
pip install -e .[meta]
```

This adds the NiMARE dependency stack.

### Enhanced HTML Cleaning

Install this for `readabilipy`-based HTML extraction:

```bash
pip install -e .[readability]
```

Notes:

- `readabilipy` also requires Node.js on the system.
- If it is not installed, Autonima falls back to basic HTML cleaning.

### Documentation Tooling

Install this if you want to build the docs site locally:

```bash
pip install -e .[docs]
```

## Recommended Dev Install

If you are actively working on the package and docs:

```bash
pip install -e .[dev,llm,docs]
```

Add `meta` and `readability` only if you need those workflows locally.
