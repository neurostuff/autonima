# cbma-skills

An experiment. Can a general-purpose coding agent (Claude Code, Codex) run a
coordinate-based neuroimaging meta-analysis end to end, with only a set of skills
and no specialized LLM pipeline?

The package holds no LLM client and needs no API key. The agent does every
judgment itself, through subagents:
- abstract screening
- full-text screening
- splitting coordinate tables into analyses
- selecting analyses for each target

A few small standard-library scripts do the bookkeeping that an agent should not do
from memory:
- searching PubMed completely
- normalizing full texts
- validating and recording decisions
- checking extracted coordinates against their tables
- exporting NiMADS

This package does not import or wrap autonima. It lives in the autonima repo for
now so it can be benchmarked against autonima's results, and it is meant to move to
its own repository.

See [PLAN.md](PLAN.md) for the hypothesis, the benchmark protocol and the MCP
decision.

## Skills

| Skill | Role |
|---|---|
| [`cbma-review`](skills/cbma-review/SKILL.md) | orchestrator: stage order, batching protocol, audit checklist; holds the ledger |
| [`pubmed-search`](skills/pubmed-search/SKILL.md) | complete, de-duplicated PubMed search via E-utilities |
| [`fulltext-sources`](skills/fulltext-sources/SKILL.md) | PMC open access plus any local folders of pre-downloaded articles |
| [`screen-studies`](skills/screen-studies/SKILL.md) | abstract and full-text screening against numbered criteria |
| [`extract-coordinates`](skills/extract-coordinates/SKILL.md) | table → analyses → points, using autonima's analysis-boundary rules |
| [`select-analyses`](skills/select-analyses/SKILL.md) | analysis × target inclusion decisions |
| [`cbma-nimare`](skills/cbma-nimare/SKILL.md) | NiMADS export and NiMARE meta-analysis |

## Install

The skills are plain folders, and they are installed as a set because the
orchestrator refers to the others by relative path.

```bash
# Claude Code: project-level (inside the folder where you will run reviews) or user-level
./install.sh /path/to/reviews-workspace          # links into .claude/skills and .agents/skills
# or by hand:
mkdir -p ~/.claude/skills && ln -s "$PWD"/skills/* ~/.claude/skills/
```

- **Codex:** it reads skills from its own skills directory. `install.sh` also links
  them into `.agents/skills`, but check the Codex version you use for the current
  location.
- **Dependencies:** `pip install pyyaml` for every stage, plus `pip install nimare`
  for the meta-analysis stage only.
- **Network:** outbound access to `eutils.ncbi.nlm.nih.gov` for search and PMC.

## Use

In the agent, from a workspace containing a review folder:

> Run the cbma-review skill on ./wm_schizophrenia. Pilot abstract screening on 50 records first.

The orchestrator walks the stages. The loop at every judged stage is:

```bash
python SKILLS/cbma-review/scripts/ledger.py batches REVIEW --stage abstract --size 25
#   one subagent per batch file, each following the stage skill
python SKILLS/cbma-review/scripts/ledger.py ingest  REVIEW --stage abstract --agent "claude-code/<model>"
python SKILLS/cbma-review/scripts/ledger.py status  REVIEW
```

If the harness can't spawn subagents, or to run a large stage headless,
`skills/cbma-review/scripts/run_batches.sh` runs each batch in a fresh
`claude -p` or `codex exec` session.

## What the scripts enforce

These are the lessons from autonima's issue tracker and code review, made
mechanical:

| Rule | Autonima failure it prevents |
|---|---|
| esearch windows split past 9,999; retrieved must equal reported | silent truncation at 10k (#53) |
| every abstract section, full titles, the article's own DOI, string PMIDs | truncated abstracts and titles, wrong DOIs, integer PMIDs giving 0 studies (#56) |
| failures are never decisions; rejected output stays pending and is retried | failed screenings cached forever |
| unavailable ≠ excluded, reported separately in PRISMA | retrieval failures counted as exclusions |
| a missing selection decision is never exported as `false` | missing annotations exported as "exclude" |
| every extracted xyz must appear in its table row, or the point is dropped | sign errors and invented foci reaching the meta-analysis |
| duplicate tables flagged at normalization | double-weighted studies (#60) |
| reference lists removed from full text | bibliography sent to the model (#67) |
| unknown `review.yaml` keys are errors | typos silently ignored (#58) |
| criteria IDs numbered per stage; decisions keyed on hashes of criteria, skill text and input | renumbered IDs reusing stale decisions; hand-bumped prompt versions |
| full-text evidence quotes checked against the text | ungrounded reasons (#38) |

## Tests

```bash
pip install pytest pyyaml
pytest experiments/cbma-skills/tests
```

The tests run offline against fixtures. They cover PubMed parsing, date-window
splitting, retries, JATS and HTML normalization, point verification, and one review
through every stage, with fake subagent output.
