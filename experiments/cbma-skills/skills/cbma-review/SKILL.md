---
name: cbma-review
description: Run a coordinate-based neuroimaging meta-analysis as a systematic review, from a PubMed query to NiMARE maps, with the agent itself doing the abstract screening, full-text screening, coordinate-table extraction and analysis selection. Use when the user asks to run, resume, pilot or check the status of a neuroimaging meta-analysis or systematic review (fMRI/PET activation coordinates, PRISMA, NiMADS, NiMARE), or points at a folder containing a review.yaml.
---

# CBMA review (orchestrator)

You are running a systematic review whose output is a coordinate-based
meta-analysis (CBMA). You do the judgment work yourself, through subagents. Small
bundled scripts do the bookkeeping. No LLM API keys are involved: every judgment is
made by you or by a subagent you start.

This skill is the entry point. It tells you the order of stages and the rules. Each
stage has its own skill with the detailed instructions:

| Stage | Skill | Who does it |
|---|---|---|
| 1. Search | `pubmed-search` | a script |
| 2. Abstract screening | `screen-studies` | subagents, in batches |
| 3. Full-text retrieval | `fulltext-sources` | a script (PMC, then any local folders) |
| 4. Full-text screening | `screen-studies` | subagents, in batches |
| 5. Coordinate extraction | `extract-coordinates` | subagents, one study per item |
| 6. Analysis selection | `select-analyses` | subagents, one study per item |
| 7. Meta-analysis | `cbma-nimare` | a script (NiMARE) |

`SKILLS` below means the directory that holds these skill folders (the parent of
this file's directory). `REVIEW` means the review folder, the one with
`review.yaml`.

## Non-negotiable rules

1. **The ledger is the only writer of decisions.** Subagents write their raw output
   to the `output` path named in their batch file, and nowhere else. You then run
   `ledger.py ingest`, which validates it and records it. Never edit anything under
   `REVIEW/decisions/` or `REVIEW/analyses/` by hand, and never write decisions
   yourself outside a batch.
2. **A failure is not a decision.** If a subagent errors, times out, returns
   malformed output, or `ingest` rejects an item, that item stays pending. Re-batch
   and retry it. Never mark it excluded to make progress.
3. **Unavailable is not excluded.** A study with no retrievable full text is reported
   as unavailable. It is never counted as a full-text exclusion.
4. **Criteria come from `review.yaml` only.** If screening shows a criterion is
   ambiguous, stop and propose the edit to the user. Do not reinterpret it silently.
   Once `review.yaml` or a stage skill is edited, the ledger re-opens the affected
   decisions automatically.
5. **Record who judged.** Pass `--agent "<harness>/<model id>"` to every `ingest`,
   for example `--agent "claude-code/claude-opus-5-5"`. Use your real harness and
   model identifiers.
6. **Keep state on disk, not in your context.** After any interruption or context
   compaction, run `ledger.py status REVIEW` and continue from what it reports.

## Setup

```bash
pip install pyyaml            # the only dependency until the meta-analysis stage
python SKILLS/cbma-review/scripts/ledger.py init REVIEW
```

If `REVIEW/review.yaml` does not exist, write it with the user first. The format is
in `references/review-spec.md`, and `references/example-review.yaml` is a starting
point. `init` rejects unknown keys, so a typo fails loudly. Show the user the
numbered criteria that `init` prints before spending any effort on screening.

## The loop for every judged stage

The same five steps run for abstract screening, full-text screening, extraction and
selection:

1. **Make batches.**
   `python SKILLS/cbma-review/scripts/ledger.py batches REVIEW --stage <stage> --size <n>`
   Only pending items are batched, so re-running this after an ingest picks up
   exactly the retries.
   Suggested sizes: abstract 25, fulltext 2, extraction 1, selection 1.
   Add `--limit N` for a pilot.
2. **Dispatch one subagent per batch file.** Run several at once if your harness
   allows it, but keep it to about 8 at a time. Give each subagent this prompt,
   with the paths filled in:

   > Read `SKILLS/<skill>/SKILL.md` and follow it exactly. Your batch file is
   > `<batch path>`. Process every item in it and write your output to the path in
   > the batch's `output` field. Do not read or write any other part of the review
   > folder except the files the batch names. When done, reply with only the number
   > of items you wrote.

   `<skill>` is the batch's `skill` field.

   - **No in-session subagents** (some Codex setups): run each batch in a fresh
     headless session instead, one at a time or a few in parallel. See
     `scripts/run_batches.sh`.
   - **Never process batches in your own context** beyond a pilot of one or two.
     Fresh context per batch is what keeps the thousandth decision as careful as
     the first.
3. **Ingest.**
   `python SKILLS/cbma-review/scripts/ledger.py ingest REVIEW --stage <stage> --agent "<harness>/<model>"`
   It prints accepted and rejected counts and every rejection reason. Exit code 2
   means at least one item was rejected.
4. **Retry the rejected items.** Go back to step 1. If the same item fails twice
   for the same reason, read the rejection and the item yourself. If the problem
   is the input (a broken table, a garbled text), tell the user and leave the item
   pending.
5. **Audit before moving on.** Run the checks in the next section.

## Stage by stage

1. **Search:** follow `pubmed-search`. When it finishes, run `ledger.py status` and
   report the record count. It must equal PubMed's reported count, and the script
   fails if it does not.
2. **Abstract screening:** run the loop with `--stage abstract`. Records without an
   abstract are screened on title and metadata; the `screen-studies` skill covers
   this.
3. **Full-text retrieval:** first run
   `python SKILLS/cbma-review/scripts/ledger.py needs-fulltext REVIEW`, then follow
   `fulltext-sources`. Report available, incomplete and unavailable counts by source.
4. **Full-text screening:** run the loop with `--stage fulltext`. Items whose text is
   unavailable are never batched.
5. **Extraction:** run the loop with `--stage extraction`, one study per batch.
6. **Selection:** run the loop with `--stage selection`. Skip it if `review.yaml`
   defines no targets.
7. **Export and meta-analysis:**
   `python SKILLS/cbma-review/scripts/ledger.py export REVIEW`, then follow
   `cbma-nimare`.

## Audit checklist (do this between stages; it is part of the method)

- **`ledger.py status REVIEW`: pending must be 0** before the next stage, unless
  the user agreed to leave items pending.
- **Abstract screening:** read 10 random excludes and 10 random includes from
  `decisions/abstract.jsonl`. If you disagree with more than one, show the user
  the cases before continuing.
- **Full-text screening:** check `decisions_with_ungrounded_evidence`, which counts
  evidence quotes the ledger could not find in the text. A high count means
  subagents are paraphrasing instead of quoting. Re-batch those studies.
- **Extraction:**
  - `points_unverified` should be near 0. Unverified points are dropped at export.
  - Open two extracted studies and compare them with their tables by eye.
- **Selection:** confirm each target has enough studies for a CBMA (usually at least
  17 for ALE and 10 as an absolute floor). Say so if a target falls short.

## Finishing

Report the final `results/prisma.json` counts to the user. Name every stage where
items remain pending or unavailable, and list the files under `results/`. For a
benchmark run, also follow `SKILLS/../benchmark/README.md` if it exists.

## Layout of REVIEW (for orientation; the scripts create it)

```
review.yaml                 the protocol (criteria, sources, targets)
search/records.jsonl        one record per PMID; search/search_log.json
fulltext/index.jsonl        retrieval outcome per PMID; fulltext/raw/ cached downloads
docs/<pmid>/                normalized text.md, tables/<id>.json, meta.json
work/<stage>/batch_*.json   batches waiting for subagents (outputs land beside them)
decisions/<stage>.jsonl     accepted decisions (append-only; the ledger writes these)
analyses/<pmid>.json        accepted, verified coordinate extractions
results/                    prisma.json, criteria.json, nimads/, meta/
```
