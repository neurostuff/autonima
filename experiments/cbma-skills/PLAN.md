# Experiment plan: skills-only CBMA versus autonima

## Hypothesis

A general-purpose coding agent with a set of skills can reproduce autonima's
workflow with accuracy that is not practically worse. It needs no specialized LLM
pipeline and no API keys. The workflow is PubMed search, screening, coordinate
extraction, analysis selection and meta-analysis.

"Specialized" means autonima's machinery:
- its prompt construction, structured-output clients and retry logic;
- its cache and signature system;
- its parallel executor.

The skills-only arm may use small generic helpers. Examples are an E-utilities
client, an HTML-to-text normalizer, a JSONL validator, and NiMARE. What it may not
use is a bespoke LLM harness. Every judgment is made by the agent.

**Outcomes that would falsify the hypothesis:**
- final-inclusion recall falls well below autonima's on the same benchmark;
- coordinate recall falls well below autonima's;
- running a benchmark-sized review is infeasible within normal subscription usage
  limits.

## Scope and phases

1. **Phase 1 (this package):** PMC open access as the full-text source; all four
   judged stages done by agents; NiMADS export; NiMARE.
2. **Phase 1b (built; exercised later):** generic local sources. Any folder of
   pre-downloaded articles (publisher HTML from Elsevier or Springer downloads,
   ACE-scraped HTML, JATS XML, plain text) becomes a full-text source.
   - It is declared in `review.yaml` with a path, a glob, and a rule for reading
     the PMID, DOI or PMCID from the filename, folder, a regex or a sidecar JSON.
   - Local files only fill in text for studies the search found, and those studies
     then go through full-text screening like any other.
   - Sources are tried in the configured order.
   - Implemented in `fulltext-sources/scripts/gather_fulltext.py` and tested with a
     local HTML folder.
3. **Phase 2 (decide after the pilot):** whether an MCP server would buy anything.
   See the last section.

## Arms

| Arm | System | Judging model |
|---|---|---|
| A | autonima v0.1.0, results already in autonima-results | as in the manuscript |
| B | cbma-skills in Claude Code | the session's model, recorded per decision |
| C | cbma-skills in Codex | the session's model, recorded per decision |

**Controls:**
- **Same skills:** arms B and C use the same commit of `experiments/cbma-skills`.
  Record the commit SHA in the run notes.
- **Same inputs:** each arm gets the same `review.yaml` criteria, translated
  one-for-one from the autonima config of that project, and the same PubMed query
  and date limits.
- **Blinding:** the agent never sees the gold standard. Keep the gold files outside
  the workspace the agent can read.

## Benchmarks

1. **autonima-results benchmark:** the projects used in the manuscript.
   - For each, translate the autonima config into `review.yaml`: objective,
     abstract and full-text criteria, and annotations as selection targets.
   - Gold: the manuscript's human-labelled includes, plus the analysis-level labels
     where available.
2. **NeuroMetaBench:** the published meta-analyses and their included studies.
   - Gold: each meta-analysis's included-study list, plus coordinates where the
     benchmark provides them.
   - Write a small adapter per benchmark to `benchmark/compare.py`'s CSV format.
     Keep it next to the benchmark data so the mapping can be reviewed.
3. **One difficult case, run first.** Pick the project where autonima did worst
   or where the criteria are hardest. The social-cognition annotation case
   (autonima#39) is the obvious candidate: analysis-level selection with fine
   contrast distinctions. Starting there tells us early whether it is worth
   running the rest.

## Protocol per project

1. **Setup:** write `review.yaml`, run `ledger.py init`, and check the numbered
   criteria against autonima's.
2. **Search:** run the search and check the count against the manuscript's. If it
   differs, record why (the date of the search, and any PubMed indexing changes
   since).
3. **Screening pilot:** screen the first 50 records only (`--limit 50`) and read
   every decision.
   - Fix unclear wording in `review.yaml` now, before the full run.
   - Record every change to `review.yaml` in the run notes. Changes after the pilot
     are protocol deviations and must be reported.
4. **Full run:** run all stages, following the orchestrator's audit checklist
   between stages.
5. **Scoring:** run
   `python benchmark/compare.py REVIEW --gold gold.csv --autonima AUTONIMA_RUN --gold-nimads gold_studyset.json --out report.json`.
6. **Repeatability:** re-run abstract screening and extraction on a fixed random
   subset in a second fresh session. Use about 200 abstracts and about 20 studies.
   - The agreement between the two sessions is the run-to-run variance.
   - Without it, a difference between arms cannot be interpreted.

## Metrics

| Stage | Metric |
|---|---|
| Search | recall of gold includes among retrieved records |
| Abstract | recall (primary: no eligible study lost), precision, workload saved |
| Full-text retrieval | coverage by source; gold includes without text |
| Full-text screening | recall, precision and F1 of final includes, over all gold and over gold with retrievable text |
| Extraction | point recall and precision against gold coordinates (±2 mm), analyses per study, share of points unverified |
| Selection | analysis-level agreement with gold labels where available |
| Meta-analysis | per target: spatial correlation and Dice overlap of thresholded maps with the gold-standard maps |
| Agreement with autonima | Cohen's kappa per stage on studies both systems judged |
| Cost | wall-clock time, subagent count, usage-limit interruptions, human interventions |
| Repeatability | session-to-session agreement on the fixed subset |

Report each with its denominator. The unavailable and incomplete counts are always
reported, never folded into the exclusions.

## Things to record in every run

- The harness and version, and the model ID. The ledger stores `--agent` on every
  decision.
- The cbma-skills commit SHA.
- The date of the PubMed search.
- The batch sizes used per stage, and any batches re-run by hand.
- Each `review.yaml` change, with its reason.

## Known risks to watch

- **Usage limits.** A 5,000-abstract review is about 200 abstract batches at 25
  each. Track where limits hit. This is a real feasibility outcome, not noise.
- **Drift in long sessions.** Mitigated by a fresh subagent per batch. If the
  orchestrator starts processing batches itself after context compaction, treat
  that as a protocol failure.
- **HTML tables.** Local publisher HTML can yield broken grids, and the coordinate
  check then drops good points. Watch `points_unverified` per source.
- **PMC coverage.** PMC open access alone will miss many included studies, so
  report recall given retrievable text separately. Phase 1b, with local sources,
  is how that gap closes.

## Would an MCP server buy anything? (decide after the pilot)

Build one only if the pilot shows a problem it would fix. Concretely:

| Observation in the pilot | What MCP would add |
|---|---|
| Subagents write outside their output files, or edit the ledger | a hard write boundary: tools are the only way to record a decision |
| The agent skips `ingest` or the audit steps, or invents hashes | server-side sequencing and validation that cannot be bypassed |
| Clients without a shell (chat apps) need to run reviews | tools usable without Bash or file access |
| The same review is run from Claude and from Codex, or by several people | one shared ledger service instead of files on one machine |
| Credentials for local publisher sources must stay out of the agent's view | secrets held by the server |

If none of these show up, the skills-plus-scripts design is enough. The ledger
already validates everything that matters; it just relies on the agent calling it.
Also note: moving the ledger into an MCP server is a thin wrapper, because
`ledger.py` already exposes these operations: init, batches, ingest, status,
needs-fulltext and export.
