# review.yaml reference

`ledger.py init` validates this file. **Unknown keys are errors** at every level.

```yaml
name: wm_schizophrenia          # short identifier, used in output names
objective: >                    # REQUIRED. What the meta-analysis is for; shown to every screener.
  Identify task-based fMRI studies of working memory in adults with schizophrenia
  that report whole-brain activation coordinates for patients versus controls.

search:
  query: '(schizophrenia[tiab]) AND ("working memory"[tiab]) AND (fMRI[tiab] OR "functional magnetic resonance"[tiab])'
  date_from: 2000/01/01         # optional, publication date, YYYY/MM/DD
  date_to: 2025/12/31           # optional
  pmids: ["12345678"]           # optional; unioned with the query results
  pmids_file: seeds.txt         # optional; one PMID per line, relative to the review folder
  email: you@example.org        # recommended; NCBI asks tools to identify a contact

screening:
  abstract:                     # permissive: exclude only when the abstract clearly fails
    inclusion:                  # REQUIRED, becomes I1, I2, ...
      - Human participants
      - Functional neuroimaging (fMRI or PET) of brain activation
    exclusion:                  # optional, becomes E1, E2, ...
      - Review, meta-analysis, commentary or protocol without new data
    instructions: >             # optional, free text appended to the screening skill
      Case reports are excluded under E1.
  fulltext:                     # strict: include only when every inclusion criterion is met
    inclusion:
      - Adults with a DSM or ICD diagnosis of schizophrenia
      - A working-memory task during fMRI
      - Whole-brain results reported as stereotaxic coordinates (MNI or Talairach)
    exclusion:
      - ROI-only analyses without whole-brain coordinates
      - Sample overlaps with a larger included study from the same group (flag, do not guess)
    instructions: ""

fulltext:
  sources:                      # tried in order; the first complete document wins
    - type: pmc
    - type: local
      name: publisher_html
      path: /data/publisher_html
      pattern: "**/*.html"
      id_from: filename         # filename | parent_dir | regex | sidecar
      id_kind: pmid             # pmid | doi | pmcid
      format: auto              # auto | html | jats | text

extraction:
  drop_unverified: true         # default true: export only points found in their table row
  instructions: ""              # optional, appended to the extraction skill

selection:                      # analysis-level selection for each meta-analytic target
  global:                       # apply to every target; become GI1.., GE1..
    inclusion:
      - Whole-brain analysis (not restricted to a region of interest)
    exclusion:
      - Conjunction analyses
  targets:
    - name: wm_patients_vs_controls      # identifier: letters, digits, underscore
      description: Working-memory load effects contrasting patients with controls
      inclusion:                         # become I1.., E1.. within this target
        - Between-group contrast of patients and controls
        - The contrast isolates working-memory load or task versus baseline
      exclusion:
        - Resting-state or connectivity maps
  instructions: ""

meta:
  estimator: mkdadensity        # mkdadensity | ale | kda
  corrector: fdr                # fdr | montecarlo | bonferroni
  estimator_args: {}
  corrector_args: {}
```

## How changes propagate

Every decision stores two hashes, and it stays valid only while both still match:
- **Criteria hash:** the stage's objective, criteria, instructions and the text of
  the stage's skill file.
- **Input hash:** the title and abstract, or the normalized full text and tables,
  or the accepted analyses.

| You change | Re-opened |
|---|---|
| an abstract criterion | abstract decisions (full-text IDs are unaffected: IDs are per stage) |
| a full-text criterion | full-text decisions |
| a full-text source, or re-gathering yields a different text | that study's full-text decision and extraction |
| a target's criteria | selection decisions |
| a skill file under `SKILLS/` | every decision of the stages that skill instructs |

Nothing else re-opens a decision. Re-running a stage after a change batches only
the re-opened items.
