---
name: cbma-nimare
description: Export a review's verified, selected coordinates as a NiMADS studyset and annotation, and run coordinate-based meta-analyses (MKDA density, ALE, KDA) with NiMARE, one per target. Use when a cbma-review has finished analysis selection, or when asked to meta-analyse a NiMADS studyset.
---

# CBMA with NiMARE

1. **Export** (the ledger refuses while any study is pending):
   ```bash
   python SKILLS/cbma-review/scripts/ledger.py export REVIEW
   ```
   This writes `REVIEW/results/nimads/studyset.json` and `annotation.json`.
   - **What is exported:** only full-text-included studies, and only points
     verified against a row of their table.
   - **Annotation:** one boolean column per target.
   - **Missing decisions:** an analysis without a selection decision is left out
     entirely, never exported as `false`.
   - **The export report:** check `dropped`. Unverified points and analyses that
     are left with no points are listed there, so you can see what the checks
     removed.
   - **Pending studies:** `--allow-pending` exports only the finished studies.
     Use it only for pilots, and say so in the results.

2. **Run the meta-analyses** (NiMARE is the only heavy dependency):
   ```bash
   pip install nimare
   python SKILLS/cbma-nimare/scripts/run_meta.py REVIEW
   ```
   - **Settings:** the estimator and corrector come from `review.yaml` `meta:`
     (defaults: `mkdadensity`, `fdr`). Override them with `--estimator` and
     `--corrector`.
   - **Small targets:** targets with fewer than `--min-studies` studies (default
     10) are skipped, and the reason is recorded.
   - **Outputs:** maps go to `REVIEW/results/meta/<target>/`, with a
     `summary.json` per target and one overall.

3. **Report per target:** the study and analysis counts, the estimator and
   corrector, and where the maps are. Mention any target that was skipped for too
   few studies.

## Checks before interpreting results

- **Mixed spaces:** if a target mixes MNI and Talairach, confirm NiMARE's space
  handling suits the analysis. NiMARE transforms Talairach to MNI when it builds the
  dataset.
- **Repeated samples:** a study contributing many analyses to one target
  over-weights it in some estimators. Report the maximum number of analyses per
  study for each target.
- **Uninterpretable results:** if a map is empty or implausible, look at the
  analyses behind it before trusting or dismissing it.
