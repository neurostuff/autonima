# Meta-Analysis

Autonima can run coordinate-based meta-analysis on the NiMADS outputs produced by the pipeline.

## Install Prerequisites

```bash
pip install -e .[meta]
```

## Input Folder

`autonima meta` expects the folder that directly contains:

- `nimads_studyset.json`
- `nimads_annotation.json`

For standard pipeline runs, that is usually:

```text
<output-folder>/outputs
```

## Basic Usage

```bash
autonima meta projects/cue_reactivity/default/outputs
```

## Optional Parameters

- `--estimator`
  Choices: `ale`, `mkdadensity`, `kda`
- `--estimator-args`
  JSON string of estimator arguments
- `--corrector`
  Choices: `fdr`, `montecarlo`, `bonferroni`
- `--corrector-args`
  JSON string of corrector arguments
- `--include-ids`
  Newline-delimited study IDs or PMIDs to include

Example:

```bash
autonima meta projects/cue_reactivity/default/outputs \
  --estimator ale \
  --corrector montecarlo
```

## Output Layout

Meta-analysis results are written under:

```text
<nimads-input-folder>/meta_analysis_results/
```

This directory contains:

- one folder per boolean annotation column
- NiMARE reports and statistical images
- summary artifacts for each analysis set

## Common Failure Modes

- NiMARE is not installed
- the input folder does not contain both NiMADS JSON files
- JSON argument strings are invalid
- include-ID filtering removes all analyses for a target annotation
