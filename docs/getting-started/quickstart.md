# Quickstart

## 1. Generate a Sample Config

```bash
autonima create-sample-config > config.yaml
```

The generated file is the canonical sample config used by the docs and repository example.
It follows the standard path:

- **Search** NeuroStore's index of neuroimaging studies with your PubMed-style query.
- **Screen abstracts** as usual.
- **Review each remaining study from NeuroStore's record of it**: its analyses with the
  features NeuroStore extracted (contrast, task, whole brain or region of interest,
  participants, design). One call per study screens it and selects its analyses for your
  annotations. No article is read and no table is parsed; the coordinates are NeuroStore's.

Only studies NeuroStore has extracted features for are reviewed. The rest are listed in
`outputs/documents_unavailable.csv` with the reason. To review those from their articles,
add `fallback: fulltext` under `retrieval.records`; see
[NeuroStore records](../guides/records.md#neurostore-records).

To read every article in full and parse its tables instead, start from
`autonima create-sample-config --fulltext`.

## 2. Edit the Config

At minimum, update:

- `search.query`
- `screening.abstract.objective`
- `screening.abstract.inclusion_criteria`
- `screening.fulltext.objective`
- `screening.fulltext.inclusion_criteria`
- `annotation.annotations`: the meta-analyses you want, each with the criteria an analysis
  must meet to be selected for it

If you want to skip a screening stage, set `skip_stage: true` for that stage instead of leaving required fields empty.

## 3. Validate Before Running

```bash
autonima validate config.yaml
```

You can also validate with an explicit output folder:

```bash
autonima validate config.yaml runs/my_review
```

## 4. Run the Pipeline

```bash
autonima run config.yaml
```

Or with an explicit output folder:

```bash
autonima run config.yaml runs/my_review
```

For stage-limited pilot runs:

```bash
autonima run-search config.yaml
autonima run-abstract config.yaml
```

- `run-search` executes only literature search.
- `run-abstract` executes search plus abstract screening, then stops before retrieval.
- `run-abstract` reruns upstream stages for each invocation (no cache-only abstract mode).

## Default Output Folder

If you omit `OUTPUT_FOLDER`, the CLI derives it from the config filename:

- Input config: `projects/cue_reactivity/default.yaml`
- Default output folder: `projects/cue_reactivity/default/`

This is separate from the `output.directory` value stored in the YAML. The CLI will override that runtime path.

## 5. Run Meta-Analysis

`autonima meta` takes either the run's output folder or its `outputs/` directory, which holds `nimads_studyset.json` and `nimads_annotation.json`:

```bash
autonima meta projects/cue_reactivity/default/outputs
```

Install the meta extra first:

```bash
pip install -e ".[meta]"
```

## Next Steps

- [Configuration Guide](../guides/configuration.md)
- [CLI Usage Guide](../guides/cli.md)
- [Outputs Guide](../guides/outputs.md)
- [Interpreting Outputs](../guides/interpreting-outputs.md)
