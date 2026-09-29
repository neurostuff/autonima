---
name: extract-coordinates
description: Extract stereotaxic activation coordinates from the tables of a neuroimaging paper and group them into the distinct statistical analyses (contrasts, groups, directions, sessions) the tables report. Use when given a cbma-review batch file whose stage is "extraction", or when asked to pull MNI/Talairach peak coordinates out of fMRI/PET result tables.
---

# Extract coordinates

You are a neuroimaging data curator. For each study in the batch, read its tables.
Extract every valid stereotaxic peak, group the peaks into the separate statistical
analyses (maps) they come from, and write one JSON file per study.

## Inputs

The batch's `items` each contain:
- `pmid` and `text_file`: the full text, for context.
- `tables`: one entry per table, with `table_id`, `file`, `label`, `caption`,
  `coordinate_candidate` and `duplicate_of`.
- `must_report`: table IDs you must give a status for. These are the coordinate
  candidates that are not duplicates.

Each table `file` is JSON:
- `grid`: rows of cell strings. Merged cells are already expanded, so column k
  means the same column in every row.
- `tsv`: the same grid as tab-separated text.
- `caption` and `footer`.

Read `caption` and `footer` too; they often define the contrast, the sign
convention and the space.

Also check the text for tables that the candidate flag missed. Look for a table
whose caption mentions peaks or coordinates but whose header was unusual. You may
report extra tables; you must report every table in `must_report`.

Skip tables with a `duplicate_of` value. They repeat an earlier table.

## Per-table status

- **`parsed`:** the table has coordinates; give the analyses.
- **`no_coordinates`:** it looked like a candidate but holds no stereotaxic
  coordinates (for example demographics, or behavioural results).
- **`not_applicable`:** it has coordinates, but not activation results from this
  study. Examples: an ROI definition table, coordinates quoted from other papers,
  or a meta-analysis of other studies.
- **`failed`:** you cannot read it reliably, for example because columns are
  scrambled beyond recovery. Add a `note`. The ledger treats this as retry-later,
  not as empty.

## Analysis boundaries

Work in two passes. First, list the table's analysis-defining axes and result
blocks without extracting any points. Then go through the table and assign every
coordinate to the right combination of those axes.

**Start a separate analysis when an explicit label changes any of:**
- **Contrast or comparison,** including its direction (A > B versus A < B).
- **Sign of the effect:** activation versus deactivation, positive versus negative,
  increase versus decrease.
- **Participant group or cohort,** including "all groups" and each separately
  reported subgroup.
- **Study variable:** treatment, task condition, session, time point, or parametric
  effect.
- **A contrast label in a column or a multi-level header.**

**Tables can encode analyses in columns as well as rows.**
- **Label columns:** columns named Contrast, Comparison, Group, Condition,
  Treatment, Session or Effect define analyses. When several such axes are present,
  combine them explicitly in the analysis name.
- **Wide tables:** repeated statistic/X/Y/Z column groups under different top-level
  headers are separate analyses. One row can therefore contribute one peak to
  several analyses. Never pool coordinates from different header blocks, and never
  let blank cells in one block shift values into the next.
- **Sign-coded direction:** positive and negative statistics can encode opposite
  directions in one table. When the caption, header or footnote defines what the
  sign means, create the two directional analyses and route each peak by its sign.
- **Inherited labels:** a blank label cell inherits the last applicable label.
  Continuation rows and sub-peaks (local maxima) belong to the analysis above them.
- **Section-header rows:** a row that repeats one non-numeric label across most
  columns is context, not data. If it names a group, condition, direction, session
  or time point, it starts that analysis.
- **Groups and subgroups:** when "all groups" and the named subgroups each report
  coordinates, give each block its own analysis. Do not pool subgroups into "all".

**Do not split on descriptive anatomy.**
- Region, lobe, cluster, hemisphere (left/right), local maximum, and a-priori or
  "predicted" region labels subdivide one map. They are not analyses.
- Split on one of these only when the table explicitly calls it a separate contrast
  or map.

**ROIs:**
- A region described as an ROI stays in its current analysis.
- Separately labelled "ROI analysis" and "whole-brain analysis" blocks are different
  analyses, even for the same contrast.
- An ROI block starts at its header and does not reach back over earlier rows.

**No labels:** with no analysis-defining label, the whole table is one analysis.
Use only labels that appear in the table, caption or footer. Combine them to make
analyses distinguishable, and never invent a contrast or a group.

## Coordinates

- **One point per row.** Each valid coordinate row gives exactly one point.
  Check for dropped continuation rows before you finish.
- **Coordinates come only from X, Y and Z columns,** or an explicit equivalent such
  as "MNI coordinates (x, y, z)". Never take a coordinate from Cluster size,
  volume, BA, a T, Z or F statistic, a p-value, ALE, or any other number.
- **A column headed Z may be the statistic, not the axis.** Decide from the grouped
  header and its X and Y neighbours.
- **A point needs three numbers in x, y, z order.** Skip incomplete rows.
- **Copy numbers exactly as printed:** keep signs and decimals. Unicode minus signs
  (−) are negative. Do not round, convert or "fix" any number.
- **Several values in one cell:** when the X, Y and Z cells each hold the same
  number of clearly separated values, pair them by position. If the pairing is
  ambiguous, skip those values.
- **Values over 200 mm in magnitude** are malformed extraction, not coordinates.
- **The ledger checks your numbers.** Every x, y, z you report must appear in one
  row of that table. Points it cannot find are dropped from the meta-analysis, so
  copy, don't compute.

## Space

- **`"MNI"`:** the table, caption, footer or Methods say MNI (including "Montreal
  Neurological Institute", SPM templates stated as MNI, or ICBM152).
- **`"TAL"`:** they say Talairach.
- **`null`:** they say neither. Unlike autonima's parser, you may use the Methods
  text for space, because you have the full paper. Put the sentence that settles
  it in the table's `note`.
- **Never infer space from coordinate magnitudes.**

## Statistics

For each point, give the same-row statistics, each with a `kind` from this list:

| `kind` | Use for |
|---|---|
| `z-statistic` | Z scores |
| `t-statistic` | T values |
| `f-statistic` | F values |
| `p-value` | p-values |
| `beta` | beta or effect-size estimates |
| `correlation` | correlation coefficients |
| `other` | any other statistic |

Cluster size, volume, BA and ALE are not statistics; leave them out. Omit `values`
when a row has none.

## Output

For each item, write `<output>/<pmid>.json`, where `<output>` is the batch's
`output` directory (create it if it is missing):

```json
{
  "pmid": "12345678",
  "tables": [
    {
      "table_id": "tbl2",
      "status": "parsed",
      "space": "MNI",
      "note": "Methods: 'normalized to the MNI template'",
      "analyses": [
        {
          "name": "Patients > Controls, 2-back > 0-back",
          "description": "Between-group load effect; positive T values",
          "points": [
            {"xyz": [-42, 18, 30], "values": [{"kind": "t-statistic", "value": 5.12}]},
            {"xyz": [38, 22, 28], "values": [{"kind": "t-statistic", "value": 4.40}]}
          ]
        }
      ]
    },
    {"table_id": "tbl1", "status": "no_coordinates", "note": "demographics"}
  ]
}
```

- **Point fields:** `xyz` holds numbers, not strings. A point may override the
  table's space with its own `"space"`.
- **Names and descriptions:** name each analysis with the labels the table uses;
  put the direction and group in the description when the name is terse.
- **Before writing, check:**
  - no analysis pools different contrasts, groups, signs, ROI and whole-brain
    blocks, or sessions;
  - no two analyses differ only by region, hemisphere or cluster;
  - every repeated X/Y/Z block became its own analysis, or is intentionally empty.

Reply with only the number of study files written.
