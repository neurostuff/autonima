# Records

A record source replaces each article's full text with a record that another tool produced
from it: a structured extraction record, a summary, a compressed rendering. Full-text
screening and annotation read the record instead of the article. Nothing else in the pipeline
changes.

Autonima doesn't produce records. It reads a directory that some other tool wrote before the
run, so any tool can supply one by writing the layout below.

## Two kinds

| kind | the source supplies | coordinates come from |
|---|---|---|
| `text` | one document per study | the article, retrieved and parsed as usual |
| `analyses` | one document per study, plus the analyses it refers to | the source; the article is not retrieved |
| `neurostore` | built by autonima from NeuroStore: each analysis with the features NeuroStore extracted | NeuroStore; the article is not retrieved |

Use `text` when the record is only something to read, like a summary. Use `analyses` when the
record describes the study's analyses individually. Its analyses must then travel with it.
Re-parsing the article's tables would produce a different list, and every reference in the
document would point at the wrong contrast without anything noticing.

## Configuration

Records are configured under `retrieval`:

```yaml
retrieval:
  records:
    enabled: true
    kind: analyses              # text | analyses
    root: /data/records/arm1    # the directory described below
    description: "a structured extraction record of the article"
```

`description` says what the records are. Prompts show it wherever they would otherwise say
"full text", so a summary and an extraction record read differently to the model. If you
leave it out, a generic description is used.

These combinations are refused when the config loads, before the search stage runs:

- `kind: analyses` together with `parsing.parse_coordinates: true`. The records already hold
  the analyses, and re-parsing them is the silent mis-attachment described above.
- `annotation.annotations` configured without `study_fulltext` in
  `annotation.metadata_fields`. Annotation would never read the records, so a `text` run
  would annotate exactly as the article run did.
- a `root` that is not a directory, an unknown `kind`, or an unknown key under
  `retrieval.records`.
- a top-level `documents:` section, the name this block had before it moved under
  `retrieval`.

## Directory layout

The directory is flat, with files named by PMID:

```text
<root>/
├── 12345678.md               the document: .md, .txt, .json, .yaml or .yml
└── 12345678.analyses.json    `analyses` only: the analyses the document refers to
```

The document is passed to the model verbatim, so a JSON or YAML record goes in exactly as it
was written. A PMID with two documents, for example `12345678.md` and `12345678.json`, stops
the run with an error that names both files.

### The analyses file

```json
{
  "document_sha256": "<sha256 of 12345678.md's bytes>",
  "space": "MNI",
  "analyses": [
    {
      "key": "t2#1",
      "name": "PTSD > controls",
      "description": "Whole-brain group contrast",
      "table_id": "t2",
      "points": [
        {"coordinates": [-24, 8, 52], "values": [{"value": 4.1, "kind": "t-statistic"}]}
      ],
      "document": {"effect": "contrast", "spatial_scope": "whole brain"}
    }
  ],
  "tables": [
    {"table_id": "t2", "table_label": "Table 2", "table_caption": "Group differences"}
  ]
}
```

- `document_sha256` is required. It pins this file to the document it was produced with. If
  the document is edited or re-rendered without this file being rewritten, the study is
  reported as unavailable instead of being paired with analyses that no longer match.
- `key` is the source's own name for the analysis. Keys must be unique within the file.
- `points` use the same shape as autonima's coordinates. A point's `space` falls back to the
  analysis's `space`, then to the file's.
- `document` is optional: the analysis's own record. It can be a string or any JSON value,
  and a JSON value is serialized before the model sees it. Add `analysis_document` to
  `annotation.metadata_fields` to show it beside each analysis during annotation.
- `tables` is optional. Any table that an analysis names but that isn't listed gets an entry
  with only its id.

Order matters: analysis *i* in the file becomes the study's analysis *i*, with autonima's id
`<pmid>_analysis_<i>`.

### Referring to analyses

In the document, and in any per-analysis `document`, refer to an analysis as
`{{analysis:<key>}}`:

```markdown
- {{analysis:t2#1}}: PTSD > controls
- {{analysis:t2#2}}: controls > PTSD, the sign split of {{analysis:t2#1}}
```

Before the model reads the document, autonima replaces each reference with its own id for that
analysis. The model then sees the same ids it has to answer with, so it can't reply in the
source's address space. A reference to a key that the file doesn't supply marks the study as
unavailable. A `text` document that contains references is refused too, because a `text`
source supplies no analyses to resolve them against.

## What happens during a run

Documents are attached at the end of the retrieval stage. For each study:

- the document, with its references resolved, is written to `<output>/documents/<pmid>.<ext>`,
  and `full_text_path` points at it. Every full-text cache check hashes that file, so changing
  a document re-screens only the studies it affects.
- for `analyses`, the source's analyses and tables replace anything retrieval left, and they
  are marked as ingested, so coordinate parsing never touches them.

A study with no usable document is treated like one whose full text wasn't retrieved. It isn't
screened, it's listed in `missing_fulltexts.csv` with source `document:<kind>`, and
`outputs/documents_unavailable.csv` records why:

| reason | meaning |
|---|---|
| `no_document` | no file for this PMID |
| `no_pmid` | the study has no numeric PMID to look up |
| `pairing_mismatch` | the analyses file is missing, its `document_sha256` doesn't match, or a reference is unresolved |
| `identity_conflict` | duplicate analysis keys |
| `invalid_document` | the file is empty, isn't UTF-8, or doesn't follow the layout |

Such a study never falls back to its article, because that would mix two kinds of input in one
run. With `screening.abstract.skip_stage: true` there is no abstract decision on file, so this
list is the only record of why those studies never reached screening.

`final_results.json` records each study's document: its source path, content hash, kind, and
the mapping from source keys to autonima ids. It never stores the document text.

## NeuroStore records

`kind: neurostore` needs no directory. Autonima builds each included study's record from
NeuroStore, so full-text screening and annotation judge NeuroStore's extracted values instead
of the article:

```yaml
retrieval:
  records:
    enabled: true
    kind: neurostore
    neurostore:                 # all optional
      sources: [study_schema]   # which NeuroStore version's analyses, as parsing.neurostore
      roles: [result]
      base_url: https://neurostore.org/api
parsing:
  parse_coordinates: false      # the analyses come with the records
```

Per study, the record is the title and abstract, then one block per analysis giving what
NeuroStore's extraction states about it: the contrast and its two sides, the task, whole brain
or region of interest, the effect, the participants and n, the medical condition, the measure
and the design. A feature the extraction is silent on is left out. Each block is also the
analysis's own document, for `annotation.metadata_fields: [analysis_document]`.

The analyses and coordinates are chosen as `parsing.coordinate_source: neurostore` chooses
them. The features come from NeuroStore's analysis query. Both are frozen in the run,
`outputs/neurostore_record_analyses.json` and `outputs/neurostore_features.json`, and later
runs read them instead of the server; delete a study's entry, or the file, to fetch it again.
The records themselves are written to `<output>/neurostore_records/`.

A study without a usable record is listed in `documents_unavailable.csv` and is not screened
from anything else:

| reason | meaning |
|---|---|
| `not_in_neurostore` | no NeuroStore study has this PMID |
| `no_coordinates`, `no_coordinates_from_configured_sources` | as for NeuroStore coordinates |
| `no_features` | NeuroStore has coordinates for the study but no features |
| `features_for_other_version` | the features describe another version of the study than the one the coordinates came from |
| `fetch_failed`, `features_fetch_failed` | a request failed; the study is fetched again next run |

### Full-text fallback

```yaml
retrieval:
  sources: [pubget]             # where articles come from, as in any full-text run
  records:
    enabled: true
    kind: neurostore
    fallback: fulltext
```

With `fallback: fulltext`, a study NeuroStore has coordinates for but no features
(`no_features`, `features_for_other_version`) is reviewed from its article instead of being
left unavailable. Autonima retrieves the article for just those studies, from
`retrieval.sources` and `retrieval.full_text_sources`, and the study's record becomes its
title, abstract and NeuroStore's list of analyses, followed by the article's full text. The
model still selects among NeuroStore's analyses, so every study's coordinates come from
NeuroStore whichever record it was judged from.

- **Which studies:** each fallback record's analyses file says `"input": "fulltext"` under
  `source`, with `article_source` naming where the article came from. The run's
  `execution_stats.retrieval` counts `documents_fallback_candidates` and
  `documents_from_full_text`.
- **Still unavailable:** a study whose article cannot be retrieved keeps its reason, with
  "full-text fallback: article not retrieved" in the detail. Studies NeuroStore has no
  coordinates for are not fallback studies: there would be no analyses to select.
- **Caching:** the fallback enters the retrieval stage's signature only when it is on, and
  leaves every feature record byte for byte as it was. Turning it on for an existing run
  therefore reviews only the fallback studies; every other decision is reused.
- **Cost:** a fallback record is as long as the article, so each fallback study costs about
  what a full-text run spends on it.

## Prompts

When a study has a document, full-text screening uses a prompt that says the model is reading
a document that stands in for the full text, and gives the document's `description`. The
article prompt's completeness check, which looks for introduction, methods, results and
discussion sections, is replaced by one that flags `fulltext_incomplete` only when the document
itself reports that the article couldn't be read. A derived document never has those sections,
and it's as complete as the process that produced it.

Annotation prompts label the document in the same way. Studies without a document get the
exact prompts, and keep the exact cache signatures, that they had before records existed.
