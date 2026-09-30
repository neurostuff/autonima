# Documents

A document source replaces each article's full text with a document that another tool
produced from it: a structured extraction record, a summary, a compressed rendering. Full-text
screening and annotation read the document instead of the article. Nothing else in the
pipeline changes.

Autonima doesn't produce documents. It reads a directory that some other tool wrote before
the run, so any tool can supply one by writing the layout below.

## Two kinds

| kind | the source supplies | coordinates come from |
|---|---|---|
| `text` | one document per study | the article, retrieved and parsed as usual |
| `records` | one document per study, plus the analyses it refers to | the source; the article is not retrieved |

Use `text` when the document is only something to read, like a summary. Use `records` when the
document describes the study's analyses individually. Its analyses must then travel with it.
Re-parsing the article's tables would produce a different list, and every reference in the
document would point at the wrong contrast without anything noticing.

## Configuration

```yaml
documents:
  enabled: true
  kind: records               # text | records
  root: /data/records/arm1    # the directory described below
  description: "a structured extraction record of the article"
```

`description` says what the documents are. Prompts show it wherever they would otherwise say
"full text", so a summary and an extraction record read differently to the model. If you
leave it out, a generic description is used.

These combinations are refused when the config loads, before the search stage runs:

- `kind: records` together with `parsing.parse_coordinates: true`. The records already hold
  the analyses, and re-parsing them is the silent mis-attachment described above.
- `annotation.annotations` configured without `study_fulltext` in
  `annotation.metadata_fields`. Annotation would never read the documents, so a `text` run
  would annotate exactly as the article run did.
- a `root` that is not a directory, an unknown `kind`, or an unknown key under `documents`.

## Directory layout

The directory is flat, with files named by PMID:

```text
<root>/
├── 12345678.md               the document: .md, .txt, .json, .yaml or .yml
└── 12345678.analyses.json    records only: the analyses the document refers to
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
- for `records`, the source's analyses and tables replace anything retrieval left, and they
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

## Prompts

When a study has a document, full-text screening uses a prompt that says the model is reading
a document that stands in for the full text, and gives the document's `description`. The
article prompt's completeness check, which looks for introduction, methods, results and
discussion sections, is replaced by one that flags `fulltext_incomplete` only when the document
itself reports that the article couldn't be read. A derived document never has those sections,
and it's as complete as the process that produced it.

Annotation prompts label the document in the same way. Studies without a document get the
exact prompts, and keep the exact cache signatures, that they had before documents existed.
