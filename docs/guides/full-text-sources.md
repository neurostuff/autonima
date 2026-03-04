# Full-Text Sources

Use `retrieval.full_text_sources` when your full text already exists on disk or when PubGet is only part of the retrieval path.

## Purpose

Each item in `full_text_sources` maps PMIDs to local files and, optionally, to associated coordinate/table data.

## Common Fields

- `root_path`
  Root directory containing the content source
- `pmid_source`
  One of:
  - `file_name`
  - `folder_name`
  - `json`
- `text_path_templates`
  Relative file templates used to locate full text for `folder_name` and `json` sources
- `allowed_extensions`
  Allowed file extensions when `pmid_source: file_name`
- `json_filename`
  Metadata filename for `json` mode
- `json_pmid_key`
  Key in the JSON file that contains the PMID
- `processed_data_path`
  Path to pubget-like processed coordinate/table CSVs
- `coordinates_path_templates`
  Relative coordinate-file templates when coordinates are stored near each source item

## Source Patterns

## `pmid_source: file_name`

Use this when files are named by PMID:

```yaml
full_text_sources:
  - root_path: "/data/fulltexts"
    pmid_source: "file_name"
    allowed_extensions: [".txt", ".html"]
```

## `pmid_source: folder_name`

Use this when each publication is stored in a PMID-named directory:

```yaml
full_text_sources:
  - root_path: "/data/fulltexts"
    pmid_source: "folder_name"
    text_path_templates:
      - "fulltext.txt"
      - "text.txt"
```

## `pmid_source: json`

Use this when each publication directory contains metadata describing the PMID:

```yaml
full_text_sources:
  - root_path: "/data/fulltexts"
    pmid_source: "json"
    json_filename: "identifiers.json"
    json_pmid_key: "pmid"
    text_path_templates:
      - "processed/pubget/text.txt"
      - "text.txt"
```

## Coordinates and Tables

You have two ways to attach coordinate/table context:

1. `processed_data_path`
   Use a directory containing pubget-like processed CSV outputs such as `coordinates.csv` and `tables.csv`.
2. `coordinates_path_templates`
   Use direct coordinate-file templates inside each source item.

Do not use both in the same source entry. Current validation treats them as mutually exclusive.

## When to Use This Feature

- you already scraped or licensed the full text separately
- you want Autonima to screen a local corpus
- you want to attach coordinate/table data from a custom preprocessing pipeline
