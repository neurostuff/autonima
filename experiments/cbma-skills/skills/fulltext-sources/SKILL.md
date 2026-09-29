---
name: fulltext-sources
description: Retrieve and normalize full texts for a systematic review, from PubMed Central open access (no key) and from any folders of pre-downloaded articles (publisher HTML, JATS XML, plain text), into one per-study layout of text plus parsed tables. Use when a review needs full texts gathered, when adding a local folder of downloaded papers as a source, or when checking which studies have usable text.
---

# Full-text sources

Every downstream stage reads one normalized layout, whatever the source:

```
REVIEW/docs/<pmid>/
  text.md            the article as markdown; the reference list is removed; tables and
                     figures appear as one-line [TABLE id: caption] / [FIGURE: caption] markers
  tables/<id>.json   label, caption, footer, `grid` (merged cells expanded), `tsv`,
                     `coordinate_candidate`, `duplicate_of`, `content_sha256`
  meta.json          source, original file, hashes, table counts, completeness flag
```

## Configure sources in review.yaml

```yaml
fulltext:
  sources:
    - type: pmc                       # PubMed Central open access, via E-utilities; no key
    - type: local                     # a folder of files you already have
      name: elsevier_html
      path: /data/elsevier_html       # absolute, or relative to REVIEW
      pattern: "**/*.html"
      id_from: filename               # filename | parent_dir | regex | sidecar
      id_kind: pmid                   # pmid | doi | pmcid
      format: auto                    # auto | html | jats | text
```

### Source order

Sources are tried in the order listed, and the first one that yields a complete
article wins. If none is complete, the longest incomplete text is kept and flagged.

Put your most reliable source first. Pre-downloaded publisher HTML often has
better tables than PMC; if so, list it first.

### How local files map to PMIDs

| `id_from` | Where the identifier comes from | Example |
|---|---|---|
| `filename` | the file stem | `12345678.html` |
| `parent_dir` | the containing folder's name | `12345678/article.html` |
| `regex` | the named group `id` in `regex`, matched against the path relative to `path` | `regex: "pmid_(?P<id>\\d+)"` |
| `sidecar` | a key in a JSON file beside the article | `sidecar: identifiers.json`, `sidecar_key: pmid` |

With `id_kind: doi` or `pmcid`, the identifier is matched to a PMID through the
search records. Files that match nothing are reported as unmapped.

Local files only supply text for studies the search found. A file whose PMID is not
in `search/records.jsonl` is never added to the review. Studies found through a
local source go through full-text screening like any other.

## Run

1. **Check the mapping before a long run** (it does not download anything):
   ```bash
   python SKILLS/fulltext-sources/scripts/gather_fulltext.py REVIEW --index-only
   ```
   It reports, per local source, the files mapped, the PMIDs that are in the
   search, and some unmapped examples. Fix `id_from` and `regex` until the numbers
   look right.
2. **Gather the studies that passed abstract screening:**
   ```bash
   python SKILLS/cbma-review/scripts/ledger.py needs-fulltext REVIEW
   python SKILLS/fulltext-sources/scripts/gather_fulltext.py REVIEW --pmids-file REVIEW/fulltext/needed.txt
   ```
3. **Report the outcomes** it prints (available, incomplete and unavailable, by
   source) to the user.

PMC downloads are cached under `REVIEW/fulltext/raw/pmc/`, so a re-run does not
download again. To re-parse after changing sources, run the same command.

## Outcomes

- **`available`:** a complete article from some source.
- **`incomplete`:** only a short or partial text was found, such as an abstract
  page, a paywall notice, or a PMC record without an open-access body.
  - It is still screened, and the screener may return `incomplete`.
  - Look for another source for it: a local folder, or ask the user.
- **`unavailable`:** no source had anything. `fulltext/index.jsonl` gives each
  source's reason.
  - These are reported in PRISMA as "not retrieved".
  - They are never exclusions.

## When the normalizer gets a table wrong

Publisher HTML varies. If a coordinate table's `grid` is clearly broken (columns
shifted or cells merged into one), note it for the user. The extraction subagent
can still read the original file listed in `meta.json` as `origin`, but its numbers
are checked against `grid`. So fix the source or the parsing, rather than letting
extraction work around it.

Individual files can be normalized for debugging:
```bash
python SKILLS/fulltext-sources/scripts/docnorm.py FILE --pmid 12345678 --out /tmp/docs
```
