---
name: pubmed-search
description: Search PubMed through NCBI E-utilities without an API key and save complete, de-duplicated records (full structured abstracts, DOI, PMCID, MeSH, publication types) for a systematic review. Use when a review needs its search stage run or re-run, when a user wants to test or refine a PubMed query, or when seeding a review from a PMID list.
---

# PubMed search

Run the bundled script. Do not scrape the PubMed website, and do not hand-write
E-utilities calls for the review record. The script's log is what makes the search
reproducible and auditable.

```bash
python SKILLS/pubmed-search/scripts/pubmed_search.py REVIEW
```

It reads `search:` from `REVIEW/review.yaml` (the query, the date range, and
`pmids`/`pmids_file`). You can also pass `--query`, `--pmids-file`, `--date-from`
and `--date-to` directly.

## Before running a new query

1. **Check the count first.** For a new or edited query, get only the count, and
   show it to the user before fetching thousands of records:
   ```bash
   curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=0&term=<url-encoded query>"
   ```
2. **Check the syntax.** Use field tags (`[tiab]`, `[mh]`, `[pt]`), group with
   parentheses, and use explicit booleans. Test sub-queries if the count surprises
   you.
3. **Run the script.**

## What it guarantees

- **Nothing is silently truncated.** Result sets over 9,999 are split by
  publication date until every window fits. If the number retrieved differs from
  the number PubMed reports, the script exits non-zero.
- **Abstracts are complete.** All sections of a structured abstract are kept and
  labelled (for example `METHODS: ...`), and titles keep text inside italics and
  sub/superscripts.
- **The DOI is the article's own,** from its id list or ELocationID, never from the
  reference list.
- **The query and the PMID list are unioned and de-duplicated.** Each record's
  `found_by` says which source produced it.
- **Failed requests are retried with backoff,** then reported. A batch is never
  skipped silently.

## Outputs

- **`REVIEW/search/records.jsonl`:** one record per PMID with
  `pmid, title, abstract, authors, journal, year, doi, pmcid, publication_types,
  mesh, keywords, language, has_abstract, found_by`.
- **`REVIEW/search/search_log.json`:**
  - the query and dates;
  - PubMed's reported count, the date windows and the retrieved count;
  - PMIDs that efetch did not return (withdrawn records and book chapters);
  - the number of records without an abstract;
  - the request count.

## Report to the user

Give the reported count, the retrieved count, and the number of records without an
abstract. If any PMIDs are listed as missing from efetch, name them.

## Notes

- **API key:** none is needed. The limit is 3 requests per second; setting
  `NCBI_API_KEY` raises it to 10. The script keeps to the limit.
- **Email:** set `search.email` in `review.yaml`. NCBI asks tools to identify a
  contact.
- **Re-running:** re-running rewrites `records.jsonl`. Abstract decisions for
  records whose title and abstract did not change stay valid, and new records are
  pending.
