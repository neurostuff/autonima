# NeuroStore

[NeuroStore](https://neurostore.org) indexes neuroimaging studies and the coordinates
extracted from them. Autonima can use it in two places, separately or together:

| Stage | Setting | What changes |
|---|---|---|
| Search | `search.database: neurostore` | the candidate studies come from NeuroStore's index, not all of PubMed |
| Coordinates | `parsing.coordinate_source: neurostore` | each included study's analyses come from NeuroStore, not from parsing its tables with an LLM |

Everything else is unchanged. Full-text screening and annotation still read the article,
which NeuroStore does not share, so retrieval runs as usual.

To have them read NeuroStore's extracted values instead, each analysis with its contrast,
task, spatial scope, participants and design, use `retrieval.records.kind: neurostore`; see
[Records](records.md#neurostore-records). It replaces both the article and
`parsing.coordinate_source`, since the analyses come with the records.

## Search

```yaml
search:
  database: neurostore
  query: '("emotion regulation"[tiab] OR reappraisal[tiab]) AND fMRI[tiab]'
  date_from: "2000/01/01"     # year granularity in NeuroStore
  date_to: "2019/12/31"
  email: you@example.org      # study metadata is still fetched from PubMed
  neurostore:                 # all optional
    search_full_text: false   # also match the paper's text, not only title and abstract
    data_type: coordinate     # coordinate (default) | image | both | null for all studies
    query: null               # a query in NeuroStore's own syntax, used instead of `query`
```

`query` is the PubMed query you would otherwise run, translated. NeuroStore matches whole
words after stemming, with quoted phrases, `AND`, `OR`, `NOT`, `-` and parentheses, so:

- **text field tags** (`[tiab]`, `[Title/Abstract]`) are removed;
- **MeSH** inside an `OR` group becomes the heading as a phrase; MeSH `AND`-ed onto the
  query (`AND Humans[mh]`) is dropped, since as a phrase it would demand the word in the text;
- **publication type, language and date tags** are dropped; dates come from
  `date_from`/`date_to`;
- **truncation** (`reactiv*`) becomes the prefix with a few common endings, which the
  server's stemmer folds onto real words;
- **`NOT` clauses** at the top of the query stay limited to titles and abstracts when
  `search_full_text` is on. NeuroStore applies `NOT` to every field it searches, so sent
  as written, `NOT "systematic review"[tiab]` would drop every paper whose text mentions a
  systematic review. Autonima searches the rest of the query in full text and removes what a
  title-and-abstract search of the `NOT` clauses finds. A native `neurostore.query` is sent
  as written, so its `NOT` applies to the text too.

Every dropped or approximated clause, the NeuroStore query sent, the totals and any base
study without a PMID are written to `outputs/neurostore_search_log.json`. Studies without a
PMID are not carried, because the rest of the pipeline is keyed on PMIDs. To control the
query exactly, write it in NeuroStore syntax under `neurostore.query`.

The search pages through every result and fails if the count retrieved differs from the
total NeuroStore reported.

## Coordinates

```yaml
parsing:
  parse_coordinates: true
  coordinate_source: neurostore      # tables (default) | neurostore
  neurostore:                        # all optional
    sources: [study_schema, nuextract-v21, llm, neurostore, pubget, nuextract-v19, neuroquery, neurosynth]
    roles: [result]
```

A NeuroStore study has several versions, each one source's extraction of its coordinates.
Each included study uses one: the first in `sources` order that has coordinates. Its
analyses are filtered to `roles`. NeuroStore labels a paper's own findings `result`, while
seed regions (`anchor`), coordinates cited from other work (`reference`) and region
definitions (`localization`) are left out by default. An analysis with no role counts as a
result.

Every included study is looked up, including studies whose tables were not retrieved. A study
without usable coordinates keeps no analyses and is listed with the reason in
`outputs/neurostore_unavailable.csv`:

| reason | meaning |
|---|---|
| `not_in_neurostore` | no NeuroStore study has this PMID |
| `no_coordinates` | NeuroStore has the study, without coordinates |
| `no_coordinates_from_configured_sources` | coordinates exist, only from sources not in `sources` |
| `fetch_failed` | the request failed; the study is fetched again next run |

NeuroStore's extractions change as it re-ingests papers. What a run used, down to the
version id and its update time, is frozen in `outputs/neurostore_analyses.json`, and later
runs read that snapshot instead of the server. Delete a study's entry, or the file, to fetch
again. Changing `sources` or `roles` invalidates the snapshot.

## Caching

The new settings enter a stage's cache signature only when set, so existing configs keep
their caches. Switching `coordinate_source` invalidates the parsing stage and the stages
after it, nothing before it.
