# Benchmarking cbma-skills

`compare.py` scores a finished review against:
- a gold standard (`--gold`);
- an autonima run on the same project (`--autonima`);
- gold coordinates (`--gold-nimads`).

```bash
python benchmark/compare.py REVIEW \
  --gold gold.csv \
  --autonima /path/to/autonima-results/<project>/<run> \
  --gold-nimads gold_studyset.json \
  --out REVIEW/results/benchmark.json
```

## Gold file format

```csv
pmid,abstract_included,included
11111111,1,1
22222222,0,0
```

- **`included`** is final inclusion and is the main outcome.
- **`abstract_included`** is optional. When it is absent, the final includes are
  used as the abstract-stage positives. That tests recall only, which is the
  property that matters at that stage.
- **A plain text file** with one PMID per line is read as the final includes.

Each benchmark needs a short adapter from its own files to this shape. Keep the
adapter beside the benchmark data, not in this package, so the mapping is
reviewable.

## Keeping the agent blind

Put gold files outside the agent's workspace, and run `compare.py` yourself after
the review is finished. If you run it through an agent, use a fresh session that
did not run the review.

See [../PLAN.md](../PLAN.md) for the full protocol and metrics.
