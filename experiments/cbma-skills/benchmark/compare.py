"""Score a cbma-skills review against a gold standard and/or an autonima run.

    python compare.py REVIEW --gold gold.csv [--autonima RUN_DIR] [--out report.json]

Gold standard (CSV, header required). Any of these columns may be present:
    pmid                 required
    abstract_included    1/0, true/false: should pass abstract screening
    included             1/0: final (full-text) inclusion; the main outcome
A plain text file with one PMID per line is read as the list of finally
included studies. Adapt the benchmark's own files to this shape with a few lines
of pandas; keep the adapter next to the benchmark so the mapping is reviewable.

Autonima run (optional): a folder with outputs/abstract_screening_results.json and
outputs/fulltext_screening_results.json, as autonima writes them. Its decisions are
scored against the same gold standard, and the two systems are compared directly
(agreement and Cohen's kappa on the studies both judged).

Coordinates (optional): --gold-nimads STUDYSET.json compares extracted peaks per
study, matching points within --tolerance mm (default 2).

Scoring notes:
  * The universe is the review's search records. Gold PMIDs that the search never
    found count as misses at every stage and are listed ("not_retrieved_by_search"),
    because a search miss is a real recall loss.
  * "uncertain" at the abstract stage counts as passing, because it goes on to full text.
  * Full-text "unavailable" and "incomplete" are reported separately. They are not
    exclusions; recall is reported both over all gold studies and over those with
    retrievable text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


def _truthy(v) -> Optional[bool]:
    if v is None or str(v).strip() == "":
        return None
    return str(v).strip().lower() in {"1", "true", "yes", "y", "include", "included"}


def load_gold(path: Path) -> Dict[str, dict]:
    text = path.read_text()
    if "," not in text.splitlines()[0] and "pmid" not in text.splitlines()[0].lower():
        return {str(int(line.strip())): {"included": True} for line in text.splitlines() if line.strip()}
    gold = {}
    for row in csv.DictReader(text.splitlines()):
        pmid = str(int(row["pmid"]))
        gold[pmid] = {"abstract_included": _truthy(row.get("abstract_included")),
                      "included": _truthy(row.get("included"))}
    return gold


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def skills_decisions(review: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "cbma-review" / "scripts"))
    import ledger  # noqa: E402
    rv = ledger.Review(review)
    abstract = {k[0]: r["decision"] for k, r in rv.valid_decisions("abstract").items()}
    fulltext = {k[0]: r["decision"] for k, r in rv.valid_decisions("fulltext").items()}
    retrieval = {p: e["status"] for p, e in rv.fulltext_index.items()}
    return {"universe": set(rv.records), "abstract_pass": {p for p, d in abstract.items() if d != "exclude"},
            "abstract_judged": set(abstract), "fulltext_include": {p for p, d in fulltext.items() if d == "include"},
            "fulltext_judged": {p for p, d in fulltext.items() if d in ("include", "exclude")},
            "retrieval": retrieval}


def autonima_decisions(run: Path) -> dict:
    def load(name):
        path = run / "outputs" / name
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {str(r["study_id"]): r["decision"] for r in data.get("screening_results", [])}
    abstract = load("abstract_screening_results.json")
    fulltext = load("fulltext_screening_results.json")
    return {"universe": set(abstract),
            "abstract_pass": {p for p, d in abstract.items() if d.startswith("included")},
            "abstract_judged": {p for p, d in abstract.items() if d.startswith(("included", "excluded"))},
            "fulltext_include": {p for p, d in fulltext.items() if d == "included_fulltext"},
            "fulltext_judged": {p for p, d in fulltext.items() if d in ("included_fulltext", "excluded_fulltext")},
            "retrieval": {}}


def binary_metrics(predicted: Set[str], judged: Set[str], positives: Set[str], universe: Set[str]) -> dict:
    tp = len(predicted & positives)
    fp = len(predicted - positives)
    fn = len(positives - predicted)
    tn = len((judged & universe) - predicted - positives)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": _r(prec), "recall": _r(rec), "f1": _r(f1),
            "missed": sorted(positives - predicted, key=int)[:50],
            "false_positives": sorted(predicted - positives, key=int)[:50]}


def _r(x):
    return None if x is None else round(x, 4)


def kappa(a: Set[str], b: Set[str], common: Set[str]) -> dict:
    n = len(common)
    if not n:
        return {"n": 0}
    agree = sum((p in a) == (p in b) for p in common)
    pa, pb = len(a & common) / n, len(b & common) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    po = agree / n
    return {"n": n, "agreement": _r(po), "kappa": _r((po - pe) / (1 - pe)) if pe < 1 else None,
            "only_first": sorted((a - b) & common, key=int)[:50], "only_second": sorted((b - a) & common, key=int)[:50]}


def score(system: dict, gold: Dict[str, dict]) -> dict:
    final_pos = {p for p, g in gold.items() if g.get("included")}
    abs_pos = {p for p, g in gold.items() if g.get("abstract_included")} or final_pos
    universe = system["universe"]
    out = {
        "not_retrieved_by_search": sorted(final_pos - universe, key=int),
        "search_recall_of_final_includes": _r(len(final_pos & universe) / len(final_pos)) if final_pos else None,
        "abstract": binary_metrics(system["abstract_pass"], system["abstract_judged"], abs_pos, universe),
        "final": binary_metrics(system["fulltext_include"], system["fulltext_judged"], final_pos, universe),
    }
    if system["retrieval"]:
        retrievable = {p for p, s in system["retrieval"].items() if s in ("available", "incomplete")}
        pos_r = final_pos & retrievable
        out["final_given_retrievable_text"] = binary_metrics(system["fulltext_include"] & retrievable,
                                                             system["fulltext_judged"] & retrievable, pos_r, universe)
        out["gold_includes_without_text"] = sorted((final_pos & universe) - retrievable, key=int)
    return out


def compare_coordinates(review: Path, gold_nimads: Path, tol: float) -> dict:
    ours = json.loads((review / "results" / "nimads" / "studyset.json").read_text())
    theirs = json.loads(gold_nimads.read_text())

    def points(studyset):
        by = {}
        for s in studyset["studies"]:
            pmid = str(s.get("pmid") or s["id"])
            by.setdefault(pmid, []).extend(tuple(p["coordinates"]) for a in s["analyses"] for p in a["points"])
        return by
    a, b = points(ours), points(theirs)
    rows, tp_all, na, nb = [], 0, 0, 0
    for pmid in sorted(set(a) & set(b), key=int):
        pa, pb = list(a[pmid]), list(b[pmid])
        used, tp = set(), 0
        for p in pa:
            best = min(((math.dist(p, q), j) for j, q in enumerate(pb) if j not in used), default=None)
            if best and best[0] <= tol:
                used.add(best[1])
                tp += 1
        rows.append({"pmid": pmid, "ours": len(pa), "gold": len(pb), "matched": tp})
        tp_all, na, nb = tp_all + tp, na + len(pa), nb + len(pb)
    return {"studies_compared": len(rows), "point_precision": _r(tp_all / na) if na else None,
            "point_recall": _r(tp_all / nb) if nb else None, "tolerance_mm": tol,
            "only_ours": sorted(set(a) - set(b), key=int), "only_gold": sorted(set(b) - set(a), key=int),
            "per_study": rows}


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--autonima", type=Path)
    ap.add_argument("--gold-nimads", type=Path)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)

    report: dict = {"review": str(args.review)}
    ours = skills_decisions(args.review)
    theirs = autonima_decisions(args.autonima) if args.autonima else None
    if args.gold:
        gold = load_gold(args.gold)
        report["gold_final_includes"] = sum(1 for g in gold.values() if g.get("included"))
        report["skills"] = score(ours, gold)
        if theirs:
            report["autonima"] = score(theirs, gold)
    if theirs:
        report["agreement"] = {
            "abstract": kappa(ours["abstract_pass"], theirs["abstract_pass"],
                              ours["abstract_judged"] & theirs["abstract_judged"]),
            "final": kappa(ours["fulltext_include"], theirs["fulltext_include"],
                           ours["fulltext_judged"] & theirs["fulltext_judged"]),
        }
    if args.gold_nimads:
        report["coordinates"] = compare_coordinates(args.review, args.gold_nimads, args.tolerance)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
