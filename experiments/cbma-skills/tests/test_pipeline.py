"""Offline tests for the cbma-skills scripts. No network: NCBI calls go to fakes.

    pip install pytest pyyaml
    pytest experiments/cbma-skills/tests
"""

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
FIX = Path(__file__).parent / "fixtures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pubmed = load("pubmed_search", SKILLS / "pubmed-search/scripts/pubmed_search.py")
docnorm = load("docnorm", SKILLS / "fulltext-sources/scripts/docnorm.py")
gather = load("gather_fulltext", SKILLS / "fulltext-sources/scripts/gather_fulltext.py")
ledger = load("ledger", SKILLS / "cbma-review/scripts/ledger.py")
compare = load("compare", ROOT / "benchmark/compare.py")


# --------------------------------------------------------------------------- #
# PubMed parsing and search
# --------------------------------------------------------------------------- #

def test_pubmed_parsing_keeps_everything_autonima_lost():
    recs = {r["pmid"]: r for r in pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())}
    a = recs["11111111"]
    assert a["title"].startswith("Effects of BDNF Val66Met")          # text inside <i> kept
    assert a["abstract"].count("\n") == 2                             # all three sections
    assert "METHODS: Thirty patients" in a["abstract"] and "n-back" in a["abstract"]
    assert a["doi"] == "10.1016/j.neuroimage.2019.001"                # from ELocationID
    assert a["pmcid"] == "PMC7000001"
    assert a["authors"] == ["Smith J", "WM Consortium"]
    assert a["mesh"] == ["Schizophrenia"]
    b = recs["22222222"]
    assert b["doi"] is None                                           # not the cited paper's DOI
    assert b["year"] == 2001 and not b["has_abstract"]
    assert recs["33333333"]["doi"] == "10.1002/hbm.33333"


def test_normalize_pmid():
    assert pubmed.normalize_pmid(12345) == "12345"
    assert pubmed.normalize_pmid(" PMID: 0012345 ") == "12345"
    with pytest.raises(pubmed.SearchError):
        pubmed.normalize_pmid("abc")


class FakeEutils:
    """esearch over a fixed corpus with publication dates; efetch from the fixture."""

    def __init__(self, n=25000):
        import datetime as dt
        self.dates = {str(10_000_000 + i): dt.date(2000, 1, 1) + dt.timedelta(days=i % 9000) for i in range(n)}
        self.calls = 0

    def __call__(self, url, params):
        import datetime as dt
        self.calls += 1
        if url.endswith("esearch.fcgi"):
            ids = sorted(self.dates, key=int)
            if "mindate" in params:
                lo = dt.date(*map(int, params["mindate"].split("/")))
                hi = dt.date(*map(int, params["maxdate"].split("/")))
                ids = [p for p in ids if lo <= self.dates[p] <= hi]
            retmax = int(params["retmax"])
            return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids[:retmax]}}).encode()
        raise AssertionError(url)


def test_search_splits_past_the_9999_cap(monkeypatch):
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: None)
    fake = FakeEutils(25000)
    client = pubmed.Client(http=fake)
    log = {}
    ids = pubmed.search_ids(client, "anything", log=log)
    assert len(ids) == 25000 == log["reported_count"]
    assert all(w["count"] <= pubmed.ESEARCH_CAP for w in log["windows"])


def test_client_retries_then_raises(monkeypatch):
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: None)
    attempts = []

    def flaky(url, params):
        attempts.append(1)
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    with pytest.raises(pubmed.SearchError):
        pubmed.Client(http=flaky, retries=3).call("esearch", {})
    assert len(attempts) == 3


# --------------------------------------------------------------------------- #
# Document normalization
# --------------------------------------------------------------------------- #

def test_jats_normalization():
    parsed = docnorm.parse_jats((FIX / "pmc_11111111.xml").read_bytes())
    assert parsed["complete"]
    assert "cited paper about schizophrenia" not in parsed["text_md"]          # reference list dropped
    assert "## Participants" in parsed["text_md"] or "### Participants" in parsed["text_md"]
    tables = {t["table_id"]: t for t in parsed["tables"]}
    assert not tables["tbl1"]["coordinate_candidate"]
    t2 = tables["tbl2"]
    assert t2["coordinate_candidate"]
    assert t2["grid"][0][2:5] == ["MNI coordinates"] * 3                       # colspan expanded
    assert t2["grid"][1][0] == "Region"                                        # rowspan expanded
    assert t2["grid"][2][2] == "−42"
    assert "FWE" in t2["footer"]


def test_html_normalization_drops_boilerplate_and_flags_duplicates():
    parsed = docnorm.parse_html((FIX / "publisher_33333333.html").read_bytes())
    text = parsed["text_md"]
    assert "Journal menu" not in text and "var x" not in text and "Copyright" not in text
    assert "nobody should read" not in text
    assert "Talairach" in text
    assert len(parsed["tables"]) == 2
    first, second = parsed["tables"]
    assert first["label"] == "Table 1" and "Load effect" in first["caption"] and "0.001" in first["footer"]
    assert first["coordinate_candidate"]
    assert second["duplicate_of"] == first["table_id"]


# --------------------------------------------------------------------------- #
# Point verification
# --------------------------------------------------------------------------- #

def test_verify_point_catches_sign_flips_and_inventions():
    grid = docnorm.parse_jats((FIX / "pmc_11111111.xml").read_bytes())["tables"][1]["grid"]
    assert ledger.verify_point([-42, 18, 30], grid) == "row"
    assert ledger.verify_point([42, 18, 30], grid) == "unverified"     # sign flipped
    assert ledger.verify_point([-42, 22, 28], grid) == "table"         # mixed rows
    assert ledger.verify_point([10, 10, 10], grid) == "unverified"


# --------------------------------------------------------------------------- #
# End to end through the ledger
# --------------------------------------------------------------------------- #

REVIEW_YAML = """
name: test_review
objective: fMRI studies of working memory in schizophrenia with coordinates
search:
  query: schizophrenia working memory fmri
screening:
  abstract:
    inclusion: [Human participants with schizophrenia, Task fMRI]
    exclusion: [Review article]
  fulltext:
    inclusion: [Adults with schizophrenia, Whole-brain coordinates reported]
fulltext:
  sources:
    - type: pmc
    - type: local
      name: publisher_html
      path: local_html
      pattern: "*.html"
      id_from: regex
      regex: "publisher_(?P<id>\\\\d+)"
selection:
  targets:
    - name: patients_gt_controls
      inclusion: [Patients greater than controls]
"""


@pytest.fixture
def review(tmp_path, monkeypatch):
    monkeypatch.setattr(gather.time, "sleep", lambda s: None)
    rv = tmp_path / "review"
    rv.mkdir()
    (rv / "review.yaml").write_text(REVIEW_YAML)
    (rv / "search").mkdir()
    recs = pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())
    (rv / "search" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (rv / "local_html").mkdir()
    (rv / "local_html" / "publisher_33333333.html").write_bytes((FIX / "publisher_33333333.html").read_bytes())
    return rv


def fake_pmc(url, params):
    if url.endswith("efetch.fcgi") and params["id"] == "7000001":
        return (FIX / "pmc_11111111.xml").read_bytes()
    if url.endswith("elink.fcgi"):
        return json.dumps({"linksets": [{"linksetdbs": []}]}).encode()
    raise AssertionError((url, params))


def run_ledger(*args):
    return ledger.main([str(a) for a in args])


def test_spec_rejects_typos(review):
    text = (review / "review.yaml").read_text().replace("selection:", "selecton:")
    (review / "review.yaml").write_text(text)
    assert run_ledger("init", review) == 1


def test_full_flow(review, capsys):
    assert run_ledger("init", review) == 0
    capsys.readouterr()
    criteria = json.loads((review / "results" / "criteria.json").read_text())
    assert list(criteria["abstract"]["criteria"]) == ["I1", "I2", "E1"]
    assert list(criteria["fulltext"]["criteria"]) == ["I1", "I2"]       # numbered per stage

    # ---- abstract stage: one good line, one inconsistent line, one missing
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "10") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    assert {i["pmid"] for i in batch["items"]} == {"11111111", "22222222", "33333333"}
    out = Path(batch["output"])
    out.write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "11111111", "decision": "include", "criteria": {"I1": "met", "I2": "met", "E1": "not_met"}, "reason": "I1, I2 met."},
        {"pmid": "22222222", "decision": "include", "criteria": {"I1": "unclear", "I2": "unclear", "E1": "met"}, "reason": "oops"},
    ]) + "\n")
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test/fake") == 2
    report = last_json(capsys)
    assert report["accepted"] == 1 and report["rejected"] == 2

    # retry: only the two failures come back
    assert run_ledger("batches", review, "--stage", "abstract") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    assert {i["pmid"] for i in batch["items"]} == {"22222222", "33333333"}
    Path(batch["output"]).write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "22222222", "decision": "exclude", "criteria": {"I1": "unclear", "I2": "unclear", "E1": "met"}, "reason": "Review (E1)."},
        {"pmid": "33333333", "decision": "uncertain", "criteria": {"I1": "met", "I2": "met", "E1": "unclear"}, "reason": "x"},
    ]) + "\n")
    # 'uncertain' is fine when only an exclusion is unclear (is it a review? can't tell).
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test/fake") == 0
    capsys.readouterr()

    # ---- full text: PMC for one study, the local HTML folder for the other
    assert run_ledger("needs-fulltext", review) == 0
    needed = (review / "fulltext/needed.txt").read_text().split()
    assert needed == ["11111111", "33333333"]
    spec = gather.load_spec(review)
    records = gather.load_records(review)
    sources = gather.build_sources(spec, review, records, http=fake_pmc)
    entries = gather.gather(review, needed, sources)
    gather.write_index(review, entries)
    by = {e["pmid"]: e for e in entries}
    assert by["11111111"]["status"] == "available" and by["11111111"]["source"] == "pmc"
    assert by["33333333"]["source"] == "publisher_html"
    assert by["33333333"]["attempts"][0]["source"] == "pmc"             # PMC tried first, failed, recorded

    # ---- full-text screening with an evidence quote
    assert run_ledger("batches", review, "--stage", "fulltext", "--size", "5") == 0
    capsys.readouterr()
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    Path(batch["output"]).write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "11111111", "decision": "include", "criteria": {"I1": "met", "I2": "met"}, "reason": "I1, I2.",
         "evidence": [{"criterion": "I1", "quote": "Thirty patients with DSM-IV schizophrenia"},
                      {"criterion": "I2", "quote": "a sentence that is not in the paper"}]},
        {"pmid": "33333333", "decision": "include", "criteria": {"I1": "met", "I2": "met"}, "reason": "I1, I2.",
         "evidence": [{"criterion": "I2", "quote": "Coordinates are reported in Talairach space."}]},
    ]) + "\n")
    assert run_ledger("ingest", review, "--stage", "fulltext", "--agent", "test/fake") == 0
    capsys.readouterr()
    dec = ledger.Review(review).valid_decisions("fulltext")
    assert dec[("11111111",)]["ungrounded_evidence"] == ["I2"]
    assert dec[("33333333",)]["ungrounded_evidence"] == []

    # ---- extraction: one study with a sign-flipped point, one missing a must-report table
    assert run_ledger("batches", review, "--stage", "extraction", "--size", "5") == 0
    capsys.readouterr()
    batch = json.loads((review / "work/extraction/batch_0001.json").read_text())
    outdir = Path(batch["output"])
    outdir.mkdir()
    (outdir / "11111111.json").write_text(json.dumps({"pmid": "11111111", "tables": [
        {"table_id": "tbl1", "status": "no_coordinates"},
        {"table_id": "tbl2", "status": "parsed", "space": "MNI", "analyses": [
            {"name": "Controls > Patients", "points": [{"xyz": [-42, 18, 30]}, {"xyz": [38, 22, 28]}]},
            {"name": "Patients > Controls", "points": [{"xyz": [-4, -62, 44], "values": [{"kind": "t-statistic", "value": 3.95}]}]},
        ]}]}))
    (outdir / "33333333.json").write_text(json.dumps({"pmid": "33333333", "tables": []}))
    assert run_ledger("ingest", review, "--stage", "extraction", "--agent", "test/fake") == 2
    capsys.readouterr()
    a = json.loads((review / "analyses/11111111.json").read_text())
    assert [x["analysis_id"] for x in a["analyses"]] == ["11111111-tbl2-a1", "11111111-tbl2-a2"]
    assert a["analyses"][1]["points"][0]["verification"] == "unverified"     # 4 was reported as -4
    assert not (review / "analyses/33333333.json").exists()                  # rejected, still pending
    assert ledger.Review(review).pending("extraction") == ["33333333"]

    # ---- export refuses while anything is pending
    assert run_ledger("export", review) == 1
    capsys.readouterr()

    # ---- selection for the finished study, then export the finished part
    assert run_ledger("batches", review, "--stage", "selection") == 0
    capsys.readouterr()
    batch = json.loads((review / "work/selection/batch_0001.json").read_text())
    Path(batch["output"]).write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "11111111", "analysis_id": "11111111-tbl2-a1", "target": "patients_gt_controls",
         "include": False, "criteria": {"I1": "not_met"}, "reason": "Controls > patients (I1 not met)."},
        {"pmid": "11111111", "analysis_id": "11111111-tbl2-a2", "target": "patients_gt_controls",
         "include": True, "criteria": {"I1": "met"}, "reason": "I1."},
    ]) + "\n")
    assert run_ledger("ingest", review, "--stage", "selection", "--agent", "test/fake") == 0
    capsys.readouterr()
    assert run_ledger("export", review, "--allow-pending") == 0
    report = last_json(capsys)
    assert report["studies"] == 1 and report["analyses"] == 1
    assert report["dropped"]["points_unverified"] == 1                        # the flipped point
    assert report["dropped"]["analysis_no_verified_points"] == 1
    studyset = json.loads((review / "results/nimads/studyset.json").read_text())
    ann = json.loads((review / "results/nimads/annotation.json").read_text())
    assert [a["id"] for a in studyset["studies"][0]["analyses"]] == ["11111111-tbl2-a1"]
    assert ann["notes"] == [{"analysis_id": "11111111-tbl2-a1", "annotation_id": "cbma_skills",
                             "note": {"patients_gt_controls": False}}]

    # ---- status is consistent
    assert run_ledger("status", review) == 0
    status = last_json(capsys)
    assert status["abstract_screening"] == {"screened": 3, "include": 1, "uncertain": 1, "exclude": 1, "pending": 0}
    assert status["fulltext_retrieval"]["by_source"] == {"pmc": 1, "publisher_html": 1}
    assert status["fulltext_screening"]["decisions_with_ungrounded_evidence"] == 1
    assert status["extraction"]["pending"] == 1

    # ---- benchmark scoring against a gold file and a fake autonima run
    gold = review.parent / "gold.csv"
    gold.write_text("pmid,included\n11111111,1\n33333333,0\n44444444,1\n")
    run = review.parent / "autonima_run" / "outputs"
    run.mkdir(parents=True)
    (run / "abstract_screening_results.json").write_text(json.dumps({"screening_results": [
        {"study_id": "11111111", "decision": "included_abstract"},
        {"study_id": "22222222", "decision": "excluded_abstract"},
        {"study_id": "33333333", "decision": "excluded_abstract"}]}))
    (run / "fulltext_screening_results.json").write_text(json.dumps({"screening_results": [
        {"study_id": "11111111", "decision": "included_fulltext"}]}))
    assert compare.main([str(review), "--gold", str(gold), "--autonima", str(run.parent)]) == 0
    bench = last_json(capsys)
    assert bench["skills"]["not_retrieved_by_search"] == ["44444444"]
    assert bench["skills"]["final"]["tp"] == 1 and bench["skills"]["final"]["fp"] == 1
    assert bench["skills"]["final"]["recall"] == 0.5
    assert bench["autonima"]["final"]["precision"] == 1.0
    assert bench["agreement"]["final"]["n"] == 1

    # ---- editing a full-text criterion re-opens exactly the full-text decisions
    text = (review / "review.yaml").read_text().replace("Whole-brain coordinates reported",
                                                        "Whole-brain peak coordinates reported")
    (review / "review.yaml").write_text(text)
    rv = ledger.Review(review)
    assert rv.pending("abstract") == []
    assert rv.pending("fulltext") == ["11111111", "33333333"]


def last_json(capsys):
    """The last JSON document the CLI printed."""
    out = capsys.readouterr().out
    decoder, pos, last = json.JSONDecoder(), 0, None
    while (start := out.find("{", pos)) != -1:
        try:
            last, pos = decoder.raw_decode(out, start)
        except json.JSONDecodeError:
            pos = start + 1
    return last
