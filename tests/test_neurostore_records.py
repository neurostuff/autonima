"""retrieval.records.kind: neurostore, offline: a fake client stands in for the API."""

import csv
import json
from copy import deepcopy

import pytest
import yaml

from autonima.config import ConfigManager, ConfigurationError
from autonima.documents import NeuroStoreRecordSource, attach_documents, create_document_source
from autonima.documents.neurostore import RECORD_ANALYSES_FILENAME, RECORDS_DIRNAME
from autonima.execution import stage_hashes
from autonima.models.types import Study, StudyStatus
from autonima.neurostore.client import NeuroStoreClient, NeuroStoreError
from autonima.neurostore.features import FEATURES_FILENAME, feature_lines


def _point(xyz, space="MNI"):
    return {"coordinates": xyz, "space": space, "values": [{"kind": "z", "value": 4.2}]}


def _analysis(aid, name, points, role="result", order=0):
    return {"id": aid, "name": name, "description": None, "role": role, "order": order,
            "table_id": "T1", "metadata": {"table_id": "2"}, "points": points}


WHOLE_BRAIN = {
    "name": "Reappraise > Look", "contrast": "reappraise > look", "task": ["Emotion Regulation Task"],
    "spatial_scope": "whole_brain", "effect_kind": "contrast", "group_count": 1, "n": 24,
    "has_patients": False, "has_healthy": True, "medical_condition": [],
    "measure_type": "bold_response", "measure_family": "functional_bold",
    "assignment_structure": "single_group", "allocation": "not_applicable", "blinding": None,
    "coordinate_space": "MNI", "contrast_positive_facets": ["regulation:reappraise"],
    "contrast_negative_facets": ["valence:negative"],
}
ROI = {**WHOLE_BRAIN, "name": "Amygdala ROI", "contrast": "look > reappraise", "spatial_scope": "roi"}


def _paper(pmid, base, version, analyses, source="study_schema"):
    return {
        "listing": {"results": [{"id": base, "pmid": pmid, "versions": [
            {"id": version, "source": source, "has_coordinates": True, "updated_at": "2026-11-01"}]}]},
        "study": {"id": version, "source": source, "updated_at": "2026-11-01", "analyses": analyses},
    }


PAPERS = {
    # Two analyses with coordinates, both with features; an anchor without a result role.
    "111": _paper("111", "B1", "V1", [
        _analysis("A1", "Reappraise > Look", [_point([1, 2, 3]), _point([4, 5, 6])], order=0),
        _analysis("A2", "Amygdala ROI", [_point([-20, -4, -16])], order=1),
        _analysis("A0", "Seed", [_point([0, 0, 0])], role="anchor", order=2),
    ]),
    # Coordinates, but NeuroStore has no features for the paper.
    "222": _paper("222", "B2", "V2", [_analysis("C1", "Task > Rest", [_point([7, 8, 9])])]),
    # Coordinates from an older extraction; the features belong to the study_schema version.
    "333": _paper("333", "B3", "V3OLD", [_analysis("D1", "x > y", [_point([9, 9, 9])])], source="nuextract-v21"),
}
FEATURES = {
    "B1": {"study": "V1", "base_study": "B1", "analyses": [
        {"id": "A1", "name": "Reappraise > Look", "matched_by": {}, "features": WHOLE_BRAIN},
        {"id": "A2", "name": "Amygdala ROI", "matched_by": {}, "features": ROI},
        {"id": "A0", "name": "Seed", "matched_by": {}, "features": WHOLE_BRAIN},
    ]},
    "B3": {"study": "V3NEW", "base_study": "B3", "analyses": [
        {"id": "E1", "name": "x > y", "matched_by": {}, "features": WHOLE_BRAIN}]},
}


class FakeNeuroStore(NeuroStoreClient):
    def __init__(self, papers=PAPERS, features=FEATURES, fail_features=False):
        super().__init__("https://example.test/api")
        self.papers, self.features, self.fail_features = papers, features, fail_features
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(("GET", path))
        params = params or {}
        if path == "base-studies/":
            paper = self.papers.get(str(params.get("pmid")))
            return deepcopy(paper["listing"]) if paper else {"results": []}
        for paper in self.papers.values():
            if path == f"studies/{paper['study']['id']}":
                return deepcopy(paper["study"])
        return None

    def post(self, path, body):
        self.calls.append(("POST", path))
        assert path == "analysis-queries/" and body["query"] == {} and body["paginate"] is False
        if self.fail_features:
            raise NeuroStoreError("analysis-queries returned 500")
        results = [deepcopy(self.features[b]) for b in body["base_study_ids"] if b in self.features]
        return {"metadata": {"total_count": len(results), "analysis_count": 0}, "query": {}, "results": results}


def _study(pmid):
    return Study(pmid=pmid, title=f"Study {pmid}", abstract=f"Abstract of {pmid}.", authors=["A"],
                 journal="J", publication_date="2020", status=StudyStatus.INCLUDED_ABSTRACT)


def _source(tmp_path, client=None, fallback=None, **options):
    return NeuroStoreRecordSource(tmp_path / "run", options=options or None,
                                  client=client or FakeNeuroStore(), fallback=fallback)


def _unavailable(run):
    with open(run / "outputs" / "documents_unavailable.csv", newline="") as handle:
        return {row["pmid"]: row["reason"] for row in csv.DictReader(handle)}


def test_a_record_lists_each_analysis_with_its_features_under_autonima_ids(tmp_path):
    study = _study("111")
    source = _source(tmp_path)

    report = attach_documents([study], source, output_dir=tmp_path / "run")

    assert report.attached == 1
    # The anchor is not a result, so it is not an analysis of the study.
    assert [a.name for a in study.analyses] == ["Reappraise > Look", "Amygdala ROI"]
    assert [len(a.points) for a in study.analyses] == [2, 1]
    text = study.full_text
    assert text.startswith("NEUROSTORE RECORD")
    assert "Title: Study 111" in text and "Abstract: Abstract of 111." in text
    assert "Analysis 111_analysis_0: Reappraise > Look" in text
    assert "Model fit over: whole brain" in text and "Model fit over: region of interest" in text
    assert "Contrast: reappraise > look" in text
    assert "Participants: n = 24; healthy participants" in text
    assert "{{analysis:" not in text and "A1" not in text
    document = study.document
    assert document.kind == "neurostore" and document.provides_analyses
    assert document.analysis_keys == ["A1", "A2"]
    assert "Model fit over: region of interest" in document.analysis_documents[1]
    assert study.full_text_source == "document:neurostore"


def test_studies_without_usable_features_are_unavailable_with_their_reason(tmp_path):
    studies = [_study(p) for p in ("111", "222", "333", "999")]

    report = attach_documents(studies, _source(tmp_path), output_dir=tmp_path / "run")

    assert report.attached == 1
    assert _unavailable(tmp_path / "run") == {
        "222": "no_features",
        "333": "features_for_other_version",
        "999": "not_in_neurostore",
    }
    for study in studies[1:]:
        assert study.fulltext_available is False and study.analyses == []


def test_the_snapshots_are_reused_without_asking_neurostore_again(tmp_path):
    first = FakeNeuroStore()
    attach_documents([_study("111")], _source(tmp_path, first), output_dir=tmp_path / "run")
    again = FakeNeuroStore(papers={}, features={})  # a server that has forgotten everything

    study = _study("111")
    attach_documents([study], _source(tmp_path, again), output_dir=tmp_path / "run")

    assert again.calls == []
    assert [a.name for a in study.analyses] == ["Reappraise > Look", "Amygdala ROI"]
    outputs = tmp_path / "run" / "outputs"
    assert (outputs / RECORD_ANALYSES_FILENAME).is_file() and (outputs / FEATURES_FILENAME).is_file()


def test_a_failed_features_fetch_is_not_cached(tmp_path):
    report = attach_documents(
        [_study("111")], _source(tmp_path, FakeNeuroStore(fail_features=True)), output_dir=tmp_path / "run"
    )
    assert report.attached == 0
    assert _unavailable(tmp_path / "run") == {"111": "features_fetch_failed"}

    study = _study("111")
    report = attach_documents([study], _source(tmp_path), output_dir=tmp_path / "run")
    assert report.attached == 1 and len(study.analyses) == 2


def test_a_study_no_longer_ready_is_not_served_from_old_files(tmp_path):
    attach_documents([_study("111")], _source(tmp_path), output_dir=tmp_path / "run")
    records = tmp_path / "run" / RECORDS_DIRNAME
    assert (records / "111.md").is_file()
    features = json.loads((tmp_path / "run" / "outputs" / FEATURES_FILENAME).read_text())
    features["papers"]["B1"]["studies"] = {}
    (tmp_path / "run" / "outputs" / FEATURES_FILENAME).write_text(json.dumps(features))

    study = _study("111")
    report = attach_documents([study], _source(tmp_path), output_dir=tmp_path / "run")

    assert report.attached == 0 and not (records / "111.md").exists()


def test_the_record_does_not_change_between_runs(tmp_path):
    first, second = _study("111"), _study("111")
    attach_documents([first], _source(tmp_path), output_dir=tmp_path / "run")
    attach_documents([second], _source(tmp_path), output_dir=tmp_path / "run")
    assert first.full_text == second.full_text
    assert first.document.content_hash == second.document.content_hash


def test_a_silent_feature_is_left_out():
    lines = feature_lines({"contrast": "a > b", "spatial_scope": None, "n": None,
                           "has_patients": None, "has_healthy": None})
    assert lines == ["Contrast: a > b"]


def test_the_analysis_query_listing_is_checked_against_its_count():
    class Short(NeuroStoreClient):
        def post(self, path, body):
            return {"metadata": {"total_count": 2}, "results": [{"study": "V1", "base_study": "B1", "analyses": []}]}

    with pytest.raises(NeuroStoreError, match="listed 1 studies but reported 2"):
        Short("https://example.test/api").analysis_features(["B1", "B2"])


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "search": {"database": "pubmed", "query": "x", "email": "a@b.c"},
    "screening": {
        "abstract": {"objective": "o", "inclusion_criteria": ["i"], "exclusion_criteria": ["e"]},
        "fulltext": {"objective": "o", "inclusion_criteria": ["i"], "exclusion_criteria": ["e"]},
    },
    "parsing": {"parse_coordinates": False},
}


def _load(tmp_path, records, **sections):
    config = deepcopy(BASE_CONFIG) | {"retrieval": {"records": records}, "output": {"directory": str(tmp_path / "run")}}
    config.update(sections)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return ConfigManager().load_from_file(path)


def test_a_neurostore_records_config_needs_no_root(tmp_path):
    config = _load(tmp_path, {"enabled": True, "kind": "neurostore", "neurostore": {"sources": ["study_schema"]}})
    source = create_document_source(config.documents, output_dir=config.output.directory)
    assert isinstance(source, NeuroStoreRecordSource)
    assert source.options.sources == ("study_schema",)


@pytest.mark.parametrize("records, message", [
    ({"enabled": True, "kind": "neurostore", "root": "/tmp"}, "remove retrieval.records.root"),
    ({"enabled": True, "kind": "neurostore", "neurostore": {"sauces": []}}, "Unknown retrieval.records.neurostore key"),
    ({"enabled": True, "kind": "neurostore", "neurostore": {"sources": ["nope"]}}, "Unknown retrieval.records.neurostore source"),
    ({"enabled": True, "kind": "text", "root": "/tmp", "neurostore": {}}, "kind is not 'neurostore'"),
])
def test_neurostore_records_settings_are_checked(tmp_path, records, message):
    with pytest.raises(ConfigurationError, match=message):
        _load(tmp_path, records)


def test_neurostore_records_refuse_reparsing_the_article(tmp_path):
    with pytest.raises(ConfigurationError, match="parse_coordinates must be false"):
        _load(tmp_path, {"enabled": True, "kind": "neurostore"}, parsing={"parse_coordinates": True})


def test_the_version_choice_enters_the_retrieval_signature(tmp_path):
    default = stage_hashes(_load(tmp_path, {"enabled": True, "kind": "neurostore"}))
    narrowed = stage_hashes(_load(tmp_path, {"enabled": True, "kind": "neurostore",
                                             "neurostore": {"sources": ["study_schema"]}}))
    assert default["retrieval"] != narrowed["retrieval"]
    assert default["abstract"] == narrowed["abstract"]


# The shape NeuroStore serves since its analysis-feature release (2026-11-10), trimmed.
NESTED = {
    "spatial_scope": "roi", "coordinate_space": "MNI", "n": 24, "has_patients": False, "has_healthy": True,
    "species": ["human"], "measure_type": "bold_response", "measure_family": "functional_bold", "is_meta_analysis": False,
    "contrast": {"key": {"ids": ["x|regulation:reappraise|vs:y"]}, "kind": "contrast", "groups": "within",
                 "positive": {"name": ["Distance"], "facet": ["regulation:reappraise"]},
                 "negative": {"name": ["Attend"], "facet": ["y"]}},
    "task": {"ids": ["hedtsk_pavlovian_fear_conditioning"], "family_ids": ["c"], "text": ["Pavlovian Fear Conditioning Task"]},
    "tasks": {"stimuli": ["Negative IAPS pictures"], "design_type": ["event_related"], "instructions": ["Distance or attend."]},
    "groups": {"age_mean": [24.8], "female_count": 24, "male_count": 0},
    "details": {"connectivity_method": "gppi", "seed_regions": ["left vlPFC"]},
    "regions": {"name": ["midbrain ROI"], "definition_method": ["prior_literature"]},
    "inference_settings": {"multiple_comparison_method": ["FWE small-volume correction"], "correction_scope": "roi"},
    "coordinate_sets": {"role": ["result"], "result_kind": ["connectivity"]},
    "medical_condition": {"ids": ["MONDO:1", "MONDO:2"], "text": ["PTSD", "depression"],
                          "role_ids": ["primary|MONDO:1", "comorbid|MONDO:2"]},
    "design": {"allocation": "not_applicable", "assignment_structure": "single_group"},
    "acquisitions": {"modality": ["fmri"], "repetition_time_seconds": [2.0]},
}


def test_nested_features_read_as_words_never_as_dicts():
    lines = feature_lines(NESTED)
    text = "\n".join(lines)
    assert "Contrast: Distance > Attend [contrast, within-group]" in lines
    assert "Result: connectivity" in lines
    assert "Model fit over: region of interest" in lines
    assert "Inference restricted to: region of interest" in lines
    assert "Thresholding: FWE small-volume correction" in lines
    assert "Regions: midbrain ROI (defined by prior literature)" in lines
    assert "Medical condition: primary: PTSD; comorbid: depression" in lines
    assert "connectivity method gppi" in text and "Pavlovian Fear Conditioning Task" in text
    assert "{" not in text and "ids" not in text
    assert "repetition" not in text  # acquisition details are left out


def test_lines_every_analysis_shares_are_printed_once():
    from autonima.neurostore.features import analysis_lines, render_record

    shared = analysis_lines(NESTED, 3, "Table 1")
    other = analysis_lines({**NESTED, "spatial_scope": "whole_brain"}, 5, None)
    text = render_record("T", "A", [("A1", "first", shared), ("A2", "second", other)])
    assert text.count("Participants: n = 24") == 1
    assert "Shared by all analyses:\n" in text
    assert "Analysis A2: second\n  Model fit over: whole brain" in text
    assert text.count("Reported peaks") == 2  # never shared


def test_model_fit_and_inference_scope_are_told_apart():
    from autonima.neurostore.features import analysis_lines, render_record

    # Whole-brain estimation, inference within ROIs only: a small-volume correction.
    lines = analysis_lines({**NESTED, "spatial_scope": "whole_brain"}, 3, None)
    text = render_record("T", "A", [("A1", "first", lines)])
    assert "Model fit over: whole brain" in lines
    assert "Inference restricted to: region of interest" in lines
    assert "where significance was tested" in text
    # Nothing tells the judge which of the two a criterion means; the criteria decide.
    assert "does not make" not in text and "Spatial scope" not in text
    whole = feature_lines({**NESTED, "inference_settings": {"correction_scope": "whole_brain"}})
    assert "Inference over: whole brain" in whole


# ---------------------------------------------------------------------------
# full-text fallback (retrieval.records.fallback: fulltext)
# ---------------------------------------------------------------------------

def _with_article(study, tmp_path, text="Methods. We scanned 20 adults. Table 2 lists task > rest."):
    """The study as article retrieval leaves it: its text on disk and marked available."""
    path = tmp_path / "articles" / f"{study.pmid}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    study.full_text_path, study.full_text_source = str(path), "pubget"
    study.full_text_output_dir, study.fulltext_available = str(tmp_path / "run"), True
    return study


def test_the_fallback_reviews_a_study_without_features_from_its_article(tmp_path):
    with_article = _with_article(_study("222"), tmp_path)
    no_article = _study("333")
    not_in_neurostore = _with_article(_study("999"), tmp_path)

    report = attach_documents([with_article, no_article, not_in_neurostore],
                              _source(tmp_path, fallback="fulltext"), output_dir=tmp_path / "run")

    assert report.attached == 1
    text = with_article.full_text
    assert text.startswith("NEUROSTORE RECORD WITH THE ARTICLE'S FULL TEXT")
    assert "Analysis 222_analysis_0: Task > Rest" in text and "Reported peaks: 1 (Table 2)" in text
    assert text.rstrip().endswith("Table 2 lists task > rest.")
    assert "No features were extracted" not in text
    # The analyses, and so the exported coordinates, are still NeuroStore's.
    assert [a.name for a in with_article.analyses] == ["Task > Rest"]
    assert with_article.analyses[0].points[0].coordinates == [7, 8, 9]
    payload = json.loads((tmp_path / "run" / RECORDS_DIRNAME / "222.analyses.json").read_text())
    assert payload["source"]["input"] == "fulltext" and payload["source"]["article_source"] == "pubget"
    # No article: still unavailable, saying the fallback found none. Not in NeuroStore: no
    # analyses to select among, so the article does not help.
    with open(tmp_path / "run" / "outputs" / "documents_unavailable.csv", newline="") as handle:
        rows = {row["pmid"]: row for row in csv.DictReader(handle)}
    assert rows["333"]["reason"] == "features_for_other_version"
    assert "article not retrieved" in rows["333"]["detail"]
    assert rows["999"]["reason"] == "not_in_neurostore"


def test_the_fallback_leaves_feature_records_byte_for_byte(tmp_path):
    off, on = _study("111"), _study("111")
    attach_documents([off], _source(tmp_path / "off"), output_dir=tmp_path / "off" / "run")
    attach_documents([on], _source(tmp_path / "on", fallback="fulltext"), output_dir=tmp_path / "on" / "run")
    assert off.full_text == on.full_text
    assert off.document.content_hash == on.document.content_hash
    assert off.document.analyses_hash == on.document.analyses_hash


def test_fallback_candidates_are_the_studies_with_coordinates_but_no_features(tmp_path):
    pmids = [111, 222, 333, 999]
    assert _source(tmp_path, fallback="fulltext").fallback_candidates(pmids) == [222, 333]
    assert _source(tmp_path).fallback_candidates(pmids) == []


@pytest.mark.parametrize("records, message", [
    ({"enabled": True, "kind": "neurostore", "fallback": "abstract"}, "fallback must be one of"),
    ({"enabled": True, "kind": "text", "root": "/tmp", "fallback": "fulltext"}, "kind is not 'neurostore'"),
])
def test_the_fallback_setting_is_checked(tmp_path, records, message):
    with pytest.raises(ConfigurationError, match=message):
        _load(tmp_path, records)


def test_the_fallback_needs_somewhere_to_fetch_articles_from(tmp_path):
    with pytest.raises(ConfigurationError, match="no sources"):
        _load(tmp_path, {"enabled": True, "kind": "neurostore", "fallback": "fulltext"},
              retrieval={"sources": [], "records": {"enabled": True, "kind": "neurostore", "fallback": "fulltext"}})


def test_the_fallback_enters_the_retrieval_signature_only_when_on(tmp_path):
    off = _load(tmp_path, {"enabled": True, "kind": "neurostore"})
    on = _load(tmp_path, {"enabled": True, "kind": "neurostore", "fallback": "fulltext"})
    assert "fallback" not in off.to_dict()["retrieval"]["records"]
    assert on.to_dict()["retrieval"]["records"]["fallback"] == "fulltext"
    off_hashes, on_hashes = stage_hashes(off), stage_hashes(on)
    assert off_hashes["retrieval"] != on_hashes["retrieval"]
    # Feature records are unchanged, so screening's cache stays valid for them.
    assert off_hashes["fulltext"] == on_hashes["fulltext"]


def test_the_pipeline_retrieves_articles_only_for_the_fallback_studies(tmp_path, monkeypatch):
    import asyncio

    from autonima.config import get_sample_config_text
    from autonima.pipeline import AutonimaPipeline

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = yaml.safe_load(get_sample_config_text())
    raw["output"]["directory"] = str(tmp_path / "run")
    raw["retrieval"]["records"]["fallback"] = "fulltext"
    pipeline = AutonimaPipeline(ConfigManager().load_from_dict(raw))
    pipeline._document_source = NeuroStoreRecordSource(
        tmp_path / "run", client=FakeNeuroStore(), fallback="fulltext"
    )
    pipeline.results.studies = [_study(p) for p in ("111", "222", "333")]
    asked = []

    async def retrieve_articles(studies, pmids, load_excluded):
        asked.extend(sorted(s.pmid for s in studies))
        _with_article(next(s for s in studies if s.pmid == "222"), tmp_path)
        return []

    pipeline._retrieve_articles = retrieve_articles
    asyncio.run(pipeline._execute_retrieval_phase())

    assert asked == ["222", "333"]
    available = {s.pmid: s.fulltext_available for s in pipeline.results.studies}
    assert available == {"111": True, "222": True, "333": False}
    stats = pipeline.results.execution_stats["retrieval"]
    assert stats["documents_fallback_candidates"] == 2 and stats["documents_from_full_text"] == 1
