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


def _source(tmp_path, client=None, **options):
    return NeuroStoreRecordSource(tmp_path / "run", options=options or None, client=client or FakeNeuroStore())


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
    assert "Spatial scope: whole brain" in text and "Spatial scope: region of interest" in text
    assert "Contrast: reappraise > look" in text
    assert "Participants: n = 24; healthy participants" in text
    assert "{{analysis:" not in text and "A1" not in text
    document = study.document
    assert document.kind == "neurostore" and document.provides_analyses
    assert document.analysis_keys == ["A1", "A2"]
    assert "Spatial scope: region of interest" in document.analysis_documents[1]
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
