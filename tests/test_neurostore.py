"""NeuroStore search and coordinates, offline: a fake client stands in for the API."""

import asyncio
import json
from copy import deepcopy

import pytest
import yaml

from autonima.config import ConfigManager, ConfigurationError
from autonima.execution import stage_hashes
from autonima.models.types import ActivationTable, Study, StudyStatus
from autonima.neurostore.analyses import (
    SNAPSHOT_FILENAME,
    NeuroStoreOptions,
    NeuroStoreSnapshot,
    fetch_record,
    to_analyses,
)
from autonima.neurostore.client import NeuroStoreClient, NeuroStoreError
from autonima.neurostore.query import QueryTranslationError, translate_pubmed_query
from autonima.pipeline import AutonimaPipeline
from autonima.search.neurostore import NeuroStoreSearch
from autonima.config import get_sample_config_text


# ---------------------------------------------------------------------------
# query translation
# ---------------------------------------------------------------------------

def test_text_tags_are_removed_and_phrases_kept():
    t = translate_pubmed_query('"emotion regulation"[tiab] OR reappraisal[Title/Abstract]')
    assert t.query == '"emotion regulation" OR reappraisal'
    assert t.dropped == [] and t.approximated == []


def test_anded_mesh_is_dropped_but_ored_mesh_becomes_a_phrase():
    t = translate_pubmed_query(
        '(fMRI[tiab] OR "Magnetic Resonance Imaging, Functional"[mh]) AND Humans[mh]'
    )
    # NeuroStore matching ignores case; the heading keeps its own.
    assert t.query.lower() == 'fmri or "functional magnetic resonance imaging"'
    assert t.dropped == ["Humans[mh]"]
    assert t.approximated == ['"Magnetic Resonance Imaging, Functional"[mh]']


def test_publication_type_language_and_dates_are_dropped():
    t = translate_pubmed_query(
        '(reappraisal[tiab] AND ("2000/01/01"[Date - Publication] : "2019/12/31"[Date - Publication]))'
        ' AND English[lang] NOT (Review[pt] OR Comment[pt])'
    )
    assert t.query == "reappraisal"
    assert "English[lang]" in t.dropped and "Review[pt]" in t.dropped
    assert any("Date - Publication" in d for d in t.dropped)


def test_not_keeps_the_part_that_can_be_searched():
    t = translate_pubmed_query('PTSD[tiab] NOT (review[pt] OR "systematic review"[tiab])')
    assert t.query == 'PTSD NOT "systematic review"'


def test_truncation_becomes_word_endings():
    t = translate_pubmed_query("densit*[tiab]")
    assert t.query.startswith("densit OR ")
    assert "density" in t.query
    assert t.approximated == ["densit*[tiab]"]
    phrase = translate_pubmed_query('"drug cue*"[tiab]')
    assert '"drug cue"' in phrase.query and '"drug cuey"' in phrase.query


def test_untagged_words_bind_tighter_than_or():
    t = translate_pubmed_query("frontotemporal dementia OR bvFTD")
    assert t.query == "(frontotemporal dementia) OR bvFTD"


def test_tagged_words_run_together_are_a_phrase():
    assert translate_pubmed_query("cue reactivity[tiab]").query == '"cue reactivity"'


def test_booleans_evaluate_left_to_right_like_pubmed():
    # PubMed reads A OR B AND C as (A OR B) AND C; the output says so explicitly.
    assert translate_pubmed_query("a OR b AND c").query == "(a OR b) AND c"


def test_surplus_and_missing_parentheses_are_tolerated_and_reported():
    t = translate_pubmed_query("(fMRI[tiab] AND PTSD[tiab]))")
    assert t.query == "fMRI AND PTSD"
    assert any("unmatched ')'" in a for a in t.approximated)
    t = translate_pubmed_query("((fMRI[tiab] AND PTSD[tiab])")
    assert t.query == "fMRI AND PTSD"
    assert any("unclosed '('" in a for a in t.approximated)


def test_a_query_of_only_filters_cannot_be_translated():
    with pytest.raises(QueryTranslationError):
        translate_pubmed_query("Humans[mh] AND English[lang]")


def test_single_quoted_phrases():
    assert translate_pubmed_query("'social cognition'[Title/Abstract]").query == '"social cognition"'


# ---------------------------------------------------------------------------
# a fake NeuroStore
# ---------------------------------------------------------------------------

def _point(xyz, space="MNI"):
    return {"coordinates": xyz, "space": space, "values": [{"kind": "z", "value": 4.2}]}


def _analysis(aid, name, points, role="result", table_id="T1", order=0):
    return {
        "id": aid, "name": name, "description": None, "role": role, "order": order,
        "table_id": table_id, "metadata": {"table_id": "2"}, "points": points,
    }


class FakeClient(NeuroStoreClient):
    """Routes get() to canned responses; records every call."""

    def __init__(self, routes):
        super().__init__("https://example.test/api")
        self.routes = routes
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        handler = self.routes.get(path)
        if handler is None:
            return None
        return handler(params or {}) if callable(handler) else deepcopy(handler)


VERSIONS = {
    "results": [{
        "id": "BASE1", "pmid": "111",
        "versions": [
            {"id": "V_NS", "source": "neurosynth", "has_coordinates": True, "updated_at": "2024-01-01"},
            {"id": "V_SCHEMA", "source": "study_schema", "has_coordinates": True, "updated_at": "2026-10-31"},
            {"id": "V_EMPTY", "source": "llm", "has_coordinates": False},
        ],
    }]
}
STUDY_SCHEMA = {
    "id": "V_SCHEMA", "source": "study_schema", "updated_at": "2026-10-31",
    "analyses": [
        _analysis("A2", "Patients > Controls", [_point([1, 2, 3]), _point([4, 5, 6], "TAL")], order=1),
        _analysis("A1", "Seed", [_point([0, 0, 0])], role="anchor", order=0),
        _analysis("A3", "Task > Rest", [_point([7, 8, 9])], role=None, table_id=None, order=2),
    ],
}


def _routes():
    return {
        "base-studies/": lambda p: VERSIONS if p.get("pmid") == "111" else {"results": []},
        "studies/V_SCHEMA": STUDY_SCHEMA,
        "studies/V_NS": {"id": "V_NS", "source": "neurosynth", "analyses": [_analysis("N1", "x", [_point([9, 9, 9])])]},
    }


def test_the_first_configured_source_with_coordinates_is_used():
    record = fetch_record(FakeClient(_routes()), "111", NeuroStoreOptions())
    assert record["source"] == "study_schema" and record["version_id"] == "V_SCHEMA"
    # The anchor analysis is not a finding; an analysis with no role counts as a result.
    assert [a["name"] for a in record["analyses"]] == ["Patients > Controls", "Task > Rest"]
    assert record["analyses"][0]["points"][1]["space"] == "TAL"
    assert record["analyses"][0]["points"][0]["values"] == [{"value": 4.2, "kind": "z-statistic"}]


def test_source_order_is_configurable():
    record = fetch_record(FakeClient(_routes()), "111", NeuroStoreOptions(sources=("neurosynth", "study_schema")))
    assert record["source"] == "neurosynth"


def test_a_study_not_in_neurostore_is_recorded_as_unavailable():
    record = fetch_record(FakeClient(_routes()), "999", NeuroStoreOptions())
    assert record["unavailable"] == "not_in_neurostore"


def test_only_unconfigured_sources_is_a_distinct_reason():
    record = fetch_record(FakeClient(_routes()), "111", NeuroStoreOptions(sources=("pubget",)))
    assert record["unavailable"] == "no_coordinates_from_configured_sources"
    assert record["sources_seen"] == ["neurosynth", "study_schema"]


def test_records_become_analyses_with_a_table_entry_for_each_table():
    record = fetch_record(FakeClient(_routes()), "111", NeuroStoreOptions())
    analyses, tables = to_analyses(record)
    assert [a.table_id for a in analyses] == ["neurostore-T1", None]
    assert [(t.table_id, t.table_label) for t in tables] == [("neurostore-T1", "Table 2")]
    assert analyses[0].points[0].coordinates == [1.0, 2.0, 3.0]


def test_the_snapshot_is_reused_and_options_change_invalidates_it(tmp_path):
    path = tmp_path / SNAPSHOT_FILENAME
    client = FakeClient(_routes())
    first = NeuroStoreSnapshot(path, NeuroStoreOptions(), client.base_url)
    assert first.ensure(client, ["111", "999"], workers=1) == {"fetched": 2, "reused": 0, "failed": 0}
    calls = len(client.calls)
    again = NeuroStoreSnapshot(path, NeuroStoreOptions(), client.base_url)
    assert again.ensure(client, ["111", "999"], workers=1) == {"fetched": 0, "reused": 2, "failed": 0}
    assert len(client.calls) == calls
    other = NeuroStoreSnapshot(path, NeuroStoreOptions(roles=("result", "anchor")), client.base_url)
    assert other.records == {}


def test_a_failed_fetch_is_not_cached(tmp_path):
    class Failing(FakeClient):
        def get(self, path, params=None):
            raise NeuroStoreError("down")

    snapshot = NeuroStoreSnapshot(tmp_path / SNAPSHOT_FILENAME, NeuroStoreOptions(), "x")
    assert snapshot.ensure(Failing({}), ["111"], workers=1)["failed"] == 1
    assert snapshot.get("111") is None


# ---------------------------------------------------------------------------
# client paging
# ---------------------------------------------------------------------------

class PagedClient(NeuroStoreClient):
    def __init__(self, pages, totals):
        super().__init__("https://example.test/api")
        self.pages, self.totals = pages, totals

    def get(self, path, params=None):
        page = params["page"]
        return {"metadata": {"total_count": self.totals[page - 1]}, "results": self.pages[page - 1]}


def test_search_pages_to_the_reported_total():
    client = PagedClient([[{"pmid": "1"}, {"pmid": "2"}], [{"pmid": "3"}]], [3, 3])
    assert [r["pmid"] for r in client.search_base_studies("x", page_size=2)] == ["1", "2", "3"]


def test_a_total_that_changes_while_paging_fails():
    client = PagedClient([[{"pmid": "1"}, {"pmid": "2"}], [{"pmid": "3"}]], [3, 4])
    with pytest.raises(NeuroStoreError, match="changed while paging"):
        list(client.search_base_studies("x", page_size=2))


def test_a_short_result_set_fails():
    client = PagedClient([[{"pmid": "1"}], []], [3, 3])
    with pytest.raises(NeuroStoreError, match="retrieved 1"):
        list(client.search_base_studies("x", page_size=1))


# ---------------------------------------------------------------------------
# search engine
# ---------------------------------------------------------------------------

def _config(tmp_path, **sections):
    config = yaml.safe_load(get_sample_config_text())
    config["output"]["directory"] = str(tmp_path / "run")
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    return ConfigManager().load_from_dict(deepcopy(config))


def test_search_translates_the_query_and_logs_what_it_lost(tmp_path, monkeypatch):
    config = _config(tmp_path, search={
        "database": "neurostore", "query": 'reappraisal[tiab] AND Humans[mh]',
        "date_from": "2000/01/01", "date_to": "2019/12/31", "pmids_list": None,
    })
    seen = {}

    class SearchClient(FakeClient):
        def search_base_studies(self, search, **kwargs):
            seen.update(search=search, **kwargs)
            return iter([{"id": "B1", "pmid": "111"}, {"id": "B2", "pmid": None, "doi": "10.1/x"},
                         {"id": "B3", "pmid": "111"}])

    engine = NeuroStoreSearch(config.search, output_dir=config.output.directory, client=SearchClient({}))
    pmids = asyncio.run(engine._execute_search(config.search.query))
    assert pmids == ["111"]
    assert seen["search"] == "reappraisal" and seen["year_min"] == 2000 and seen["year_max"] == 2019
    assert seen["data_type"] == "coordinate"
    log = json.loads((tmp_path / "run" / "outputs" / "neurostore_search_log.json").read_text())
    assert log["translation"]["dropped"] == ["Humans[mh]"]
    assert log["without_pmid"] == [{"id": "B2", "doi": "10.1/x", "name": None}]


def test_a_native_query_is_used_as_written(tmp_path):
    config = _config(tmp_path, search={
        "database": "neurostore", "query": "", "pmids_list": None,
        "neurostore": {"query": '"emotion regulation" -review', "search_full_text": True},
    })
    seen = {}

    class SearchClient(FakeClient):
        def search_base_studies(self, search, **kwargs):
            seen.update(search=search, **kwargs)
            return iter([])

    engine = NeuroStoreSearch(config.search, output_dir=config.output.directory, client=SearchClient({}))
    asyncio.run(engine._execute_search(""))
    assert seen["search"] == '"emotion regulation" -review' and seen["search_full_text"] is True


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_unknown_neurostore_keys_are_errors(tmp_path):
    with pytest.raises(ConfigurationError, match="Unknown search.neurostore"):
        _config(tmp_path, search={"database": "neurostore", "neurostore": {"querry": "x"}})
    with pytest.raises(ConfigurationError, match="Unknown parsing.neurostore source"):
        _config(tmp_path, parsing={"parse_coordinates": True, "coordinate_source": "neurostore",
                                   "neurostore": {"sources": ["study_schma"]}})


def test_neurostore_options_need_neurostore_in_use(tmp_path):
    with pytest.raises(ConfigurationError, match="search.database is not 'neurostore'"):
        _config(tmp_path, search={"neurostore": {"search_full_text": True}})
    with pytest.raises(ConfigurationError, match="coordinate_source is not 'neurostore'"):
        _config(tmp_path, parsing={"neurostore": {"roles": ["result"]}})


def test_configs_without_neurostore_keep_their_stage_hashes(tmp_path):
    base = _config(tmp_path)
    hashes = stage_hashes(base)
    serialized = base.to_dict()
    assert "neurostore" not in serialized["search"]
    assert "coordinate_source" not in serialized["parsing"] and "neurostore" not in serialized["parsing"]
    explicit = _config(tmp_path, parsing={"coordinate_source": "tables"})
    assert stage_hashes(explicit) == hashes


def test_neurostore_coordinates_change_only_the_parsing_hash(tmp_path):
    base = stage_hashes(_config(tmp_path, parsing={"parse_coordinates": True}))
    ns = stage_hashes(_config(tmp_path, parsing={"parse_coordinates": True, "coordinate_source": "neurostore"}))
    assert {k for k in base if base[k] != ns[k]} == {"parsing"}


# ---------------------------------------------------------------------------
# pipeline stage
# ---------------------------------------------------------------------------

def test_the_parsing_stage_attaches_neurostore_analyses(tmp_path, monkeypatch):
    config = _config(tmp_path, parsing={"parse_coordinates": True, "coordinate_source": "neurostore"})
    pipeline = AutonimaPipeline(config)
    pipeline.results.studies = [
        Study(pmid=pmid, title="t", abstract="a", authors=[], journal="j", publication_date="2020",
              status=StudyStatus.INCLUDED_FULLTEXT,
              activation_tables=[ActivationTable(table_id="tbl1", table_label="Table 1"),
                                 # Older retrievals left some tables without an id.
                                 ActivationTable(table_id=None, table_label="Table 2")])
        for pmid in ("111", "999")
    ]
    monkeypatch.setattr(
        "autonima.neurostore.client.NeuroStoreClient",
        lambda base_url: FakeClient(_routes()),
    )
    asyncio.run(pipeline._execute_coordinate_parsing())

    found, missing = pipeline.results.studies
    assert [a.name for a in found.analyses] == ["Patients > Controls", "Task > Rest"]
    # The article's own tables stay; NeuroStore's are added beside them.
    assert [t.table_id for t in found.activation_tables] == ["tbl1", None, "neurostore-T1"]
    assert missing.analyses == []
    stats = pipeline.results.execution_stats["coordinate_parsing"]
    assert stats["studies_with_analyses"] == 1 and stats["unavailable"] == {"not_in_neurostore": 1}
    outputs = tmp_path / "run" / "outputs"
    assert "999,not_in_neurostore" in (outputs / "neurostore_unavailable.csv").read_text()
    assert json.loads((outputs / SNAPSHOT_FILENAME).read_text())["studies"]["111"]["version_id"] == "V_SCHEMA"

    # A second run reads the snapshot and does not accumulate tables.
    asyncio.run(pipeline._execute_coordinate_parsing())
    assert [t.table_id for t in found.activation_tables] == ["tbl1", None, "neurostore-T1"]
    assert pipeline.results.execution_stats["coordinate_parsing"]["studies_reused"] == 2


# ---------------------------------------------------------------------------
# cache: the snapshot travels and is invalidated with the parsing stage
# ---------------------------------------------------------------------------

def _ns_config(directory, roles=None):
    from autonima.config import ConfigManager as Manager

    config = yaml.safe_load(get_sample_config_text())
    config["output"]["directory"] = str(directory)
    config["parsing"] = {"parse_coordinates": True, "coordinate_source": "neurostore",
                         **({"neurostore": {"roles": roles}} if roles else {})}
    return Manager().load_from_dict(config)


def _signed(path, stage, stage_hash):
    from autonima.execution import CACHE_SCHEMA_VERSION

    path.write_text(json.dumps({"studies": [], "cache_signature": {
        "schema_version": CACHE_SCHEMA_VERSION, "stage": stage, "stage_hash": stage_hash}}))


def test_the_snapshot_is_copied_from_a_seed_run_and_cleared_when_options_change(tmp_path):
    from autonima.execution import prepare_execution

    seed = tmp_path / "seed"
    config = _ns_config(seed)
    prepare_execution(config, seed)
    _signed(seed / "outputs" / "coordinate_parsing_results.json", "parsing", stage_hashes(config)["parsing"])
    (seed / "outputs" / SNAPSHOT_FILENAME).write_text("{}")

    arm = tmp_path / "arm"
    prepare_execution(_ns_config(arm), arm, copy_valid_cache_from=str(seed))
    assert (arm / "outputs" / SNAPSHOT_FILENAME).exists()

    prepare_execution(_ns_config(arm, roles=["result", "anchor"]), arm)
    assert not (arm / "outputs" / SNAPSHOT_FILENAME).exists()


def test_a_snapshot_left_by_an_interrupted_stage_does_not_block_the_rerun(tmp_path):
    from autonima.execution import prepare_execution

    run = tmp_path / "run"
    prepare_execution(_ns_config(run), run)
    # The parsing stage died after fetching: a snapshot but no coordinate_parsing_results.json.
    (run / "outputs" / SNAPSHOT_FILENAME).write_text("{}")
    prepare_execution(_ns_config(run), run)  # must not raise UnsupportedCacheError


def test_top_level_not_clauses_are_split_out():
    t = translate_pubmed_query('(PTSD[tiab] AND VBM[tiab]) NOT (review[pt] OR "systematic review"[tiab]) NOT case[ti]')
    assert t.query == '((PTSD AND VBM) NOT "systematic review") NOT case'
    assert t.include == "PTSD AND VBM"
    assert t.exclude == ['"systematic review"', "case"]


def test_full_text_search_applies_not_to_titles_and_abstracts_only(tmp_path):
    config = _config(tmp_path, search={
        "database": "neurostore", "query": 'PTSD[tiab] NOT "systematic review"[tiab]', "pmids_list": None,
        "neurostore": {"search_full_text": True},
    })
    calls = []

    class SearchClient(FakeClient):
        def search_base_studies(self, search, search_full_text=False, **kwargs):
            calls.append((search, search_full_text))
            if search == "PTSD":
                # B2 only mentions a systematic review in its text; B3 is one.
                return iter([{"id": "B1", "pmid": "1"}, {"id": "B2", "pmid": "2"}, {"id": "B3", "pmid": "3"}])
            return iter([{"id": "B3", "pmid": "3"}])

    engine = NeuroStoreSearch(config.search, output_dir=config.output.directory, client=SearchClient({}))
    assert asyncio.run(engine._execute_search(config.search.query)) == ["1", "2"]
    assert calls == [("PTSD", True), ('"systematic review"', False)]
