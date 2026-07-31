import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from autonima.retrieval import utils


def _install_fake_ace(monkeypatch, calls):
    def extract_and_export(files, output_dir, **kwargs):
        calls.append([str(path) for path in files])
        output_dir = Path(output_dir)
        raw_dir = output_dir / "tables" / "123"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "1.html"
        raw_path.write_text(
            "<table><tr><th>x</th><th>y</th><th>z</th></tr>"
            "<tr><td>0</td><td>0</td><td>10</td></tr></table>",
            encoding="utf-8",
        )
        pd.DataFrame(columns=[
            "pmid", "table_id", "table_label", "x", "y", "z",
            "p_value", "region", "size", "statistic", "groups",
        ]).to_csv(output_dir / "coordinates.csv", index=False)
        pd.DataFrame([{
            "pmid": 123,
            "table_id": 1,
            "table_label": "Table 1",
            "table_caption": "Coordinates",
            "table_foot": "",
            "n_header_rows": 1,
            "table_raw_file": "tables/123/1.html",
        }]).to_csv(output_dir / "tables.csv", index=False)
        return {
            "articles": 1,
            "tables": 1,
            "coordinates": 0,
            "missing_sources": [],
        }

    def classify_coordinate_table(**kwargs):
        return SimpleNamespace(candidate=True, reasons=("xyz_headers",))

    fake_ace = SimpleNamespace(__version__="test")
    monkeypatch.setattr(
        utils,
        "_load_ace_api",
        lambda: (
            fake_ace,
            extract_and_export,
            "test-gate",
            classify_coordinate_table,
        ),
    )


def test_html_source_without_processed_path_runs_ace_and_reuses_cache(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "html"
    source_dir.mkdir()
    article_path = source_dir / "123.html"
    article_path.write_text("<html>full text</html>", encoding="utf-8")
    source_before = article_path.read_bytes()
    generated_path = tmp_path / "run" / "retrieval" / "ace" / "source"
    calls = []
    _install_fake_ace(monkeypatch, calls)

    kwargs = {
        "root_path": source_dir,
        "pmid_source": "file_name",
        "allowed_extensions": [".html"],
        "pmids_to_include": {123},
        "generated_processed_data_path": generated_path,
    }
    text_paths, analyses, tables = utils._map_pmids_to_text(**kwargs)
    cached = utils._map_pmids_to_text(**kwargs)

    assert text_paths[123] == article_path
    assert analyses == {}
    assert len(tables[123]) == 1
    assert tables[123][0]["table_raw_path"].endswith("tables/123/1.html")
    assert len(calls) == 1
    assert cached[2][123][0]["table_id"] == "1"
    assert article_path.read_bytes() == source_before
    assert (generated_path / "ace_manifest.json").exists()


def test_incomplete_configured_export_uses_run_managed_export(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "html"
    source_dir.mkdir()
    (source_dir / "123.html").write_text("<html/>", encoding="utf-8")
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "coordinates.csv").write_text("pmid,x,y,z\n")
    generated = tmp_path / "run" / "ace"
    calls = []
    _install_fake_ace(monkeypatch, calls)

    _, _, tables = utils._map_pmids_to_text(
        root_path=source_dir,
        pmid_source="file_name",
        allowed_extensions=[".html"],
        processed_data_path=str(configured),
        generated_processed_data_path=generated,
        pmids_to_include={123},
    )

    assert len(calls) == 1
    assert 123 in tables
    assert not (configured / "tables.csv").exists()


def test_html_source_without_matching_pmids_does_not_invoke_ace(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "html"
    source_dir.mkdir()
    (source_dir / "123.html").write_text("<html/>", encoding="utf-8")
    calls = []
    _install_fake_ace(monkeypatch, calls)

    result = utils._map_pmids_to_text(
        root_path=source_dir,
        pmid_source="file_name",
        allowed_extensions=[".html"],
        generated_processed_data_path=tmp_path / "run" / "ace",
        pmids_to_include={456},
    )

    assert result == ({}, {}, {})
    assert calls == []


def test_elsevier_sibling_candidate_is_added(tmp_path, monkeypatch):
    article_dir = tmp_path / "456"
    table_dir = article_dir / "tables"
    table_dir.mkdir(parents=True)
    (article_dir / "text.txt").write_text("article", encoding="utf-8")
    (article_dir / "coordinates.json").write_text(
        json.dumps({"studyset": {"studies": []}}),
        encoding="utf-8",
    )
    missed_table = table_dir / "02_t0010.csv"
    missed_table.write_text(
        "Region,x,y,z\nInsula,-32,18,4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils,
        "_is_coordinate_candidate_path",
        lambda path: path == missed_table,
    )

    _, _, tables = utils._map_pmids_to_text(
        root_path=tmp_path,
        pmid_source="folder_name",
        text_path_templates=["text.txt"],
        coordinates_path_templates=["coordinates.json"],
        pmids_to_include={456},
    )

    assert tables[456][0]["table_id"] == "02_t0010"
    assert tables[456][0]["table_data_path"] == str(missed_table.resolve())


def test_elsevier_sibling_candidate_deduplicates_prefixed_table_id(
    tmp_path,
    monkeypatch,
):
    article_dir = tmp_path / "456"
    table_dir = article_dir / "tables"
    table_dir.mkdir(parents=True)
    (article_dir / "text.txt").write_text("article", encoding="utf-8")
    (article_dir / "coordinates.json").write_text(json.dumps({
        "studyset": {
            "studies": [{
                "analyses": [{
                    "name": "analysis",
                    "points": [{"coordinates": [1, 2, 3]}],
                    "metadata": {
                        "table_id": "t0010",
                        "table_label": "Table 2",
                        "raw_table_xml": (
                            "<table><tr><th>x</th><th>y</th><th>z</th></tr>"
                            "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
                        ),
                    },
                }],
            }],
        },
    }), encoding="utf-8")
    duplicate = table_dir / "02_t0010.csv"
    duplicate.write_text("x,y,z\n1,2,3\n", encoding="utf-8")
    monkeypatch.setattr(
        utils,
        "_is_coordinate_candidate_path",
        lambda path: path == duplicate,
    )

    _, _, tables = utils._map_pmids_to_text(
        root_path=tmp_path,
        pmid_source="folder_name",
        text_path_templates=["text.txt"],
        coordinates_path_templates=["coordinates.json"],
        pmids_to_include={456},
    )

    assert [table["table_id"] for table in tables[456]] == ["t0010"]


def test_pubget_gate_adds_rejected_candidate(tmp_path, monkeypatch):
    raw_dir = tmp_path / "tables"
    raw_dir.mkdir()
    selected = raw_dir / "selected.csv"
    missed = raw_dir / "missed.csv"
    selected.write_text("x,y,z\n1,2,3\n", encoding="utf-8")
    missed.write_text("x,y,z\n-32,18,4\n", encoding="utf-8")
    pd.DataFrame([{
        "pmcid": 999,
        "table_id": "selected",
        "table_label": "Selected",
        "table_caption": "",
        "table_foot": "",
        "table_data_file": "tables/selected.csv",
    }, {
        "pmcid": 999,
        "table_id": "missed",
        "table_label": "Missed",
        "table_caption": "",
        "table_foot": "",
        "table_data_file": "tables/missed.csv",
    }]).to_csv(tmp_path / "tables.csv", index=False)
    pd.DataFrame([{
        "pmcid": 999,
        "table_id": "selected",
        "table_label": "Selected",
        "x": 1,
        "y": 2,
        "z": 3,
    }]).to_csv(tmp_path / "coordinates.csv", index=False)
    monkeypatch.setattr(
        utils,
        "_is_coordinate_candidate_path",
        lambda path: path == missed,
    )

    _, tables = utils.load_activation_table_map(
        processed_data_path=tmp_path,
        filter_by_coordinates=True,
        identifier_key="pmcid",
        fallback_candidate_gate=True,
    )

    assert {table["table_id"] for table in tables[999]} == {
        "selected",
        "missed",
    }
