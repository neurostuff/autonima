"""Tests source-specific table metadata derivation in retrieval utils."""

from pathlib import Path

from autonima.retrieval.utils import load_activation_table_map


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_pubget_metadata_sets_tables_xml_path(tmp_path):
    _write(
        tmp_path / "tables.csv",
        "\n".join(
            [
                "pmcid,table_id,table_label,table_caption,table_foot,n_header_rows,table_data_file",
                "123456,tbl1,Table 1,Caption,Foot,1,articles/abc/pmcid_123456/tables/table_000.csv",
            ]
        ),
    )
    _write(tmp_path / "articles/abc/pmcid_123456/tables/table_000.csv", "Region,x,y,z\nA,1,2,3\n")
    _write(tmp_path / "articles/abc/pmcid_123456/tables/tables.xml", "<extracted-tables-set></extracted-tables-set>")

    _, tables = load_activation_table_map(
        processed_data_path=tmp_path,
        filter_by_coordinates=False,
        identifier_key="pmcid",
    )
    key = 123456 if 123456 in tables else "123456"
    assert key in tables
    table = tables[key][0]
    assert table["table_source_type"] == "pubget_xml"
    assert table["table_xml_path"].endswith("tables.xml")
    assert table["table_data_path"].endswith("table_000.csv")


def test_ace_metadata_sets_html_source(tmp_path):
    _write(
        tmp_path / "tables.csv",
        "\n".join(
            [
                "pmid,table_id,table_label,table_caption,table_foot,n_header_rows,table_raw_file",
                "1111,12,Table 2,Caption,Foot,1,tables/1111/12.html",
            ]
        ),
    )
    _write(tmp_path / "tables/1111/12.html", "<table><tr><td>a</td></tr></table>")

    _, tables = load_activation_table_map(
        processed_data_path=tmp_path,
        filter_by_coordinates=False,
        identifier_key="pmid",
    )
    key = 1111 if 1111 in tables else "1111"
    assert key in tables
    table = tables[key][0]
    assert table["table_source_type"] == "ace_html"
    assert table["table_html_path"].endswith("12.html")
