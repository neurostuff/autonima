"""Tests for canonical table normalization across Pubget and ACE sources."""

from pathlib import Path

from autonima.coordinates.table_normalizer import (
    expand_table_grid,
    normalize_table,
    parse_pubget_tables_xml,
)
from autonima.models.types import ActivationTable


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_expand_table_grid_handles_rowspan_colspan():
    html = """
    <table>
      <thead>
        <tr><th rowspan="2">Region</th><th colspan="3">MNI coordinates</th></tr>
        <tr><th>x</th><th>y</th><th>z</th></tr>
      </thead>
      <tbody>
        <tr><td>A</td><td>10</td><td>20</td><td>30</td></tr>
      </tbody>
    </table>
    """
    grid_data = expand_table_grid(html)
    assert grid_data["column_count"] == 4
    assert grid_data["grid"][0][0] == "Region"
    assert grid_data["grid"][1][0] == "Region"
    assert grid_data["grid"][1][1] == "x"


def test_parse_pubget_tables_xml_prefers_original_table(tmp_path):
    xml_path = _write(
        tmp_path / "tables.xml",
        """<?xml version='1.0' encoding='UTF-8'?>
<extracted-tables-set>
  <extracted-table>
    <table-id>tbl1</table-id>
    <original-table>
      <table-wrap>
        <table><tbody><tr><td>ORIGINAL</td></tr></tbody></table>
      </table-wrap>
    </original-table>
    <transformed-table>
      <div><table><tbody><tr><td>TRANSFORMED</td></tr></tbody></table></div>
    </transformed-table>
  </extracted-table>
</extracted-tables-set>
""",
    )
    fragment = parse_pubget_tables_xml(str(xml_path), "tbl1")
    assert "ORIGINAL" in fragment
    assert "TRANSFORMED" not in fragment


def test_normalize_table_pubget_and_ace_share_schema(tmp_path):
    common_table_html = """
    <table>
      <thead>
        <tr><th>Region</th><th>x</th><th>y</th><th>z</th><th>T</th></tr>
      </thead>
      <tbody>
        <tr><td>emotional > neutral</td><td></td><td></td><td></td><td></td></tr>
        <tr><td>Occipital Pole</td><td>20</td><td>-94</td><td>-2</td><td>7.33</td></tr>
        <tr><td>neutral > emotional</td><td></td><td></td><td></td><td></td></tr>
        <tr><td>Precuneus</td><td>-14</td><td>-60</td><td>14</td><td>6.20</td></tr>
      </tbody>
    </table>
    """

    # ACE
    ace_html_path = _write(tmp_path / "ace" / "table.html", common_table_html)
    ace_table = ActivationTable(
        table_id="ace_tbl",
        table_label="Table 1",
        table_caption="Activation table",
        table_foot="",
        table_source_type="ace_html",
        table_html_path=str(ace_html_path),
    )
    ace_canonical = normalize_table(ace_table)

    # Pubget XML
    pubget_xml_path = _write(
        tmp_path / "pubget" / "tables.xml",
        f"""<?xml version='1.0' encoding='UTF-8'?>
<extracted-tables-set>
  <extracted-table>
    <table-id>pub_tbl</table-id>
    <original-table>
      <table-wrap>
        {common_table_html}
      </table-wrap>
    </original-table>
  </extracted-table>
</extracted-tables-set>
""",
    )
    pub_table = ActivationTable(
        table_id="pub_tbl",
        table_label="Table 2",
        table_caption="Activation table",
        table_foot="",
        table_source_type="pubget_xml",
        table_xml_path=str(pubget_xml_path),
    )
    pub_canonical = normalize_table(pub_table)

    expected_top_keys = {
        "table_identity",
        "source_parser",
        "grid",
        "header_hierarchy",
        "rows",
        "coordinate_column_detection",
        "space_hints",
        "boundary_markers",
    }
    assert set(ace_canonical.keys()) == expected_top_keys
    assert set(pub_canonical.keys()) == expected_top_keys

    assert ace_canonical["coordinate_column_detection"]["has_xyz"]
    assert pub_canonical["coordinate_column_detection"]["has_xyz"]
    assert ace_canonical["boundary_markers"]["segment_count"] == 2
    assert pub_canonical["boundary_markers"]["segment_count"] == 2

