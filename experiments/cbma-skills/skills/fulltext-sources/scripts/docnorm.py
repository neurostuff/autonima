"""Normalize a full-text article (PMC JATS XML, publisher HTML, or plain text)
into the source-agnostic document layout every later stage reads:

    docs/<pmid>/
        meta.json          source, original path, sha256 of the original, counts
        text.md            article body as markdown; the reference list is dropped,
                           tables and figures appear as one-line placeholders
        tables/<id>.json   one file per table: label, caption, footer, the cell grid
                           with row/col spans expanded, a TSV rendering, and flags

Standard library only. Import it, or run it directly on one file:

    python docnorm.py ARTICLE_FILE --pmid 12345 --out docs/ [--format auto|jats|html|text]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

# A table counts as a coordinate candidate when its header names coordinates, or
# when some row holds three adjacent cells that look like a stereotaxic triple.
_COORD_HEADER = re.compile(
    r"\b(mni|talairach|tal|coordinates?|x\s*,\s*y\s*,\s*z|peak)\b|^\s*[xyz]\s*(\(mm\))?\s*$",
    re.I,
)
_SIGNED_NUM = re.compile(r"^[\s(]*[-−–+]?\d{1,3}(\.\d+)?[\s)]*$")
_REF_HEADING = re.compile(r"^\s*(references?|bibliography|literature cited|works cited)\s*$", re.I)
_REF_CLASS = re.compile(r"(^|[\s_-])(ref-?list|references?|bibliography|citations?)($|[\s_-])", re.I)
_DROP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "form", "button", "svg", "iframe"}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "dd", "dt", "blockquote", "figcaption", "pre"}


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Table helpers shared by the JATS and HTML paths
# --------------------------------------------------------------------------- #

def expand_grid(rows: List[List[dict]]) -> List[List[str]]:
    """Turn rows of {text, colspan, rowspan} cells into a rectangular grid.

    A spanned cell's text is repeated into every position it covers, so column
    k of row r always means the same column, which is what coordinate
    verification and a reader both need.
    """
    grid: Dict[tuple, str] = {}
    n_cols = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            colspan = max(1, int(cell.get("colspan") or 1))
            rowspan = max(1, int(cell.get("rowspan") or 1))
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = cell["text"]
            c += colspan
            n_cols = max(n_cols, c)
    n_rows = max((r for r, _ in grid), default=-1) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def is_coordinate_candidate(grid: List[List[str]], caption: str = "") -> bool:
    header_text = " ".join(grid[0]) if grid else ""
    if _COORD_HEADER.search(caption) and re.search(r"\b(mni|talairach|coordinates?)\b", caption, re.I):
        return True
    for cell in (grid[0] if grid else []) + (grid[1] if len(grid) > 1 else []):
        if _COORD_HEADER.search(cell):
            return True
    for row in grid:
        run = 0
        for cell in row:
            run = run + 1 if cell and _SIGNED_NUM.match(cell) else 0
            if run >= 3:
                return True
    return bool(re.search(r"\bx\b.*\by\b.*\bz\b", header_text, re.I))


def table_record(table_id: str, label: str, caption: str, footer: str,
                 rows: List[List[dict]], source_markup: str) -> dict:
    grid = expand_grid(rows)
    tsv = "\n".join("\t".join(cell.replace("\t", " ") for cell in row) for row in grid)
    return {
        "table_id": table_id,
        "label": label,
        "caption": caption,
        "footer": footer,
        "n_rows": len(grid),
        "n_cols": len(grid[0]) if grid else 0,
        "grid": grid,
        "tsv": tsv,
        "has_data": bool(grid),
        "coordinate_candidate": bool(grid) and is_coordinate_candidate(grid, caption),
        "content_sha256": _sha256_bytes(tsv.encode("utf-8")),
        "duplicate_of": None,
        "source_markup": source_markup,
    }


def mark_duplicates(tables: List[dict]) -> None:
    """Flag tables whose cell content repeats an earlier table in the same document.

    Re-ingested duplicate tables otherwise reach the meta-analysis twice and
    double-weight the study (autonima issue #60).
    """
    seen: Dict[str, str] = {}
    for t in tables:
        if not t["has_data"]:
            continue
        key = t["content_sha256"]
        if key in seen:
            t["duplicate_of"] = seen[key]
        else:
            seen[key] = t["table_id"]


# --------------------------------------------------------------------------- #
# JATS (PMC)
# --------------------------------------------------------------------------- #

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _jats_text(el: Optional[ET.Element]) -> str:
    """Text of an element, skipping nested tables, figures and xrefs' markup noise."""
    if el is None:
        return ""
    parts: List[str] = []

    def walk(node: ET.Element) -> None:
        if _local(node.tag) in {"table-wrap", "fig", "table-wrap-group", "fig-group"}:
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return _clean("".join(parts))


def _jats_table(wrap: ET.Element, index: int) -> dict:
    label = _clean("".join(wrap.find("label").itertext())) if wrap.find("label") is not None else ""
    caption_el = wrap.find("caption")
    caption = _clean(" ".join(caption_el.itertext())) if caption_el is not None else ""
    foot_el = wrap.find("table-wrap-foot")
    footer = _clean(" ".join(foot_el.itertext())) if foot_el is not None else ""
    rows: List[List[dict]] = []
    markup = ""
    table = next((n for n in wrap.iter() if _local(n.tag) == "table"), None)
    if table is not None:
        markup = ET.tostring(table, encoding="unicode")
        for tr in (n for n in table.iter() if _local(n.tag) == "tr"):
            cells = []
            for cell in tr:
                if _local(cell.tag) in {"td", "th"}:
                    cells.append({
                        "text": _clean(" ".join(cell.itertext())),
                        "colspan": cell.get("colspan", 1),
                        "rowspan": cell.get("rowspan", 1),
                    })
            rows.append(cells)
    table_id = wrap.get("id") or f"T{index}"
    return table_record(table_id, label, caption, footer, rows, markup)


def parse_jats(xml_bytes: bytes) -> dict:
    """Parse one PMC article. Returns {title, abstract, text_md, tables, complete, reason}."""
    root = ET.fromstring(xml_bytes)
    article = root if _local(root.tag) == "article" else next(
        (n for n in root.iter() if _local(n.tag) == "article"), None
    )
    if article is None:
        return {"title": "", "abstract": "", "text_md": "", "tables": [], "complete": False,
                "reason": "no <article> element in XML"}

    def find(path_tags: List[str], start: ET.Element) -> Optional[ET.Element]:
        node = start
        for tag in path_tags:
            node = next((c for c in node.iter() if _local(c.tag) == tag), None)
            if node is None:
                return None
        return node

    title_el = find(["front", "article-title"], article)
    title = _clean(" ".join(title_el.itertext())) if title_el is not None else ""
    abstract_el = find(["front", "abstract"], article)
    abstract = _clean(" ".join(abstract_el.itertext())) if abstract_el is not None else ""
    body = next((c for c in article if _local(c.tag) == "body"), None)
    back = next((c for c in article if _local(c.tag) == "back"), None)

    lines: List[str] = [f"# {title}", ""]
    if abstract:
        lines += ["## Abstract", "", abstract, ""]
    tables: List[dict] = []

    def emit(node: ET.Element, depth: int) -> None:
        tag = _local(node.tag)
        if tag in {"ref-list", "fn-group"}:
            return
        if tag == "sec":
            title_child = next((c for c in node if _local(c.tag) == "title"), None)
            heading = _clean(" ".join(title_child.itertext())) if title_child is not None else ""
            if _REF_HEADING.match(heading):
                return
            if heading:
                lines.extend(["#" * min(6, depth + 1) + " " + heading, ""])
            for child in node:
                if child is not title_child:
                    emit(child, depth + 1)
            return
        if tag == "p":
            text = _jats_text(node)
            if text:
                lines.extend([text, ""])
            for child in node.iter():
                if child is not node and _local(child.tag) in {"table-wrap", "fig"}:
                    emit(child, depth)
            return
        if tag == "table-wrap":
            t = _jats_table(node, len(tables) + 1)
            tables.append(t)
            lines.extend([f"[TABLE {t['table_id']}: {t['label']} {t['caption']}]".replace("  ", " "), ""])
            return
        if tag == "fig":
            label = node.find("label")
            caption = node.find("caption")
            text = _clean(" ".join(caption.itertext())) if caption is not None else ""
            lab = _clean("".join(label.itertext())) if label is not None else "Figure"
            lines.extend([f"[FIGURE: {lab} {text}]", ""])
            return
        for child in node:
            emit(child, depth)

    if body is not None:
        for child in body:
            emit(child, 1)
    if back is not None:
        for child in back:
            emit(child, 1)
    # Tables sometimes live in a floats-group outside body.
    for floats in (c for c in article if _local(c.tag) == "floats-group"):
        for child in floats:
            emit(child, 1)

    mark_duplicates(tables)
    complete = body is not None and len(_jats_text(body)) > 1500
    reason = None
    if body is None:
        reason = "PMC XML has no <body>: the article is not in the open-access subset, or is embargoed"
    elif not complete:
        reason = "PMC XML body is very short; likely incomplete"
    return {"title": title, "abstract": abstract, "text_md": "\n".join(lines).strip() + "\n",
            "tables": tables, "complete": complete, "reason": reason}


# --------------------------------------------------------------------------- #
# HTML (publisher pages, ACE downloads, anything a lab pre-downloaded)
# --------------------------------------------------------------------------- #

class _Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict, parent: Optional["_Node"]):
        self.tag, self.attrs, self.children, self.parent = tag, attrs, [], parent

    def text(self) -> str:
        out: List[str] = []

        def walk(n: "_Node") -> None:
            for c in n.children:
                if isinstance(c, str):
                    out.append(c)
                elif c.tag not in _DROP_TAGS:
                    if c.tag in {"br", "p", "div", "tr", "li"}:
                        out.append(" ")
                    walk(c)
        walk(self)
        return _clean("".join(out))

    def iter(self, tag: Optional[str] = None):
        for c in self.children:
            if isinstance(c, _Node):
                if tag is None or c.tag == tag:
                    yield c
                yield from c.iter(tag)

    def cls(self) -> str:
        return f"{self.attrs.get('class', '')} {self.attrs.get('id', '')}"


_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {}, None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.children.append(node)
        if tag not in _VOID:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(_Node(tag, {k: (v or "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        node = self.cur
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.cur = node.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def _html_table(table: _Node, index: int) -> dict:
    caption_el = next(table.iter("caption"), None)
    caption = caption_el.text() if caption_el is not None else ""
    label = ""
    footer = ""
    # Publisher markup usually wraps a table in a container that also holds the
    # label, caption and footnotes; look one or two levels up.
    container = table.parent
    for _ in range(2):
        if container is None or container.tag == "root":
            break
        for n in container.iter():
            c = n.cls().lower()
            if _is_inside(n, table):
                continue
            if not label and re.search(r"\blabel\b", c):
                label = n.text()
            elif not caption and re.search(r"caption|title", c):
                caption = n.text()
            elif not footer and re.search(r"foot|legend|notes?\b", c):
                footer = n.text()
        if caption or label:
            break
        container = container.parent
    if not label:
        m = re.match(r"^(table\s+[\w.]+)[.:]?\s*", caption, re.I)
        if m:
            label = m.group(1)
    tfoot = next(table.iter("tfoot"), None)
    if tfoot is not None and not footer:
        footer = tfoot.text()
    rows: List[List[dict]] = []
    for tr in table.iter("tr"):
        if tfoot is not None and _is_inside(tr, tfoot):
            continue
        cells = [
            {"text": c.text(), "colspan": c.attrs.get("colspan", 1) or 1, "rowspan": c.attrs.get("rowspan", 1) or 1}
            for c in tr.children if isinstance(c, _Node) and c.tag in {"td", "th"}
        ]
        if cells:
            rows.append(cells)
    table_id = table.attrs.get("id") or (table.parent.attrs.get("id") if table.parent else "") or f"T{index}"
    return table_record(table_id, label, caption, footer, rows, "")


def _is_inside(node: _Node, ancestor: _Node) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def parse_html(html_bytes: bytes) -> dict:
    builder = _TreeBuilder()
    builder.feed(html_bytes.decode("utf-8", errors="replace"))
    root = builder.root

    title_el = next(root.iter("h1"), None) or next(root.iter("title"), None)
    title = title_el.text() if title_el is not None else ""
    tables: List[dict] = []
    lines: List[str] = [f"# {title}", ""]
    in_refs = {"flag": False}

    def is_ref_container(n: _Node) -> bool:
        if _REF_CLASS.search(n.cls()):
            return True
        heading = next((c for c in n.children if isinstance(c, _Node) and re.match(r"h[1-6]$", c.tag)), None)
        return heading is not None and bool(_REF_HEADING.match(heading.text()))

    def emit(n: _Node) -> None:
        for c in n.children:
            if isinstance(c, str):
                continue
            if c.tag in _DROP_TAGS:
                continue
            if c.tag in {"section", "div", "ol", "ul"} and is_ref_container(c):
                continue
            if re.match(r"h[1-6]$", c.tag):
                heading = c.text()
                if _REF_HEADING.match(heading):
                    in_refs["flag"] = True
                    continue
                in_refs["flag"] = False
                if heading and heading != title:
                    lines.extend(["#" * int(c.tag[1]) + " " + heading, ""])
                continue
            if in_refs["flag"]:
                continue
            if c.tag == "table":
                t = _html_table(c, len(tables) + 1)
                tables.append(t)
                lines.extend([f"[TABLE {t['table_id']}: {t['label']} {t['caption']}]".replace("  ", " "), ""])
                continue
            if c.tag == "figure" and next(c.iter("table"), None) is None:
                cap = next(c.iter("figcaption"), None)
                lines.extend([f"[FIGURE: {cap.text() if cap is not None else ''}]", ""])
                continue
            has_block_child = any(isinstance(g, _Node) and (g.tag in _BLOCK_TAGS or g.tag == "table"
                                  or re.match(r"h[1-6]$", g.tag)) for g in c.children)
            if c.tag in _BLOCK_TAGS and not has_block_child:
                text = c.text()
                if text:
                    lines.extend([text, ""])
            else:
                emit(c)

    body = next(root.iter("body"), None) or root
    emit(body)
    mark_duplicates(tables)
    text_md = "\n".join(lines).strip() + "\n"
    complete = len(text_md) > 3000
    return {"title": title, "abstract": "", "text_md": text_md, "tables": tables, "complete": complete,
            "reason": None if complete else "HTML yields very little body text; likely an abstract page or paywall"}


def parse_text(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace")
    complete = len(text) > 3000
    return {"title": "", "abstract": "", "text_md": text.strip() + "\n", "tables": [], "complete": complete,
            "reason": None if complete else "text file is very short; likely incomplete"}


def detect_format(path: Path, raw: bytes) -> str:
    head = raw[:4000].decode("utf-8", errors="replace").lower()
    if "<pmc-articleset" in head or "<!doctype article" in head or re.search(r"<article[\s>][^<]*(dtd-version|xmlns:xlink)", head):
        return "jats"
    if path.suffix.lower() in {".html", ".htm", ".xhtml"} or "<html" in head:
        return "html"
    if path.suffix.lower() == ".xml":
        return "jats"
    return "text"


def normalize(raw: bytes, fmt: str) -> dict:
    if fmt == "jats":
        return parse_jats(raw)
    if fmt == "html":
        return parse_html(raw)
    if fmt == "text":
        return parse_text(raw)
    raise ValueError(f"unknown format {fmt!r}")


def write_doc(out_root: Path, pmid: str, parsed: dict, *, source: str, origin: str, raw: bytes, fmt: str) -> dict:
    """Write docs/<pmid>/ and return the index entry describing it."""
    doc_dir = out_root / pmid
    tables_dir = doc_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for old in tables_dir.glob("*.json"):
        old.unlink()
    (doc_dir / "text.md").write_text(parsed["text_md"], encoding="utf-8")
    for t in parsed["tables"]:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", t["table_id"])
        t["table_id"] = safe
        (tables_dir / f"{safe}.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
    meta = {
        "pmid": pmid,
        "source": source,
        "origin": origin,
        "format": fmt,
        "original_sha256": _sha256_bytes(raw),
        "text_sha256": _sha256_bytes(parsed["text_md"].encode("utf-8")),
        "title": parsed["title"],
        "n_tables": len(parsed["tables"]),
        "n_coordinate_candidates": sum(1 for t in parsed["tables"] if t["coordinate_candidate"] and not t["duplicate_of"]),
        "n_duplicate_tables": sum(1 for t in parsed["tables"] if t["duplicate_of"]),
        "complete": parsed["complete"],
        "incomplete_reason": parsed["reason"],
    }
    (doc_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--pmid", required=True)
    ap.add_argument("--out", type=Path, required=True, help="the review's docs/ directory")
    ap.add_argument("--format", default="auto", choices=["auto", "jats", "html", "text"])
    ap.add_argument("--source", default="manual")
    args = ap.parse_args(argv)
    raw = args.file.read_bytes()
    fmt = detect_format(args.file, raw) if args.format == "auto" else args.format
    meta = write_doc(args.out, str(args.pmid), normalize(raw, fmt), source=args.source,
                     origin=str(args.file), raw=raw, fmt=fmt)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
