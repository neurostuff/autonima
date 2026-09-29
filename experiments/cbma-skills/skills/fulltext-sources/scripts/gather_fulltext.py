"""Gather full texts for a review from the sources listed in review.yaml, in order.

    python gather_fulltext.py REVIEW_DIR --pmids-file REVIEW_DIR/fulltext/needed.txt
    python gather_fulltext.py REVIEW_DIR --all          # every record in search/records.jsonl
    python gather_fulltext.py REVIEW_DIR --index-only   # just report what local sources hold

Sources (review.yaml, `fulltext.sources`), tried in the order listed. The first one
that yields a complete document wins; if none does, the longest incomplete one is
kept and marked incomplete.

    fulltext:
      sources:
        - type: pmc                      # PMC open-access JATS via E-utilities (no key)
        - type: local                    # any folder of pre-downloaded files
          name: elsevier_html
          path: /data/elsevier_html      # relative paths resolve against REVIEW_DIR
          pattern: "**/*.html"           # glob, default "**/*"
          id_from: filename              # filename | parent_dir | regex | sidecar
          id_kind: pmid                  # pmid | doi | pmcid (doi/pmcid map via search records)
          # regex: "PMID(?P<id>\\d+)"    # for id_from: regex, matched against the relative path
          # sidecar: identifiers.json    # for id_from: sidecar, a JSON file next to the article
          # sidecar_key: pmid
          format: auto                   # auto | html | jats | text

Writes docs/<pmid>/ (see docnorm.py) and fulltext/index.jsonl, one line per PMID:
    {pmid, status: available|incomplete|unavailable, source, attempts: [...], ...}

A PMID with no usable text is recorded as unavailable with the reasons. That is a
retrieval outcome, not an eligibility decision, and the ledger never counts it as an
exclusion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docnorm  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "cbma-skills"
Http = Callable[[str, Dict[str, str]], bytes]


def _default_http(url: str, data: Dict[str, str]) -> bytes:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(),
                                 headers={"User-Agent": TOOL})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read()


class PMCSource:
    name = "pmc"

    def __init__(self, review_dir: Path, records: Dict[str, dict], http: Optional[Http] = None,
                 api_key: Optional[str] = None, email: Optional[str] = None):
        self.cache = review_dir / "fulltext" / "raw" / "pmc"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.records = records
        self.http = http or _default_http
        self.api_key = api_key
        self.email = email
        self.interval = 0.11 if api_key else 0.34
        self._last = 0.0

    def _call(self, endpoint: str, params: Dict[str, str]) -> bytes:
        params = dict(params, tool=TOOL)
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        delay = 1.0
        for attempt in range(4):
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                return self.http(f"{EUTILS}/{endpoint}.fcgi", params)
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500 or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    def pmcid_for(self, pmid: str) -> Optional[str]:
        pmcid = (self.records.get(pmid) or {}).get("pmcid")
        if pmcid:
            return pmcid
        data = json.loads(self._call("elink", {"dbfrom": "pubmed", "db": "pmc", "id": pmid,
                                               "linkname": "pubmed_pmc", "retmode": "json"}))
        for linkset in data.get("linksets", []):
            for db in linkset.get("linksetdbs", []):
                if db.get("links"):
                    return f"PMC{db['links'][0]}"
        return None

    def fetch(self, pmid: str) -> dict:
        pmcid = self.pmcid_for(pmid)
        if not pmcid:
            return {"result": "not_found", "reason": "no PMC record linked to this PMID"}
        path = self.cache / f"{pmcid}.xml"
        if not path.exists():
            raw = self._call("efetch", {"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "xml"})
            path.write_bytes(raw)
        raw = path.read_bytes()
        return {"result": "file", "raw": raw, "format": "jats", "origin": str(path), "pmcid": pmcid}


class LocalSource:
    def __init__(self, spec: dict, review_dir: Path, records: Dict[str, dict]):
        self.name = spec.get("name") or f"local:{spec['path']}"
        root = Path(spec["path"]).expanduser()
        self.root = root if root.is_absolute() else (review_dir / root)
        self.pattern = spec.get("pattern", "**/*")
        self.id_from = spec.get("id_from", "filename")
        self.id_kind = spec.get("id_kind", "pmid")
        self.regex = re.compile(spec["regex"]) if spec.get("regex") else None
        self.sidecar = spec.get("sidecar")
        self.sidecar_key = spec.get("sidecar_key", "pmid")
        self.format = spec.get("format", "auto")
        if self.id_from == "regex" and not self.regex:
            raise SystemExit(f"source {self.name}: id_from: regex needs a regex with a named group (?P<id>...)")
        if self.id_from == "sidecar" and not self.sidecar:
            raise SystemExit(f"source {self.name}: id_from: sidecar needs sidecar: <filename>")
        by_doi = {r["doi"].lower(): p for p, r in records.items() if r.get("doi")}
        by_pmcid = {r["pmcid"].upper(): p for p, r in records.items() if r.get("pmcid")}
        self.index: Dict[str, List[Path]] = {}
        self.unmapped: List[str] = []
        if not self.root.exists():
            raise SystemExit(f"source {self.name}: {self.root} does not exist")
        for path in sorted(self.root.glob(self.pattern)):
            if not path.is_file() or path.name == self.sidecar:
                continue
            raw_id = self._raw_id(path)
            pmid = self._to_pmid(raw_id, by_doi, by_pmcid) if raw_id else None
            if pmid:
                self.index.setdefault(pmid, []).append(path)
            else:
                self.unmapped.append(str(path.relative_to(self.root)))

    def _raw_id(self, path: Path) -> Optional[str]:
        rel = str(path.relative_to(self.root))
        if self.id_from == "filename":
            return path.stem
        if self.id_from == "parent_dir":
            return path.parent.name
        if self.id_from == "regex":
            m = self.regex.search(rel)
            return m.group("id") if m else None
        if self.id_from == "sidecar":
            side = path.parent / self.sidecar
            if side.exists():
                value = json.loads(side.read_text()).get(self.sidecar_key)
                return str(value) if value is not None else None
            return None
        raise SystemExit(f"source {self.name}: unknown id_from {self.id_from!r}")

    def _to_pmid(self, raw: str, by_doi: Dict[str, str], by_pmcid: Dict[str, str]) -> Optional[str]:
        raw = raw.strip()
        if self.id_kind == "pmid":
            digits = re.sub(r"\D", "", raw)
            return str(int(digits)) if digits else None
        if self.id_kind == "doi":
            doi = urllib.parse.unquote(raw).lower().replace("_", "/")
            return by_doi.get(doi)
        if self.id_kind == "pmcid":
            key = raw.upper() if raw.upper().startswith("PMC") else f"PMC{raw}"
            return by_pmcid.get(key)
        raise SystemExit(f"source {self.name}: unknown id_kind {self.id_kind!r}")

    def fetch(self, pmid: str) -> dict:
        paths = self.index.get(pmid)
        if not paths:
            return {"result": "not_found", "reason": f"no file for this PMID under {self.root}"}
        # Several files for one PMID (e.g. article.html and supplement.html): use the largest.
        path = max(paths, key=lambda p: p.stat().st_size)
        raw = path.read_bytes()
        fmt = docnorm.detect_format(path, raw) if self.format == "auto" else self.format
        return {"result": "file", "raw": raw, "format": fmt, "origin": str(path),
                "other_files": [str(p) for p in paths if p != path]}


def load_spec(review_dir: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("review.yaml needs PyYAML: pip install pyyaml") from exc
    return yaml.safe_load((review_dir / "review.yaml").read_text()) or {}


def load_records(review_dir: Path) -> Dict[str, dict]:
    path = review_dir / "search" / "records.jsonl"
    if not path.exists():
        return {}
    return {r["pmid"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def build_sources(spec: dict, review_dir: Path, records: Dict[str, dict], http: Optional[Http] = None) -> list:
    import os
    source_specs = (spec.get("fulltext") or {}).get("sources") or [{"type": "pmc"}]
    sources = []
    for s in source_specs:
        if s.get("type") == "pmc":
            sources.append(PMCSource(review_dir, records, http=http, api_key=os.environ.get("NCBI_API_KEY"),
                                     email=(spec.get("search") or {}).get("email")))
        elif s.get("type") == "local":
            sources.append(LocalSource(s, review_dir, records))
        else:
            raise SystemExit(f"unknown full-text source type {s.get('type')!r}; use pmc or local")
    return sources


def gather(review_dir: Path, pmids: List[str], sources: list) -> List[dict]:
    docs_root = review_dir / "docs"
    entries = []
    for n, pmid in enumerate(pmids, 1):
        attempts, best = [], None
        for src in sources:
            try:
                got = src.fetch(pmid)
            except Exception as exc:  # noqa: BLE001 -- record every failure, never drop the PMID
                attempts.append({"source": src.name, "result": "error", "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if got["result"] != "file":
                attempts.append({"source": src.name, "result": got["result"], "reason": got.get("reason")})
                continue
            try:
                parsed = docnorm.normalize(got["raw"], got["format"])
            except Exception as exc:  # noqa: BLE001
                attempts.append({"source": src.name, "result": "parse_error", "origin": got["origin"],
                                 "reason": f"{type(exc).__name__}: {exc}"})
                continue
            attempt = {"source": src.name, "result": "complete" if parsed["complete"] else "incomplete",
                       "origin": got["origin"], "reason": parsed["reason"], "chars": len(parsed["text_md"])}
            attempts.append(attempt)
            candidate = (src, got, parsed)
            if parsed["complete"]:
                best = candidate
                break
            if best is None or len(parsed["text_md"]) > len(best[2]["text_md"]):
                best = candidate
        entry = {"pmid": pmid, "attempts": attempts,
                 "gathered_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        if best is None:
            entry.update(status="unavailable", source=None)
        else:
            src, got, parsed = best
            meta = docnorm.write_doc(docs_root, pmid, parsed, source=src.name, origin=got["origin"],
                                     raw=got["raw"], fmt=got["format"])
            entry.update(status="available" if parsed["complete"] else "incomplete", source=src.name,
                         text_sha256=meta["text_sha256"], n_tables=meta["n_tables"],
                         n_coordinate_candidates=meta["n_coordinate_candidates"])
        entries.append(entry)
        if n % 25 == 0 or n == len(pmids):
            print(f"  {n}/{len(pmids)} gathered", file=sys.stderr)
    return entries


def write_index(review_dir: Path, entries: List[dict]) -> Path:
    path = review_dir / "fulltext" / "index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                existing[e["pmid"]] = e
    for e in entries:
        existing[e["pmid"]] = e
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for pmid in sorted(existing, key=int):
            fh.write(json.dumps(existing[pmid]) + "\n")
    tmp.replace(path)
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pmids-file", type=Path)
    group.add_argument("--all", action="store_true")
    group.add_argument("--index-only", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    spec = load_spec(args.review_dir)
    records = load_records(args.review_dir)
    sources = build_sources(spec, args.review_dir, records)

    if args.index_only:
        for s in sources:
            if isinstance(s, LocalSource):
                in_search = sum(1 for p in s.index if p in records)
                print(json.dumps({"source": s.name, "files_mapped": sum(len(v) for v in s.index.values()),
                                  "pmids": len(s.index), "pmids_in_search": in_search,
                                  "unmapped_files": len(s.unmapped), "unmapped_examples": s.unmapped[:5]}))
        return 0

    if args.all:
        pmids = sorted(records, key=int)
    else:
        pmids = [str(int(line.strip())) for line in args.pmids_file.read_text().splitlines() if line.strip()]
    entries = gather(args.review_dir, pmids, sources)
    write_index(args.review_dir, entries)
    summary: Dict[str, int] = {}
    for e in entries:
        key = f"{e['status']}:{e['source']}" if e["source"] else e["status"]
        summary[key] = summary.get(key, 0) + 1
    print(json.dumps({"requested": len(pmids), "outcomes": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
