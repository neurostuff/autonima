"""The review ledger: batches work for subagents, validates what they return, and is
the only thing that writes decisions. Standard library plus PyYAML.

    python ledger.py init     REVIEW_DIR                   # check review.yaml, assign criteria IDs
    python ledger.py batches  REVIEW_DIR --stage abstract --size 25
    python ledger.py ingest   REVIEW_DIR --stage abstract --agent "claude-code/<model id>"
    python ledger.py needs-fulltext REVIEW_DIR             # writes fulltext/needed.txt
    python ledger.py status   REVIEW_DIR [--json]          # PRISMA-style counts, writes results/prisma.json
    python ledger.py export   REVIEW_DIR                   # NiMADS studyset + annotation for NiMARE

Stages, in order: abstract, fulltext, extraction, selection.

Design rules (each one closes a failure mode found in autonima):
  * Subagents never write to decisions/. They write raw output next to their batch
    file; `ingest` validates it and appends accepted records. One writer, no races.
  * A decision is valid only while both its criteria hash and its input hash match
    the current review.yaml, skill text and input documents. Change a criterion, the
    skill instructions, or the text a decision was based on, and that item is
    pending again. Nothing else invalidates it.
  * A failure is never a decision. Output that is missing, malformed or
    inconsistent is rejected and the item stays pending.
  * Criteria IDs are numbered per stage (and per target), so adding an abstract
    criterion cannot renumber the full-text criteria.
  * Missing data is never exported as "excluded".
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

STAGES = ["abstract", "fulltext", "extraction", "selection"]
SKILLS_ROOT = Path(__file__).resolve().parents[2]
STAGE_SKILL = {
    "abstract": "screen-studies",
    "fulltext": "screen-studies",
    "extraction": "extract-coordinates",
    "selection": "select-analyses",
}
CRITERION_STATES = {"met", "not_met", "unclear"}
POINT_VALUE_KINDS = {"z-statistic", "t-statistic", "f-statistic", "p-value", "beta", "correlation", "other"}


class LedgerError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def sha(obj) -> str:
    data = obj if isinstance(obj, bytes) else json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{path}:{n}: not valid JSON ({exc.msg})") from exc
    return out


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


# --------------------------------------------------------------------------- #
# Review spec and criteria
# --------------------------------------------------------------------------- #

def load_spec(review: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("ledger.py needs PyYAML: pip install pyyaml") from exc
    path = review / "review.yaml"
    if not path.exists():
        raise LedgerError(f"{path} not found")
    spec = yaml.safe_load(path.read_text()) or {}
    validate_spec(spec)
    return spec


_ALLOWED = {
    "": {"name", "objective", "search", "screening", "fulltext", "extraction", "selection", "meta", "notes"},
    "search": {"query", "date_from", "date_to", "pmids", "pmids_file", "email"},
    "screening": {"abstract", "fulltext"},
    "screening.stage": {"inclusion", "exclusion", "instructions"},
    "fulltext": {"sources"},
    "extraction": {"drop_unverified", "instructions"},
    "selection": {"global", "targets", "instructions"},
    "selection.global": {"inclusion", "exclusion"},
    "selection.target": {"name", "description", "inclusion", "exclusion"},
    "meta": {"estimator", "corrector", "estimator_args", "corrector_args"},
}


def _check_keys(obj: dict, allowed_key: str, where: str) -> None:
    if not isinstance(obj, dict):
        raise LedgerError(f"{where or 'review.yaml'} must be a mapping")
    unknown = set(obj) - _ALLOWED[allowed_key]
    if unknown:
        raise LedgerError(f"unknown key(s) in {where or 'review.yaml top level'}: {sorted(unknown)} "
                          f"(allowed: {sorted(_ALLOWED[allowed_key])})")


def validate_spec(spec: dict) -> None:
    """Reject unknown keys everywhere. A typo must fail loudly, not be ignored."""
    _check_keys(spec, "", "")
    if not spec.get("objective"):
        raise LedgerError("review.yaml needs an objective")
    _check_keys(spec.get("search") or {}, "search", "search")
    search = spec.get("search") or {}
    if not (search.get("query") or search.get("pmids") or search.get("pmids_file")):
        raise LedgerError("search needs a query, pmids or pmids_file")
    screening = spec.get("screening") or {}
    _check_keys(screening, "screening", "screening")
    for stage in ("abstract", "fulltext"):
        block = screening.get(stage)
        if not block:
            raise LedgerError(f"screening.{stage} is required")
        _check_keys(block, "screening.stage", f"screening.{stage}")
        if not block.get("inclusion"):
            raise LedgerError(f"screening.{stage}.inclusion needs at least one criterion")
    _check_keys(spec.get("fulltext") or {}, "fulltext", "fulltext")
    _check_keys(spec.get("extraction") or {}, "extraction", "extraction")
    selection = spec.get("selection") or {}
    _check_keys(selection, "selection", "selection")
    _check_keys(selection.get("global") or {}, "selection.global", "selection.global")
    names = set()
    for i, t in enumerate(selection.get("targets") or []):
        _check_keys(t, "selection.target", f"selection.targets[{i}]")
        if not t.get("name") or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t["name"]):
            raise LedgerError(f"selection.targets[{i}].name must be an identifier like working_memory")
        if t["name"] in names:
            raise LedgerError(f"duplicate target name {t['name']}")
        names.add(t["name"])
    _check_keys(spec.get("meta") or {}, "meta", "meta")


def number(inclusion: List[str], exclusion: List[str], prefix: str = "") -> Dict[str, str]:
    ids = {f"{prefix}I{i}": c for i, c in enumerate(inclusion or [], 1)}
    ids.update({f"{prefix}E{i}": c for i, c in enumerate(exclusion or [], 1)})
    return ids


def skill_hash(stage: str) -> str:
    """Hash of the skill text that instructs this stage, so editing it re-opens decisions."""
    path = SKILLS_ROOT / STAGE_SKILL[stage] / "SKILL.md"
    return sha(path.read_bytes()) if path.exists() else "no-skill-file"


def stage_criteria(spec: dict, stage: str) -> dict:
    if stage in ("abstract", "fulltext"):
        block = spec["screening"][stage]
        crit = number(block.get("inclusion"), block.get("exclusion"))
        payload = {"objective": spec["objective"], "criteria": crit, "instructions": block.get("instructions")}
    elif stage == "extraction":
        crit = {}
        payload = {"instructions": (spec.get("extraction") or {}).get("instructions")}
    elif stage == "selection":
        sel = spec.get("selection") or {}
        glob = sel.get("global") or {}
        crit = number(glob.get("inclusion"), glob.get("exclusion"), prefix="G")
        targets = {
            t["name"]: {"description": t.get("description"), "criteria": number(t.get("inclusion"), t.get("exclusion"))}
            for t in sel.get("targets") or []
        }
        payload = {"objective": spec["objective"], "global": crit, "targets": targets,
                   "instructions": sel.get("instructions")}
    else:
        raise LedgerError(f"unknown stage {stage}")
    payload["skill"] = skill_hash(stage)
    return {"criteria": crit, "payload": payload, "hash": sha(payload)}


# --------------------------------------------------------------------------- #
# Review state
# --------------------------------------------------------------------------- #

class Review:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.spec = load_spec(self.root)
        self.records = {r["pmid"]: r for r in read_jsonl(self.root / "search" / "records.jsonl")}
        self.fulltext_index = {e["pmid"]: e for e in read_jsonl(self.root / "fulltext" / "index.jsonl")}
        self.criteria = {s: stage_criteria(self.spec, s) for s in STAGES}

    # paths
    def decisions_path(self, stage: str) -> Path:
        return self.root / "decisions" / f"{stage}.jsonl"

    def work_dir(self, stage: str) -> Path:
        return self.root / "work" / stage

    def doc_dir(self, pmid: str) -> Path:
        return self.root / "docs" / pmid

    # input hashes: what a decision was based on
    def input_hash(self, stage: str, pmid: str) -> Optional[str]:
        if stage == "abstract":
            r = self.records.get(pmid)
            return sha([r.get("title"), r.get("abstract")]) if r else None
        if stage in ("fulltext", "extraction"):
            meta_path = self.doc_dir(pmid) / "meta.json"
            if not meta_path.exists():
                return None
            meta = json.loads(meta_path.read_text())
            tables = sorted(
                json.loads(p.read_text())["content_sha256"] for p in (self.doc_dir(pmid) / "tables").glob("*.json")
            )
            return sha([meta["text_sha256"], tables])
        if stage == "selection":
            path = self.root / "analyses" / f"{pmid}.json"
            return sha(path.read_bytes()) if path.exists() else None
        raise LedgerError(stage)

    def valid_decisions(self, stage: str) -> Dict[tuple, dict]:
        """Latest record per key whose criteria and input hashes are both current."""
        current = self.criteria[stage]["hash"]
        inputs: Dict[str, Optional[str]] = {}
        out: Dict[tuple, dict] = {}
        for rec in read_jsonl(self.decisions_path(stage)):
            pmid = rec["pmid"]
            if pmid not in inputs:
                inputs[pmid] = self.input_hash(stage, pmid)
            if rec.get("criteria_hash") != current or rec.get("input_hash") != inputs[pmid]:
                continue
            key = (pmid, rec["analysis_id"], rec["target"]) if stage == "selection" else (pmid,)
            out[key] = rec
        return out

    def abstract_passed(self) -> List[str]:
        dec = self.valid_decisions("abstract")
        return sorted((k[0] for k, r in dec.items() if r["decision"] in ("include", "uncertain")), key=int)

    def fulltext_included(self) -> List[str]:
        dec = self.valid_decisions("fulltext")
        return sorted((k[0] for k, r in dec.items() if r["decision"] == "include"), key=int)

    def analyses(self, pmid: str) -> Optional[dict]:
        path = self.root / "analyses" / f"{pmid}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def pending(self, stage: str) -> List[str]:
        if stage == "abstract":
            done = {k[0] for k in self.valid_decisions("abstract")}
            return [p for p in sorted(self.records, key=int) if p not in done]
        if stage == "fulltext":
            done = {k[0] for k in self.valid_decisions("fulltext")}
            return [p for p in self.abstract_passed()
                    if p not in done and self.fulltext_index.get(p, {}).get("status") in ("available", "incomplete")]
        if stage == "extraction":
            out = []
            for p in self.fulltext_included():
                a = self.analyses(p)
                if a is None or a.get("input_hash") != self.input_hash("extraction", p) \
                        or a.get("criteria_hash") != self.criteria["extraction"]["hash"]:
                    out.append(p)
            return out
        if stage == "selection":
            dec = self.valid_decisions("selection")
            targets = list(self.criteria["selection"]["payload"]["targets"])
            out = []
            for p in self.fulltext_included():
                a = self.analyses(p)
                if a is None or not a["analyses"]:
                    continue
                needed = {(p, x["analysis_id"], t) for x in a["analyses"] for t in targets}
                if not needed <= set(dec):
                    out.append(p)
            return out
        raise LedgerError(stage)


# --------------------------------------------------------------------------- #
# Batch construction
# --------------------------------------------------------------------------- #

def _abstract_item(rv: Review, pmid: str) -> dict:
    r = rv.records[pmid]
    return {k: r.get(k) for k in ("pmid", "title", "abstract", "journal", "year", "publication_types", "mesh")}


def _fulltext_item(rv: Review, pmid: str) -> dict:
    d = rv.doc_dir(pmid)
    meta = json.loads((d / "meta.json").read_text())
    return {
        "pmid": pmid,
        "title": rv.records.get(pmid, {}).get("title") or meta.get("title"),
        "text_file": str(d / "text.md"),
        "tables_dir": str(d / "tables"),
        "source": meta["source"],
        "text_flagged_incomplete": not meta["complete"],
        "incomplete_reason": meta.get("incomplete_reason"),
    }


def _extraction_item(rv: Review, pmid: str) -> dict:
    d = rv.doc_dir(pmid)
    tables = []
    for p in sorted((d / "tables").glob("*.json")):
        t = json.loads(p.read_text())
        tables.append({"table_id": t["table_id"], "file": str(p), "label": t["label"],
                       "caption": t["caption"][:300], "coordinate_candidate": t["coordinate_candidate"],
                       "duplicate_of": t["duplicate_of"], "has_data": t["has_data"]})
    return {"pmid": pmid, "title": rv.records.get(pmid, {}).get("title"), "text_file": str(d / "text.md"),
            "tables": tables,
            "must_report": [t["table_id"] for t in tables if t["coordinate_candidate"] and not t["duplicate_of"]]}


def _selection_item(rv: Review, pmid: str) -> dict:
    a = rv.analyses(pmid)
    r = rv.records.get(pmid, {})
    return {"pmid": pmid, "title": r.get("title"), "abstract": r.get("abstract"),
            "text_file": str(rv.doc_dir(pmid) / "text.md"),
            "analyses": [{k: x.get(k) for k in ("analysis_id", "table_id", "table_label", "table_caption",
                                                 "name", "description", "n_points")} for x in a["analyses"]]}


ITEM_BUILDERS = {"abstract": _abstract_item, "fulltext": _fulltext_item,
                 "extraction": _extraction_item, "selection": _selection_item}


def cmd_batches(rv: Review, stage: str, size: int, limit: Optional[int], discard: bool) -> List[Path]:
    work = rv.work_dir(stage)
    work.mkdir(parents=True, exist_ok=True)
    unfinished = [p for p in work.glob("batch_*.json") if _output_path(p, stage).exists()]
    if unfinished and not discard:
        raise LedgerError(f"{len(unfinished)} batch outputs in {work} are not ingested yet; "
                          f"run `ingest --stage {stage}` first (or pass --discard)")
    for p in work.glob("batch_*"):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    pending = rv.pending(stage)
    if limit:
        pending = pending[:limit]
    crit = rv.criteria[stage]
    paths = []
    for i in range(0, len(pending), size):
        chunk = pending[i:i + size]
        n = i // size + 1
        path = work / f"batch_{n:04d}.json"
        batch = {
            "stage": stage,
            "batch_id": f"{stage}-{n:04d}",
            "skill": STAGE_SKILL[stage],
            "objective": rv.spec["objective"],
            "criteria": crit["criteria"],
            "instructions": crit["payload"].get("instructions"),
            "output": str(_output_path(path, stage)),
            "items": [ITEM_BUILDERS[stage](rv, p) for p in chunk],
            # The ledger fills these in at ingest; the subagent never copies them.
            "_criteria_hash": crit["hash"],
            "_input_hashes": {p: rv.input_hash(stage, p) for p in chunk},
        }
        if stage == "selection":
            batch["targets"] = crit["payload"]["targets"]
        write_json(path, batch)
        paths.append(path)
    return paths


def _output_path(batch_path: Path, stage: str) -> Path:
    if stage == "extraction":
        return batch_path.with_suffix(".out")          # a directory: one <pmid>.json per study
    return batch_path.with_suffix(".out.jsonl")


# --------------------------------------------------------------------------- #
# Validation of subagent output
# --------------------------------------------------------------------------- #

def _check_criteria(obj: dict, ids: Iterable[str], where: str) -> List[str]:
    errs = []
    crit = obj.get("criteria")
    ids = list(ids)
    if not isinstance(crit, dict):
        return [f"{where}: 'criteria' must map every criterion ID to met/not_met/unclear"]
    missing = [i for i in ids if i not in crit]
    extra = [k for k in crit if k not in ids]
    bad = [k for k, v in crit.items() if v not in CRITERION_STATES]
    if missing:
        errs.append(f"{where}: criteria missing {missing}")
    if extra:
        errs.append(f"{where}: unknown criterion IDs {extra}")
    if bad:
        errs.append(f"{where}: criterion states must be met/not_met/unclear, got {[(k, crit[k]) for k in bad]}")
    return errs


def validate_screening(rec: dict, stage: str, ids: List[str], text: Optional[str]) -> Tuple[List[str], dict]:
    where = f"pmid {rec.get('pmid')}"
    allowed = {"abstract": {"include", "exclude", "uncertain"},
               "fulltext": {"include", "exclude", "incomplete"}}[stage]
    errs = []
    decision = rec.get("decision")
    if decision not in allowed:
        errs.append(f"{where}: decision must be one of {sorted(allowed)}, got {decision!r}")
    if not isinstance(rec.get("reason"), str) or not rec["reason"].strip():
        errs.append(f"{where}: reason is required")
    errs += _check_criteria(rec, ids, where)
    flags: dict = {}
    if errs:
        return errs, flags
    crit = rec["criteria"]
    any_excl = any(v == "met" for k, v in crit.items() if k.startswith("E"))
    any_incl_fail = any(v == "not_met" for k, v in crit.items() if k.startswith("I"))
    any_incl_unclear = any(v == "unclear" for k, v in crit.items() if k.startswith("I"))
    if decision == "include" and (any_excl or any_incl_fail):
        errs.append(f"{where}: decision is include but criteria show an exclusion met or an inclusion not met")
    elif decision == "include" and any_incl_unclear:
        fix = "use uncertain" if stage == "abstract" else "an inclusion not shown by the full text is not_met"
        errs.append(f"{where}: decision is include but an inclusion criterion is unclear; {fix}")
    if decision == "exclude" and not (any_excl or any_incl_fail):
        errs.append(f"{where}: decision is exclude but no exclusion is met and no inclusion is not_met; "
                    "use uncertain (abstract) or record which criterion failed")
    if stage == "abstract" and decision == "uncertain" and (any_excl or any_incl_fail):
        errs.append(f"{where}: uncertain is for insufficient information; a clearly failed criterion means exclude")
    if stage == "fulltext" and text is not None:
        ev = rec.get("evidence") or []
        if not isinstance(ev, list):
            errs.append(f"{where}: evidence must be a list of {{criterion, quote}}")
        else:
            haystack = norm_text(text)
            ungrounded = [e.get("criterion") for e in ev
                          if not isinstance(e, dict) or norm_text(e.get("quote", ""))[:200] not in haystack]
            flags["n_evidence"] = len(ev)
            flags["ungrounded_evidence"] = ungrounded
    return errs, flags


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _cell_numbers(cell: str) -> List[float]:
    s = unicodedata.normalize("NFKC", cell).replace("−", "-").replace("–", "-").replace("—", "-")
    return [float(x) for x in _NUM.findall(s)]


def verify_point(xyz: List[float], grid: List[List[str]]) -> str:
    """'row' if all three values appear in one table row, 'table' if only somewhere
    in the table, else 'unverified'. Catches sign flips and invented foci."""
    want = Counter(round(float(v), 2) for v in xyz)
    table_counts: Counter = Counter()
    for row in grid:
        nums = Counter(round(v, 2) for cell in row for v in _cell_numbers(cell))
        table_counts.update(nums)
        if all(nums[v] >= c for v, c in want.items()):
            return "row"
    return "table" if all(table_counts[v] >= c for v, c in want.items()) else "unverified"


def validate_extraction(out: dict, item: dict, rv: Review) -> Tuple[List[str], Optional[dict]]:
    pmid = item["pmid"]
    where = f"pmid {pmid}"
    errs = []
    if str(out.get("pmid")) != pmid:
        errs.append(f"{where}: output pmid is {out.get('pmid')!r}")
    tables_out = out.get("tables")
    if not isinstance(tables_out, list):
        return errs + [f"{where}: 'tables' must be a list"], None
    by_id = {t.get("table_id"): t for t in tables_out if isinstance(t, dict)}
    known = {t["table_id"]: t for t in item["tables"]}
    missing = [t for t in item["must_report"] if t not in by_id]
    if missing:
        errs.append(f"{where}: no status reported for coordinate-candidate tables {missing}")
    unknown = [t for t in by_id if t not in known]
    if unknown:
        errs.append(f"{where}: unknown table_id(s) {unknown}")
    analyses: List[dict] = []
    tables_summary = []
    for tid, t in by_id.items():
        if tid not in known:
            continue
        status = t.get("status")
        if status not in {"parsed", "no_coordinates", "not_applicable", "failed"}:
            errs.append(f"{where} {tid}: status must be parsed/no_coordinates/not_applicable/failed")
            continue
        if status == "failed":
            errs.append(f"{where} {tid}: extraction reported failed ({t.get('note')}); retry this study")
            continue
        grid = json.loads(Path(known[tid]["file"]).read_text())["grid"]
        table_space = t.get("space")
        if table_space not in ("MNI", "TAL", None):
            errs.append(f"{where} {tid}: space must be MNI, TAL or null")
        n_pts = n_row = n_tab = n_bad = 0
        tab_analyses = t.get("analyses") or []
        if status == "parsed" and not tab_analyses:
            errs.append(f"{where} {tid}: status parsed but no analyses")
        for k, a in enumerate(tab_analyses, 1):
            pts_out = []
            for j, p in enumerate(a.get("points") or []):
                xyz = p.get("xyz")
                if not (isinstance(xyz, list) and len(xyz) == 3 and all(isinstance(v, (int, float)) for v in xyz)):
                    errs.append(f"{where} {tid} analysis {k} point {j}: xyz must be three numbers")
                    continue
                if any(abs(v) > 200 for v in xyz):
                    errs.append(f"{where} {tid} analysis {k} point {j}: |coordinate| > 200 mm")
                    continue
                space = p.get("space", table_space)
                if space not in ("MNI", "TAL", None):
                    errs.append(f"{where} {tid} analysis {k} point {j}: space must be MNI, TAL or null")
                    continue
                values = []
                for v in p.get("values") or []:
                    if isinstance(v, dict) and v.get("kind") in POINT_VALUE_KINDS:
                        values.append({"kind": v["kind"], "value": v.get("value")})
                check = verify_point(xyz, grid)
                n_pts += 1
                n_row += check == "row"
                n_tab += check == "table"
                n_bad += check == "unverified"
                pts_out.append({"xyz": [float(v) for v in xyz], "space": space, "values": values,
                                "verification": check})
            if not pts_out:
                continue
            analyses.append({
                "analysis_id": f"{pmid}-{tid}-a{k}",
                "table_id": tid,
                "table_label": known[tid]["label"],
                "table_caption": known[tid]["caption"],
                "name": a.get("name"),
                "description": a.get("description"),
                "n_points": len(pts_out),
                "points": pts_out,
            })
        tables_summary.append({"table_id": tid, "status": status, "space": table_space, "note": t.get("note"),
                               "points": n_pts, "verified_row": n_row, "verified_table_only": n_tab,
                               "unverified": n_bad})
    if errs:
        return errs, None
    return [], {"pmid": pmid, "tables": tables_summary, "analyses": analyses}


def validate_selection(lines: List[dict], item: dict, targets: Dict[str, dict], global_ids: List[str]) -> Tuple[List[str], List[dict]]:
    pmid = item["pmid"]
    where = f"pmid {pmid}"
    errs = []
    needed = {(a["analysis_id"], t) for a in item["analyses"] for t in targets}
    seen = {}
    for rec in lines:
        key = (rec.get("analysis_id"), rec.get("target"))
        if key not in needed:
            errs.append(f"{where}: unexpected analysis/target {key}")
            continue
        if not isinstance(rec.get("include"), bool):
            errs.append(f"{where} {key}: include must be true or false")
            continue
        ids = global_ids + list(targets[key[1]]["criteria"])
        errs += _check_criteria(rec, ids, f"{where} {key}")
        if not isinstance(rec.get("reason"), str) or not rec["reason"].strip():
            errs.append(f"{where} {key}: reason is required")
        crit = rec.get("criteria") or {}
        failed = any(v == "met" for k, v in crit.items() if re.match(r"G?E\d", k)) or \
            any(v == "not_met" for k, v in crit.items() if re.match(r"G?I\d", k))
        unclear = any(v == "unclear" for k, v in crit.items() if re.match(r"G?I\d", k))
        if rec.get("include") is True and (failed or unclear):
            errs.append(f"{where} {key}: include is true but a criterion failed or an inclusion is unclear")
        if rec.get("include") is False and not (failed or unclear):
            errs.append(f"{where} {key}: include is false but every criterion passed; record which one fails")
        seen[key] = rec
    missing = needed - set(seen)
    if missing:
        errs.append(f"{where}: {len(missing)} analysis x target pairs have no decision, e.g. {sorted(missing)[:3]}")
    return errs, list(seen.values())


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

def cmd_ingest(rv: Review, stage: str, agent: str) -> dict:
    work = rv.work_dir(stage)
    done_dir = work / "done"
    report = {"stage": stage, "batches": 0, "accepted": 0, "rejected": 0, "missing_output": 0, "errors": []}
    for batch_path in sorted(work.glob("batch_*.json")):
        batch = json.loads(batch_path.read_text())
        out_path = _output_path(batch_path, stage)
        if not out_path.exists():
            report["missing_output"] += 1
            continue
        report["batches"] += 1
        if batch["_criteria_hash"] != rv.criteria[stage]["hash"]:
            report["errors"].append(f"{batch['batch_id']}: criteria or skill changed since this batch was made; "
                                    "its output is discarded")
            report["rejected"] += len(batch["items"])
            _archive(batch_path, out_path, done_dir)
            continue
        accepted_rows: List[dict] = []
        items = {it["pmid"]: it for it in batch["items"]}
        base = {"stage": stage, "batch_id": batch["batch_id"], "agent": agent, "ingested_at": now(),
                "criteria_hash": batch["_criteria_hash"]}

        if stage in ("abstract", "fulltext"):
            try:
                lines = read_jsonl(out_path)
            except LedgerError as exc:
                report["errors"].append(f"{batch['batch_id']}: {exc}")
                report["rejected"] += len(items)
                _archive(batch_path, out_path, done_dir)
                continue
            ids = list(batch["criteria"])
            seen = set()
            for rec in lines:
                pmid = str(rec.get("pmid", ""))
                if pmid not in items:
                    report["errors"].append(f"{batch['batch_id']}: pmid {pmid!r} is not in this batch")
                    continue
                if pmid in seen:
                    report["errors"].append(f"{batch['batch_id']}: pmid {pmid} appears twice; first kept")
                    continue
                seen.add(pmid)
                text = Path(items[pmid]["text_file"]).read_text() if stage == "fulltext" else None
                errs, flags = validate_screening(rec, stage, ids, text)
                if batch["_input_hashes"][pmid] != rv.input_hash(stage, pmid):
                    errs.append(f"pmid {pmid}: the input changed after the batch was made")
                if errs:
                    report["errors"] += [f"{batch['batch_id']}: {e}" for e in errs]
                    continue
                row = dict(base, pmid=pmid, input_hash=batch["_input_hashes"][pmid],
                           decision=rec["decision"], criteria=rec["criteria"], reason=rec["reason"].strip(),
                           **({"evidence": rec.get("evidence")} if stage == "fulltext" else {}),
                           **({"model": rec["model"]} if rec.get("model") else {}), **flags)
                accepted_rows.append(row)
            not_returned = set(items) - seen
            if not_returned:
                report["errors"].append(f"{batch['batch_id']}: no output for {sorted(not_returned)[:5]}"
                                        f"{'...' if len(not_returned) > 5 else ''} ({len(not_returned)} items)")
            append_jsonl(rv.decisions_path(stage), accepted_rows)

        elif stage == "extraction":
            for pmid, item in items.items():
                f = out_path / f"{pmid}.json"
                if not f.exists():
                    report["errors"].append(f"{batch['batch_id']}: no output file for pmid {pmid}")
                    continue
                try:
                    out = json.loads(f.read_text())
                except json.JSONDecodeError as exc:
                    report["errors"].append(f"{batch['batch_id']}: {f.name} is not valid JSON ({exc.msg})")
                    continue
                errs, record = validate_extraction(out, item, rv)
                if batch["_input_hashes"][pmid] != rv.input_hash(stage, pmid):
                    errs.append(f"pmid {pmid}: the document changed after the batch was made")
                if errs:
                    report["errors"] += [f"{batch['batch_id']}: {e}" for e in errs]
                    continue
                record.update(base, input_hash=batch["_input_hashes"][pmid])
                write_json(rv.root / "analyses" / f"{pmid}.json", record)
                accepted_rows.append(record)

        elif stage == "selection":
            try:
                lines = read_jsonl(out_path)
            except LedgerError as exc:
                report["errors"].append(f"{batch['batch_id']}: {exc}")
                report["rejected"] += len(items)
                _archive(batch_path, out_path, done_dir)
                continue
            by_pmid: Dict[str, List[dict]] = {}
            for rec in lines:
                by_pmid.setdefault(str(rec.get("pmid", "")), []).append(rec)
            global_ids = list(batch["criteria"])
            for pmid, item in items.items():
                errs, recs = validate_selection(by_pmid.get(pmid, []), item, batch["targets"], global_ids)
                if batch["_input_hashes"][pmid] != rv.input_hash(stage, pmid):
                    errs.append(f"pmid {pmid}: analyses changed after the batch was made")
                if errs:
                    # All or nothing per study: a partial study stays pending, never half-exported.
                    report["errors"] += [f"{batch['batch_id']}: {e}" for e in errs]
                    continue
                rows = [dict(base, pmid=pmid, input_hash=batch["_input_hashes"][pmid], analysis_id=r["analysis_id"],
                             target=r["target"], include=r["include"], criteria=r["criteria"],
                             reason=r["reason"].strip()) for r in recs]
                append_jsonl(rv.decisions_path(stage), rows)
                accepted_rows.extend(rows)

        n_items = len(items)
        n_ok = len({r["pmid"] for r in accepted_rows})
        report["accepted"] += n_ok
        report["rejected"] += n_items - n_ok
        _archive(batch_path, out_path, done_dir)
    return report


def _archive(batch_path: Path, out_path: Path, done_dir: Path) -> None:
    done_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    for p in (batch_path, out_path):
        if p.exists():
            shutil.move(str(p), str(done_dir / f"{stamp}_{p.name}"))


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

def cmd_status(rv: Review) -> dict:
    log_path = rv.root / "search" / "search_log.json"
    search_log = json.loads(log_path.read_text()) if log_path.exists() else {}
    a = Counter(r["decision"] for r in rv.valid_decisions("abstract").values())
    passed = rv.abstract_passed()
    ft_status = Counter(rv.fulltext_index.get(p, {}).get("status", "not_gathered") for p in passed)
    ft_source = Counter(rv.fulltext_index[p]["source"] for p in passed
                        if rv.fulltext_index.get(p, {}).get("source"))
    f_dec = rv.valid_decisions("fulltext")
    f = Counter(r["decision"] for r in f_dec.values())
    ungrounded = sum(1 for r in f_dec.values() if r.get("ungrounded_evidence"))
    included = rv.fulltext_included()
    ext = {"studies": len(included), "extracted": 0, "with_analyses": 0, "analyses": 0, "points": 0,
           "points_verified_row": 0, "points_verified_table_only": 0, "points_unverified": 0,
           "pending": 0}
    ext_pending = set(rv.pending("extraction"))
    ext["pending"] = len(ext_pending)
    for p in included:
        an = rv.analyses(p)
        if an is None or p in ext_pending:
            continue
        ext["extracted"] += 1
        ext["with_analyses"] += bool(an["analyses"])
        ext["analyses"] += len(an["analyses"])
        for t in an["tables"]:
            ext["points"] += t["points"]
            ext["points_verified_row"] += t["verified_row"]
            ext["points_verified_table_only"] += t["verified_table_only"]
            ext["points_unverified"] += t["unverified"]
    sel_dec = rv.valid_decisions("selection")
    targets = {}
    for t in rv.criteria["selection"]["payload"]["targets"]:
        inc = [k for k, r in sel_dec.items() if k[2] == t and r["include"]]
        targets[t] = {"analyses_included": len(inc), "studies_included": len({k[0] for k in inc})}
    status = {
        "identification": {
            "records": len(rv.records),
            "reported_by_pubmed": (search_log.get("query_result") or {}).get("reported_count"),
            "found_by_query_and_list": search_log.get("in_both_query_and_list"),
            "without_abstract": sum(1 for r in rv.records.values() if not r.get("has_abstract")),
        },
        "abstract_screening": {"screened": sum(a.values()), "include": a["include"], "uncertain": a["uncertain"],
                               "exclude": a["exclude"], "pending": len(rv.pending("abstract"))},
        "fulltext_retrieval": {"sought": len(passed), **{k: v for k, v in ft_status.items()},
                               "by_source": dict(ft_source)},
        "fulltext_screening": {"screened": sum(f.values()), "include": f["include"], "exclude": f["exclude"],
                               "text_incomplete": f["incomplete"], "pending": len(rv.pending("fulltext")),
                               "decisions_with_ungrounded_evidence": ungrounded},
        "extraction": ext,
        "selection": {"pending_studies": len(rv.pending("selection")), "targets": targets},
        "criteria_hashes": {s: rv.criteria[s]["hash"] for s in STAGES},
        "generated_at": now(),
    }
    write_json(rv.root / "results" / "prisma.json", status)
    return status


# --------------------------------------------------------------------------- #
# NiMADS export
# --------------------------------------------------------------------------- #

def cmd_export(rv: Review, allow_pending: bool, include_table_only: bool) -> dict:
    """Write results/nimads/{studyset,annotation}.json from full-text-included studies.

    Points are exported only when verified against their table (row-level by
    default). An analysis with a missing selection decision is left out of the
    studyset entirely rather than exported as false; without --allow-pending the
    export refuses while any are missing.
    """
    drop_unverified = (rv.spec.get("extraction") or {}).get("drop_unverified", True)
    targets = list(rv.criteria["selection"]["payload"]["targets"])
    sel = rv.valid_decisions("selection")
    pending_sel = rv.pending("selection")
    pending_ext = rv.pending("extraction")
    if (pending_sel or pending_ext) and not allow_pending:
        raise LedgerError(f"{len(pending_ext)} studies await extraction and {len(pending_sel)} await selection; "
                          "finish them or pass --allow-pending to export only completed studies")
    ok_checks = {"row", "table"} if include_table_only else {"row"}
    studies, notes = [], []
    dropped = Counter()
    for pmid in rv.fulltext_included():
        a = rv.analyses(pmid)
        if a is None or pmid in pending_ext or pmid in pending_sel:
            dropped["study_pending"] += 1
            continue
        r = rv.records.get(pmid, {})
        nimads_analyses = []
        for x in a["analyses"]:
            if not all((pmid, x["analysis_id"], t) in sel for t in targets):
                dropped["analysis_without_decisions"] += 1
                continue
            pts = [p for p in x["points"] if p["verification"] in ok_checks or not drop_unverified]
            dropped["points_unverified"] += len(x["points"]) - len(pts)
            if not pts:
                dropped["analysis_no_verified_points"] += 1
                continue
            nimads_analyses.append({
                "id": x["analysis_id"], "name": x.get("name") or x["analysis_id"],
                "description": x.get("description"), "weights": [], "conditions": [], "images": [],
                "points": [{"id": f"{x['analysis_id']}-p{i}", "coordinates": p["xyz"], "space": p["space"],
                            "kind": None, "label_id": None, "values": p["values"],
                            "analysis_id": x["analysis_id"]} for i, p in enumerate(pts)],
                "study_id": pmid,
            })
            notes.append({"analysis_id": x["analysis_id"], "annotation_id": "cbma_skills",
                          "note": {t: sel[(pmid, x["analysis_id"], t)]["include"] for t in targets}})
        if not nimads_analyses:
            continue
        studies.append({
            "id": pmid, "pmid": pmid, "doi": r.get("doi"), "name": r.get("title"), "authors": ", ".join(r.get("authors") or []),
            "year": r.get("year"), "publication": r.get("journal"), "description": None,
            "metadata": {"fulltext_source": rv.fulltext_index.get(pmid, {}).get("source")},
            "analyses": nimads_analyses,
        })
    name = rv.spec.get("name") or rv.root.name
    studyset = {"id": f"cbma_skills_{name}", "name": name, "description": rv.spec["objective"],
                "publication": None, "doi": None, "pmid": None, "studies": studies}
    annotation = {"id": "cbma_skills", "name": "cbma_skills_selection",
                  "description": "Analysis selection decisions recorded by the cbma-skills ledger",
                  "studyset_id": studyset["id"], "note_keys": {t: "boolean" for t in targets}, "notes": notes}
    out = rv.root / "results" / "nimads"
    write_json(out / "studyset.json", studyset)
    write_json(out / "annotation.json", annotation)
    return {"studies": len(studies), "analyses": len(notes), "targets": targets, "dropped": dict(dropped),
            "written": [str(out / "studyset.json"), str(out / "annotation.json")]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "needs-fulltext", "status"):
        p = sub.add_parser(name)
        p.add_argument("review_dir", type=Path)
        if name == "status":
            p.add_argument("--json", action="store_true")
    p = sub.add_parser("batches")
    p.add_argument("review_dir", type=Path)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--size", type=int, default=25)
    p.add_argument("--limit", type=int, help="batch at most this many pending items (for pilots)")
    p.add_argument("--discard", action="store_true", help="drop un-ingested batch outputs")
    p = sub.add_parser("export")
    p.add_argument("review_dir", type=Path)
    p.add_argument("--allow-pending", action="store_true")
    p.add_argument("--include-table-only", action="store_true",
                   help="also export points verified only somewhere in the table, not within one row")
    p = sub.add_parser("ingest")
    p.add_argument("review_dir", type=Path)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--agent", default=os.environ.get("CBMA_AGENT", "unknown"),
                   help="harness/model that produced the output, e.g. claude-code/<model id>")
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        rv = Review(args.review_dir)
        if args.cmd == "init":
            for d in ("search", "docs", "fulltext", "decisions", "work", "analyses", "results"):
                (rv.root / d).mkdir(parents=True, exist_ok=True)
            write_json(rv.root / "results" / "criteria.json",
                       {s: {"criteria": rv.criteria[s]["criteria"], "hash": rv.criteria[s]["hash"]} for s in STAGES})
            print(json.dumps({s: rv.criteria[s]["criteria"] for s in STAGES}, indent=1))
            print("review.yaml is valid; criteria IDs written to results/criteria.json", file=sys.stderr)
        elif args.cmd == "batches":
            paths = cmd_batches(rv, args.stage, args.size, args.limit, args.discard)
            print(json.dumps({"stage": args.stage, "batches": [str(p) for p in paths],
                              "items": sum(len(json.loads(p.read_text())["items"]) for p in paths)}, indent=1))
        elif args.cmd == "ingest":
            report = cmd_ingest(rv, args.stage, args.agent)
            print(json.dumps(report, indent=1))
            return 0 if not report["errors"] else 2
        elif args.cmd == "needs-fulltext":
            path = rv.root / "fulltext" / "needed.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            passed = rv.abstract_passed()
            path.write_text("".join(f"{p}\n" for p in passed))
            print(f"{len(passed)} PMIDs passed abstract screening -> {path}")
        elif args.cmd == "export":
            print(json.dumps(cmd_export(rv, args.allow_pending, args.include_table_only), indent=1))
        elif args.cmd == "status":
            status = cmd_status(rv)
            print(json.dumps(status, indent=1))
    except LedgerError as exc:
        print(f"LEDGER ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
