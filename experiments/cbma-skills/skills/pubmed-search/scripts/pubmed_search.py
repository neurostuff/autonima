"""PubMed search through NCBI E-utilities. Standard library only; no API key needed.

    python pubmed_search.py REVIEW_DIR                  # reads search: from REVIEW_DIR/review.yaml
    python pubmed_search.py REVIEW_DIR --query "..."    # or pass the query directly
    python pubmed_search.py REVIEW_DIR --pmids-file ids.txt

Writes:
    REVIEW_DIR/search/records.jsonl   one record per unique PMID, sorted by PMID
    REVIEW_DIR/search/search_log.json what was asked, what NCBI reported, what came back

Behaviour that matters for a systematic review:
  * No silent truncation. PubMed's esearch stops at 9,999 records per query, so a
    larger result set is split into publication-date windows until every window
    fits. The log records the reported count and the retrieved count; they must
    match or the script exits non-zero.
  * Every abstract section is kept (structured abstracts arrive as several
    AbstractText elements), and titles keep text inside inline markup.
  * The DOI comes only from the article's own ArticleIdList or ELocationID, never
    from the reference list.
  * PMIDs are strings. Query results and a PMID list are unioned and de-duplicated,
    and the log says which source each PMID came from.
  * A failed fetch batch is retried, then reported as an error. It is never
    skipped.

Set NCBI_API_KEY to raise the rate limit from 3 to 10 requests per second. It is optional.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_CAP = 9999
FETCH_BATCH = 200
TOOL = "cbma-skills"

Http = Callable[[str, Dict[str, str]], bytes]


class SearchError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Client:
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None,
                 http: Optional[Http] = None, retries: int = 4):
        self.email = email
        self.api_key = api_key
        self.retries = retries
        self._http = http or self._urlopen
        self._interval = 0.11 if api_key else 0.34
        self._last = 0.0
        self.n_requests = 0

    @staticmethod
    def _urlopen(url: str, data: Dict[str, str]) -> bytes:
        body = urllib.parse.urlencode(data).encode()
        req = urllib.request.Request(url, data=body, headers={"User-Agent": TOOL})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    def call(self, endpoint: str, params: Dict[str, str]) -> bytes:
        params = dict(params, tool=TOOL)
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{EUTILS}/{endpoint}.fcgi"
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1
            try:
                return self._http(url, params)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.retries:
                    raise SearchError(f"{endpoint} failed with HTTP {exc.code}: {exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.retries:
                    raise SearchError(f"{endpoint} failed after {attempt} attempts: {exc}") from exc
            time.sleep(delay)
            delay *= 2
        raise SearchError(f"{endpoint} failed")  # unreachable


# --------------------------------------------------------------------------- #
# esearch with date splitting
# --------------------------------------------------------------------------- #

def _esearch(client: Client, term: str, mindate: Optional[str], maxdate: Optional[str],
             retmax: int = 0) -> Tuple[int, List[str]]:
    params = {"db": "pubmed", "term": term, "retmax": str(retmax), "retmode": "json"}
    if mindate or maxdate:
        params.update(datetype="pdat", mindate=mindate or "1800/01/01", maxdate=maxdate or "3000/12/31")
    data = json.loads(client.call("esearch", params))
    result = data.get("esearchresult", {})
    if "ERROR" in result:
        raise SearchError(f"esearch error: {result['ERROR']}")
    return int(result.get("count", 0)), [str(x) for x in result.get("idlist", [])]


def _parse_date(s: str) -> dt.date:
    parts = [int(p) for p in re.split(r"[/-]", s)]
    while len(parts) < 3:
        parts.append(1)
    return dt.date(*parts[:3])


def _fmt(d: dt.date) -> str:
    return d.strftime("%Y/%m/%d")


def search_ids(client: Client, term: str, date_from: Optional[str] = None,
               date_to: Optional[str] = None, log: Optional[dict] = None) -> List[str]:
    """Every PMID matching term, splitting date windows until each fits under the cap."""
    total, ids = _esearch(client, term, date_from, date_to, retmax=ESEARCH_CAP)
    if log is not None:
        log["reported_count"] = total
        log["windows"] = []
    if total <= ESEARCH_CAP:
        if log is not None:
            log["windows"].append({"from": date_from, "to": date_to, "count": total})
        return sorted(set(ids), key=int)
    start = _parse_date(date_from) if date_from else dt.date(1800, 1, 1)
    # Issue dates can sit a year or so in the future; the default end must cover them.
    end = _parse_date(date_to) if date_to else dt.date.today() + dt.timedelta(days=3 * 365)
    out: List[str] = []
    stack = [(start, end)]
    while stack:
        lo, hi = stack.pop()
        count, ids = _esearch(client, term, _fmt(lo), _fmt(hi), retmax=ESEARCH_CAP)
        if count <= ESEARCH_CAP:
            out.extend(ids)
            if log is not None:
                log["windows"].append({"from": _fmt(lo), "to": _fmt(hi), "count": count})
            continue
        if lo >= hi:
            raise SearchError(
                f"{count} records share the single publication date {_fmt(lo)}; "
                "narrow the query, esearch cannot page past 9,999 within one day"
            )
        mid = lo + (hi - lo) // 2
        stack.append((mid + dt.timedelta(days=1), hi))
        stack.append((lo, mid))
    return sorted(set(out), key=int)


# --------------------------------------------------------------------------- #
# efetch + parsing
# --------------------------------------------------------------------------- #

def _text(el: Optional[ET.Element]) -> str:
    """All text inside el, including text inside inline markup like <i> and <sup>."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_pubmed_xml(xml_bytes: bytes) -> List[dict]:
    root = ET.fromstring(xml_bytes)
    records = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        pmid = _text(cit.find("PMID"))
        article = cit.find("Article")
        if article is None or not pmid:
            continue

        sections = []
        for ab in article.findall("Abstract/AbstractText"):
            label = ab.get("Label") or ab.get("NlmCategory")
            body = _text(ab)
            if body:
                sections.append(f"{label.upper()}: {body}" if label else body)
        abstract = "\n".join(sections)

        authors = []
        for a in article.findall("AuthorList/Author"):
            if a.find("CollectiveName") is not None:
                authors.append(_text(a.find("CollectiveName")))
            else:
                last, init = _text(a.find("LastName")), _text(a.find("Initials"))
                if last:
                    authors.append(f"{last} {init}".strip())

        pubdate = article.find("Journal/JournalIssue/PubDate")
        year = _text(pubdate.find("Year")) if pubdate is not None else ""
        if not year and pubdate is not None:
            m = re.search(r"\d{4}", _text(pubdate.find("MedlineDate")))
            year = m.group(0) if m else ""

        ids: Dict[str, str] = {}
        pubmed_data = art.find("PubmedData")
        if pubmed_data is not None:
            # Only the article's own id list; never .//ArticleId, which also
            # matches every entry in ReferenceList.
            for aid in pubmed_data.findall("ArticleIdList/ArticleId"):
                kind = aid.get("IdType")
                if kind and aid.text:
                    ids.setdefault(kind, aid.text.strip())
        doi = ids.get("doi")
        if not doi:
            for loc in article.findall("ELocationID"):
                if loc.get("EIdType") == "doi" and loc.text:
                    doi = loc.text.strip()
                    break
        pmcid = ids.get("pmc")

        records.append({
            "pmid": pmid,
            "title": _text(article.find("ArticleTitle")),
            "abstract": abstract,
            "authors": authors,
            "journal": _text(article.find("Journal/Title")),
            "year": int(year) if year.isdigit() else None,
            "doi": doi.lower() if doi else None,
            "pmcid": pmcid if not pmcid or pmcid.startswith("PMC") else f"PMC{pmcid}",
            "publication_types": [_text(p) for p in article.findall("PublicationTypeList/PublicationType")],
            "mesh": [_text(d) for d in cit.findall("MeshHeadingList/MeshHeading/DescriptorName")],
            "keywords": [_text(k) for k in cit.findall("KeywordList/Keyword")],
            "language": [_text(lang) for lang in article.findall("Language")],
            "has_abstract": bool(abstract),
        })
    return records


def fetch_records(client: Client, pmids: List[str]) -> Tuple[List[dict], List[str]]:
    records: Dict[str, dict] = {}
    for i in range(0, len(pmids), FETCH_BATCH):
        batch = pmids[i:i + FETCH_BATCH]
        xml = client.call("efetch", {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
        for rec in parse_pubmed_xml(xml):
            records[rec["pmid"]] = rec
        print(f"  fetched {min(i + FETCH_BATCH, len(pmids))}/{len(pmids)}", file=sys.stderr)
    missing = [p for p in pmids if p not in records]
    return [records[p] for p in sorted(records, key=int)], missing


def normalize_pmid(value) -> str:
    s = str(value).strip()
    s = re.sub(r"^(pmid:?\s*)", "", s, flags=re.I)
    if not s.isdigit():
        raise SearchError(f"not a PMID: {value!r}")
    return str(int(s))


def read_pmids(path: Path) -> List[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(normalize_pmid(line))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def load_search_spec(review_dir: Path) -> dict:
    spec_path = review_dir / "review.yaml"
    if not spec_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("review.yaml needs PyYAML: pip install pyyaml") from exc
    return (yaml.safe_load(spec_path.read_text()) or {}).get("search", {}) or {}


def run(review_dir: Path, query: Optional[str], pmids: List[str], date_from: Optional[str],
        date_to: Optional[str], email: Optional[str], client: Optional[Client] = None) -> dict:
    client = client or Client(email=email, api_key=os.environ.get("NCBI_API_KEY"))
    log: dict = {
        "query": query, "date_from": date_from, "date_to": date_to,
        "pmid_list_size": len(pmids), "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "eutils": EUTILS,
    }
    source: Dict[str, List[str]] = {}
    if query:
        qlog: dict = {}
        for p in search_ids(client, query, date_from, date_to, log=qlog):
            source.setdefault(p, []).append("query")
        log["query_result"] = qlog
        log["query_retrieved"] = sum(1 for v in source.values() if "query" in v)
        if log["query_retrieved"] != qlog.get("reported_count", 0):
            raise SearchError(
                f"PubMed reported {qlog.get('reported_count')} records but {log['query_retrieved']} were retrieved"
            )
    for p in pmids:
        source.setdefault(p, []).append("pmid_list")
    all_ids = sorted(source, key=int)
    records, missing = fetch_records(client, all_ids)
    for rec in records:
        rec["found_by"] = source[rec["pmid"]]
    out = review_dir / "search"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "records.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out / "records.jsonl")
    log.update({
        "unique_pmids": len(all_ids),
        "records_written": len(records),
        "in_both_query_and_list": sum(1 for v in source.values() if len(v) > 1),
        "missing_from_efetch": missing,
        "without_abstract": sum(1 for r in records if not r["has_abstract"]),
        "requests": client.n_requests,
    })
    (out / "search_log.json").write_text(json.dumps(log, indent=1))
    return log


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    ap.add_argument("--query")
    ap.add_argument("--pmids-file", type=Path)
    ap.add_argument("--date-from", help="YYYY/MM/DD, publication date")
    ap.add_argument("--date-to", help="YYYY/MM/DD, publication date")
    ap.add_argument("--email", help="contact address NCBI asks tools to send")
    args = ap.parse_args(list(argv) if argv is not None else None)

    spec = load_search_spec(args.review_dir)
    query = args.query or spec.get("query")
    pmids = [normalize_pmid(p) for p in spec.get("pmids", [])]
    pmids_file = args.pmids_file or (args.review_dir / spec["pmids_file"] if spec.get("pmids_file") else None)
    if pmids_file:
        pmids += read_pmids(Path(pmids_file))
    if not query and not pmids:
        ap.error("give a query or PMIDs, on the command line or under search: in review.yaml")
    try:
        log = run(args.review_dir, query, pmids, args.date_from or spec.get("date_from"),
                  args.date_to or spec.get("date_to"), args.email or spec.get("email"))
    except SearchError as exc:
        print(f"SEARCH FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in log.items() if k != "query_result"}, indent=1))
    if log["missing_from_efetch"]:
        print(f"WARNING: {len(log['missing_from_efetch'])} PMIDs returned no record from efetch "
              "(withdrawn, book chapters, or not PubMed articles); see search_log.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
