"""
Purpose: Extract existing coordinates from PMC studies' supplementary files

Pipeline Methodology:
    1. If given PMID, uses NCBI's ID Converter to find corresponding PMCID. If given PMCID, uses PMCID directly.
    2. Fetches that article's JATS XML from PubMed Central (Open Access subset) using pubget's Entrez request/retry logic
    3. Locates the associated supplementary-material files and downloads them
    4. Converts each supplementary file to structured tables (via Docling)
    5. Scans the resulting tables for stereotactic coordinates (x, y, z triplets)
    6. Writes one combined CSV per article and stores in specified output folder

File Dependencies:
    - _entrez.py: Fetches article using pubget's EntrezClient handling NCBI's rate limiting, retries with backoff, and response validation.
        (Reference: _entrez.py from https://github.com/neuroquery/pubget)
    - ace_coordinate_extraction.py: Finds tables and extracts coordinates using ACE
        (Reference: https://github.com/neurosynth/ACE)

Design Notes:
    - Only works for articles that (a) exist in PMC, and (b) are in the PMC Open Access subset.

Requirements:
    - Packages: requests, lxml, pandas, docling

How to run:
    python extraction-pipeline.py PMID_OR_PMCID [--work-dir DIR] [--output-dir DIR] [--quiet-tables]

    Example:
        python extraction-pipeline.py 30921611 --work-dir directory --output-dir results
"""

from __future__ import annotations

import re, logging, requests, time
import pandas as pd

from pathlib import Path
from lxml import etree
from _entrez import EntrezClient
from ace_coordinate_extraction import (
    _dataframe_to_rows,
    classify_coordinate_table,
    is_coordinate_table_candidate,
    extract_coordinates_from_table,
    extract_coordinates_from_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extraction-pipeline")


#=======================================#
#           GLOBAL VARIABLES            #
#=======================================#

_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
_IDCONV_REQUEST_DELAY = 0.34
_IDCONV_MAX_ATTEMPTS = 5
_IDCONV_BACKOFF_BASE_SECONDS = 1.0
_last_idconv_request_time = [0.0]

_SUPP_REQUEST_DELAY = 0.34
_last_supp_request_time = [0.0]

_S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

_CONTACT_EMAIL = "anthonyang2027@utexas.edu"
_USER_AGENT = f"extraction-pipeline/1.0 (contact: {_CONTACT_EMAIL})"
_HTTP_HEADERS = {"User-Agent": _USER_AGENT}

# Supplementary files with any other extension are ignored.
ALLOWED_SUPPLEMENT_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".csv", ".pdf"}


def _throttle_idconv() -> None:
    """
    Purpose:
        Pause if needed so idconv requests don't go out faster than
        _IDCONV_REQUEST_DELAY seconds apart, then record this call's time.

    Inputs:
        None.

    Outputs:
        None.
    """
    elapsed = time.monotonic() - _last_idconv_request_time[0]
    if elapsed < _IDCONV_REQUEST_DELAY:
        time.sleep(_IDCONV_REQUEST_DELAY - elapsed)
    _last_idconv_request_time[0] = time.monotonic()


def _throttle_supp_download() -> None:
    """
    Purpose:
        Pause if needed so supplementary-file/S3 requests don't go out
        faster than _SUPP_REQUEST_DELAY seconds apart, then record this
        call's time.

    Inputs:
        None.

    Outputs:
        None.
    """
    elapsed = time.monotonic() - _last_supp_request_time[0]
    if elapsed < _SUPP_REQUEST_DELAY:
        time.sleep(_SUPP_REQUEST_DELAY - elapsed)
    _last_supp_request_time[0] = time.monotonic()


def normalize_pmid(pmid: str) -> str:
    """
    Purpose:
        Clean up a PMID string by removing a leading "PMID" label or a
        leading colon, so downstream code always gets bare digits.

    Inputs:
        pmid (str): the raw identifier, e.g. "PMID:12345" or ":12345".

    Outputs:
        str: the cleaned PMID, e.g. "12345".
    """
    pmid = str(pmid).strip()
    return pmid[4:] if pmid.upper().startswith("PMID") else pmid.lstrip(":").strip()


def normalize_pmcid(pmcid: str) -> str:
    """
    Purpose:
        Make sure a PMCID string starts with "PMC".

    Inputs:
        pmcid (str): the raw identifier, e.g. "1234567" or "PMC1234567".

    Outputs:
        str: the PMCID with a "PMC" prefix, e.g. "PMC1234567".
    """
    return pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"

_PMCID_FORMAT_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


def looks_like_pmcid(identifier: str) -> bool:
    """
    Purpose:
        Check whether an identifier looks like a PMCID (starts with "PMC")
        rather than a PMID, so the pipeline knows which lookup to skip.

    Inputs:
        identifier (str): the raw identifier entered by the user.

    Outputs:
        bool: True if it starts with "PMC".
    """
    return str(identifier).strip().upper().startswith("PMC")


def is_valid_pmcid(pmcid: str) -> bool:
    """
    Purpose:
        Check that a string is a properly formatted PMCID: "PMC" followed
        by one or more digits, and nothing else.

    Inputs:
        pmcid (str): the identifier to check.

    Outputs:
        bool: True if it matches the "PMC" + digits format.
    """
    return bool(_PMCID_FORMAT_RE.match(str(pmcid).strip()))


def _idconv_lookup(identifier: str) -> dict:
    """
    Purpose:
        Look up a PMID or PMCID in NCBI's ID Converter service and return
        its matching record, retrying with backoff if the request fails
        or NCBI asks us to slow down.

    Inputs:
        identifier (str): a PMID or PMCID to look up.

    Outputs:
        dict: the first matching record, which may contain "pmid",
            "pmcid", "doi", etc. depending on what NCBI has on file.
            Returns {} if there was no match.

    Notes:
        Retries up to _IDCONV_MAX_ATTEMPTS times, waiting longer each time
        (or honoring the server's Retry-After header), on a network error
        or an HTTP 429/5xx response. Raises RuntimeError if it still
        hasn't completed after all attempts.
    """
    for attempt in range(1, _IDCONV_MAX_ATTEMPTS + 1):
        _throttle_idconv()
        try:
            resp = requests.get(
                _IDCONV_URL, params={"ids": identifier, "format": "json"}, headers=_HTTP_HEADERS, timeout=30
            )
        except requests.RequestException as e:
            if attempt == _IDCONV_MAX_ATTEMPTS:
                raise
            wait = _IDCONV_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "idconv lookup for %r failed (%s) on attempt %d/%d -- retrying in %.1fs",
                identifier, e, attempt, _IDCONV_MAX_ATTEMPTS, wait,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == _IDCONV_MAX_ATTEMPTS:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else None
            except ValueError:
                wait = None
            if wait is None:
                wait = _IDCONV_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "idconv lookup for %r got HTTP %d on attempt %d/%d -- retrying in %.1fs",
                identifier, resp.status_code, attempt, _IDCONV_MAX_ATTEMPTS, wait,
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        records = data.get("records", [])
        return records[0] if records else {}

    raise RuntimeError(f"idconv lookup for {identifier!r} did not complete after {_IDCONV_MAX_ATTEMPTS} attempts")


def pmid_to_pmcid(pmid: str) -> str | None:
    """
    Purpose:
        Look up the PMCID that corresponds to a given PMID.

    Inputs:
        pmid (str): the PMID to look up.

    Outputs:
        str or None: the matching PMCID, or None if NCBI has no PMCID on
            file for this PMID.
    """
    pmid = normalize_pmid(pmid)
    _throttle_idconv()
    resp = requests.get(_IDCONV_URL, params={"ids": pmid, "format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    if not records or "pmcid" not in records[0]:
        return None
    return records[0]["pmcid"]

def pmcid_to_pmid(pmcid: str) -> str | None:
    """
    Purpose:
        Look up the PMID that corresponds to a given PMCID.

    Inputs:
        pmcid (str): the PMCID to look up.

    Outputs:
        str or None: the matching PMID, or None if NCBI has no PMID on
            file for this PMCID.
    """
    pmcid = normalize_pmcid(pmcid)
    return _idconv_lookup(pmcid).get('pmid')


def fetch_jats_xml(client: EntrezClient, pmcid: str, batch_dir: Path) -> etree._Element:
    """
    Purpose:
        Download an article's JATS XML from PubMed Central via Entrez and
        return the <article> element that matches this PMCID.

    Inputs:
        client (EntrezClient): handles the actual Entrez requests, rate
            limiting, and retries.
        pmcid (str): the article's PMCID.
        batch_dir (Path): folder to download the Entrez XML batch into.

    Outputs:
        lxml.etree._Element: the <article> element for this PMCID.

    Raises:
        RuntimeError: if the PMCID isn't in the PMC Open Access subset, or
            no <article> elements were found in the downloaded batch.
    """
    pmcid = normalize_pmcid(pmcid)
    numeric_id = int(pmcid.upper().replace("PMC", ""))

    search_result = client.epost([numeric_id])
    if not search_result or int(search_result.get("count", 0)) == 0:
        raise RuntimeError(
            f"{pmcid} was not found in the PMC Open Access subset "
            "(or the Entrez request failed -- check logs above)."
        )

    batch_dir.mkdir(parents=True, exist_ok=True)
    client.efetch(batch_dir, search_result=search_result)

    articles = []
    for batch_file in sorted(batch_dir.glob("articleset_*.xml")):
        root = etree.parse(str(batch_file)).getroot()
        articles.extend(root.findall("article"))

    if not articles:
        raise RuntimeError(f"No <article> elements found in the downloaded batch(es) for {pmcid}.")

    if len(articles) == 1:
        return articles[0]

    # More than one article came back in the batch -- match by the
    # article's own PMC id instead of just assuming the first one is ours.
    for article in articles:
        for article_id in article.findall(".//article-id"):
            if article_id.get("pub-id-type") != "pmc":
                continue
            digits_only = re.sub(r"\D", "", article_id.text or "")
            if digits_only == str(numeric_id):
                return article

    log.warning(
        "Got %d articles back for %s but couldn't match by article-id; using the first one.",
        len(articles), pmcid,
    )
    return articles[0]


class SupplementaryFile:
    """One supplementary-material file referenced in a paper's JATS XML."""

    def __init__(self, pmcid: str, label: str, href: str, mimetype: str = ""):
        """
        Purpose:
            Store the information needed to locate and download one
            supplementary file.

        Inputs:
            pmcid (str): the article's PMCID this file belongs to.
            label (str): the file's label/caption from the JATS XML (e.g.
                "Supplementary Table 2").
            href (str): the file reference (usually a filename) from the
                JATS XML.
            mimetype (str, optional): the file's MIME type, if known.

        Outputs:
            None.
        """
        self.pmcid = pmcid
        self.label = label
        self.href = href
        self.mimetype = mimetype


def find_supplementary_links_in_jats(article: etree._Element, pmcid: str) -> list[SupplementaryFile]:
    """
    Purpose:
        Find every unique supplementary-material file referenced in an
        article's JATS XML.

    Inputs:
        article (lxml.etree._Element): the article's <article> element.
        pmcid (str): the article's PMCID, stored on each result.

    Outputs:
        list of SupplementaryFile: one entry per unique file reference
            (href). A file linked more than once in the XML is only
            included once.
    """
    results = []
    seen_hrefs = set()
    for supp in article.iter("supplementary-material"):
        label_el = supp.find(".//label")
        label = label_el.text if label_el is not None else "supplementary-material"
        for ref_tag in ("media", "ext-link", "graphic"):
            for el in supp.iter(ref_tag):
                href = el.get("{http://www.w3.org/1999/xlink}href")
                if not href:
                    continue
                if href in seen_hrefs:  # same file linked again elsewhere in the XML -- skip it
                    log.info(
                        "%s: %r is linked more than once in the JATS (label=%r) -- keeping the first reference.",
                        pmcid, href, label,
                    )
                    continue
                seen_hrefs.add(href)
                mimetype = f"{el.get('mimetype', '')}/{el.get('mime-subtype', '')}"
                results.append(SupplementaryFile(pmcid=pmcid, label=label, href=href, mimetype=mimetype))
    return results


def find_article_s3_prefix(pmcid: str) -> str | None:
    """
    Purpose:
        Find this article's folder (key prefix) in the PMC Open Access S3
        bucket, picking the newest package version if more than one
        exists.

    Inputs:
        pmcid (str): the article's PMCID.

    Outputs:
        str or None: the S3 key prefix (e.g. "PMC1234567.2"), or None if
            the article has no package in the bucket.
    """
    import requests

    _throttle_supp_download()
    resp = requests.get(
        _S3_BASE + "/",
        params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
        headers=_HTTP_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    root = etree.fromstring(resp.content)
    prefixes = [el.text for el in root.findall(".//s3:CommonPrefixes/s3:Prefix", _S3_NS)]
    if not prefixes:
        return None

    def _version_number(prefix: str) -> int:
        # The package version is the trailing number after the last "."
        # (e.g. "PMC1234567.2/" -> 2). Non-numeric suffixes sort last.
        suffix = prefix.rstrip("/").rsplit(".", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    return max(prefixes, key=_version_number).rstrip("/")


def fetch_article_s3_metadata(prefix: str) -> dict:
    """
    Purpose:
        Download and parse an article's metadata JSON file from its S3
        package (this is where the real supplementary-file URLs live).

    Inputs:
        prefix (str): the article's S3 key prefix, from
            find_article_s3_prefix.

    Outputs:
        dict: the parsed metadata JSON.
    """
    _throttle_supp_download()
    resp = requests.get(f"{_S3_BASE}/{prefix}/{prefix}.json", headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _s3_url_to_https(url: str) -> str:
    """
    Purpose:
        Convert an "s3://bucket/key" URL into a plain https:// URL that
        can be downloaded with a normal GET request.

    Inputs:
        url (str): the URL to convert.

    Outputs:
        str: the https:// URL, or the original string unchanged if it
            didn't start with "s3://".
    """
    if url.startswith("s3://"):
        without_scheme = url[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


def download_supplementary_files(files: list[SupplementaryFile], out_dir: Path) -> list[Path]:
    """
    Purpose:
        Download each wanted supplementary file to out_dir, matching each
        JATS file reference to its real download URL in the article's S3
        package.

    Inputs:
        files (list of SupplementaryFile): the files to download, as found
            by find_supplementary_links_in_jats.
        out_dir (Path): folder to save downloaded files into.

    Outputs:
        list of Path: local paths of the files actually saved. Files that
            couldn't be matched or downloaded are skipped (and logged),
            not included here.

    Notes:
        Skips files already queued for the same article under the same
        filename, and files whose extension isn't in
        ALLOWED_SUPPLEMENT_EXTENSIONS. When a file's exact filename isn't
        found in the article's S3 metadata, falls back to matching the
        next unclaimed S3 file of the same extension, in order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop duplicate references to the same file (same article, same
    # filename) before doing any network work.
    deduped_files = []
    seen_targets = set()
    for f in files:
        target_key = (f.pmcid, Path(f.href).name)
        if target_key in seen_targets:
            log.info("Already queued %s for %s -- skipping duplicate reference.", f.href, f.pmcid)
            continue
        seen_targets.add(target_key)
        deduped_files.append(f)
    files = deduped_files

    # Only keep files whose extension we know how to convert later.
    allowed_files = []
    for f in files:
        ext = Path(f.href).suffix.lower()
        if ext in ALLOWED_SUPPLEMENT_EXTENSIONS:
            allowed_files.append(f)
        else:
            log.info("Skipping %s (extension %r not in %s)", f.href, ext, sorted(ALLOWED_SUPPLEMENT_EXTENSIONS))
    files = allowed_files

    saved = []
    media_urls_by_pmcid: dict[str, list[str]] = {}
    unclaimed_by_pmcid_ext: dict[tuple[str, str], list[str]] = {}

    for f in files:
        if f.pmcid not in media_urls_by_pmcid:
            prefix = find_article_s3_prefix(f.pmcid)
            if prefix is None:
                log.warning("No S3 article version found for %s", f.pmcid)
                media_urls_by_pmcid[f.pmcid] = []
            else:
                metadata = fetch_article_s3_metadata(prefix)
                media_urls_by_pmcid[f.pmcid] = metadata.get("media_urls") or []

        media_urls = media_urls_by_pmcid[f.pmcid]
        target_basename = Path(f.href).name
        target_ext = Path(f.href).suffix.lower()
        match_url = next(
            (u for u in media_urls if Path(u.split("?")[0]).name == target_basename),
            None,
        )

        if match_url is None:
            # No exact filename match in S3 -- fall back to matching the
            # next S3 file of the same extension that no other JATS
            # reference has already claimed exactly.
            ext_key = (f.pmcid, target_ext)
            if ext_key not in unclaimed_by_pmcid_ext:
                exactly_claimed = {
                    Path(u.split("?")[0]).name
                    for other in files
                    if other.pmcid == f.pmcid and Path(other.href).suffix.lower() == target_ext
                    for u in media_urls
                    if Path(u.split("?")[0]).name == Path(other.href).name
                }
                unclaimed_by_pmcid_ext[ext_key] = [
                    u for u in media_urls
                    if Path(u.split("?")[0]).suffix.lower() == target_ext
                    and Path(u.split("?")[0]).name not in exactly_claimed
                ]
            remaining = unclaimed_by_pmcid_ext[ext_key]
            if remaining:
                match_url = remaining.pop(0)
                log.info(
                    "%s has no exact filename match among %s's S3 files -- matched positionally to %s instead.",
                    target_basename, f.pmcid, Path(match_url.split("?")[0]).name,
                )

        if match_url is None:
            log.warning("Could not find %s among %s's S3 media files -- skipping", target_basename, f.pmcid)
            continue

        try:
            _throttle_supp_download()
            resp = requests.get(_s3_url_to_https(match_url), headers=_HTTP_HEADERS, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Could not download %s: %s", match_url, e)
            continue

        fname = out_dir / target_basename
        fname.write_bytes(resp.content)
        saved.append(fname)
        log.info("Downloaded %s", fname)

    return saved


def convert_with_docling(file_path: Path) -> tuple[list[pd.DataFrame], str]:
    """
    Purpose:
        Convert a document (PDF, DOCX, etc.) into its tables and its full
        text, using Docling.

    Inputs:
        file_path (Path): the file to convert.

    Outputs:
        tuple of (list of pandas.DataFrame, str):
            the tables Docling found in the document, and the document's
            full text rendered as markdown ("" if text export failed).
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )
    result = converter.convert(str(file_path))

    tables = []
    for table in result.document.tables:
        try:
            tables.append(table.export_to_dataframe(doc=result.document))
        except Exception as e:
            log.warning("Failed exporting a table from %s: %s", file_path, e)

    try:
        text = result.document.export_to_markdown()
    except Exception as e:
        log.warning("Could not export text from %s: %s", file_path, e)
        text = ""

    return tables, text


def load_tables_and_text(file_path: Path) -> tuple[list[pd.DataFrame], str]:
    """
    Purpose:
        Load a supplementary file's tables (and text, where available),
        choosing the right method for the file's type.

    Inputs:
        file_path (Path): the file to load.

    Outputs:
        tuple of (list of pandas.DataFrame, str):
            the tables found in the file, and its text (only non-empty
            for PDF/DOCX/DOC/PPTX/HTML files, which go through Docling).
            Returns ([], "") for an unsupported file type.
    """
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return list(pd.read_excel(file_path, sheet_name=None).values()), ""
    if suffix == ".csv":
        return [pd.read_csv(file_path)], ""
    if suffix in (".pdf", ".docx", ".doc", ".pptx", ".html", ".htm"):
        return convert_with_docling(file_path)
    log.warning("Unsupported file type for %s, skipping", file_path)
    return [], ""


def table_looks_like_coordinates(df: pd.DataFrame) -> bool:
    """
    Purpose:
        Quick check for whether a DataFrame looks like a coordinate table.

    Inputs:
        df (pandas.DataFrame): the table to check.

    Outputs:
        bool: True if the table is a coordinate-table candidate.
    """
    return is_coordinate_table_candidate(rows=_dataframe_to_rows(df))


def process_pmid(pmid: str, client: EntrezClient, work_dir: Path, verbose: bool = True) -> tuple[str, pd.DataFrame]:
    """
    Purpose:
        Run the full pipeline for one article: resolve its PMID/PMCID,
        fetch its JATS XML, download its supplementary files, and extract
        coordinates from every table (and text) found in them.

    Inputs:
        pmid (str): a PMID or a PMCID (PMCIDs are detected automatically
            and used as-is).
        client (EntrezClient): used to fetch the article's JATS XML.
        work_dir (Path): folder to store downloaded files into.
        verbose (bool, optional): if True, log per-table/per-row detail
            from ace_coordinate_extraction.py. Defaults to True.

    Outputs:
        tuple of (str, pandas.DataFrame):
            the resolved PMCID, and a DataFrame of every coordinate found
            across all of its supplementary files (empty if none were
            found).

    Raises:
        RuntimeError: if the identifier is an invalid PMCID, or a given
            PMID has no corresponding PMCID in PMC.
    """
    identifier = str(pmid).strip()
    log.info("Processing %s", identifier)

    if looks_like_pmcid(identifier):
        pmcid = normalize_pmcid(identifier)
        if not is_valid_pmcid(pmcid):
            log.warning(
                "%r is not a valid PMCID (expected 'PMC' followed by digits). Skipping.",
                identifier,
            )
            raise RuntimeError(f"{identifier!r} is not a valid PMCID.")
        log.info("%s is already a PMCID.", pmcid)
        pmid = pmcid_to_pmid(pmcid) or ""
        if pmid:
            log.info("%s -> PMID %s", pmcid, pmid)
        else:
            log.info(
                "%s has no corresponding PMID on file with NCBI "
                "-- leaving the PMID field blank.",
                pmcid,
            )
    else:
        pmid = normalize_pmid(identifier)
        pmcid = pmid_to_pmcid(pmid)
        if pmcid is None:
            raise RuntimeError(f"PMID {pmid} does not have a corresponding PMCID.")
        log.info("PMID %s -> %s", pmid, pmcid)

    article = fetch_jats_xml(client, pmcid, batch_dir=work_dir / pmcid / "articleset")
    supp_files = find_supplementary_links_in_jats(article, pmcid)
    if not supp_files:
        log.info("No <supplementary-material> links found in JATS for %s", pmcid)

    downloaded = download_supplementary_files(supp_files, work_dir / pmcid / "supplementary")

    all_coords = []
    for path in downloaded:
        try:
            tables, doc_text = load_tables_and_text(path)
        except Exception as e:
            log.error("Could not convert %s, skipping: %s", path.name, e)
            continue

        log.info("%s: Docling found %d table(s)", path.name, len(tables))

        for i, df in enumerate(tables):
            decision = classify_coordinate_table(rows=_dataframe_to_rows(df))
            log.info(
                "%s table #%d: shape=%s columns=%s -> candidate=%s reasons=%s",
                path.name, i, df.shape, list(df.columns), decision.candidate, decision.reasons,
            )
            if not decision.candidate:
                continue

            coords = extract_coordinates_from_table(
                df, source_file=path.name, pmcid=pmcid, pmid=pmid, verbose=verbose
            )
            log.info("%s table #%d: extracted %d coordinate row(s)", path.name, i, len(coords))
            if not coords.empty:
                all_coords.append(coords)

        if doc_text:
            text_coords = extract_coordinates_from_text(
                doc_text, source_file=path.name, pmcid=pmcid, pmid=pmid, verbose=verbose
            )
            log.info("%s: text-fallback extraction found %d coordinate row(s)", path.name, len(text_coords))
            if not text_coords.empty:
                all_coords.append(text_coords)

    if not all_coords:
        log.info("No coordinate tables found for %s", pmcid)
        return pmcid, pd.DataFrame(columns=["pmid", "pmcid", "source_file", "x", "y", "z", "region", "group"])

    return pmcid, pd.concat(all_coords, ignore_index=True)


def run_batch(pmids: list[str], work_dir: str = "pmc_downloads", output_dir: str = "extracted-coordinates", verbose: bool = True,) -> dict[str, pd.DataFrame]:
    """
    Purpose:
        Run process_pmid for a list of identifiers, writing one
        coordinates CSV per article that produced any rows.

    Inputs:
        pmids (list of str): PMIDs and/or PMCIDs to process.
        work_dir (str, optional): folder to store downloads into.
            Defaults to "pmc_downloads".
        output_dir (str, optional): folder to write per-article CSVs
            into. Defaults to "extracted-coordinates".
        verbose (bool, optional): passed through to process_pmid.
            Defaults to True.

    Outputs:
        dict mapping PMCID (str) to pandas.DataFrame: the coordinates
            found for every identifier that didn't raise an error.
            Identifiers that failed are logged and skipped, not included.
    """
    work_dir_path = Path(work_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    client = EntrezClient(failed_requests_dump_dir=work_dir_path / "failed_requests_dumps")

    results = {}
    for identifier in pmids:
        identifier = normalize_pmid(identifier)
        try:
            pmcid, df = process_pmid(identifier, client, work_dir_path, verbose=verbose)
        except Exception as e:
            log.error("Failed on %s: %s", identifier, e)
            continue

        if len(df) == 0:
            log.warning(f"0 coordinate rows extracted. No table was created for {pmcid}")
        else:
            out_path = output_dir_path / f"{pmcid}-coordinates.csv"
            df.to_csv(out_path, index=False)
            log.info("Wrote %d coordinate rows to %s", len(df), out_path)
        results[pmcid] = df

    return results


def _parse_args() -> "argparse.Namespace":
    """
    Purpose:
        Parse command-line arguments for running this script directly.

    Inputs:
        None (reads from sys.argv).

    Outputs:
        argparse.Namespace: the parsed arguments (pmids, work_dir,
            output_dir, quiet_tables).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch PMC articles, locate supplementary data, and extract coordinate tables."
    )
    parser.add_argument(
        "pmids",
        nargs="+",
        help=(
            "One or more identifiers, each either a PMID or a PMCID."
            "A PMCID is detected by its 'PMC' prefix and used as-is without the "
            "PMID -> PMCID lookup; anything else is treated as a PMID."
        ),
    )
    parser.add_argument("--work-dir", default="pmc_downloads", help="Directory to store downloads.")
    parser.add_argument(
        "--output-dir", default="extracted-coordinates", help="Directory to write per-PMID coordinate CSVs."
    )
    parser.add_argument(
        "--quiet-tables",
        action="store_true",
        help=(
            "Suppress the per-table/per-row debug logging from "
            "ace_coordinate_extraction.py (raw table contents, resolved "
            "column roles, and each row's kept/dropped reason). On by "
            "default so you can see what was extracted from each "
            "supplementary file; pass this flag to quiet it for a large "
            "batch run."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_batch(args.pmids, work_dir=args.work_dir, output_dir=args.output_dir, verbose=not args.quiet_tables)