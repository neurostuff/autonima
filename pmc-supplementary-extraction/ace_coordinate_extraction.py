"""
Purpose:
    Find and extract neuroimaging coordinate tables (the kind with a column
    of 3D brain coordinates like Talairach or MNI x, y, z, next to a region
    name, Brodmann area, and a statistic) from the tables that Docling
    pulls out of a paper's full text or supplementary DOCX/PDF/DOC files.
    Cleans up the many ways Docling can damage a table's header, then turns
    each coordinate table into tidy rows of (x, y, z, region, group, ...).

Origin:
    Ported from the neurosynth/ACE project: https://github.com/neurosynth/ACE. 
    The table-classification and column-identification
    logic (classify_coordinate_table, identify_standard_columns,
    identify_repeating_groups) is kept close to ACE's original. The row-
    extraction and table-orchestration functions were rewritten to work
    with pandas DataFrames instead of ACE's original HTML-table parser and
    ORM objects. Everything else (the `_repair_*` functions and the raw-
    text fallback) is new, written to handle header damage specific to
    Docling's DOCX/PDF/DOC conversion.

Requirements:
    pip install regex pandas --break-system-packages

How to run:
    This is a library module, not a standalone script. In this pipeline
    it is called by extraction-pipeline.py, once per table Docling finds.
    To be used on its own, refer to below:

        import pandas as pd
        from ace_coordinate_extraction import extract_coordinates_from_table

        df = pd.DataFrame(...)  # a table from Docling, or built by hand
        coords = extract_coordinates_from_table(
            df, source_file="mmc1.docx", pmcid="PMC1234567", pmid="12345678",
        )
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict, namedtuple

import regex
import pandas as pd

logger = logging.getLogger(__name__)

# One table classification result: whether it's worth inspecting, and the specific heuristics that fired
CoordinateTableDecision = namedtuple("CoordinateTableDecision", ["candidate", "reasons"])

# --------------------------------------------------------------------------
# Shared regex building blocks
# --------------------------------------------------------------------------

# Unicode minus/dash variants that documents sometimes use instead of a plain ASCII "-" for negative coordinates
_MINUS_CHARS = "֊‐‑⁃﹣－‒–—﹘−"

# A single bare number, e.g. "-6" or "88.5".
_NUMBER_RE = regex.compile(r"(?<![\d.])[+\-%s]?\s*\d{1,3}(?:\.\d+)?(?![\d.])" % _MINUS_CHARS)

# Three numbers in a row separated by a comma, semicolon, slash, pipe, tab, or whitespace
_TRIPLET_RE = regex.compile(
    r"(?<!\d)[+\-%s]?\s*\d{1,3}(?:\.\d+)?"
    r"\s*[,;/|\t ]+\s*[+\-%s]?\s*\d{1,3}(?:\.\d+)?"
    r"\s*[,;/|\t ]+\s*[+\-%s]?\s*\d{1,3}(?:\.\d+)?(?!\d)"
    % (_MINUS_CHARS, _MINUS_CHARS, _MINUS_CHARS),
    regex.IGNORECASE,
)

# Words that signal "this table is about brain coordinates"
_COORD_CONTEXT_RE = regex.compile(
    r"\b(mni|talairach|stereotaxic|stereotactic|peak(?:\s+voxel)?"
    r"\s+coordinates?|coordinates?\s*\(?(?:mm)?\)?)\b",
    regex.IGNORECASE,
)

# Pulls an x/y/z triplet out of one cell's text, e.g. "-6, -88, 8" or "-6/-88/8" -> ("-6", "-88", "8")
_COORD_TRIPLET_IN_CELL_RE = regex.compile(
    r"\n*([-]?\d{1,3}\.?\d{0,2})[,;/|\s]+([-]?\d{1,3}\.?\d{0,2})[,;/|\s]+([-]?\d{1,3}\.?\d{0,2})"
)


def _clean_label(label) -> str:
    """
    Purpose:
        Clean up a column header or section-title label pulled from a
        Docling table: remove markdown emphasis (**bold**, *italic*),
        unescape HTML entities (&gt; -> >), and collapse extra whitespace.

    Inputs:
        label: the raw label text (any type; converted to a string).

    Outputs:
        str: the cleaned label.
    """
    text = str(label).replace("*", "")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    text = regex.sub(r"\s+", " ", text).strip()
    return text


def classify_coordinate_table(rows=None, caption=None, footnotes=None) -> CoordinateTableDecision:
    """
    Purpose:
        Decide whether a table looks like a coordinate table worth parsing,
        by checking for x/y/z-style headers, coordinate-related words (MNI,
        Talairach, etc.), and rows that look like numeric triplets.

    Inputs:
        rows (list of list of str): the table's rows, header row(s)
            included. See _dataframe_to_rows for building this from a
            DataFrame.
        caption (str, optional): the table's caption text, if any.
        footnotes (str, optional): the table's footnote text, if any.

    Outputs:
        CoordinateTableDecision: a named tuple of
            candidate (bool): True if this table is worth parsing.
            reasons (tuple of str): which checks matched.
    """
    normalized_rows = []
    for row in rows or []:
        normalized_rows.append([regex.sub(r"\s+", " ", str(value or "")).strip() for value in row])

    if not normalized_rows:
        return CoordinateTableDecision(False, ())

    # Only look at the first few rows for header/context words
    header_rows = normalized_rows[: min(6, len(normalized_rows))]
    header_text = " ".join(" ".join(row) for row in header_rows)
    context_text = " ".join(v for v in (header_text, caption or "", footnotes or "") if v)
    context_lower = context_text.lower()

    axis_hits = {axis for axis in ("x", "y", "z") if regex.search(r"(?<![a-z])%s(?![a-z])" % axis, context_lower)}
    has_xyz_headers = axis_hits == {"x", "y", "z"}
    has_coord_context = bool(_COORD_CONTEXT_RE.search(context_text))
    has_combined_header = bool(regex.search(r"\bx\b.{0,30}\by\b.{0,30}\bz\b", context_text, regex.IGNORECASE))

    plausible_numeric_rows = 0
    triplet_rows = 0
    for row in normalized_rows:
        row_text = " | ".join(row)
        if _TRIPLET_RE.search(row_text):
            triplet_rows += 1
        # Count cells that are a single plausible coordinate value (a lone
        # number under 100 in magnitude), not a p-value or a crowded cell
        numeric_cells = 0
        for value in row:
            matches = _NUMBER_RE.findall(value)
            if len(matches) == 1:
                try:
                    number = float(
                        regex.sub(r"\s+", "", matches[0]).translate(str.maketrans({c: "-" for c in _MINUS_CHARS}))
                    )
                except ValueError:
                    continue
                if abs(number) < 100:
                    numeric_cells += 1
        if numeric_cells >= 3:
            plausible_numeric_rows += 1

    reasons = []
    if has_xyz_headers and plausible_numeric_rows:
        reasons.append("xyz_headers")
    if has_combined_header and triplet_rows:
        reasons.append("combined_xyz_header")
    if has_coord_context and (triplet_rows or plausible_numeric_rows):
        reasons.append("coordinate_context")
    if triplet_rows >= 2 and (has_coord_context or has_xyz_headers):
        reasons.append("repeated_coordinate_triplets")

    return CoordinateTableDecision(bool(reasons), tuple(reasons))


def is_coordinate_table_candidate(rows=None, caption=None, footnotes=None) -> bool:
    """
    Purpose:
        Shortcut for classify_coordinate_table that returns just the
        yes/no decision.

    Inputs:
        rows (list of list of str): see classify_coordinate_table.
        caption (str, optional): see classify_coordinate_table.
        footnotes (str, optional): see classify_coordinate_table.

    Outputs:
        bool: True if the table is a coordinate-table candidate.
    """
    return classify_coordinate_table(rows=rows, caption=caption, footnotes=footnotes).candidate


def identify_standard_columns(labels):
    """
    Purpose:
        Match each column label to a standard role (region, x, y, z, ba,
        hemisphere, size, statistic, p_value, coord_triplet), so later code
        knows what kind of value each column holds.

    Inputs:
        labels (list of str): column headers, already lowercased.

    Outputs:
        list of str or None: one role per label, in the same order.
            None means the column's role wasn't recognized.
    """
    standardized = [None] * len(labels)
    found_coords = False
    for i, lab in enumerate(labels):
        if regex.search(r"(^\s*ba$)|(^\s*brodmann(\s*area)?\s*$)", lab):
            # Anchored to the *whole* label, not a substring -- "Region
            # (Brodmann Area)" also contains "brodmann" but is a region
            # descriptor, not a standalone BA-code column
            s = "ba"
        elif regex.search(r"region|anatom|location|area", lab):
            s = "region"
        elif regex.search(r"sphere|(^\s*h$)|^\s*hem|^\s*side", lab):
            s = "hemisphere"
        elif regex.search(r"(^k$)|(mm.*?3)|volume|voxels|size|extent", lab):
            s = "size"
        elif (
            regex.search(r"\bx\b.*\by\b.*\bz\b", lab)
            or regex.search(r"(peak\s*voxel\s*coordinate|talairach\s*coordinates?|mni\s*coordinates?)", lab)
            or (regex.search(r"coordinates?", lab) and not regex.search(r"cluster|score|value", lab))
        ):
            # A single column holding all three values together
            s = "coord_triplet"
            found_coords = True
        elif regex.search(r"^\s*[xy]\b", lab):
            found_coords = True
            s = "y" if regex.search(r"^\s*y\b", lab) else "x"
        elif regex.search(r"^\s*z\b", lab):
            # "Z" is ambiguous -- treat it as the z-coordinate only if the
            # column right before it was already resolved to "y"
            s = "statistic" if not found_coords or (i > 0 and standardized[i - 1] != "y") else "z"
        elif regex.search(r"rdinate", lab):
            s = "coord_triplet"
            found_coords = True
        elif lab == "t" or regex.search(r"^(max.*(z|t).*|.*(z|t).*(score|value|max))$", lab):
            s = "statistic"
        elif regex.search(r"p[\-\s]+.*val", lab) or regex.search(r"^\s*p\b.*\b(fwe|fdr|corr)", lab):
            # Second pattern catches "P FWE-cor" / "P FDR-corrected" style
            # headers that don't contain the substring "val"
            s = "p_value"
        else:
            s = None
        standardized[i] = s
    return standardized


def identify_repeating_groups(labels):
    """
    Purpose:
        Find blocks of columns that repeat side by side, such as a
        (t, p, peak x/y/z) block repeated once per experimental condition,
        rather than stacked as separate rows.

    Inputs:
        labels (list of str): column headers.

    Outputs:
        list of str: one entry per group, formatted as "start/length"
            (e.g. "1/3" means a group starts at column 1 and spans 3
            columns).
    """
    groups = []
    n_labels = len(labels)
    label_counts = Counter(labels)
    rep_labels = {k for k, v in label_counts.items() if v > 1}
    label_seqs = defaultdict(list)

    # For each position that starts a run of labels seen more than once
    # overall, record the sequence until a different/non-repeated label
    # breaks it -- a candidate "one repetition" of the group
    for i, lab in enumerate(labels):
        if lab not in rep_labels:
            continue
        current_seq = [lab]
        for j in range(i + 1, n_labels):
            lab_j = labels[j]
            if lab_j not in rep_labels or lab_j == lab:
                break
            current_seq.append(lab_j)
        if len(current_seq) > 1:
            label_seqs["###".join(current_seq)].append(i)

    # Only keep sequences that actually repeat (start at more than one column index)
    label_seqs = {k: v for k, v in label_seqs.items() if len(v) > 1}
    seq_starts = [None] * n_labels
    for k, v in label_seqs.items():
        for start in v:
            seq_starts[start] = k.split("###")

    # A repeating block like (t, p, peak-x/y/z) cycling three times also
    # reads, one column later, as the rotated sequence (p, peak-x/y/z, t).
    # Each rotation looks like "a sequence seen more than once", so
    # `seq_starts` has a candidate group anchored at every column, not just
    # the real block boundaries. Rejecting a candidate as soon as *one*
    # column in its window is already claimed (any(...), not all(...))
    # keeps only the real, non-overlapping groups.
    labels_used = [False] * n_labels
    for i, lab in enumerate(labels):
        if seq_starts[i] is not None:
            seq_size = len(seq_starts[i])
            if not any(labels_used[i:(i + seq_size)]):
                labels_used[i:(i + seq_size)] = [True] * seq_size
                groups.append("%d/%d" % (i, seq_size))
    return groups


def _extract_triplet(value):
    """
    Purpose:
        Pull an (x, y, z) triplet out of one cell's text, if it has one.

    Inputs:
        value: the cell's contents (any type; converted to a string).

    Outputs:
        list of 3 str, or None if no triplet was found.
    """
    # Strip a trailing "sentence-ending" period (one not between two
    # digits) so a cell like "(-6, -88, 8)." still matches.
    clean_val = regex.sub(r"(?<!\d)\.(?!\d)", "", str(value))
    match = _COORD_TRIPLET_IN_CELL_RE.search(clean_val)
    if not match:
        return None
    return [regex.sub(r"-\s+", "-", c.strip()) for c in match.groups()]


def extract_row(row_values, labels, standard_cols, group_label=None):
    """
    Purpose:
        Build one coordinate record from a single table row, filling in
        x/y/z/region/hemisphere/size/ba/statistic/p_value according to
        each column's role.

    Inputs:
        row_values (list): the row's cell values, in column order.
        labels (list of str): the column headers, in the same order.
        standard_cols (list of str or None): each column's role, from
            identify_standard_columns.
        group_label (str, optional): a label (e.g. a contrast name) to
            attach to this row.

    Outputs:
        dict with keys:
            x, y, z (float or None)
            region (str or None)
            group (str or None)
            problems (list of str): anything that failed to parse.
            extra (dict): every column's raw value, by label.
            _valid (bool): True if there were no problems and x, y, z are
                all present.
    """
    record = {"x": None, "y": None, "z": None, "region": None, "group": group_label, "problems": [], "extra": {}}
    coords_from_triplet = False

    for i, col in enumerate(row_values):
        col = str(col)
        # Normalize unicode minus/dash variants to a plain ASCII "-"
        for c in _MINUS_CHARS:
            if c in col:
                col = col.replace(c, "-")
        if "*" in col:
            col = col.replace("*", "")

        sc = standard_cols[i] if i < len(standard_cols) else None

        if sc in ("coord_triplet", "x", "y", "z"):
            triplet = _extract_triplet(col)
            if triplet is not None:
                record["x"], record["y"], record["z"] = triplet
                coords_from_triplet = True
                record["extra"][labels[i]] = col
                continue
            if sc == "coord_triplet":
                # Records that column that was supposed to hold a full triplet actually didn't
                record["problems"].append(
                    f"Could not parse an x/y/z triplet out of '{col}' in column '{labels[i]}'"
                )
                record["extra"][labels[i]] = col
                continue

        if sc in ("x", "y", "z"):
            if coords_from_triplet and col.strip() == "":
                record["extra"][labels[i]] = col
                continue
            m = regex.match(r"([-])\s?(\d+\.*\d*)$", col)
            if m:
                col = f"{m.group(1)}{m.group(2)}"
            if not regex.match(r"([-]*\d+)\.*\d*$", col):
                record["problems"].append(f"Value '{col}' in {sc} column is not valid")
                record["extra"][labels[i]] = col
                continue
            record[sc] = float(col)
        elif sc == "region":
            if regex.search(r"[a-zA-Z]", col):
                record["region"] = col
            elif col.strip():
                record["problems"].append("Value in region column is not a string")
        elif sc in ("hemisphere", "size", "ba", "statistic", "p_value"):
            record[sc] = col

        record["extra"][labels[i]] = col

        if sc is None:
            # Unrecognized column -- still check whether its raw text happens to contain a full triplet
            triplet = _extract_triplet(col)
            if triplet is not None:
                record["x"], record["y"], record["z"] = triplet

    record["_valid"] = not record["problems"] and all(record.get(ax) is not None for ax in ("x", "y", "z"))
    return record


def _dataframe_to_rows(df: pd.DataFrame) -> list[list[str]]:
    """
    Purpose:
        Convert a DataFrame into ACE-style rows (a list of lists of
        strings, header row included), for classify_coordinate_table.

    Inputs:
        df (pandas.DataFrame): the table.

    Outputs:
        list of list of str: the header row, followed by each data row.
    """
    return [[str(c) for c in df.columns]] + df.astype(str).values.tolist()


# Matches a study citation header like "Smith, 2012" or "Smith, 2012 (N=20)".
_STUDY_HEADER_RE = regex.compile(r"^(?P<name>[A-Za-z][\w.\-' ]*?,\s*\d{4})\s*(?:\(N\s*=\s*\d+\))?\s*$")
# Matches a line that is just three whitespace-separated numbers
_TRIPLET_LINE_RE = regex.compile(
    r"^\s*([+\-]?\d{1,3}(?:\.\d+)?)\s+([+\-]?\d{1,3}(?:\.\d+)?)\s+([+\-]?\d{1,3}(?:\.\d+)?)\s*$"
)


def _preview_dataframe(df: pd.DataFrame, max_rows: int = 25) -> str:
    """
    Purpose:
        Render a DataFrame as readable text for log output.

    Inputs:
        df (pandas.DataFrame): the table to render.
        max_rows (int, optional): maximum rows to show. Defaults to 25.

    Outputs:
        str: the rendered table, with a note if rows were cut off.
    """
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 40):
        preview = df.astype(str).head(max_rows).to_string(index=False)
    if len(df) > max_rows:
        preview += f"\n... ({len(df) - max_rows} more row(s) not shown)"
    return preview


def _log_row_outcome(source_file, record, verbose, group_context=""):
    """
    Purpose:
        Log whether one row was kept or dropped, and why.

    Inputs:
        source_file (str): name of the file the row came from, for the
            log message.
        record (dict): the row's record, from extract_row.
        verbose (bool): if False, do nothing.
        group_context (str, optional): extra text to add to the log tag,
            e.g. which repeating column-group this row came from.

    Outputs:
        None.
    """
    if not verbose:
        return
    tag = f"{source_file}{(' ' + group_context) if group_context else ''}"
    if record["_valid"]:
        logger.info(
            "%s: kept row -> x=%s y=%s z=%s region=%s group=%s",
            tag, record["x"], record["y"], record["z"], record["region"], record["group"],
        )
    else:
        logger.info(
            "%s: dropped row (x=%s y=%s z=%s) -- %s",
            tag, record["x"], record["y"], record["z"],
            "; ".join(record["problems"]) if record["problems"] else "no x/y/z value found in this row",
        )


def extract_coordinates_from_text(
    text: str, source_file: str, pmcid: str, pmid: str = "", verbose: bool = True
) -> pd.DataFrame:
    """
    Purpose:
        Fallback extraction over plain text (e.g. an article's body text,
        as opposed to a parsed table): scan line by line for a bare x/y/z
        triplet, tracking the nearest section heading and study citation
        as context.

    Inputs:
        text (str): the text to scan.
        source_file (str): name of the file this text came from.
        pmcid (str): the article's PMC ID.
        pmid (str, optional): the article's PubMed ID.
        verbose (bool, optional): if True, log progress. Defaults to True.

    Outputs:
        pandas.DataFrame: one row per triplet found, with columns pmid,
            pmcid, source_file, x, y, z, region, group, study.
    """
    records = []
    current_task = None
    current_study = None

    if verbose:
        logger.info(
            "%s: intermediate text-fallback input (%d line(s)):\n%s",
            source_file, len(text.splitlines()),
            "\n".join(text.splitlines()[:40]) + ("\n... (truncated)" if len(text.splitlines()) > 40 else ""),
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):  # skip blank lines and markdown table rows
            continue

        triplet_match = _TRIPLET_LINE_RE.match(line)
        if triplet_match:
            x, y, z = (float(v) for v in triplet_match.groups())
            records.append({
                "pmid": pmid, "pmcid": pmcid, "source_file": source_file,
                "x": x, "y": y, "z": z, "region": None,
                "group": current_task, "study": current_study,
            })
            if verbose:
                logger.info(
                    "%s: text line %r -> kept triplet x=%s y=%s z=%s (task=%s, study=%s)",
                    source_file, line, x, y, z, current_task, current_study,
                )
            continue

        study_match = _STUDY_HEADER_RE.match(line)
        if study_match:
            current_study = study_match.group("name")
            continue

        # A short, numberless line is treated as a section heading, carried
        # forward as context for any triplet found after it.
        if not regex.search(r"\d", line) and len(line.split()) <= 8:
            current_task = line

    if not records:
        if verbose:
            logger.info("%s: text-fallback extraction found 0 coordinate row(s)", source_file)
        return pd.DataFrame(columns=["pmid", "pmcid", "source_file", "x", "y", "z", "region", "group", "study"])

    out = pd.DataFrame.from_records(records)
    if verbose:
        logger.info("%s: text-fallback intermediate result:\n%s", source_file, _preview_dataframe(out))
    return out


def _resolve_compound_header(label: str) -> str:
    """
    Purpose:
        Get the per-column part of a flattened "Parent.Child" header (e.g.
        Docling turning a two-row header into "MNI Coordinates (mm).X").
        Using just "X" instead of the whole string lets later code match
        it to the right column role.

    Inputs:
        label (str): the column header, possibly in "Parent.Child" form.

    Outputs:
        str: the child (tail) segment, or the original label unchanged if
            it has no "." in it.
    """
    if "." not in label:
        return label
    tail = label.rsplit(".", 1)[-1].strip()
    return tail if tail else label


def _compound_header_prefix(label: str) -> str | None:
    """
    Purpose:
        Get the parent part of a flattened "Parent.Child" header (e.g.
        "nC-F group" out of "nC-F group.Peak at x/y/z"), used to label
        which repeating column-group a row came from.

    Inputs:
        label (str): the column header, possibly in "Parent.Child" form.

    Outputs:
        str or None: the parent (head) segment, or None if there's no "."
            in the label.
    """
    if "." not in label:
        return None
    head = label.split(".", 1)[0].strip()
    return head or None


def _repair_repeated_triplet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Fix a table where a merged X/Y/Z header (e.g. "MNI coordinates peak
        voxel" spanning 3 columns) got exported as that same text repeated
        as the name of all 3 columns, with the real x/y/z sub-header stuck
        in the first data row instead. Without this fix, all 3 columns look
        identical and every row fails to parse.

    Inputs:
        df (pandas.DataFrame): the table to check and possibly fix.

    Outputs:
        pandas.DataFrame: the repaired table (sub-header row promoted to
            column names, columns renamed to "<label>.x" / ".y" / ".z"),
            or the original df unchanged if this pattern wasn't found.

    Notes:
        Only fires on a run of exactly 3 consecutive, identical, non-blank
        column names whose first data row holds the literal text "x", "y",
        "z" at those positions. When it fires, any other column that is
        still blank (e.g. because an earlier repair already removed a
        title row) also picks up its real label from that same row.
    """
    labels_original = [str(c) for c in df.columns]
    n = len(labels_original)
    if df.empty or n == 0:
        return df

    first_row = [("" if pd.isna(v) else str(v)) for v in df.iloc[0]]
    new_labels = list(labels_original)
    repaired_any = False

    i = 0
    while i < n:
        # Find the run of consecutive columns sharing this exact label.
        j = i
        while j + 1 < n and labels_original[j + 1] == labels_original[i] and labels_original[i].strip():
            j += 1
        run = range(i, j + 1)
        if len(run) == 3:
            sub_values = [_clean_label(first_row[k]).strip().lower() for k in run]
            if sorted(sub_values) == ["x", "y", "z"]:
                for k, axis in zip(run, sub_values):
                    new_labels[k] = f"{labels_original[k]}.{axis}"
                repaired_any = True
        i = j + 1

    if not repaired_any:
        return df

    # The same row can also carry real labels for other, still-blank
    # columns (e.g. "Region", "BA", "T-Value") when an earlier repair
    # already removed a title row above it.
    for k in range(n):
        if new_labels[k].strip():
            continue
        text = _clean_label(first_row[k]).strip()
        if text:
            new_labels[k] = text

    repaired = df.iloc[1:].reset_index(drop=True)
    repaired.columns = new_labels
    return repaired


# Matches a cell containing *only* a signed number (e.g. "-35"), as opposed
# to merely containing one somewhere in more text (e.g. "Age < 41").
_PLAIN_NUMBER_RE = regex.compile(r"^[+\-%s]?\s*\d{1,3}(?:\.\d+)?$" % _MINUS_CHARS)


def _is_plain_number_cell(value: str) -> bool:
    """
    Purpose:
        Check whether a cell's entire text is nothing but a signed number.

    Inputs:
        value (str): the cell's text.

    Outputs:
        bool: True if the cell is only a number.
    """
    return bool(_PLAIN_NUMBER_RE.match(value.strip()))


def _repair_collapsed_title_header(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Purpose:
        Fix a table where every column ended up with the exact same name
        (the table's own caption text) instead of any column being left
        blank, with the real multi-row header (group labels, then
        per-column labels) stranded several rows down in the data instead
        of in df.columns.

    Inputs:
        df (pandas.DataFrame): the table to check and possibly fix.

    Outputs:
        tuple of (pandas.DataFrame, str or None):
            the repaired table and the caption text used as a group label,
            or (df, None) unchanged if this pattern wasn't found.

    Notes:
        Since no header text survives to say which columns are x, y, and
        z, this finds them by content instead: the first row where three
        consecutive columns are each nothing but a plain number is taken
        as where real data starts, confirmed by the same three columns
        also being numeric in a later row. Other columns' names are
        rebuilt from whatever non-blank, non-caption text sits above that
        row.
    """
    labels_original = [str(c) for c in df.columns]
    n = len(labels_original)
    if df.empty or n < 3:
        return df, None
    if any(not lab.strip() for lab in labels_original):
        return df, None
    non_blank = {lab.strip() for lab in labels_original}
    if len(non_blank) != 1:
        return df, None
    title_text = next(iter(non_blank))

    nrows = len(df)

    def cell(ridx, cidx):
        v = df.iat[ridx, cidx]
        return "" if pd.isna(v) else str(v).strip()

    # Scan rows top to bottom for the first 3-column window that's all
    # plain numbers, then require it to repeat in a nearby later row
    # before trusting it as the real x/y/z columns.
    data_row_idx = None
    xyz_cols = None
    for ridx in range(nrows):
        found_start = None
        for start in range(n - 2):
            if all(_is_plain_number_cell(cell(ridx, start + k)) for k in range(3)):
                found_start = start
                break
        if found_start is None:
            continue
        confirmed = any(
            all(_is_plain_number_cell(cell(r2, found_start + k)) for k in range(3))
            for r2 in range(ridx + 1, min(ridx + 5, nrows))
        )
        if confirmed:
            data_row_idx = ridx
            xyz_cols = (found_start, found_start + 1, found_start + 2)
            break

    if data_row_idx is None:
        return df, None

    # Rebuild every non-x/y/z column's name from whatever text appears above the data boundary, 
    # skipping the (unhelpful) repeated title.
    new_labels = [""] * n
    for cidx in range(n):
        if cidx in xyz_cols:
            continue
        parts = []
        for ridx in range(data_row_idx):
            text = _clean_label(cell(ridx, cidx))
            if not text or text == title_text:
                continue
            if not parts or parts[-1] != text:
                parts.append(text)
        new_labels[cidx] = ".".join(parts)

    for cidx, axis in zip(xyz_cols, ("x", "y", "z")):
        new_labels[cidx] = axis

    repaired = df.iloc[data_row_idx:].reset_index(drop=True)
    repaired.columns = new_labels
    return repaired, title_text


def _repair_degenerate_header(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Purpose:
        Fix a table where Docling kept a spanning title row (e.g. a
        section title like "SDI>HC group comparison") as the column
        header, leaving the table's real per-column header ("Region",
        "Size", "X mm", ...) sitting in the first data row instead.

    Inputs:
        df (pandas.DataFrame): the table to check and possibly fix.

    Outputs:
        tuple of (pandas.DataFrame, str or None):
            the repaired table and the spanning title text (to use as a
            group label), or (df, None) unchanged if this pattern wasn't
            found.

    Notes:
        Only fires when at least one column name is blank and the first
        data row, once cleaned, resolves to at least one coordinate role.
    """
    labels_original = [str(c) for c in df.columns]
    if df.empty or not any(lab.strip() == "" for lab in labels_original):
        return df, None

    candidate_row = [("" if pd.isna(v) else str(v)) for v in df.iloc[0]]
    candidate_labels = [_resolve_compound_header(_clean_label(v)).strip().lower() for v in candidate_row]
    candidate_roles = identify_standard_columns(candidate_labels)
    if not any(role in ("x", "y", "z", "coord_triplet") for role in candidate_roles):
        return df, None

    leading_title = next((_clean_label(lab) for lab in labels_original if lab.strip()), None)
    repaired = df.iloc[1:].reset_index(drop=True)
    repaired.columns = candidate_row
    return repaired, leading_title


# Matches a cohort-size note like "(n=20)", stripped out of region text scraped from a raw blob
_COHORT_SIZE_RE = regex.compile(r"\(\s*[nN]\s*=\s*\d+\s*\)")

# One "row" in a flattened text blob: an optional region name, an optional
# Brodmann-area number, a z-score, then the x/y/z triplet. The z-score's
# decimal point is what tells it apart from the BA number and from x/y/z
# (which are always bare integers here), so matches don't get misaligned.
_ROW_RE = regex.compile(
    r"(?P<region>.*?)"
    r"(?:(?<![\d.])(?P<ba>\d{1,3})(?![\d.])\s+)?"
    r"(?<![\d.])(?P<zscore>\d{1,3}\.\d{1,2})(?![\d.])"
    r"\s+(?<![\d.])(?P<x>[+\-]?\d{1,3})(?![\d.])"
    r"\s+(?<![\d.])(?P<y>[+\-]?\d{1,3})(?![\d.])"
    r"\s+(?<![\d.])(?P<z>[+\-]?\d{1,3})(?![\d.])"
)


def _extract_triplets_from_blob(blob: str) -> list[dict]:
    """
    Purpose:
        Last-resort recovery for a table Docling failed to tabularize at
        all (e.g. a legacy .doc file that comes back as a single giant
        cell holding the whole table's text). Scans the flattened text
        directly for coordinate triplets.

    Inputs:
        blob (str): all of the table's cell text, joined into one string.

    Outputs:
        list of dict: one dict per coordinate found, with keys x, y, z,
            region, group (always None), ba, statistic, problems (always
            empty), extra (always empty), _valid (always True). Returns an
            empty list if fewer than 2 triplets were found.

    Notes:
        This is a best-effort text scrape, not a table parse, so region
        text can be noisy (e.g. a cohort heading folded into it). It
        always finds a valid, complete x/y/z triplet for every record it
        returns.
    """
    flat = str(blob)
    for c in _MINUS_CHARS:
        flat = flat.replace(c, "-")
    flat = regex.sub(r"\s+", " ", flat).strip()

    raw_matches = list(_ROW_RE.finditer(flat))
    if len(raw_matches) < 2:
        return []

    records = []
    current_region = None
    for m in raw_matches:
        region_text = regex.sub(r"\s+", " ", _COHORT_SIZE_RE.sub(" ", _clean_label(m.group("region")))).strip()
        if region_text:
            current_region = region_text

        records.append({
            "x": float(m.group("x")), "y": float(m.group("y")), "z": float(m.group("z")),
            "region": current_region, "group": None,
            "ba": m.group("ba"), "statistic": m.group("zscore"),
            "problems": [], "extra": {}, "_valid": True,
        })

    return records


def extract_coordinates_from_table(
    df: pd.DataFrame, source_file: str, pmcid: str, pmid: str = "", caption: str = "", footnotes: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Purpose:
        Main entry point: take one table from Docling, repair its header
        if needed, classify its columns, and extract a tidy coordinate row
        for each valid activation.

    Inputs:
        df (pandas.DataFrame): the table to extract from.
        source_file (str): name of the file this table came from, used in
            log messages and in the output.
        pmcid (str): the article's PMC ID, added to every output row.
        pmid (str, optional): the article's PubMed ID, added to every
            output row.
        caption (str, optional): the table's caption text (currently
            unused here, kept for interface consistency with
            classify_coordinate_table).
        footnotes (str, optional): the table's footnote text (also
            currently unused here).
        verbose (bool, optional): if True, log progress and per-row
            outcomes. Defaults to True.

    Outputs:
        pandas.DataFrame: one row per coordinate found, with columns pmid,
            pmcid, source_file, x, y, z, region, group, plus any extra
            columns the table had (hemisphere, size, ba, statistic,
            p_value). Empty (0 rows) if no coordinates were found.

    Notes:
        Runs the table through up to four header-repair passes before
        extraction, each one handling a different way Docling can lose or
        scramble a table's header -- see each _repair_* function's
        docstring for the specific pattern it fixes. If normal column-
        based extraction still finds nothing, falls back to a raw-text
        triplet scrape in case Docling failed to tabularize the table at
        all.
    """
    # Pass 1: unflatten a merged X/Y/Z header repeated across 3 columns,
    # if the table already has that shape before any other repair runs.
    repeated_triplet_fix = _repair_repeated_triplet_columns(df)
    if repeated_triplet_fix is not df and verbose:
        logger.info(
            "%s: a repeated column name (e.g. 'MNI coordinates peak voxel' x3) hid a real X/Y/Z "
            "sub-header one row down -- repaired before extraction.",
            source_file,
        )
    df = repeated_triplet_fix

    # Pass 2: recover a header Docling left as a spanning title row.
    repaired_df, leading_group_label = _repair_degenerate_header(df)
    if repaired_df is not df and verbose:
        logger.info(
            "%s: table's real header was found one row down from where Docling put it "
            "(leading section title: %r) -- repaired before extraction.",
            source_file, leading_group_label,
        )
    df = repaired_df

    # Pass 3: a table can need both repairs in sequence -- peeling off one
    # spanning title row (pass 2) can reveal a repeated-triplet-column
    # pattern that wasn't visible in the original, unrepaired columns.
    # Calling this a second time is a no-op whenever pass 1 already
    # handled it, or there was never a blank-title row to begin with.
    second_pass = _repair_repeated_triplet_columns(df)
    if second_pass is not df and verbose:
        logger.info(
            "%s: peeling off the spanning title row above revealed a further repeated X/Y/Z "
            "sub-header one row down -- repaired before extraction.",
            source_file,
        )
    df = second_pass

    # Pass 4: a harder variant where every column ended up with the same
    # name (the table's caption text) rather than any being left blank,
    # with the real multi-row header buried several data rows down.
    collapsed_df, collapsed_title = _repair_collapsed_title_header(df)
    if collapsed_df is not df and verbose:
        logger.info(
            "%s: every column shared the same name (the table's caption text) with the real "
            "header buried several rows further down -- repaired before extraction.",
            source_file,
        )
    df = collapsed_df
    if leading_group_label is None:
        leading_group_label = collapsed_title

    labels_original = [_clean_label(c) for c in df.columns]
    labels = [_resolve_compound_header(lab).strip().lower() for lab in labels_original]
    standard_cols = identify_standard_columns(labels)
    group_cols = identify_repeating_groups(labels)

    if verbose:
        logger.info(
            "%s: intermediate table before extraction (%d rows x %d cols):\n%s",
            source_file, df.shape[0], df.shape[1], _preview_dataframe(df),
        )
        logger.info(
            "%s: resolved column roles -> %s",
            source_file, dict(zip(labels_original, standard_cols)),
        )
        if group_cols:
            logger.info("%s: detected repeating column-groups at %s", source_file, group_cols)

    cols_in_group = [False] * len(labels)
    for g in group_cols:
        onset, length = (int(x) for x in g.split("/"))
        for i in range(onset, onset + length):
            cols_in_group[i] = True

    records = []
    group_row = leading_group_label
    for row in df.itertuples(index=False, name=None):
        row = [("" if pd.isna(v) else str(v)) for v in row]

        if row and row[0].strip() and not "".join(row[1:]).strip():
            # A row where only the first cell has text is a section-title
            # row, not a data row -- `_clean_label` here (not just
            # `.strip()`) so it matches the leading title cleanup above.
            group_row = _clean_label(row[0])
            continue

        if not group_cols:
            record = extract_row(row, labels_original, standard_cols, group_row)
            _log_row_outcome(source_file, record, verbose)
            if record["_valid"]:
                records.append(record)
        else:
            for g in group_cols:
                onset, length = (int(x) for x in g.split("/"))
                group_specific_cols = set(range(onset, onset + length))
                sub_labels, sub_values, sub_scs = [], [], []
                for i, val in enumerate(row):
                    if not cols_in_group[i] or i in group_specific_cols:
                        sub_labels.append(labels_original[i])
                        sub_values.append(val)
                        sub_scs.append(standard_cols[i])
                # Tag the record with which repeating block it came from,
                # so side-by-side per-condition blocks don't all collapse
                # into rows with the identical group value.
                col_group_label = _compound_header_prefix(labels_original[onset])
                combined_group = " / ".join(g_ for g_ in (group_row, col_group_label) if g_) or None
                record = extract_row(sub_values, sub_labels, sub_scs, combined_group)
                _log_row_outcome(source_file, record, verbose, group_context=f"[{g}]")
                if record["_valid"]:
                    records.append(record)

    if not records:
        # Column-based extraction found nothing. Check whether Docling
        # actually failed to tabularize this file at all (e.g. a (1, 1)
        # table whose lone cell holds the whole table's text) before
        # giving up.
        blob = " ".join(
            str(v) for row in df.itertuples(index=False, name=None) for v in row if not pd.isna(v)
        )
        blob_records = _extract_triplets_from_blob(blob)
        if blob_records:
            if verbose:
                logger.info(
                    "%s: column-based extraction found 0 rows, but the raw cell text contains %d "
                    "embedded coordinate triplet(s) -- Docling likely failed to tabularize this file "
                    "(see its shape/columns logged above). Falling back to a text scrape of the cell "
                    "contents; region/BA/z-score are best-effort and no group/contrast label is attempted.",
                    source_file, len(blob_records),
                )
                for rec in blob_records:
                    _log_row_outcome(source_file, rec, verbose)
            records.extend(blob_records)

    if not records:
        if verbose:
            logger.info("%s: extracted 0 coordinate row(s)", source_file)
        return pd.DataFrame(columns=["pmid", "pmcid", "source_file", "x", "y", "z", "region", "group"])

    out = pd.DataFrame.from_records(records)
    out["pmid"] = pmid
    out["pmcid"] = pmcid
    out["source_file"] = source_file

    out = out[["pmid", "pmcid", "source_file", "x", "y", "z", "region", "group"] +
              [c for c in out.columns if c not in
               {"pmid", "pmcid", "source_file", "x", "y", "z", "region", "group", "problems", "extra", "_valid"}]]

    if verbose:
        logger.info(
            "%s: extracted %d coordinate row(s), intermediate result:\n%s",
            source_file, len(out), _preview_dataframe(out),
        )
    return out