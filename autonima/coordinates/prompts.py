"""Prompts for extracting coordinate analyses from neuroimaging tables."""

from __future__ import annotations

from textwrap import dedent


# Increment this whenever a prompt change can alter coordinate parsing output.
COORDINATE_PARSING_PROMPT_VERSION = "2026-07-30.annotations-v3"


def create_coordinate_parsing_prompt(
    table_text: str,
    *,
    table_caption: str = "",
    table_foot: str = "",
) -> str:
    """Build the coordinate-table parsing prompt.

    The analysis-boundary guidance is based on reviewed parser failures in the
    Autonima benchmark. In particular, it distinguishes experimental axes
    (contrast, group, direction, treatment, and session) from descriptive
    anatomical row groupings.
    """

    return dedent(
        f"""
        You are a neuroimaging data curation assistant.

        You will receive a table extracted from a published fMRI/neuroimaging
        article. Extract every valid stereotactic coordinate and group the
        coordinates into the distinct statistical analyses/maps reported by
        the table.

        Table caption:
        {table_caption}

        Table footnotes:
        {table_foot}

        Return JSON strictly matching the `parse_analyses` function schema:

        {{
          "analyses": [
            {{
              "name": <string or null>,
              "description": <string or null>,
              "points": [
                {{
                  "coordinates": [x, y, z],
                  "space": <"MNI" | "TAL" | null>,
                  "values": [
                    {{
                      "value": <float or string or null>,
                      "kind": <string or null>
                    }}
                  ]
                }}
              ]
            }}
          ]
        }}

        ANALYSIS-BOUNDARY RULES

        Use two passes. First inventory the table's analysis-defining axes and
        result blocks without extracting points. Then traverse the table and
        assign every coordinate to the appropriate combination of those axes.

        Start a separate analysis when an explicit label changes any of these:
        - contrast or comparison, including direction (A > B versus A < B);
        - activation versus deactivation, positive versus negative, or
          increase versus decrease;
        - participant group or cohort, including "all groups" and each
          separately reported subgroup;
        - treatment, task condition, session, time point, or parametric effect;
        - an analysis/contrast label encoded in a column or multi-level header.

        A table can encode analyses across columns as well as down rows. Do not
        ignore columns named Contrast, Comparison, Group, Condition, Treatment,
        Session, Effect, or similar. When multiple independent
        analysis-defining axes are present, preserve their explicit
        combinations in the analysis names.

        In a wide table, repeated statistic/X/Y/Z column groups beneath
        different top-level headers are separate analysis blocks. A single row
        may therefore contribute one coordinate to several analyses. Route
        each non-empty X/Y/Z group to its own header-defined analysis; never
        pool coordinates from different header blocks into one broad analysis.
        Blank cells in one block do not shift values into an adjacent block.

        Positive and negative statistical values may encode opposite contrast
        directions in one table. When the caption, header, or footnote defines
        what the sign means, create the corresponding directional analyses and
        route points by sign. Do not treat both signs as one map.

        Repeated blank cells inherit the most recent applicable header or
        analysis label. Continuation rows and local maxima belong to that same
        analysis until an analysis-defining label changes.

        Table exports sometimes represent a section header as a row where the
        same nonnumeric label is repeated across most or all columns. Treat
        that row as analysis context, not as data. If it names an experimental
        group, condition, contrast direction, session, or time point, start
        the corresponding analysis and combine it with any applicable parent
        context. When "all groups" and named subgroups are each explicitly
        reported with coordinates, return one analysis for every non-empty
        group block; do not pool the subgroup points into "all groups".

        Do NOT start a new analysis merely because a descriptive anatomical
        grouping changes. Brain region, lobe, cluster, hemisphere/left/right,
        local maximum/subpeak, a-priori versus non-a-priori region, and
        predicted versus non-predicted region are normally subdivisions within
        one statistical map. Split on one of these only if the table or caption
        explicitly identifies it as a distinct statistical contrast/map.
        In particular, left and right hemisphere sections alone never define
        different statistical analyses. "Predicted" and "not predicted"
        normally classify reported regions by prior hypothesis; they are not
        participant groups, contrast directions, or separate maps.

        Distinguish an anatomical ROI label from an explicit analysis-method
        block. A region merely described as an ROI stays in its current
        analysis. However, separately labeled "ROI analysis" and "whole-brain
        analysis" result blocks are distinct analyses because they report
        different statistical searches, even when their contrast is the same.
        An ROI subsection begins where its explicit header appears and applies
        only to the following rows; do not retroactively assign preceding
        whole-brain/unrestricted rows to that ROI subsection.

        If there is no explicit analysis-defining label, treat the whole table
        as one analysis. Use only labels that appear in the table, caption, or
        footnotes. Combine explicit labels when needed to make analyses
        distinguishable, but never invent a contrast or group.

        COORDINATE COMPLETENESS AND SAFETY

        - Extract each valid coordinate row exactly once. Before returning,
          check that no continuation row or valid local maximum was dropped.
        - Coordinates must come only from X, Y, Z columns, or an explicitly
          labeled equivalent such as MNI/Talairach coordinates.
        - Never use Cluster, Volume, extent, Brodmann area, ALE, T, Z-statistic,
          p-value, or another numeric column as a coordinate component.
        - A coordinate requires exactly three numeric values in [X, Y, Z]
          order. Exclude rows that do not provide a complete triple.
        - Some exported tables place several delimited numbers in one cell. If
          the X, Y, and Z cells contain the same number of clearly separated
          values, align them positionally and emit one point per triple. If the
          alignment is ambiguous, omit those values rather than guessing.
          Never concatenate multiple tokens into one coordinate number.
        - Treat implausible coordinate magnitudes above 200 mm as malformed
          table extraction, not as valid stereotactic coordinates.
        - A statistical column named Z is not the spatial Z coordinate. Resolve
          this from the grouped coordinate header and neighboring X/Y columns.
        - Preserve negative signs and decimals exactly as reported.

        SPACE

        - Use "MNI" when the table/caption/footnote identifies MNI space.
        - Use "TAL" for Talairach space.
        - Otherwise use null. Do not infer space from coordinate magnitude.

        STATISTICAL VALUES

        Include same-row statistical values when available. The `kind` must be
        one of:
        - "z-statistic"
        - "t-statistic"
        - "f-statistic"
        - "p-value"
        - "beta"
        - "correlation"
        - "other"

        Do not include cluster size, volume, Brodmann area, ALE, coordinates,
        or other non-statistical metadata as statistical values. Omit `values`
        when no statistical values are available.

        OUTPUT CHECK

        - Coordinates are always numeric [x, y, z] triples.
        - Every point is assigned to exactly one analysis.
        - Distinct experimental contrasts/groups/directions are not merged.
        - Descriptive anatomical sections are not incorrectly split.
        - Recheck every repeated X/Y/Z header block: each must either produce
          its own analysis or be intentionally empty.
        - If two outputs differ only by hemisphere, region, cluster,
          predicted/non-predicted status, or local-maximum labels, merge them
          unless the source explicitly defines separate statistical maps.
        - If one output pools points from different contrast, group, sign,
          ROI-analysis, whole-brain-analysis, session, or condition blocks,
          split it before returning.
        - Do not return fields outside the schema.

        Table:
        {table_text}
        """
    ).strip()
