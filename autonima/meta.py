"""Meta-analysis functionality for Autonima."""

import argparse
import math
import os
from pathlib import Path
import json
from typing import Iterable, Optional, Set

# Try to import NiMARE dependencies
try:
    from nimare.correct import FDRCorrector, FWECorrector
    from nimare.workflows import CBMAWorkflow
    from nimare.meta.cbma import MKDADensity, ALE, KDA
    from nimare.nimads import Studyset
    from nimare.reports.base import run_reports
except ImportError as exc:
    # Report the real failure: an installed but incompatible NiMARE used to be reported
    # as "NiMARE is not installed".
    if getattr(exc, "name", None) == "nimare":
        message = "NiMARE is not installed."
    else:
        message = f"Could not import NiMARE: {exc}. autonima meta needs nimare>=0.11."
    raise ImportError(
        f'{message} Install a supported version with: pip install "nimare>=0.11"'
    ) from exc


def create_estimator(estimator_name, estimator_args):
    """Create an estimator instance based on the name and arguments."""
    estimator_map = {
        "ale": ALE,
        "mkdadensity": MKDADensity,
        "kda": KDA,
    }

    if estimator_name not in estimator_map:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    estimator_class = estimator_map[estimator_name]
    return estimator_class(**estimator_args)


def create_corrector(corrector_name, corrector_args):
    """Create a corrector instance based on the name and arguments."""
    if corrector_name == "fdr":
        return FDRCorrector(**corrector_args)
    if corrector_name == "montecarlo":
        return FWECorrector(method="montecarlo", **corrector_args)
    if corrector_name == "bonferroni":
        return FWECorrector(method="bonferroni", **corrector_args)
    raise ValueError(f"Unsupported corrector: {corrector_name}")


def find_nimads_files(output_folder):
    """Find NiMADS StudySet and Annotation JSON files in the output folder."""
    output_path = Path(output_folder)
    studyset_file = output_path / "nimads_studyset.json"
    annotation_file = output_path / "nimads_annotation.json"

    if not studyset_file.exists():
        raise FileNotFoundError(f"StudySet file not found: {studyset_file}")

    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    return str(studyset_file), str(annotation_file)


def resolve_nimads_folder(output_folder) -> Path:
    """Return the folder that holds the NiMADS files, given a run folder or its outputs/.

    The README, the CLI help and the guides all pass ``<run>/outputs``, while the web UI
    passes the run folder itself. This used to append ``outputs/`` unconditionally, so
    following the docs looked for ``<run>/outputs/outputs/`` and failed.
    """
    folder = Path(output_folder)
    if (folder / "nimads_studyset.json").exists():
        return folder
    return folder / "outputs"


def load_include_ids(include_ids_file: Optional[os.PathLike]) -> Optional[Set[str]]:
    """
    Load study IDs (PMIDs) to include from a text file.

    The file should contain one ID per line.
    """
    if include_ids_file is None:
        return None

    include_path = Path(include_ids_file)
    if not include_path.exists():
        raise FileNotFoundError(f"Include file not found: {include_path}")

    with include_path.open("r") as f:
        include_ids = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(include_ids)} study IDs to include from {include_path}")
    return include_ids


def _normalize_include_ids(include_ids) -> Optional[Set[str]]:
    """Normalize include_ids to a set of non-empty strings."""
    if include_ids is None:
        return None

    if isinstance(include_ids, (str, os.PathLike)):
        return load_include_ids(include_ids)

    if isinstance(include_ids, set):
        normalized = {str(value).strip() for value in include_ids if str(value).strip()}
        return normalized

    if isinstance(include_ids, Iterable):
        normalized = {str(value).strip() for value in include_ids if str(value).strip()}
        return normalized

    raise TypeError(
        "include_ids must be None, a path-like object, or an iterable of IDs"
    )


def _load_and_sanitize_data(studyset_file: Path, annotation_file: Path):
    """Load NiMADS studyset + annotation payloads with minimal normalization."""

    print("Loading studyset JSON...")
    with open(studyset_file, "r") as f:
        studyset_data = json.load(f)

    print("Loading annotation JSON...")
    with open(annotation_file, "r") as f:
        annotation_data = json.load(f)

    print("Normalizing annotation data...")
    if isinstance(annotation_data, list):
        annotation_data = annotation_data[0] if annotation_data else {}

    return studyset_data, annotation_data


def _get_boolean_columns(annotation_data, columns=None):
    """Get boolean annotation columns, optionally filtered to requested columns."""
    boolean_columns = [
        col for col, col_type in annotation_data.get("note_keys", {}).items()
        if col_type == "boolean"
    ]

    if columns is None:
        return boolean_columns

    requested = [str(column) for column in columns]
    selected = []
    for column in requested:
        if column in boolean_columns:
            selected.append(column)
        else:
            print(f"Column {column} is missing or non-boolean in annotation. Skipping.")
    return selected


def _analysis_ids_for_column(annotation_data, column):
    """Get analysis IDs associated with a boolean column."""
    return [
        note.get("analysis")
        for note in annotation_data.get("notes", [])
        if isinstance(note, dict)
        and isinstance(note.get("note"), dict)
        and note["note"].get(column)
        and note.get("analysis")
    ]


def _unknown_analysis_count(annotation_data, column):
    """Count analyses whose decision for a column is null: annotation failed, not excluded."""
    return sum(
        1
        for note in annotation_data.get("notes", [])
        if isinstance(note, dict)
        and isinstance(note.get("note"), dict)
        and column in note["note"]
        and note["note"][column] is None
    )


def _analysis_has_valid_coordinates(analysis) -> bool:
    """Return True when an analysis has at least one finite (x, y, z) point.

    Accepts a NiMADS analysis dict (points carry ``coordinates``) or a NiMARE analysis
    object (points carry ``x``, ``y``, ``z``).
    """
    if isinstance(analysis, dict):
        points = analysis.get("points") or []
        triples = [tuple(p.get("coordinates") or ())[:3] for p in points if isinstance(p, dict)]
    else:
        points = getattr(analysis, "points", None) or []
        triples = [
            (getattr(p, "x", None), getattr(p, "y", None), getattr(p, "z", None))
            for p in points
        ]
    for coords in triples:
        if len(coords) == 3 and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in coords
        ):
            return True
    return False


def _select_studyset_data(studyset_data, analysis_ids, include_ids=None):
    """The studyset dict holding exactly the selected analyses, and what was dropped.

    Selection is done on the NiMADS dict rather than with ``Studyset.slice``, whose meaning
    depends on the NiMARE version: on 0.21.x it keeps every analysis of each named study
    (asking for one of a study's five analyses returns all five) and raises on an id that is
    not in the studyset. Filtering the dict behaves the same on every supported release.

    - ids not present in the studyset are ignored;
    - ``include_ids`` (study ids) restricts the result to those studies;
    - analyses without a finite (x, y, z) point are dropped and counted;
    - a repeated study or analysis record is kept once (an export can repeat a study);
    - a point with no recorded space is labelled MNI, which is how NiMARE treated it before
      0.21 (no transform applied); 0.21 refuses a dataset without a space. Counted.
    """
    wanted = {str(i) for i in analysis_ids}
    counts = {"no_coordinates": 0, "unknown_space": 0, "duplicates": 0}
    studies, seen_studies, seen_analyses = [], set(), set()
    for study in studyset_data.get("studies", []) or []:
        study_id = str(study.get("id"))
        if study_id in seen_studies:
            counts["duplicates"] += sum(
                1 for a in study.get("analyses") or [] if str(a.get("id")) in wanted
            )
            continue
        seen_studies.add(study_id)
        if include_ids is not None and study_id not in include_ids:
            continue
        kept = []
        for analysis in study.get("analyses") or []:
            analysis_id = str(analysis.get("id"))
            if analysis_id not in wanted:
                continue
            if analysis_id in seen_analyses:
                counts["duplicates"] += 1
                continue
            seen_analyses.add(analysis_id)
            if not _analysis_has_valid_coordinates(analysis):
                counts["no_coordinates"] += 1
                continue
            points = []
            unknown = False
            for point in analysis.get("points") or []:
                if isinstance(point, dict) and not point.get("space"):
                    point = dict(point, space="MNI")
                    unknown = True
                points.append(point)
            counts["unknown_space"] += int(unknown)
            kept.append(dict(analysis, name=analysis_id, points=points))
        if kept:
            studies.append(dict(study, name=study_id, analyses=kept))
    subset = dict(studyset_data)
    subset["studies"] = studies
    return subset, counts


def _post_mortem_debugger():
    """Launch post-mortem debugging, preferring ipdb when available."""
    import sys

    try:
        import ipdb as debugger
    except ImportError:
        import pdb as debugger

    debugger.post_mortem(sys.exc_info()[2])


def run_meta_analysis_for_column(
    studyset,
    annotation,
    annotation_data,
    column,
    output_dir,
    estimator_name="mkdadensity",
    estimator_args=None,
    corrector_name="fdr",
    corrector_args=None,
    include_ids=None,
    generate_reports=False,
):
    """Run meta-analysis for a specific annotation column."""
    del annotation  # kept for backward compatibility with existing call sites

    print(f"Running meta-analysis for column: {column}")

    column_type = annotation_data.get("note_keys", {}).get(column)
    if column_type is None:
        print(f"Column {column} not found in annotation. Skipping.")
        return None

    if column_type != "boolean":
        print(f"Column {column} is not boolean. Skipping.")
        return None

    analysis_ids = _analysis_ids_for_column(annotation_data, column)
    unknown = _unknown_analysis_count(annotation_data, column)
    if unknown:
        # Left out because they were never decided, not because they were excluded. Say so,
        # or the map silently loses them the way the old False default did.
        print(
            f"Warning: {unknown} analyses have no decision for column {column} "
            "(annotation failed) and are left out of this meta-analysis. "
            "Re-run the pipeline to retry them."
        )
    if not analysis_ids:
        print(f"No studies found for column {column}. Skipping.")
        return None

    if not isinstance(studyset, dict):
        # Older call sites pass a NiMARE Studyset; select on its dict form all the same.
        studyset = studyset.to_dict()
    include_ids_set = _normalize_include_ids(include_ids)
    subset, counts = _select_studyset_data(studyset, analysis_ids, include_ids_set)
    n_selected = sum(len(study["analyses"]) for study in subset["studies"])

    if counts["duplicates"]:
        print(
            f"Ignoring {counts['duplicates']} repeated analysis records for column {column} "
            "(the studyset repeats a study); each analysis is fitted once."
        )
    if counts["no_coordinates"]:
        print(
            f"Skipping {counts['no_coordinates']} analyses without valid coordinates "
            f"for column {column}."
        )
    if counts["unknown_space"]:
        print(
            f"{counts['unknown_space']} analyses for column {column} have points with no "
            "recorded coordinate space; those points are fitted as MNI."
        )
    if n_selected == 0:
        if include_ids_set is not None:
            print(f"No analyses remain after include-ids filtering for column {column}. Skipping.")
        else:
            print(f"No analyses with valid coordinates remain for column {column}. Skipping.")
        return None

    dataset = Studyset(subset).to_dataset()
    if len(dataset.ids) != n_selected:
        raise RuntimeError(
            f"NiMARE built a dataset of {len(dataset.ids)} analyses for column {column}, "
            f"but {n_selected} were selected."
        )

    estimator = create_estimator(estimator_name, estimator_args or {})
    corrector = create_corrector(corrector_name, corrector_args or {})

    workflow = CBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics="focuscounter",
        output_dir=str(output_dir),
    )

    meta_results = workflow.fit(dataset)
    if generate_reports:
        run_reports(meta_results, str(output_dir))

    return meta_results


def run_meta_analyses_from_files(
    studyset_file,
    annotation_file,
    output_dir,
    estimator_name="mkdadensity",
    estimator_args=None,
    corrector_name="fdr",
    corrector_args=None,
    include_ids=None,
    skip_existing=False,
    columns=None,
    fail_fast=False,
    debug=False,
    generate_reports=False,
):
    """
    Run meta-analyses from explicit studyset + annotation file paths.

    Parameters
    ----------
    studyset_file, annotation_file : path-like
        NiMADS studyset and annotation JSON paths.
    output_dir : path-like
        Directory where per-column outputs should be written.
    include_ids : optional
        Either a path to a newline-delimited ID file or an iterable/set of study IDs.
        IDs are applied post-hoc after annotation slicing.
    skip_existing : bool
        If True, skip columns whose output directories already exist and are non-empty.
    columns : optional list
        If provided, only these annotation columns are considered.
    fail_fast : bool
        If True, re-raise on the first column-level exception instead of continuing.
    debug : bool
        If True and fail_fast is enabled, enter post-mortem debugging on error.
    generate_reports : bool
        If True, run NiMARE HTML report generation for each completed column.
    """
    include_ids_set = _normalize_include_ids(include_ids)

    studyset_file = Path(studyset_file)
    annotation_file = Path(annotation_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    studyset_data, annotation_data = _load_and_sanitize_data(studyset_file, annotation_file)


    selected_columns = _get_boolean_columns(annotation_data, columns=columns)
    print(f"Found {len(selected_columns)} boolean columns: {selected_columns}")

    results = {}
    for column in selected_columns:
        column_output_dir = output_dir / column

        if skip_existing and column_output_dir.exists() and any(column_output_dir.iterdir()):
            print(f"⊘ Skipping {column} (output already exists)")
            continue

        column_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            meta_results = run_meta_analysis_for_column(
                studyset_data,
                None,
                annotation_data,
                column,
                str(column_output_dir),
                estimator_name,
                estimator_args,
                corrector_name,
                corrector_args,
                include_ids=include_ids_set,
                generate_reports=generate_reports,
            )
            if meta_results is not None:
                results[column] = meta_results
        except Exception as e:
            print(f"Error running meta-analysis for column {column}: {e}")
            import traceback

            traceback.print_exc()
            if fail_fast:
                if debug:
                    _post_mortem_debugger()
                raise

    return results


def run_meta_analyses(
    output_folder,
    estimator_name="mkdadensity",
    estimator_args=None,
    corrector_name="fdr",
    corrector_args=None,
    include_ids=None,
    skip_existing=False,
    columns=None,
    fail_fast=False,
    debug=False,
    generate_reports=False,
):
    """Run meta-analyses on all boolean annotation columns in the NiMADS files.

    ``output_folder`` may be the run folder or its ``outputs/`` folder. Results are written
    to ``meta_analysis_results/`` beside the NiMADS files either way.
    """
    output_folder = resolve_nimads_folder(output_folder)
    studyset_file, annotation_file = find_nimads_files(output_folder)

    output_dir = Path(output_folder) / "meta_analysis_results"
    output_dir.mkdir(exist_ok=True)

    return run_meta_analyses_from_files(
        studyset_file=studyset_file,
        annotation_file=annotation_file,
        output_dir=output_dir,
        estimator_name=estimator_name,
        estimator_args=estimator_args,
        corrector_name=corrector_name,
        corrector_args=corrector_args,
        include_ids=include_ids,
        skip_existing=skip_existing,
        columns=columns,
        fail_fast=fail_fast,
        debug=debug,
        generate_reports=generate_reports,
    )


def main():
    """Main function to run meta-analyses from command line."""
    parser = argparse.ArgumentParser(description="Run meta-analyses on autonima output")
    parser.add_argument(
        "output_folder",
        help="Run output folder, or its outputs/ folder holding the NiMADS files",
    )

    parser.add_argument(
        "--estimator",
        choices=["ale", "mkdadensity", "kda"],
        default="mkdadensity",
        help="CBMA estimator to use (default: mkdadensity)",
    )

    parser.add_argument(
        "--estimator-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the estimator (default: {})",
    )

    parser.add_argument(
        "--corrector",
        choices=["fdr", "montecarlo", "bonferroni"],
        default="fdr",
        help="Corrector to use (default: fdr)",
    )

    parser.add_argument(
        "--corrector-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the corrector (default: {})",
    )

    parser.add_argument(
        "--include-ids",
        type=Path,
        default=None,
        help="Path to text file with study IDs/PMIDs to include (one per line)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing annotation column instead of continuing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable post-mortem debugging (prefers ipdb) when used with --fail-fast",
    )
    parser.add_argument(
        "--run-reports",
        action="store_true",
        help="Generate NiMARE HTML reports (disabled by default to reduce memory use)",
    )

    args = parser.parse_args()

    try:
        estimator_args = json.loads(args.estimator_args)
        corrector_args = json.loads(args.corrector_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}")
        return

    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder {args.output_folder} does not exist")
        return

    results = run_meta_analyses(
        args.output_folder,
        estimator_name=args.estimator,
        estimator_args=estimator_args,
        corrector_name=args.corrector,
        corrector_args=corrector_args,
        include_ids=args.include_ids,
        fail_fast=args.fail_fast,
        debug=args.debug,
        generate_reports=args.run_reports,
    )
    print(f"Completed meta-analyses for {len(results)} columns")


if __name__ == "__main__":
    main()
