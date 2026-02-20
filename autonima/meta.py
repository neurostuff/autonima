"""Meta-analysis functionality for Autonima."""

import argparse
import os
from pathlib import Path
import json
from typing import Iterable, Optional, Set

# Try to import NiMARE dependencies
try:
    from nimare.correct import FDRCorrector, FWECorrector
    from nimare.workflows import CBMAWorkflow
    from nimare.meta.cbma import MKDADensity, ALE, KDA
    from nimare.nimads import Studyset, Annotation
    from nimare.reports.base import run_reports

    NIMARE_AVAILABLE = True
except ImportError:
    NIMARE_AVAILABLE = False

if not NIMARE_AVAILABLE:
    raise ImportError(
        "NiMARE is not installed. Please install with: pip install autonima[meta]"
    )


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
    """Load and sanitize NiMADS studyset + annotation payloads."""
    from .coordinates.nimads_models import sanitize_studyset_dict, sanitize_annotation_dict

    print("Loading studyset JSON...")
    with open(studyset_file, "r") as f:
        studyset_data = json.load(f)

    print("Loading annotation JSON...")
    with open(annotation_file, "r") as f:
        annotation_data = json.load(f)

    print("Sanitizing studyset data...")
    studyset_data = sanitize_studyset_dict(studyset_data)

    print("Sanitizing annotation data...")
    if isinstance(annotation_data, list):
        annotation_data = annotation_data[0] if annotation_data else {}
    annotation_data = sanitize_annotation_dict(annotation_data)

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
    if not analysis_ids:
        print(f"No studies found for column {column}. Skipping.")
        return None

    first_studyset = studyset.slice(analyses=analysis_ids)

    include_ids_set = _normalize_include_ids(include_ids)
    if include_ids_set is not None:
        filtered_analysis_ids = [
            analysis.id
            for study in first_studyset.studies
            if study.id in include_ids_set
            for analysis in study.analyses
        ]
        if not filtered_analysis_ids:
            print(f"No analyses remain after include-ids filtering for column {column}. Skipping.")
            return None
        first_studyset = first_studyset.slice(analyses=filtered_analysis_ids)

    for study in first_studyset.studies:
        study.name = study.id
        for analysis in study.analyses:
            analysis.name = analysis.id

    first_dataset = first_studyset.to_dataset()
    if len(first_dataset.ids) == 0:
        print(f"No analyses available for column {column} after slicing. Skipping.")
        return None

    estimator = create_estimator(estimator_name, estimator_args or {})
    corrector = create_corrector(corrector_name, corrector_args or {})

    workflow = CBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics="focuscounter",
        output_dir=str(output_dir),
    )

    meta_results = workflow.fit(first_dataset)
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
    """
    include_ids_set = _normalize_include_ids(include_ids)

    studyset_file = Path(studyset_file)
    annotation_file = Path(annotation_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    studyset_data, annotation_data = _load_and_sanitize_data(studyset_file, annotation_file)

    print("Creating studyset...")
    studyset = Studyset(studyset_data)

    print("Creating annotation...")
    annotation = Annotation(annotation_data, studyset)

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
                studyset,
                annotation,
                annotation_data,
                column,
                str(column_output_dir),
                estimator_name,
                estimator_args,
                corrector_name,
                corrector_args,
                include_ids=include_ids_set,
            )
            if meta_results is not None:
                results[column] = meta_results
        except Exception as e:
            print(f"Error running meta-analysis for column {column}: {e}")
            import traceback

            traceback.print_exc()

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
):
    """Run meta-analyses on all boolean annotation columns in the NiMADS files."""
    output_folder = Path(output_folder) / "outputs"
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
    )


def main():
    """Main function to run meta-analyses from command line."""
    parser = argparse.ArgumentParser(description="Run meta-analyses on autonima output")
    parser.add_argument(
        "output_folder",
        help="Path to the autonima output folder containing NiMADS files",
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
    )
    print(f"Completed meta-analyses for {len(results)} columns")


if __name__ == "__main__":
    main()
