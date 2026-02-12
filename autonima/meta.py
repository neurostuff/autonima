"""Meta-analysis functionality for Autonima."""

import argparse
import os
from pathlib import Path
import json
import importlib

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
    # Map estimator names to classes
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
    # Map corrector names to classes
    if corrector_name == "fdr":
        return FDRCorrector(**corrector_args)
    elif corrector_name == "montecarlo":
        return FWECorrector(method="montecarlo", **corrector_args)
    elif corrector_name == "bonferroni":
        return FWECorrector(method="bonferroni", **corrector_args)
    else:
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


def run_meta_analysis_for_column(studyset, annotation, annotation_data, column, output_dir, 
                                 estimator_name="mkdadensity", estimator_args=None,
                                 corrector_name="fdr", corrector_args=None):
    """Run meta-analysis for a specific annotation column."""
    print(f"Running meta-analysis for column: {column}")
    
    # Get column type from the original annotation data
    column_type = annotation_data["note_keys"].get(column)
    if column_type is None:
        print(f"Column {column} not found in annotation. Skipping.")
        return
    
    # Only process boolean columns
    if column_type != "boolean":
        print(f"Column {column} is not boolean. Skipping.")
        return
    
    # Get analysis ids for the studies to include
    analysis_ids = [
        n["analysis"] for n in annotation_data["notes"] if n["note"].get(column)
    ]
    
    if not analysis_ids:
        print(f"No studies found for column {column}. Skipping.")
        return
    
    # Slice the studyset to include only selected studies
    first_studyset = studyset.slice(analyses=analysis_ids)

    # Switch studyset.name with studyset.id to ensure uniqueness
    for study in first_studyset.studies:
        study.name = study.id

        # Switch analysis names to IDs to ensure uniqueness
        for analysis in study.analyses:
            analysis.name = analysis.id    

    
    # Convert to dataset
    first_dataset = first_studyset.to_dataset()
    
    # Set up estimator and corrector
    estimator = create_estimator(estimator_name, estimator_args or {})
    corrector = create_corrector(corrector_name, corrector_args or {})
    
    # Run meta-analysis
    workflow = CBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics="focuscounter",
        output_dir=output_dir,
    )
    
    meta_results = workflow.fit(first_dataset)

    run_reports(meta_results, output_dir)
    
    return meta_results


def run_meta_analyses(output_folder, estimator_name="mkdadensity", estimator_args=None,
                      corrector_name="fdr", corrector_args=None):
    """Run meta-analyses on all boolean annotation columns in the NiMADS files."""
    # Import sanitization functions
    from .coordinates.nimads_models import sanitize_studyset_dict, sanitize_annotation_dict
    
    # Find the NiMADS files
    output_folder = Path(output_folder) / "outputs"
    studyset_file, annotation_file = find_nimads_files(output_folder)
    
    # Create output directory
    output_dir = Path(output_folder) / "meta_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load the JSON data
    print("Loading studyset JSON...")
    with open(studyset_file, 'r') as f:
        studyset_data = json.load(f)
    
    print("Loading annotation JSON...")
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    # Sanitize the data before passing to NiMARE
    print("Sanitizing studyset data...")
    studyset_data = sanitize_studyset_dict(studyset_data)
    
    print("Sanitizing annotation data...")
    if isinstance(annotation_data, list):
        annotation_data = annotation_data[0] if annotation_data else {}
    annotation_data = sanitize_annotation_dict(annotation_data)
    
    # Process the files using NiMARE classes
    print("Creating studyset...")
    studyset = Studyset(studyset_data)
    
    print("Creating annotation...")
    annotation = Annotation(annotation_data, studyset)
    
    # Get all boolean columns from the original annotation data
    boolean_columns = [
        col for col, col_type in annotation_data["note_keys"].items() 
        if col_type == "boolean"
    ]
    
    print(f"Found {len(boolean_columns)} boolean columns: {boolean_columns}")
    
    # Run meta-analysis for each boolean column
    results = {}
    for column in boolean_columns:
        column_output_dir = output_dir / column
        column_output_dir.mkdir(exist_ok=True)    
        try:
            meta_results = run_meta_analysis_for_column(
                studyset, annotation, annotation_data, column, str(column_output_dir),
                estimator_name, estimator_args, corrector_name, corrector_args
            )
            results[column] = meta_results
        except Exception as e:
            print(f"Error running meta-analysis for column {column}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    """Main function to run meta-analyses from command line."""
    parser = argparse.ArgumentParser(description="Run meta-analyses on autonima output")
    parser.add_argument(
        "output_folder",
        help="Path to the autonima output folder containing NiMADS files"
    )
    
    # Add arguments for estimator
    parser.add_argument(
        "--estimator",
        choices=["ale", "mkdadensity", "kda"],
        default="mkdadensity",
        help="CBMA estimator to use (default: mkdadensity)"
    )
    
    parser.add_argument(
        "--estimator-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the estimator (default: {})"
    )
    
    # Add arguments for corrector
    parser.add_argument(
        "--corrector",
        choices=["fdr", "montecarlo", "bonferroni"],
        default="fdr",
        help="Corrector to use (default: fdr)"
    )
    
    parser.add_argument(
        "--corrector-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the corrector (default: {})"
    )
    
    args = parser.parse_args()
    
    # Parse estimator and corrector arguments
    try:
        estimator_args = json.loads(args.estimator_args)
        corrector_args = json.loads(args.corrector_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}")
        return
    
    # Check if output folder exists
    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder {args.output_folder} does not exist")
        return
    
    results = run_meta_analyses(
        args.output_folder,
        estimator_name=args.estimator,
        estimator_args=estimator_args,
        corrector_name=args.corrector,
        corrector_args=corrector_args
    )
    print(f"Completed meta-analyses for {len(results)} columns")


if __name__ == "__main__":
    main()