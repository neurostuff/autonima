"""Coordinate-based meta-analysis of an exported NiMADS studyset, one per target.

    pip install nimare
    python run_meta.py REVIEW_DIR [--targets working_memory ...]
                                  [--estimator mkdadensity|ale|kda] [--corrector fdr|montecarlo|bonferroni]

Reads results/nimads/studyset.json and annotation.json (written by
`ledger.py export`) and writes results/meta/<target>/ with NiMARE's maps and a
summary.json per target. Estimator and corrector default to review.yaml `meta:`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--estimator")
    ap.add_argument("--corrector")
    ap.add_argument("--min-studies", type=int, default=10,
                    help="skip targets with fewer included studies (CBMA is unstable below ~10-17)")
    args = ap.parse_args(argv)

    try:
        from nimare.correct import FDRCorrector, FWECorrector
        from nimare.meta.cbma import ALE, KDA, MKDADensity
        from nimare.nimads import Studyset
        from nimare.workflows import CBMAWorkflow
    except ImportError:
        print("NiMARE is required for this stage: pip install nimare", file=sys.stderr)
        return 1

    spec_meta = {}
    spec_path = args.review_dir / "review.yaml"
    if spec_path.exists():
        import yaml  # type: ignore
        spec_meta = (yaml.safe_load(spec_path.read_text()) or {}).get("meta") or {}
    estimator_name = args.estimator or spec_meta.get("estimator", "mkdadensity")
    corrector_name = args.corrector or spec_meta.get("corrector", "fdr")
    estimator_args = spec_meta.get("estimator_args") or {}
    corrector_args = spec_meta.get("corrector_args") or {}

    nimads = args.review_dir / "results" / "nimads"
    studyset_data = json.loads((nimads / "studyset.json").read_text())
    annotation = json.loads((nimads / "annotation.json").read_text())
    targets = args.targets or list(annotation["note_keys"])

    estimators = {"mkdadensity": MKDADensity, "ale": ALE, "kda": KDA}
    if estimator_name not in estimators:
        print(f"unknown estimator {estimator_name}", file=sys.stderr)
        return 1

    def corrector():
        if corrector_name == "fdr":
            return FDRCorrector(**corrector_args)
        if corrector_name in ("montecarlo", "bonferroni"):
            return FWECorrector(method=corrector_name, **corrector_args)
        raise SystemExit(f"unknown corrector {corrector_name}")

    summary = {}
    for target in targets:
        ids = [n["analysis_id"] for n in annotation["notes"] if n["note"].get(target) is True]
        n_studies = len({i.split("-", 1)[0] for i in ids})
        out_dir = args.review_dir / "results" / "meta" / target
        out_dir.mkdir(parents=True, exist_ok=True)
        info = {"target": target, "analyses": len(ids), "studies": n_studies,
                "estimator": estimator_name, "corrector": corrector_name}
        if n_studies < args.min_studies:
            info["skipped"] = f"only {n_studies} studies (< --min-studies {args.min_studies})"
            print(json.dumps(info))
            summary[target] = info
            (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
            continue
        studyset = Studyset(studyset_data).slice(analyses=ids)
        dataset = studyset.to_dataset()
        workflow = CBMAWorkflow(estimator=estimators[estimator_name](**estimator_args), corrector=corrector(),
                                diagnostics="focuscounter", output_dir=str(out_dir))
        workflow.fit(dataset)
        info["output_dir"] = str(out_dir)
        info["dataset_ids"] = len(dataset.ids)
        (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
        summary[target] = info
        print(json.dumps(info))
    (args.review_dir / "results" / "meta" / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
