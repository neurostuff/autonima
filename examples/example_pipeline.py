#!/usr/bin/env python3
"""
Example pipeline runner for AUTONIMA
Automated NeuroImaging Meta-Analysis + Systematic Review Tool
"""

import yaml
import os
from pathlib import Path


class AutonimaPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.results = {"included": [], "excluded": []}

    def run(self):
        self._log("Starting AUTONIMA pipeline")
        self.search()
        self.screen()
        self.retrieve()
        self.save_outputs()
        self._log("Pipeline complete")

    def search(self):
        search_cfg = self.config.get("search", {})
        query = search_cfg.get("query", "")
        db = search_cfg.get("database", "pubmed")
        self._log(f"Searching {db} for: {query}")
        # Stub: Replace with actual PubMed/Entrez API or Pubget call
        self.results["included"] = [f"PMID{i}" for i in range(1, 6)]

    def screen(self):
        screen_cfg = self.config.get("screening", {})
        model = screen_cfg.get("abstract", {}).get("model", "gpt-4")
        self._log(f"Screening abstracts with model {model}")
        # Stub: Replace with LLM screening
        # Simulate exclusions
        self.results["excluded"] = [self.results["included"].pop()]

    def retrieve(self):
        retrieval_cfg = self.config.get("retrieval", {})
        self._log("Retrieving full-texts from sources: "
                  + ", ".join(retrieval_cfg.get("sources", [])))
        # Stub: Replace with Pubget/ACE integration
        for doc in self.results["included"]:
            self._log(f"Retrieved full-text for {doc}")

    def save_outputs(self):
        output_cfg = self.config.get("output", {})
        out_dir = Path(output_cfg.get("directory", "results"))
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "results.csv"
        with open(csv_path, "w") as f:
            f.write("id,status\n")
            for inc in self.results["included"]:
                f.write(f"{inc},included\n")
            for exc in self.results["excluded"]:
                f.write(f"{exc},excluded\n")
        self._log(f"Saved results to {csv_path}")

        if output_cfg.get("prisma_diagram", False):
            self._log("Generating PRISMA diagram (stub)")

    def _log(self, msg: str):
        print(f"[AUTONIMA] {msg}")


if __name__ == "__main__":
    cfg_path = os.environ.get("AUTONIMA_CONFIG", "examples/sample_config.yaml")
    pipeline = AutonimaPipeline(cfg_path)
    pipeline.run()
