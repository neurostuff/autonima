# Autonima

**Autonima** (Automated Neuroimaging Meta-Analysis) is an LLM-powered framework for automating systematic literature reviews and meta-analyses in neuroimaging.

Given a meta-analysis specification (e.g., inclusion/exclusion criteria, study objective, search parameters), Autonima automates the full evidence-synthesis workflow, from literature search to final inclusion of studies, in line with the **PRISMA** framework.

Autonima is part of the **Neurosynth family of tools** for open, reproducible, and scalable meta-analysis.

---

## 🌟 Key Features

Autonima enables end-to-end automation of systematic review steps:

1. **Automated Search**

   * Uses PubMed to apply user-specified search parameters.
   * Fetches all potentially relevant abstracts.

2. **Abstract Screening (PRISMA Identification & Screening)**

   * Applies inclusion/exclusion criteria to abstracts using LLM-based screening.
   * Benchmarked against human/manual performance.
   * Uses function calling with Pydantic schemas for structured output.

3. **Full-Text Retrieval**

   * Downloads full texts of candidate papers via a combination of **PubGet** and **ACE**.

4. **Eligibility & Inclusion**

   * Applies full-text screening with LLMs.
   * Produces a final set of included studies for meta-analysis.
   * Uses function calling with Pydantic schemas for structured output.

---

## 📊 Benchmarking

Autonima is benchmarked against **[neurometabench](https://github.com/neurosynth/neurometabench)**, a structured dataset of neuroimaging meta-analyses designed to test automated reconstruction workflows.

Benchmarking focuses on:

* **Recall & precision** of included studies (abstract vs. full-text screening).
* **Reproducibility** of original inclusion decisions.
* **Scalability & speed** compared to human systematic reviews.

---

## 📂 Repository Structure

```
autonima/
├── autonima/             # Core Python package
├── benchmarks/
│   └── neurometabench/   # Scripts for benchmarking against neurometabench
├── tests/                # Unit and integration tests
└── README.md             # You are here
```

---

## 🚀 Quickstart

```bash
# clone repository
git clone https://github.com/neurosynth/autonima.git
cd autonima

# install
pip install -e .

# run an example review pipeline
python examples/run_pipeline.py --config examples/sample_config.yaml
```

Where `sample_config.yaml` specifies:

* Study objective
* Search parameters
* Inclusion/exclusion criteria

---

## 🔍 Example Workflow

1. Define your review specification (`YAML` or `JSON`):

   ```yaml
   objective: "Identify fMRI studies of working memory in schizophrenia"
   search:
     database: "pubmed"
     query: "schizophrenia AND working memory AND fMRI"
   inclusion_criteria:
     - Human participants
     - fMRI neuroimaging
     - Case-control design
   exclusion_criteria:
     - Animal studies
     - Review articles
   ```

2. Run Autonima pipeline:

   ```bash
   autonima run --config config.yaml
   ```

3. Output:

   * **PRISMA flow diagram** (records identified, screened, excluded, included).
   * **Final included study list** (with metadata + DOIs/PMIDs).
   * **NiMARE dataset** for downstream meta-analysis.

---

## 🧬 Related Projects

* [neurometabench](https://github.com/neurostuff/neurometabench) – benchmarking dataset
* [NiMARE](https://github.com/neurostuff/NiMARE) – neuroimaging meta-analysis in Python
* [Neurosynth Compose](https://compose.neurosynth.org) – human-in-the-loop neuroimaging meta-analysis

---
