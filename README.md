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

   * Downloads full texts of candidate papers via a combination of **PubGet**

4. **Eligibility & Inclusion**

   * Applies full-text screening with LLMs.
   * Produces a final set of included studies for meta-analysis.
   * Uses function calling with Pydantic schemas for structured output.

5. **Enhanced HTML Cleaning**

   * Uses Mozilla's Readability algorithm via `readabilipy` to extract clean text content from HTML.
   * Falls back to basic HTML cleaning when `readabilipy` is not available.
   * Install with `pip install -e .[readability]` (requires Node.js).

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

# install with enhanced HTML cleaning (requires Node.js)
pip install -e .[readability]

> Note: The enhanced HTML cleaning feature requires Node.js to be installed on your system.

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
   search:
     database: "pubmed"
     query: "schizophrenia AND working memory AND fMRI"
   screening:
     abstract:
       objective: "Identify fMRI studies of working memory in schizophrenia"
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

## 🧠 Coordinate Parsing

Autonima includes a specialized module for parsing neuroimaging coordinate tables using LLMs. This module can:

* Parse CSV files containing neuroimaging results tables
* Extract coordinate points and their associated metadata
* Structure the data according to standardized schemas
* Support parallel processing for improved performance

### Python API

```python
from autonima.coordinates import parse_tables

# Parse tables with default settings
results = parse_tables("./input_tables", "./output_json")

# Parse tables with custom model and parallel processing
results = parse_tables(
    "./input_tables",
    "./output_json",
    model="gpt-4",
    num_workers=4
)
```
