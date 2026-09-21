# FinTrust Digital Bank — ML Engineering Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-26%2F26%20Passed-brightgreen.svg)]()
[![Technical%20Tests](https://img.shields.io/badge/Technical%20Scenarios-5%2F5%20Passed-brightgreen.svg)](docs/technical_test_report.md)
[![Status](https://img.shields.io/badge/Status-Week%202%20Complete-success.svg)]()

---

## 1. Project Overview

This repository houses the Machine Learning Engineering (MLE) technical pipeline for **FinTrust Digital Bank**, developed during **Week 2 (*Analyse & Prepare*)** of the **AnalystLab Africa Experience Lab Internship Programme**. Within the larger multidisciplinary project (collaborating alongside Data Science, Data Analytics, Generative AI, and Project Management tracks), the ML Engineering track is responsible for designing, building, and automating a robust, testable technical pipeline that operationalizes risk-review predictions. 

The primary objective of this track is to prove that data can flow reliably and reproducibly from raw storage, through rigorous validation gating and feature engineering, to versioned model inference and structured output persistence. Developing, optimizing, and fine-tuning high-performance predictive models remains the designated responsibility of the Data Science track; our focus is end-to-end technical integrity, schema contracts, and pipeline resilience.

---

## 2. Repository Structure

```text
fintrust-mle/
├── data/
│   ├── raw/                 # Canonical raw CSVs (Customer & Transaction datasets)
│   └── processed/           # Model-ready dataset and versioned prediction outputs
├── docs/                    # Architecture documentation and technical test reports
├── models/                  # Versioned, serialized baseline model artifacts (.joblib)
├── notebooks/               # Exploratory data analysis and experimental scratchpads
├── src/                     # Production source code (validation, preprocessing, prediction)
│   ├── __init__.py          # Package initialization
│   ├── validation.py        # Data validation component, schemas, and referential integrity
│   ├── preprocessing.py     # Relational join, missingness imputation, and feature preparation
│   └── prediction.py        # Baseline model training, inference, and pipeline orchestrator
└── tests/                   # Automated unit test suite and technical report generator
    ├── test_validation.py   # Schema, null threshold, and relational integrity tests
    ├── test_prediction.py   # Model training, batch/single inference, and pipeline tests
    └── technical_test_report.py # Script generating the 5-scenario evaluation table
```

---

## 3. Pipeline Architecture

The technical workflow strictly follows the 7-stage sequence defined in the project specification:

$$\textbf{Data} \longrightarrow \textbf{Validation} \longrightarrow \textbf{Preprocessing} \longrightarrow \textbf{Feature Preparation} \longrightarrow \textbf{Model} \longrightarrow \textbf{Prediction} \longrightarrow \textbf{Output}$$

1. **Data Ingestion**: Reads canonical raw datasets (`FinTrust_Customer_Data.csv` and `FinTrust_Transaction_Data.csv`).
2. **Data Validation**: Enforces schema contracts, regex patterns, categorical constraints, dynamic null thresholds, and cross-dataset referential integrity. Halts execution if blocking errors are detected.
3. **Preprocessing**: Merges relational datasets on `Customer_ID` and imputes known missing values (`Device_Type` and `Location`).
4. **Feature Preparation**: Extracts temporal features (`Transaction_Hour`, `Transaction_DayOfWeek`, `Is_Weekend`), computes derived signals (`Is_Local_Transaction`), and applies One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`), persisting `FinTrust_Modelling_Ready.csv`.
5. **Model Training / Loading**: Fits a reproducible baseline classifier (`Pipeline` with `StandardScaler` + `LogisticRegression(class_weight='balanced')`) on an 80/20 stratified split, saving versioned artifacts (`models/baseline_v1.joblib`).
6. **Prediction**: Generates binary decisions (`Yes`/`No`) alongside continuous confidence probabilities for batch data or single records.
7. **Output Formatting**: Structures predictions with traceability metadata (`Transaction_ID`, `predicted_Risk_Review_Flag`, `prediction_confidence`, `prediction_timestamp`, `model_version`) and writes to `data/processed/FinTrust_Predictions.csv`.

*(Note: Detailed architectural diagrams and system design specifications are maintained in [`docs/`](docs/) from Week 1 deliverables).*

---

## 4. Installation Instructions

### Prerequisites
- Python 3.10+ installed
- Git installed

### Setup Environment
```powershell
# 1. Clone the repository
git clone https://github.com/zaidharboul25/fintrust-mle-analystlab.git
cd fintrust-mle-analystlab

# 2. Create and activate a Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 5. Usage Instructions

All pipeline stages and test suites can be executed independently from the command line:

```powershell
# 1. Run Data Validation independently (inspects raw CSVs and referential integrity)
python src/validation.py

# 2. Run Preprocessing & Feature Preparation independently (generates FinTrust_Modelling_Ready.csv)
python src/preprocessing.py

# 3. Run Full End-to-End Pipeline (Data -> Validation -> Preprocessing -> Model -> Prediction -> Output)
python src/prediction.py

# 4. Run Automated Pytest Suite (26 unit tests covering all modules)
python -m pytest tests/ -v

# 5. Run Technical Scenario Test Report (generates docs/technical_test_report.md)
python tests/technical_test_report.py
```

---

## 6. Key Technical Decisions

The following technical decisions were established during Week 2 to balance engineering rigor, business reality, and clear role boundaries:

| Decision | Reasoning | Trade-off |
| :--- | :--- | :--- |
| **1. Dynamic Missing Value Thresholds** | `Device_Type` and `Location` contain intentional missing values (~0.8%). A fixed boolean was replaced with a dynamic threshold ($5.0\%$): missingness $\le 5\%$ issues a non-blocking warning, while missingness $> 5\%$ triggers a blocking pipeline error. | Allows known training noise to pass smoothly without silently tolerating severe future data drift or distribution corruption. |
| **2. Strict Referential Integrity Gating** | Enforces that every `Customer_ID` referenced in `Transaction_Data` exists in `Customer_Data`. Orphan transactions trigger a blocking error (`severity='error'`). | Prevents silent failures, Cartesian anomalies, and feature corruption downstream during relational joins, halting bad data before model training. |
| **3. One-Hot Encoding for City and Location** | Both `City` and `Location` share the exact same 8 Nigerian cities. Because the domain cardinality is low (8 categories each, 16 columns total), One-Hot Encoding is safe and avoids dimensionality explosion. | Consumes 16 columns in the feature matrix, but maintains full linear interpretability and avoids premature loss of geographic variance. |
| **4. Feature Engineering (`Is_Local_Transaction`)** | `Is_Local_Transaction = (Location == City)` was engineered to detect out-of-town transactions. An empirical verification revealed a **12.7% match rate**, exactly matching the random baseline ($1/8 = 12.5\%$), proving that `City` and `Location` were generated independently in this synthetic dataset. | Kept in the pipeline because the engineering logic is sound and will be valuable on real-world data; documented honestly as a dataset-specific limitation for the Data Science track to validate empirically. |
| **5. Proof-of-Concept Baseline Classifier** | Logistic Regression with `class_weight='balanced'` was chosen to demonstrate end-to-end operational flow and artifact serialization. Precision is deliberately low (~0.29) and Recall is high (~0.61) due to penalizing minority class misclassifications. | Prevents a naive model from predicting "No" everywhere (which would yield a hollow 80.4% accuracy). Further metric optimization and model selection belong to Data Science. |

---

## 7. Known Limitations

In the spirit of honest technical evaluation and engineering transparency, the following limitations are explicitly noted:

1. **CLI Printing vs. Distributed Logging**: Console summaries currently use formatted `print()` and standard `logging`. In production orchestration (Airflow/Prefect), this will be replaced with structured JSON telemetry emitting to centralized observability sinks (e.g., Datadog or Prometheus).
2. **CSV Dialect & Encoding Sniffing**: Ingestion assumes standard UTF-8 comma-separated files. Automated byte-order mark (BOM) handling and delimiter sniffing (`csv.Sniffer`) are planned for future hardening.
3. **Synthetic Target Semantics**: `Risk_Review_Flag` is a synthetic educational target and does not represent an authentic banking fraud label or real risk determination.
4. **Baseline Model Scope**: The classifier in `src/prediction.py` is strictly a technical sanity check and pipeline placeholder; it must not be interpreted as an optimized production model.

---

## 8. Testing & Quality Assurance

The codebase is protected by automated unit tests and an evaluation report:

- **Pytest Suite**: **26/26 tests passing** (`tests/test_validation.py` and `tests/test_prediction.py`), verifying schema contracts, primary key uniqueness, dynamic thresholds, referential integrity, serialization/deserialization, batch prediction, and single-record inference ($N=1$).
- **Technical Scenario Tests**: **5/5 scenarios passing** according to the Week 2 assignment specification (Valid input, Missing values, Unexpected category, Incorrect data type, Empty input). Full results and diagnostic outputs are documented in [**`docs/technical_test_report.md`**](docs/technical_test_report.md).

---

## 9. Data Disclaimer

> [!IMPORTANT]
> **Data Governance Disclaimer**: All customers, transactions, accounts, labels, and financial figures in this repository are synthetic and created exclusively for educational purposes within the AnalystLab Africa Experience Lab Internship Programme. The `Risk_Review_Flag` field is an educational artifact and must **never** be presented, deployed, or interpreted as a real-world fraud determination or authentic banking risk decision.

---

## 10. Author and Track Information

- **Intern Name**: Zaid Harboul
- **Assigned Track**: Machine Learning Engineering (Track 3)
- **Programme**: AnalystLab Africa Experience Lab Internship Programme
- **Project**: FinTrust Digital Bank Support Solution
- **Milestone**: Week 2 — *Analyse & Prepare*
