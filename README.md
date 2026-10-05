# FinTrust Digital Bank — ML Engineering Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-65%2F65%20Passed-brightgreen.svg)]()
[![Technical%20Tests](https://img.shields.io/badge/Technical%20Scenarios-9%2F9%20Passed-brightgreen.svg)](docs/week4_final_test_report.md)
[![Status](https://img.shields.io/badge/Status-Week%204%20Complete%20(Ready%20for%20Submission)-success.svg)]()

---

## 1. Project Overview & FinTrust Context

This repository houses the Machine Learning Engineering (MLE) system for **FinTrust Digital Bank**, finalized during **Week 4 (*Test, Refine & Present*)** of the **AnalystLab Africa Experience Lab Internship Programme**.

FinTrust is a digital retail bank operating across 8 major Nigerian metropolitan hubs. The objective of this project is to build, validate, harden, test, and operationalize an end-to-end, reproducible Machine Learning serving system that scores incoming transaction events for risk review.

### MLE Track vs. Data Science Track Boundary
- **Machine Learning Engineering Track (Our Mandate)**: Responsible for designing, building, hardening, testing, and operationalizing an end-to-end reproducible ML system around the predictive model. This includes input validation gating, relational preprocessing, feature schema alignment, model interface abstraction, deterministic batch prediction, real-time FastAPI service integration, and automated regression testing.
- **Data Science Track**: Designated owner of exploratory data analysis, feature selection, algorithmic experimentation, hyperparameter tuning, and final predictive model optimization.

> [!IMPORTANT]
> **Responsible Use & Educational Target Disclaimer**:  
> - **FinTrust is a fictional organisation**.
> - **All customer and transaction datasets are synthetic** and created exclusively for educational purposes within AnalystLab Africa.
> - The target variable `Risk_Review_Flag` is a **synthetic educational label** and must **NEVER** be interpreted, deployed, or represented as authentic banking fraud detection, legal determination, or real-world financial-crime decision.
> - This software is an educational prototype and **NOT a production banking system**.
> - The baseline model in this repository is strictly an operational proof-of-concept pipeline placeholder to validate the engineering architecture.

---

## 2. End-to-End System Architecture

The pipeline implements an end-to-end data flow supporting both offline batch execution and online real-time REST serving:

$$\textbf{Raw Data / Client Request} \longrightarrow \textbf{Validation Gating} \longrightarrow \textbf{Preprocessing} \longrightarrow \textbf{Feature Preparation} \longrightarrow \textbf{Model Interface} \longrightarrow \textbf{Prediction} \longrightarrow \textbf{Structured Output}$$

```text
┌────────────────────────────────────────────────────────┐
│  Client Ingestion (Batch CSV or Real-Time HTTP POST)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  1. Data Validation Gating (src/validation.py)         │
│  - Primary key uniqueness & non-nullity                │
│  - Foreign key referential integrity (Customer_ID)     │
│  - Dynamic null tolerance (warning <= 5%, error > 5%)  │
│  - Data types, regex checks, and domain categories     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  2. Preprocessing & Features (src/preprocessing.py)    │
│  - Relational merge on Customer_ID                     │
│  - Imputation: Device_Type ('Unknown'), Location (City)│
│  - Temporal features: Hour, DayOfWeek, Is_Weekend (0/1)│
│  - Location signal: Is_Local_Transaction (int 0/1)     │
│  - Serialized OneHotEncoder (models/preprocessor.joblib│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  3. Model Interface & Serving (src/model_loader.py)    │
│  - Interface contract verification (predict/proba)     │
│  - Strict column reordering invariance                 │
│  - Missing-feature drift detection                     │
│  - models/baseline_v1.joblib (Case B baseline model)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  4. Structured Output                                  │
│  - Batch CSV: data/processed/FinTrust_Predictions.csv  │
│  - Real-Time JSON: { transaction_id, prediction, ... } │
│  - Audit Traceability: model_version + timestamp       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```text
fintrust-mle/
├── data/
│   ├── raw/
│   │   ├── FinTrust_Customer_Data.csv    # 1,500 rows, 12 columns
│   │   └── FinTrust_Transaction_Data.csv # 12,000 rows, 11 columns
│   └── processed/
│       ├── FinTrust_Modelling_Ready.csv  # 12,000 rows, 68 features + target
│       └── FinTrust_Predictions.csv      # 12,000 rows, 5 audit columns
├── docs/
│   ├── final_assumptions_limitations.md  # Comprehensive assumptions, limitations, and governance
│   ├── final_video_outline.md            # 5-10 minute presentation script and slide outline
│   ├── model_integration.md              # ModelInterface contract and Case B formal status
│   ├── technical_test_report.md          # Week 2 baseline scenario test report
│   ├── week3_refinement_evidence.md     # Week 3 preprocessor persistence evidence
│   ├── week3_repository_audit.md         # Week 3 audit report
│   ├── week3_summary.md                  # Week 3 technical milestone summary
│   ├── week4_final_audit.md              # Week 4 comprehensive baseline audit report
│   ├── week4_final_summary.md            # Week 4 final technical summary and deliverables
│   ├── week4_final_test_report.md        # Week 4 end-to-end 9-scenario evaluation report
│   ├── week4_refinement_evidence.md     # Week 4 empirical refinement evidence (pytest + API)
│   └── week4_reproducibility_evidence.md # Empirical reproducibility and determinism report
├── models/
│   ├── baseline_v1.joblib                # Baseline Logistic Regression pipeline artifact
│   └── preprocessor.joblib               # Persisted OneHotEncoder and preprocessor state
├── notebooks/                            # Scratch research workspace (.gitkeep)
├── src/
│   ├── __init__.py                       # Source package marker
│   ├── api.py                            # Real-time prediction service (FastAPI)
│   ├── config.py                         # Centralized configuration, paths, and thresholds
│   ├── model_loader.py                   # ModelInterface abstraction and loader contracts
│   ├── prediction.py                     # Training, batch inference, and pipeline runner
│   ├── preprocessing.py                  # Relational join, imputation, and feature extraction
│   └── validation.py                     # Schema contracts, null thresholds, and integrity gating
├── tests/
│   ├── __init__.py                       # Test package marker
│   ├── technical_test_report.py          # 5-scenario evaluation runner (Week 2 baseline)
│   ├── test_api.py                       # 11 tests for FastAPI endpoints (/, /health, /predict)
│   ├── test_model_loading.py             # 9 tests for ModelInterface and contract checks
│   ├── test_prediction.py                # 7 tests for training, batch prediction, and gating
│   ├── test_preprocessing.py             # 6 tests for imputation, temporal extraction, encoding
│   ├── test_reproducibility.py           # 4 tests for deterministic inference and weights
│   ├── test_validation.py                # 19 tests for schema drift, null thresholds, integrity
│   ├── test_week4_e2e.py                 # 9 automated tests for Week 4 E2E scenarios
│   └── week4_e2e_scenario_runner.py      # Week 4 9-scenario runner and report generator
├── pytest.ini                            # Pytest pythonpath configuration
├── README.md                             # Comprehensive repository documentation
└── requirements.txt                      # Project dependencies with minimum version constraints
```

---

## 4. Installation & Dependencies

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- Git

### Installation Steps
```powershell
# 1. Clone repository
git clone https://github.com/zaidharboul25/fintrust-mle-analystlab.git
cd fintrust-mle-analystlab

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependency Audit (`requirements.txt`)
All dependencies specify minimum version constraints (`>=`) matching codebase imports:
- `pandas>=2.0.0` & `numpy>=1.24.0`: Ingestion, vector operations, and data structures.
- `scikit-learn>=1.3.0` & `joblib>=1.3.0`: Feature encoding, estimators, and serialization.
- `openpyxl>=3.1.0`: Tabular dataset support.
- `pytest>=8.0.0`: Multi-tier test suite.
- `fastapi>=0.110.0` & `uvicorn>=0.28.0`: Real-time HTTP prediction service.
- `pydantic>=2.6.0`: Request and response contract validation.
- `httpx>=0.27.0`: TestClient HTTP simulation.

---

## 5. Dataset Placement & Data Validation

Ensure raw datasets are placed in `data/raw/`:
- `data/raw/FinTrust_Customer_Data.csv` (1,500 rows, 12 columns)
- `data/raw/FinTrust_Transaction_Data.csv` (12,000 rows, 11 columns)

### Run Data Validation Gating
```powershell
python src/validation.py
```
Validates customer schema, transaction schema, primary key uniqueness (`FT-C#####`, `FT-T######`), dynamic null thresholds (tolerates $\le 5\%$ for `Device_Type` and `Location`), and referential integrity across tables.

---

## 6. Preprocessing & Feature Preparation

### Run Preprocessing Independently
```powershell
python src/preprocessing.py
```
- Performs relational merge on `Customer_ID`.
- Imputes missing `Device_Type` as `'Unknown'` and missing `Location` as customer `City`.
- Extracts temporal features: `Transaction_Hour`, `Transaction_DayOfWeek`, and `Is_Weekend` (binary integer `0`/`1`).
- Derives location match: `Is_Local_Transaction` (binary integer `0`/`1`).
- Serializes fitted `OneHotEncoder` state to `models/preprocessor.joblib`.
- Exports 71-column dataset (68 model features + IDs + target) to `data/processed/FinTrust_Modelling_Ready.csv`.

---

## 7. Model Integration & Replacing the Baseline (Case B Status)

The model serving layer is decoupled from underlying estimators via `src.model_loader.ModelInterface`.

### Week 4 Case B Formal Determination
- No external Data Science model was delivered during Week 4.
- As mandated by project rules, we **do not fabricate** a Data Science model or invent cross-track collaboration.
- The documented development baseline model (`models/baseline_v1.joblib`) is retained through the `ModelInterface`.

### Step-by-Step Guide for Future Model Replacement:
1. Ingest `data/processed/FinTrust_Modelling_Ready.csv`.
2. Extract `feature_names = list(X.columns)`.
3. Bundle the model artifact into a dictionary:
   ```python
   bundle = {
       "model": estimator_object,
       "feature_names": feature_names,
       "model_version": "ds_candidate_v1",
   }
   ```
4. Save via joblib to `models/<model_version>.joblib`.
5. Update `DEFAULT_MODEL_VERSION` in `src/config.py`.
6. Run the model loading test:
   ```powershell
   pytest tests/test_model_loading.py -v
   ```
*(Refer to [`docs/model_integration.md`](docs/model_integration.md) for full contract details).*

---

## 8. How to Run the Complete Batch Workflow

Execute the entire end-to-end pipeline:
```powershell
python src/prediction.py
```

**What it executes**:
1. Ingests raw customer and transaction CSVs.
2. Evaluates data validation gating (halts if blocking errors occur).
3. Preprocesses data and serializes preprocessor state to `models/preprocessor.joblib`.
4. Trains/loads `models/baseline_v1.joblib` via `ModelInterface`.
5. Generates batch predictions across all 12,000 records (completed in **0.99s**).
6. Persists structured predictions to `data/processed/FinTrust_Predictions.csv`.

---

## 9. How to Run Automated Tests & Scenarios

Thanks to `pytest.ini` (`pythonpath = .`), tests run cleanly from standard commands:

```powershell
# 1. Run complete automated regression test suite (65 tests)
pytest -v

# 2. Run Week 4 end-to-end 9-scenario evaluation runner
python tests/week4_e2e_scenario_runner.py

# 3. Run Week 2/3 technical test scenarios (5 scenarios)
python tests/technical_test_report.py

# 4. Run reproducibility verification tests specifically
pytest tests/test_reproducibility.py -v
```

**Current Automated Test Results**:
- **65 / 65 Pytest Tests PASSED (100%)** in 6.70s.
- **9 / 9 Week 4 End-to-End Scenarios PASSED (100%)**.

---

## 10. Real-Time Prediction API Service (FastAPI)

### Start the Service
```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://127.0.0.1:8000/docs`

### Endpoint 1: Root Discovery (`GET /`)
Returns service metadata, version, available endpoints, and educational disclaimer:
```powershell
curl -X GET "http://127.0.0.1:8000/"
```
**Response (`200 OK`)**:
```json
{
  "service": "FinTrust Financial Intelligence & Digital Banking Support Solution",
  "track": "Machine Learning Engineering",
  "milestone": "Week 4 — Test, Refine & Present",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "documentation": "/docs",
    "openapi_schema": "/openapi.json"
  },
  "responsible_use_notice": "FinTrust is a fictional organisation and Risk_Review_Flag is a synthetic educational target, NOT an authentic fraud determination. Predictions must not be used for real financial-crime decisions."
}
```

### Endpoint 2: Health Diagnostic Check (`GET /health`)
Verifies service uptime, loaded model version, and pipeline component status:
```powershell
curl -X GET "http://127.0.0.1:8000/health"
```
**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "baseline_v1",
  "preprocessor_loaded": true,
  "features_count": 68,
  "service": "FinTrust Financial Intelligence Prediction Service",
  "timestamp": "2026-10-05T15:35:00Z"
}
```

### Endpoint 3: Single-Record Prediction (`POST /predict`)
Scores a single incoming transaction. Accepts full customer details or automatically looks up existing customers by `Customer_ID`.

```powershell
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "Transaction_ID": "FT-T000001",
      "Customer_ID": "FT-C00001",
      "Transaction_DateTime": "2024-01-15 10:30:00",
      "Transaction_Type": "Transfer",
      "Amount_NGN": 15000.0,
      "Channel": "Mobile App",
      "Device_Type": "Android",
      "Location": "Lagos",
      "International_Transaction": "No",
      "Transaction_Status": "Successful"
    },
    "customer": {
      "Customer_ID": "FT-C00001",
      "Customer_Name": "Ibrahim Adeyemi",
      "Age": 34,
      "Gender": "Male",
      "City": "Lagos",
      "Customer_Segment": "Premium",
      "Account_Type": "Savings",
      "Tenure_Months": 12,
      "Digital_Engagement_Score": 4.2,
      "Monthly_Income_Band": "500k-999k",
      "Preferred_Channel": "Mobile App",
      "Account_Status": "Active"
    }
  }'
```

**Response (`200 OK`)**:
```json
{
  "transaction_id": "FT-T000001",
  "prediction": "No",
  "confidence": 0.4431,
  "model_version": "baseline_v1",
  "prediction_timestamp": "2026-10-05T15:35:01.123456Z"
}
```

---

## 11. Prediction Output Schema & Traceability

Batch predictions are persisted to `data/processed/FinTrust_Predictions.csv` with complete audit metadata:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Transaction_ID` | String | Unique transaction identifier (`FT-T######`) linking back to raw data. |
| `predicted_Risk_Review_Flag` | String | Model binary risk decision (`Yes` or `No`). |
| `prediction_confidence` | Float | Probability of positive class ($0.0 \le p \le 1.0$). |
| `prediction_timestamp` | String | ISO 8601 UTC timestamp recording exact inference execution. |
| `model_version` | String | Identifier of model artifact used (`baseline_v1`). |

---

## 12. Reproducibility Strategy

Deterministic behavior is verified across all pipeline runs:
- **Fixed Random Seeds**: `random_state=42` across train/test splitting and classifier estimators.
- **Stratified Splitting**: 80/20 train/test split preserving target class balance (~20% positive).
- **Persistent Preprocessor**: Serialized `OneHotEncoder` vocabulary ensures identical categorical dummy columns.
- **Automated Verification**: `tests/test_reproducibility.py` validates identical model coefficients, identical predictions, and identical probabilities ($10^{-7}$ numerical tolerance).
*(Refer to [`docs/week4_reproducibility_evidence.md`](docs/week4_reproducibility_evidence.md) for full evidence).*

---

## 13. Known Technical Limitations

1. **Synthetic Data Geographic Signal**: As proven during feature evaluation, `Is_Local_Transaction` exhibits a 12.7% match rate, identical to random chance (12.5% across 8 uniform cities). The feature logic is preserved for real-world suitability, but carries negligible predictive signal on this synthetic dataset.
2. **In-Memory Customer Store**: Customer lookup fallback queries an in-memory DataFrame indexed on `Customer_ID`. In high-throughput production, this would be replaced with Redis or a relational database replica.
3. **Local Deployment**: The service runs on local Uvicorn instances; containerization or cloud serverless deployment represents future work.
*(Refer to [`docs/final_assumptions_limitations.md`](docs/final_assumptions_limitations.md) for comprehensive governance documentation).*

---

## 14. Week 4 Final Milestone Status

**Status**: **READY FOR SUBMISSION**

| Deliverable | Verification | Status |
| :--- | :--- | :---: |
| Baseline Repository Audit | `docs/week4_final_audit.md` | **COMPLETE** |
| End-to-End Workflow & Pipeline | `src/validation.py`, `preprocessing.py`, `prediction.py` | **COMPLETE** |
| Technical Scenario Evaluation (9 Scenarios) | `docs/week4_final_test_report.md` (9/9 passed) | **COMPLETE** |
| Model Integration (Case B Retained) | `docs/model_integration.md` | **COMPLETE** |
| FastAPI REST Service (/, /health, /predict) | `src/api.py`, `tests/test_api.py` (11 passed) | **COMPLETE** |
| Empirical Refinements (pytest.ini, root API) | `docs/week4_refinement_evidence.md` | **COMPLETE** |
| Reproducibility Validation Report | `docs/week4_reproducibility_evidence.md` | **COMPLETE** |
| Assumptions, Limitations & Governance | `docs/final_assumptions_limitations.md` | **COMPLETE** |
| Final Technical Summary | `docs/week4_final_summary.md` | **COMPLETE** |
| Final Video Presentation Outline (5-10 min) | `docs/final_video_outline.md` | **COMPLETE** |
| Automated Test Suite (65 Pytest Tests) | `pytest -v` (65/65 passed in 6.70s) | **COMPLETE** |

---

## 15. Author Information

- **Intern Name**: Zaid Harboul
- **Assigned Track**: Machine Learning Engineering (Track 3)
- **Programme**: AnalystLab Africa Experience Lab Internship Programme
- **Project**: FinTrust Financial Intelligence & Digital Banking Support Solution
- **Milestone**: Week 4 — *Test, Refine & Present*
