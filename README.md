# FinTrust Digital Bank — ML Engineering Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-55%2F55%20Passed-brightgreen.svg)]()
[![Technical%20Tests](https://img.shields.io/badge/Technical%20Scenarios-5%2F5%20Passed-brightgreen.svg)](docs/technical_test_report.md)
[![Status](https://img.shields.io/badge/Status-Week%203%20Complete-success.svg)]()

---

## 1. Project Overview & Track Boundary

This repository houses the Machine Learning Engineering (MLE) system for **FinTrust Digital Bank**, developed during **Week 3 (*Develop & Integrate*)** of the **AnalystLab Africa Experience Lab Internship Programme**.

### MLE Track vs. Data Science Track Boundary
- **ML Engineering Track (Our Mandate)**: Responsible for designing, building, hardening, testing, and operationalizing an end-to-end reproducible ML system around the model. This includes input validation gating, relational preprocessing, feature schema alignment, model interface abstraction, deterministic batch prediction, real-time FastAPI service integration, and automated testing.
- **Data Science Track**: Designated owner of exploratory data analysis, feature selection, algorithmic experimentation, hyperparameter tuning, and final predictive model optimization.

> [!IMPORTANT]
> **Data Governance & Educational Target Disclaimer**:  
> All customers, transactions, financial figures, and labels in this repository are synthetic and created exclusively for educational purposes within AnalystLab Africa. The target variable `Risk_Review_Flag` is a **synthetic educational label** and must **NEVER** be interpreted, deployed, or represented as authentic banking fraud detection or real-world risk determination. The baseline model in this repository is strictly an operational proof-of-concept pipeline placeholder.

---

## 2. Repository Architecture & Layout

```text
fintrust-mle/
├── data/
│   ├── raw/                      # Canonical raw CSVs (Customer & Transaction datasets)
│   └── processed/                # Model-ready dataset and versioned prediction outputs
├── docs/                         # Technical documentation and integration contracts
│   ├── model_integration.md      # Data Science hand-off and model interface protocol
│   ├── technical_test_report.md  # 5 required technical test scenarios
│   ├── week3_refinement_evidence.md # Empirical evidence of preprocessor state persistence
│   ├── week3_repository_audit.md # Systematic audit of Week 2 baseline
│   └── week3_summary.md          # Week 3 milestone summary and technical decisions
├── models/                       # Serialized model and preprocessor artifacts (.joblib)
│   ├── baseline_v1.joblib        # Baseline Logistic Regression pipeline artifact
│   └── preprocessor.joblib       # Persisted OneHotEncoder and preprocessor state
├── src/                          # Modular production source code
│   ├── __init__.py               # Package marker
│   ├── api.py                    # Real-time prediction service (FastAPI)
│   ├── config.py                 # Centralized configuration, paths, and thresholds
│   ├── model_loader.py           # Standardized ModelInterface and loader contracts
│   ├── prediction.py             # Baseline training, batch inference, and orchestrator
│   ├── preprocessing.py          # Relational join, imputation, and feature extraction
│   └── validation.py             # Schema contracts, null thresholds, and integrity gating
├── tests/                        # Automated unit and integration test suite (55 tests)
│   ├── technical_test_report.py  # 5-scenario evaluation runner
│   ├── test_api.py               # FastAPI endpoint and validation tests
│   ├── test_model_loading.py     # Artifact interface and contract tests
│   ├── test_prediction.py        # Model training and batch prediction tests
│   ├── test_preprocessing.py     # Imputation, temporal extraction, and encoding tests
│   ├── test_reproducibility.py   # Deterministic inference and weight tests
│   └── test_validation.py        # Schema drift, regex, and referential integrity tests
├── README.md                     # Comprehensive repository documentation
└── requirements.txt              # Pinned environment dependencies
```

---

## 3. End-to-End System Architecture

The pipeline implements an end-to-end data flow supporting both offline batch execution and online real-time REST serving:

$$\textbf{Raw Data / Client Request} \longrightarrow \textbf{Validation Gating} \longrightarrow \textbf{Preprocessing} \longrightarrow \textbf{Model Interface} \longrightarrow \textbf{Prediction} \longrightarrow \textbf{Output Persistence / JSON}$$

```text
┌────────────────────────────────────────────────────────┐
│  Client Ingestion (Batch CSV or Real-Time HTTP POST)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  1. Validation Gating (src/validation.py, Pydantic)    │
│  - Primary key uniqueness & non-nullity                │
│  - Foreign key referential integrity (Customer_ID)     │
│  - Dynamic null tolerance (warning <= 5%, error > 5%)  │
│  - Data types, regex checks, and domain categories     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  2. Preprocessing (src/preprocessing.py)               │
│  - Relational merge on Customer_ID                     │
│  - Imputation: Device_Type ('Unknown'), Location (City)│
│  - Temporal features: Hour, DayOfWeek, Is_Weekend      │
│  - Location signal: Is_Local_Transaction (int 0/1)     │
│  - Serialized OneHotEncoder (models/preprocessor.joblib)│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  3. Model Interface & Serving (src/model_loader.py)    │
│  - Interface contract verification (predict/proba)     │
│  - Strict column reordering invariance                 │
│  - Missing-feature drift detection                     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  4. Structured Output                                  │
│  - Batch CSV: data/processed/FinTrust_Predictions.csv  │
│  - Real-Time JSON: { transaction_id, prediction, ... } │
└────────────────────────────────────────────────────────┘
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+
- Git

### Setup Instructions
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

# 4. Install all dependencies (core pipeline + FastAPI stack)
pip install -r requirements.txt
```

---

## 5. Dataset Placement

Ensure canonical raw datasets are placed in `data/raw/`:
- `data/raw/FinTrust_Customer_Data.csv` (1,500 rows, 12 columns)
- `data/raw/FinTrust_Transaction_Data.csv` (12,000 rows, 11 columns)

---

## 6. Execution Commands

All stages can be invoked independently or as an integrated system:

```powershell
# 1. Run Data Validation independently (inspects raw CSVs and referential integrity)
python src/validation.py

# 2. Run Preprocessing & Feature Preparation independently
python src/preprocessing.py

# 3. Run Full End-to-End Batch Pipeline (Train, Predict, Format)
python src/prediction.py

# 4. Run Automated Pytest Suite (55 unit & integration tests)
python -m pytest tests/ -v

# 5. Run Technical Scenario Evaluation Report (5 scenarios)
python tests/technical_test_report.py
```

---

## 7. Real-Time Prediction API Service (FastAPI)

Week 3 introduces a lightweight, production-ready REST service running on FastAPI and Uvicorn.

### Start the Service
```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://127.0.0.1:8000/docs`

### Endpoint 1: Health Diagnostic Check (`GET /health`)
Verifies service uptime, loaded model version, and preprocessor status.

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
  "timestamp": "2026-10-01T15:30:00Z"
}
```

### Endpoint 2: Single-Record Prediction (`POST /predict`)
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
  "prediction_timestamp": "2026-10-01T15:30:01.123456Z"
}
```

---

## 8. Prediction Output Schema & Traceability

Batch predictions are saved to `data/processed/FinTrust_Predictions.csv` with complete audit metadata:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Transaction_ID` | String | Unique transaction identifier (`FT-T######`) for audit traceability. |
| `predicted_Risk_Review_Flag` | String | Model binary risk decision (`Yes` or `No`). |
| `prediction_confidence` | Float | Probability of positive class ($0.0 \le p \le 1.0$). |
| `prediction_timestamp` | String | ISO 8601 UTC timestamp recording exact inference execution. |
| `model_version` | String | Identifier of model artifact used (`baseline_v1`). |

---

## 9. Model Integration & Replacing the Development Baseline

The model layer is decoupled from Scikit-Learn Logistic Regression via `src.model_loader.ModelInterface`. When the Data Science track delivers a candidate model:

1. Package the model into a dictionary bundle with keys: `{"model": estimator, "feature_names": list_of_names, "model_version": "ds_v1"}`.
2. Save to `models/<model_version>.joblib`.
3. Set `DEFAULT_MODEL_VERSION = "<model_version>"` in `src/config.py`.
4. Validate compliance with the test suite:
   ```powershell
   python -m pytest tests/test_model_loading.py -v
   ```
*(Refer to [`docs/model_integration.md`](docs/model_integration.md) for full contract documentation).*

---

## 10. Reproducibility Strategy

Execution determinism is guaranteed across pipeline runs through:
- Fixed random seeds (`random_state=42`) across train/test splitting and estimators.
- Stratified 80/20 splitting preserving minority class balance (~20%).
- Serialized preprocessor state ensuring identical categorical dummy columns and indices.
- Automated reproducibility assertions in `tests/test_reproducibility.py` validating that repeated runs yield identical model coefficients, predictions, and confidence scores (0.0 numerical tolerance).

---

## 11. Known Technical Limitations

1. **Synthetic Data Geographic Signal**: As proven in Week 2, `Is_Local_Transaction` exhibits a 12.7% match rate, identical to random chance (12.5% across 8 uniform cities). The feature logic is retained for real-world suitability, but carries negligible predictive signal on this synthetic dataset.
2. **Synchronous File Logging**: Logging currently outputs to console and standard streams. Distributed tracing and centralized log shipping (e.g. Datadog / OpenTelemetry) are designated for future infrastructure milestones.
3. **In-Memory SQLite / Cache**: Customer lookup fallback currently queries an in-memory DataFrame indexed on `Customer_ID`. In high-throughput production, this would be replaced with Redis or a relational database replica.

---

## 12. Author Information

- **Intern Name**: Zaid Harboul
- **Assigned Track**: Machine Learning Engineering (Track 3)
- **Programme**: AnalystLab Africa Experience Lab Internship Programme
- **Project**: FinTrust Digital Bank Support Solution
- **Milestone**: Week 3 — *Develop & Integrate*
