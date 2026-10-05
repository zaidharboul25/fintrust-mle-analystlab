# FinTrust Digital Bank — Week 4 Reproducibility Evidence

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 4 — Test, Refine & Present  
**Document**: Empirical Reproducibility Validation Report  
**Date**: October 2026  
**Status**: VERIFIED & DETERMINISTIC  

---

## 1. Reproducibility Principle & Objectives

A primary mandate of Machine Learning Engineering is guaranteeing that **the entire workflow is transparent, auditable, and mathematically deterministic**. Any reviewer or engineer checking this repository must be able to:

1. Clone the repository and install all required dependencies without version conflicts.
2. Ingest raw canonical datasets and pass validation gating.
3. Preprocess and engineer features yielding identical matrix dimensions and categorical column encodings.
4. Execute model inference generating identical risk predictions and probability values ($10^{-7}$ tolerance).
5. Verify that dynamic execution metadata (such as ISO 8601 inference timestamps) are cleanly separated from core model predictions.

This document records the exact commands executed, the real outputs produced, and the mathematical evidence verifying full determinism across pipeline components.

---

## 2. Environment & Dependency Audit

### 2.1 Python Runtime
- **Python Version**: `3.12.10` (compatible with `Python >= 3.10`)
- **Operating System**: Windows (tested with PowerShell and Bash environments)
- **Execution Engine**: Virtual environment (`.venv`)

### 2.2 Dependency Specification (`requirements.txt`)
All direct dependencies utilized by the repository are declared with minimum version constraints (`>=`):

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
pytest>=8.0.0
openpyxl>=3.1.0
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.0
httpx>=0.27.0
```

Every library listed in `requirements.txt` maps directly to imports in the codebase:
- Data manipulation & validation: `pandas`, `numpy`, `openpyxl`
- Feature preparation & modeling: `scikit-learn`, `joblib`
- Real-time serving: `fastapi`, `uvicorn`, `pydantic`
- Testing & verification: `pytest`, `httpx`

### 2.3 Pytest Configuration (`pytest.ini`)
To eliminate environment-specific path ambiguities where running bare `pytest` previously failed with `ModuleNotFoundError: No module named 'src'`, `pytest.ini` was configured:

```ini
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
python_functions = test_*
filterwarnings =
    ignore::DeprecationWarning
```

Both `pytest` and `python -m pytest` execute with identical test discovery.

---

## 3. Step-by-Step Reproduction Guide

### Step 1: Environment Setup
```powershell
# 1. Clone repository
git clone https://github.com/zaidharboul25/fintrust-mle-analystlab.git
cd fintrust-mle-analystlab

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Placement & Validation Gating
Canonical raw datasets are located in `data/raw/`:
- `data/raw/FinTrust_Customer_Data.csv` (1,500 rows, 12 columns)
- `data/raw/FinTrust_Transaction_Data.csv` (12,000 rows, 11 columns)

Execute validation gating:
```powershell
python src/validation.py
```

**Real Execution Output**:
```text
======================================================================
FINTRUST DIGITAL BANK - DATA VALIDATION RUNNER (WEEK 2)
======================================================================
VALIDATION REPORT: FinTrust Customer Data
Overall Status : [PASSED] (1,500 rows, 12 columns, 0 errors, 0 warnings)

VALIDATION REPORT: FinTrust Transaction Data
Overall Status : [PASSED] (12,000 rows, 11 columns, 0 errors, 2 warnings)
  - Device_Type : 96 nulls (0.80%) [Tolerated non-blocking warning, < 5.0%]
  - Location    : 96 nulls (0.80%) [Tolerated non-blocking warning, < 5.0%]

VALIDATION REPORT: Cross-Dataset Referential Integrity
Overall Status : [PASSED] (12,000 rows, 0 orphan transactions)

PIPELINE GATE DECISION: [PASS] - Datasets are valid for ML Feature Pipeline.
```

### Step 3: Feature Preprocessing & State Persistence
Execute feature preparation and preprocessor serialization:
```powershell
python src/preprocessing.py
```

**Real Execution Output**:
```text
2026-10-05 15:34:53,693 [INFO] DataPreprocessor fitted successfully. Total feature count: 68
2026-10-05 15:34:54,372 [INFO] Saved modeling-ready dataset (12000 rows, 71 columns) to data\processed\FinTrust_Modelling_Ready.csv
2026-10-05 15:34:54,377 [INFO] Fitted DataPreprocessor serialized successfully to models\preprocessor.joblib
2026-10-05 15:34:54,378 [INFO] Sanity Check: 10,480 / 12,000 rows (87.3%) have Is_Local_Transaction = 0 (out-of-town transactions).
```

### Step 4: Batch Prediction & Pipeline Execution
Run the complete end-to-end pipeline:
```powershell
python src/prediction.py
```

**Real Execution Output**:
```text
======================================================================
PIPELINE EXECUTION SUMMARY
======================================================================
  - Pipeline Status    : COMPLETED SUCCESSFULLY
  - Execution Time     : 0.99 seconds
  - Total Predictions  : 12,000
  - Risk Flagged 'Yes' : 4,878 (40.6%)
  - Risk Flagged 'No'  : 7,122 (59.4%)
  - Model Version      : baseline_v1
  - Model Artifact     : models/baseline_v1.joblib
  - Preprocessor       : models/preprocessor.joblib
  - Output File        : data/processed/FinTrust_Predictions.csv
======================================================================
```

### Step 5: Run Automated Regression Suite
```powershell
pytest -v
```

**Execution Result**: **65 passed in 6.70s (100% PASS)** across all 7 test modules.

---

## 4. Deterministic Behavior Verification (Empirical Evidence)

### 4.1 Fixed Random Seeds
- `RANDOM_STATE = 42` is defined centrally in `src/config.py` and passed into:
  - Stratified 80/20 train/test split: `train_test_split(..., random_state=42, stratify=y)`.
  - Logistic Regression estimator: `LogisticRegression(..., random_state=42)`.
- Verified in `tests/test_reproducibility.py::test_model_training_reproducibility`:
  - Retraining on identical data produces identical intercept: `intercept_1 == intercept_2` (exact float equality).
  - Feature coefficients match with zero variance: `np.allclose(coef_1, coef_2, atol=1e-9) == True`.

### 4.2 Preprocessing State Invariance
- `OneHotEncoder` state is persisted to `models/preprocessor.joblib`.
- In `tests/test_reproducibility.py::test_preprocessing_reproducibility`:
  - Transforming customer and transaction batches across separate preprocessor instances yields identical feature matrices:
    $$\Delta X = \max |X_{\text{run 1}} - X_{\text{run 2}}| = 0.0$$
  - Column names and column ordering are identical across 68 engineered features.

### 4.3 End-to-End Prediction Determinism
In `tests/test_reproducibility.py::test_end_to_end_prediction_determinism`:
- Two separate batches scored through `ModelInterface.predict()`:
  - Binary predictions: `np.array_equal(classes_1, classes_2) == True`
  - Continuous probabilities: `np.allclose(probs_1, probs_2, atol=1e-7) == True`
  - String decisions: `decisions_1 == decisions_2 == True`

### 4.4 Dynamic Metadata Isolation
In `tests/test_reproducibility.py::test_output_formatting_metadata_separation`:
- `prediction_timestamp` records the actual system time of execution (e.g. `2026-10-05T15:35:04Z`).
- The test verifies that stripping the dynamic `prediction_timestamp` column results in **identical DataFrames**:
  ```python
  pd.testing.assert_frame_equal(
      out_1.drop(columns=["prediction_timestamp"]),
      out_2.drop(columns=["prediction_timestamp"]),
  )
  ```

---

## 5. Output Interpretation & Governance

Batch predictions in `data/processed/FinTrust_Predictions.csv` adhere strictly to the schema:

| Field | Sample Value | Description |
| :--- | :--- | :--- |
| `Transaction_ID` | `FT-T000001` | Unique transaction key enabling referential audit back to raw transactions. |
| `predicted_Risk_Review_Flag` | `No` | Binary decision mapped from model probability using threshold $0.50$. |
| `prediction_confidence` | `0.4431` | Continuous estimated probability ($P(\text{Risk\_Review\_Flag} = 1)$). |
| `prediction_timestamp` | `2026-10-05T15:35:04...` | ISO 8601 UTC timestamp of inference execution. |
| `model_version` | `baseline_v1` | Identifier tracing exact model bundle used. |

> [!CAUTION]
> **Responsible Use Boundary**:  
> `predicted_Risk_Review_Flag` is an **educational synthetic risk-review indicator** and must **NEVER** be interpreted as a legal, financial, or operational fraud determination. This system is a proof-of-concept prototype for internship training and is not authorized or suitable for production financial infrastructure.
