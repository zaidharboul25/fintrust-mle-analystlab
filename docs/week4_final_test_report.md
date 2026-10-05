# FinTrust Digital Bank — Week 4 Final Test Report

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 4 — Test, Refine & Present  
**Status**: **9/9 Technical Scenarios PASSED (100%)**  

---

## 1. Executive Summary

In Week 4, end-to-end testing was expanded from validation-only unit checks to encompass the complete operational workflow from raw data ingestion to structured output generation. All 9 mandatory workflow dimensions were tested using actual repository assets (`DataValidator`, `DataPreprocessor`, `ModelInterface`, `baseline_v1.joblib`, and `preprocessor.joblib`).

---

## 2. Technical Scenario Test Matrix

| Test | Expected Result | Actual Result | Pass/Fail | Action Taken |
| :--- | :--- | :--- | :---: | :--- |
| **1. Valid Input** | Full pipeline processes 5 valid rows: validation passes, 5 predictions generated. | Validation is_valid=True; Preprocessed=(5, 70); Predictions=5 rows. | **`PASS`** | Verified valid raw transaction and customer batch successfully completes validation, preprocessing, and inference. |
| **2. Missing Values** | Validation logs non-blocking warning (<5% null); preprocessing imputes 'Unknown' with 0 nulls. | is_valid=True, warnings=1, null features=0. | **`PASS`** | Confirmed missing Device_Type triggers non-blocking warning and is cleanly imputed as 'Unknown' during feature extraction. |
| **3. Unexpected Categories** | Validation halts on 'FraudulentAlert' (is_valid=False); preprocessor ignores unseen 'SmartRefrigerator' with 0 errors. | Validation blocked=True (1 error); Unseen aligned=68 features; Preds=1. | **`PASS`** | Verified schema gating flags invalid status while preprocessor handle_unknown='ignore' protects inference against feature drift. |
| **4. Invalid Data Types** | Validation fails with invalid_data_type error on non-numeric Amount_NGN. | is_valid=False; blocking errors=1 on column Amount_NGN. | **`PASS`** | Confirmed non-numeric string values in numerical financial fields are caught and blocked before model ingestion. |
| **5. Empty Input** | Validation fails with empty_dataset blocking error on 0-row DataFrame. | is_valid=False; detected empty_dataset error=1. | **`PASS`** | Verified zero-row inputs halt the pipeline immediately, protecting downstream matrix transformations. |
| **6. Model Loading** | ModelInterface loads successfully, detects missing feature column, and re-orders shuffled columns. | Loaded=True; Missing detected=True; Order restored=True. | **`PASS`** | Validated ModelInterface contract abstraction, column reordering invariance, and missing-feature detection. |
| **7. Prediction Generation** | Generates valid probability in [0, 1], binary class in {0, 1}, and decision in {'Yes', 'No'}. | Probability=0.5243; Class=1; Decision='Yes'. | **`PASS`** | Verified predict() and predict_proba() wrappers return bounded probabilities and calibrated string decisions. |
| **8. Output Format** | Output DataFrame contains exactly 5 audit columns, 0 nulls, and persists to CSV. | Columns match=True; Total nulls=True; Output written=True. | **`PASS`** | Confirmed structured output format conforms to production audit schema with complete transaction traceability. |
| **9. Reproducibility** | Repeated inference yields 100% deterministic classes, probabilities (tol=1e-7), and decisions. | Classes identical=True; Probabilities identical=True; Decisions identical=True. | **`PASS`** | Verified deterministic inference across repeated runs using persistent preprocessor and model artifacts. |

---

## 3. Detailed Scenario Findings

### 1. Valid Input
- **Workflow Tested**: Ingestion $\rightarrow$ Schema Validation $\rightarrow$ Relational Join $\rightarrow$ Preprocessor $\rightarrow$ Model Serving $\rightarrow$ Decision Output.
- **Result**: Pipeline successfully validates well-formed synthetic records and transforms raw columns into the exact 68-feature model matrix, generating calibrated predictions.
- **Outcome**: `PASS`

### 2. Missing Values (Tolerated vs Blocking)
- **Workflow Tested**: Dynamic missingness tolerance threshold (5.0%) on `Device_Type` and `Location`.
- **Result**: Missingness below 5% issues informative non-blocking warnings while preprocessing automatically imputes `Unknown` / customer `City`. Zero NaN values propagate to the estimator.
- **Outcome**: `PASS`

### 3. Unexpected Categories
- **Workflow Tested**: Data quality gating against invalid categorical values vs runtime inference resilience.
- **Result**: Validation correctly halts on non-standard values (`FraudulentAlert`), protecting feature schemas. Preprocessing utilizes `OneHotEncoder(handle_unknown='ignore')`, cleanly ignoring unobserved categories during inference without crashing.
- **Outcome**: `PASS`

### 4. Invalid Data Types
- **Workflow Tested**: Type validation on numerical financial attributes (`Amount_NGN`).
- **Result**: Non-numeric strings (`'forty-five-thousand'`) trigger immediate blocking validation errors, halting execution before downstream arithmetic or standard scaling.
- **Outcome**: `PASS`

### 5. Empty Input
- **Workflow Tested**: Zero-row DataFrame submission.
- **Result**: Detected by `DataValidator` and flagged with an explicit `empty_dataset` blocking error, preventing divide-by-zero or empty-matrix exceptions.
- **Outcome**: `PASS`

### 6. Model Loading & Interface Contract
- **Workflow Tested**: `ModelInterface` contract verification, missing feature detection, and column-order invariance.
- **Result**: Loaded artifact strictly enforces dictionary bundle specification (`model`, `feature_names`, `model_version`). Missing features raise `ModelInterfaceError`; column re-ordering is handled automatically.
- **Outcome**: `PASS`

### 7. Prediction Generation
- **Workflow Tested**: Single-record and batch inference probability calibration and decision thresholding.
- **Result**: Returns calibrated probabilities strictly within $[0.0, 1.0]$, binary integers $\{0, 1\}$, and string decisions $\{'Yes', 'No'\}$ at threshold $0.50$.
- **Outcome**: `PASS`

### 8. Output Format & Traceability
- **Workflow Tested**: Output persistence schema, audit metadata, and timestamp separation.
- **Result**: Schema matches production contract exactly (`Transaction_ID`, `predicted_Risk_Review_Flag`, `prediction_confidence`, `prediction_timestamp`, `model_version`) with zero nulls.
- **Outcome**: `PASS`

### 9. Reproducibility
- **Workflow Tested**: Deterministic inference stability across repeated executions.
- **Result**: Given identical inputs and persisted artifacts, repeated runs yield identical classes, identical string decisions, and probabilities matching to numerical tolerance ($10^{-7}$).
- **Outcome**: `PASS`

---

## 4. Conclusion & Operational Readiness

All **9/9 end-to-end technical scenarios passed** without regression or unexpected behavior. The complete ML Engineering pipeline is verified to be robust, secure, and production-ready for offline batch scoring and online REST serving.
