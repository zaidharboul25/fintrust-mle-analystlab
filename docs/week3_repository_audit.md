# FinTrust Digital Bank — Week 3 Repository Audit

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 3 — Develop & Integrate  
**Auditor**: Senior Machine Learning Engineer  
**Date**: October 2026  

---

## 1. Executive Summary

A comprehensive technical audit of the FinTrust ML Engineering codebase was conducted to establish baseline readiness for **Week 3 (Develop & Integrate)**. The Week 2 foundation successfully delivers data validation gating, relational join preprocessing, baseline model training, and batch prediction persistence, backed by 26 passing automated unit tests and 5 verified technical scenario tests.

However, transitioning from an exploratory batch pipeline to a robust, modular, and service-integrated ML system reveals several real architectural gaps:
1. **Decoupled Preprocessing State**: Preprocessor transformations (such as categorical encodings) are fitted during batch runs but never serialized, preventing inference on raw single records.
2. **Missing Model Integration Interface**: Model loading lacks contract validation, exposing downstream components to schema drift when integrating Data Science models.
3. **Absence of a Service/API Layer**: No HTTP endpoint exists for real-time client inference.
4. **Hardcoded Configurations**: Paths and parameters are duplicated across multiple modules.
5. **Testing & Logging Gaps**: Preprocessing, reproducibility, and model loading lack dedicated test suites, and pipeline orchestration relies heavily on unformatted console prints.

---

## 2. Technical Audit Matrix

| Component | Current State | Weakness/Gap | Evidence | Week 3 Action | Priority |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Data Preprocessing State** | `DataPreprocessor` fits `OneHotEncoder` and extracts temporal features during training. | **Decoupled Preprocessor State**: The fitted preprocessor instance is never persisted or saved alongside the model artifact. Downstream consumers cannot transform raw incoming records without access to the full training set. | `preprocess_pipeline()` returns `(model_ready_df, preprocessor)` in `src/preprocessing.py:270`, but `train_baseline_model()` in `src/prediction.py:166` only saves `model` and `feature_names`. Preprocessor is discarded. | Implement `save()` and `load()` on `DataPreprocessor`; bundle or save preprocessor alongside model artifact to enable raw inference. | **HIGH** |
| **Model Integration Interface** | Model loading is handled via basic `joblib.load()` with Scikit-Learn `Pipeline`. | **Lack of Contract Validation & Model Abstraction**: No interface validates that a loaded artifact conforms to expected input schema, feature ordering, or prediction methods (`predict`, `predict_proba`). | `load_model_artifact()` in `src/prediction.py:188` performs raw unvalidated deserialization. `predict()` assumes scikit-learn API without standard interface checks. | Create `src/model_loader.py` with `load_model()`, `validate_model_interface()`, and standard prediction wrappers. Document contract in `docs/model_integration.md`. | **HIGH** |
| **Prediction Service / API** | Batch CSV prediction only (`src/prediction.py`). | **No Service Component**: No real-time serving interface exists for external digital banking client requests. | No FastAPI/web service file exists in `src/` or repository root. | Implement `src/api.py` with `GET /health` and `POST /predict`, powered by Pydantic request/response validation and reusing existing modules. | **HIGH** |
| **Testing Coverage** | 26 unit tests covering validation and basic prediction; 5 technical scenarios. | **Testing Gaps**: No dedicated unit tests for preprocessing transformations, model loading validation, reproducibility, or API endpoints. | `tests/` contains only `test_validation.py`, `test_prediction.py`, and `technical_test_report.py`. | Add `test_preprocessing.py`, `test_model_loading.py`, `test_reproducibility.py`, and `test_api.py`. | **HIGH** |
| **Reproducibility** | Stratified split uses `random_state=42`; baseline classifier uses fixed seed. | **Unverified Reproducibility**: No automated test asserts that given the same input, identical predictions and confidence scores are deterministically produced across repeated executions. | No reproducibility test script or assertion exists in the test suite. | Implement automated reproducibility test comparing predictions and confidences while separating dynamic timestamps. | **HIGH** |
| **Configuration Management** | Paths, thresholds, model versions, and schemas are scattered across multiple files. | **Duplicated & Hardcoded Configuration**: File paths (`data/raw`, `data/processed`, `models`), thresholds (`5.0%`), and model versions are hardcoded independently in `prediction.py`, `preprocessing.py`, and `validation.py`. | `DEFAULT_MODEL_DIR = "models"` in `src/prediction.py:68`; `"data/processed/FinTrust_Modelling_Ready.csv"` in `src/preprocessing.py:257`. | Introduce centralized `src/config.py` for pipeline paths, feature names, seeds, and thresholds. | **MEDIUM** |
| **Feature Engineering Types** | `Is_Local_Transaction` is derived as a boolean expression `(Location == City)`. | **Data Type Inconsistency**: `Is_Local_Transaction` is boolean (`True`/`False`), whereas all other engineered flags (`Is_Weekend`) are binary integers (`1`/`0`). Exporting to CSV causes type ambiguity upon reload. | `merged_df["Is_Local_Transaction"] = (merged_df["Location"] == merged_df["City"])` in `src/preprocessing.py:146`. | Explicitly cast `Is_Local_Transaction` to integer `int(0 or 1)` to maintain numeric consistency with `Is_Weekend`. | **MEDIUM** |
| **Logging & Telemetry** | Pipeline relies substantially on `print()` calls for console summaries. | **Unstructured Logging**: Direct `print()` statements cannot be configured, filtered by severity, or routed to external observability sinks. | `print(">>> [STAGE 1/6] Ingesting Raw Datasets...")` in `src/prediction.py:391`. | Refactor pipeline execution to use standard Python `logging` (`logger.info`, `logger.warning`, `logger.error`), reserving `print()` strictly for CLI wrappers. | **MEDIUM** |
| **Dependencies & Environment** | `requirements.txt` contains basic data science packages (`pandas`, `numpy`, `scikit-learn`, `joblib`, `pytest`, `openpyxl`). | **Missing Service Dependencies**: `fastapi`, `uvicorn`, `pydantic`, and `httpx` are absent from `requirements.txt`. | `requirements.txt` only lists 6 libraries without web service dependencies. | Update `requirements.txt` to include `fastapi`, `uvicorn`, `pydantic`, and `httpx` with pinned or minimum version constraints. | **MEDIUM** |
| **Documentation** | Week 2 technical test report exists (`docs/technical_test_report.md`). | **Documentation Gaps for Week 3**: Lacks model integration contract, empirical refinement evidence, reproducibility report, and API usage documentation. | `docs/` contains only `technical_test_report.md`. | Add `docs/week3_repository_audit.md`, `docs/model_integration.md`, `docs/week3_refinement_evidence.md`, `docs/week3_summary.md`, and update `README.md`. | **HIGH** |

---

## 3. Recommended Week 3 Architecture

```text
Input Request (Batch CSV or Single JSON via API)
   │
   ▼
[Data Validation] (src/validation.py + Pydantic API schemas)
   │
   ▼
[Preprocessing & Feature Preparation] (src/preprocessing.py)
   ├── Missingness Imputation (Device_Type, Location)
   ├── Temporal Feature Extraction (Hour, DayOfWeek, Weekend)
   ├── Consistent Categorical Encoding (Serialized Preprocessor)
   │
   ▼
[Model Interface & Loader] (src/model_loader.py)
   ├── Artifact Validation & Contract Enforcement
   ├── Model Deserialization & Metadata Inspection
   │
   ▼
[Prediction Generation] (src/prediction.py / Model Interface)
   ├── Probability Estimation (Confidence Score)
   ├── Decision Thresholding
   │
   ▼
[Structured Output] (Batch CSV with Traceability / API JSON Response)
```

---

## 4. Conclusion & Readiness

The Week 2 foundation is technically solid and functional. The planned Week 3 improvements directly address the identified weaknesses without disrupting existing working logic. Implementation can proceed systematically according to the approved 14-phase workflow.
