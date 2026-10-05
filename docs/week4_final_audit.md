# FinTrust Digital Bank — Week 4 Final Repository Audit

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 4 — Test, Refine & Present  
**Auditor**: Senior Machine Learning Engineer  
**Date**: October 2026  
**Status**: BASELINE AUDIT COMPLETE  

---

## 1. Executive Summary

A comprehensive repository audit was conducted prior to executing Week 4 (*Test, Refine & Present*) modifications. The FinTrust ML Engineering codebase retains all operational foundations built across Weeks 1, 2, and 3:

- **End-to-End Pipeline**: Functional sequential flow: Raw Data Ingestion $\rightarrow$ Data Validation Gating $\rightarrow$ Preprocessing & Relational Join $\rightarrow$ Feature Preparation $\rightarrow$ Baseline Logistic Classifier $\rightarrow$ Prediction Generation $\rightarrow$ Structured Output Persistence.
- **Baseline Test Suite**: 55 automated unit and integration tests collected and **100% passing (55/55 passed in 5.38 seconds)** when executed via `python -m pytest tests/ -v`.
- **Technical Scenarios**: 5/5 validation technical scenarios passing in `tests/technical_test_report.py`.
- **Model Decoupling**: Persistent preprocessor (`models/preprocessor.joblib`) and baseline model (`models/baseline_v1.joblib`) managed through `ModelInterface` (`src/model_loader.py`).
- **REST Service**: FastAPI application (`src/api.py`) exposing `GET /health` and `POST /predict`.

This audit establishes the empirical baseline, identifies specific Week 4 gaps, and defines the action plan required for final delivery and presentation.

---

## 2. Current Repository Structure

```text
fintrust-mle/
├── .git/                                 # Git version control metadata (on branch 'main')
├── .gitignore                            # Python, pytest, and environment ignores
├── .pytest_cache/                        # Pytest local cache
├── data/
│   ├── raw/
│   │   ├── FinTrust_Customer_Data.csv    # 1,500 rows, 12 columns
│   │   └── FinTrust_Transaction_Data.csv # 12,000 rows, 11 columns
│   └── processed/
│       ├── FinTrust_Modelling_Ready.csv  # 12,000 rows, 68 features + target
│       └── FinTrust_Predictions.csv      # 12,000 rows, 5 audit columns
├── docs/
│   ├── model_integration.md              # DS model hand-off and ModelInterface contract
│   ├── technical_test_report.md          # Week 2/3 technical scenario test report
│   ├── week3_refinement_evidence.md     # Preprocessor serialization empirical evidence
│   ├── week3_repository_audit.md         # Week 3 audit report
│   └── week3_summary.md                  # Week 3 milestone summary
├── models/
│   ├── baseline_v1.joblib                # Baseline classifier bundle (Logistic Regression)
│   └── preprocessor.joblib               # Serialized fitted DataPreprocessor instance
├── notebooks/                            # Exploratory scratch directory (.gitkeep)
├── src/
│   ├── __init__.py                       # Package marker
│   ├── api.py                            # FastAPI serving application (health, predict)
│   ├── config.py                         # Centralized configuration, paths, thresholds
│   ├── model_loader.py                   # ModelInterface abstraction and loading validation
│   ├── prediction.py                     # Training, batch prediction, pipeline orchestrator
│   ├── preprocessing.py                  # Stateful DataPreprocessor, temporal & location features
│   └── validation.py                     # DataValidator, schemas, and referential integrity
├── tests/
│   ├── technical_test_report.py          # 5 technical validation scenarios runner
│   ├── test_api.py                       # 10 tests for FastAPI endpoints and error handling
│   ├── test_model_loading.py             # 9 tests for ModelInterface and contract checks
│   ├── test_prediction.py                # 7 tests for training, batch prediction, and gating
│   ├── test_preprocessing.py             # 6 tests for imputation, temporal extraction, encoding
│   ├── test_reproducibility.py           # 4 tests for deterministic inference and weights
│   └── test_validation.py                # 19 tests for schema drift, null thresholds, integrity
├── README.md                             # Comprehensive project documentation
└── requirements.txt                      # Project dependency specification
```

---

## 3. Baseline Automated Test Results

Executed prior to any repository modifications:

```powershell
python -m pytest tests/ -v
```

### Execution Metrics:
- **Test Session Platform**: Windows (Python 3.12.10, pytest-9.1.1, pluggy-1.6.0)
- **Tests Collected**: 55 items across 6 test modules
- **Tests Passed**: **55 passed**
- **Tests Failed**: **0**
- **Warnings**: 0
- **Execution Time**: **5.38 seconds**

### Test Distribution by Module:
| Test File | Test Focus | Tests | Status |
| :--- | :--- | :---: | :---: |
| `tests/test_validation.py` | Schema conformity, primary keys, null tolerance, referential integrity | 19 | **PASSED** |
| `tests/test_preprocessing.py` | Missing value imputation, temporal features, local transaction flag, encoding | 6 | **PASSED** |
| `tests/test_model_loading.py` | ModelInterface contract, column reordering invariance, probability bounds | 9 | **PASSED** |
| `tests/test_prediction.py` | Pipeline stages, baseline training, batch prediction, gating halt on error | 7 | **PASSED** |
| `tests/test_reproducibility.py` | Deterministic preprocessing, training weight parity, prediction determinism | 4 | **PASSED** |
| `tests/test_api.py` | Health endpoint, single predict, customer fallback, 400/503 error handling | 10 | **PASSED** |
| **Total Unit/Integration Suite** | **Comprehensive Automated Coverage** | **55** | **100% PASS** |

### Technical Scenario Baseline:
```powershell
python tests/technical_test_report.py
```
- **Scenarios Evaluated**: 5 (Valid Input, Missing Values, Unexpected Category, Incorrect Data Type, Empty Input)
- **Result**: **5/5 PASSED**

---

## 4. Week 3 Components Verified

1. **Modular Source Packages (`src/`)**:
   - `src/config.py`: Correctly centralizes paths, schema constants, seeds (`42`), and null thresholds (`5.0%`).
   - `src/validation.py`: `DataValidator` enforces schema validation, regex ID formats (`FT-T\d{6}`, `FT-C\d{5}`), dynamic null tolerance, and referential integrity.
   - `src/preprocessing.py`: `DataPreprocessor` manages imputation, temporal features (`Transaction_Hour`, `Transaction_DayOfWeek`, `Is_Weekend`), binary `Is_Local_Transaction` (0/1), and serialized `OneHotEncoder(handle_unknown='ignore')`.
   - `src/model_loader.py`: `ModelInterface` enforces the dictionary bundle contract (`model`, `feature_names`, `model_version`), handles missing-feature detection, and provides column-order invariance.
   - `src/prediction.py`: Provides `train_baseline_model()`, `predict()`, `format_and_save_predictions()`, and the orchestrator `run_pipeline()`.
   - `src/api.py`: FastAPI application exposing `/health` and `/predict` with Pydantic payload models and error handling.
2. **Artifact Decoupling (`models/`)**:
   - `models/baseline_v1.joblib` and `models/preprocessor.joblib` exist, are valid, and load cleanly.
3. **Data Assets (`data/`)**:
   - Raw CSVs exist in `data/raw/` (12,000 transactions, 1,500 customers).
   - Processed modelling dataset and predictions exist in `data/processed/`.

---

## 5. Week 4 Gaps & Technical Opportunities Identified

| Area | Current State | Week 4 Gap / Finding | Severity / Priority |
| :--- | :--- | :--- | :---: |
| **Test Execution Discovery** | `python -m pytest` passes 55/55, but running bare `pytest` fails with `ModuleNotFoundError: No module named 'src'`. | Missing pytest configuration (`pytest.ini` or `pyproject.toml` with `pythonpath = .`) means bare `pytest` invocations fail unless `PYTHONPATH` is explicitly exported. | **HIGH (Reproducibility & DX)** |
| **End-to-End Scenario Testing** | `tests/technical_test_report.py` covers 5 validation-only scenarios from Week 2. | Week 4 requires explicit end-to-end scenario validation covering 9 key workflow dimensions (Valid input, Missing values, Unexpected categories, Invalid types, Empty input, Model loading, Prediction generation, Output schema, Reproducibility). | **HIGH** |
| **Data Science Model Status** | Only `baseline_v1.joblib` is present. No external Data Science model artifact exists in `models/` or repository. | Must explicitly formalize Case B: Retain `baseline_v1` via `ModelInterface` without fabricating DS models or collaborations, and update `docs/model_integration.md`. | **HIGH (Integrity)** |
| **API Surface & Metadata** | API provides `/health` and `/predict`, but lacks root endpoint `/`. | Accessing `GET /` returns 404. Adding a lightweight `GET /` returning service metadata, version, documentation links, and responsible use disclaimers improves API ergonomics. | **MEDIUM** |
| **Reproducibility Documentation** | `test_reproducibility.py` validates determinism in code, but lacks dedicated Week 4 documentation. | Need `docs/week4_reproducibility_evidence.md` detailing step-by-step commands, dependency auditing, and exact numerical outputs. | **HIGH** |
| **Empirical Refinement** | Week 3 solved the 53-missing-features issue. Week 4 requires new evidence-based refinement. | Need to document a concrete Week 4 finding (e.g. pytest path resolution via `pytest.ini` and API root entrypoint enhancement), root cause, fix, and re-test in `docs/week4_refinement_evidence.md`. | **HIGH** |
| **Assumptions, Limitations & Governance** | Scattered across multiple Week 3 markdown files. | Need a consolidated `docs/final_assumptions_limitations.md` detailing synthetic target semantics, educational boundaries, and operational risks. | **HIGH** |
| **Packaging & CI/CD** | Repository is pure local Python scripts. | Optional `Dockerfile`, `.dockerignore`, and GitHub Actions workflow (`.github/workflows/ci.yml`) will enhance portability and CI demonstration. | **MEDIUM (Optional Enhancement)** |
| **Final Documentation & Presentation** | `README.md` documents Week 3 milestone. | Update `README.md` to reflect Week 4 completion; create `docs/week4_final_summary.md` and `docs/final_video_outline.md`. | **HIGH** |

---

## 6. Model Integration Status (Audit Verification)

- **Artifacts Present**:
  - `models/baseline_v1.joblib` (7,187 bytes)
  - `models/preprocessor.joblib` (5,916 bytes)
- **Data Science Candidate Artifacts**: **NONE PRESENT**.
- **Decision Path**: **CASE B (Standard Baseline Retained)**
  - As mandated by project boundaries, we DO NOT fabricate or simulate a Data Science model.
  - The operational workflow will continue utilizing `baseline_v1.joblib` through `ModelInterface`.
  - `docs/model_integration.md` will clearly record: *"Final Data Science model was not available during Week 4. The existing documented model component was therefore retained to validate the final Machine Learning Engineering workflow."*

---

## 7. Remaining Technical Risks

1. **Synthetic Educational Target**: `Risk_Review_Flag` is synthetic and must not be portrayed as real fraud detection.
2. **Geographic Feature Signal**: `Is_Local_Transaction` has a 12.7% positive rate in the synthetic data (statistically identical to random chance across 8 cities); pipeline code handles it safely, but it cannot provide real-world predictive lift.
3. **Environment Drift**: Ensure `requirements.txt` accurately reflects direct dependencies and version constraints.

---

## 8. Week 4 Action Plan

```text
Step 1: Environment & Test Runner Hardening
        └── Create pytest.ini (pythonpath = .) so both bare 'pytest' and 'python -m pytest' succeed.
Step 2: API Surface Refinement
        └── Add GET / to src/api.py with service metadata and educational disclaimer; add test in tests/test_api.py.
Step 3: End-to-End Scenario Suite & Execution
        └── Implement/expand technical scenario runner covering all 9 required dimensions; generate docs/week4_final_test_report.md.
Step 4: Formalize Model Integration Status (Case B)
        └── Update docs/model_integration.md with final Week 4 status.
Step 5: Reproducibility Validation & Audit
        └── Execute deterministic pipeline runs, record outputs in docs/week4_reproducibility_evidence.md.
Step 6: Document Week 4 Refinement Evidence
        └── Create docs/week4_refinement_evidence.md documenting the pytest execution configuration and API root enhancement.
Step 7: Governance & Limitations Documentation
        └── Create docs/final_assumptions_limitations.md.
Step 8: Final Summary & Video Presentation Outline
        └── Create docs/week4_final_summary.md and docs/final_video_outline.md.
Step 9: Optional Containerization & CI
        └── Add Dockerfile, .dockerignore, and .github/workflows/ci.yml.
Step 10: Final Regression Testing & README Update
        └── Re-run all automated tests and scenarios; update README.md for final Week 4 submission.
```
