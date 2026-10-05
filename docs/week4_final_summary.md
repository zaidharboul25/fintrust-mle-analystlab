# FinTrust Digital Bank — Week 4 Final Technical Summary

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Milestone**: Week 4 — Test, Refine & Present  
**Author**: Zaid Harboul (Senior Machine Learning Engineer)  
**Date**: October 2026  
**Final Milestone Status**: **COMPLETE & READY FOR SUBMISSION**  

---

## 1. Week 4 Objective

The primary objective of Week 4 was to **test, validate, refine, finalize, and present** the FinTrust Machine Learning Engineering system developed across Weeks 1, 2, and 3. Rather than rebuilding or altering functional components without cause, Week 4 focused on:
- Establishing an empirical baseline audit of the existing codebase.
- Executing comprehensive end-to-end scenario testing across 9 core workflow dimensions.
- Resolving environment path resolution and API discoverability gaps.
- Formalizing the model integration status under Case B (baseline retained).
- Providing verified reproducibility evidence, governance documentation, and video presentation planning.

---

## 2. Starting Point from Week 3

The Week 3 milestone delivered:
- A functional 6-stage batch pipeline: Data $\rightarrow$ Validation $\rightarrow$ Preprocessing $\rightarrow$ Features $\rightarrow$ Baseline Model $\rightarrow$ Prediction $\rightarrow$ Output.
- A persistent preprocessor artifact (`models/preprocessor.joblib`) solving the single-record encoding issue.
- A standardized `ModelInterface` in `src/model_loader.py` enforcing contract validation and column reordering invariance.
- A real-time FastAPI service (`src/api.py`) exposing `/health` and `/predict`.
- 55 automated unit tests passing in pytest and 5 technical scenarios passing.

---

## 3. Final Repository Audit

At the start of Week 4, an exhaustive audit was performed before code modifications:
- Verified all raw data (`12,000` transactions, `1,500` customers).
- Verified baseline automated tests: **55/55 passed** via `python -m pytest tests/ -v`.
- Identified testing gap: bare `pytest` command failed with `ModuleNotFoundError: No module named 'src'` due to missing `pythonpath` configuration.
- Identified API gap: missing `GET /` service overview entrypoint.
- Audited model artifacts: no Data Science model was present in `models/`, confirming **Case B** status.
- Recorded full findings in `docs/week4_final_audit.md`.

---

## 4. Work Completed in Week 4

1. **Environment Hardening (`pytest.ini` & `tests/__init__.py`)**: Configured automatic `pythonpath = .` and test discovery, resolving the bare `pytest` import issue.
2. **API Entrypoint & Governance (`GET /`)**: Added `RootResponse` schema and `GET /` root endpoint in `src/api.py`, returning service metadata, version, documentation links, and responsible-use notices; added `test_root_endpoint` in `tests/test_api.py`.
3. **End-to-End Scenario Suite**: Created `tests/week4_e2e_scenario_runner.py` and `tests/test_week4_e2e.py` covering all 9 required technical scenarios.
4. **Model Integration Formalization**: Updated `docs/model_integration.md` codifying Case B (baseline retained via ModelInterface) without fabricating DS artifacts.
5. **Technical Test Report**: Generated `docs/week4_final_test_report.md` capturing actual execution outputs and 100% pass rates.
6. **Empirical Refinement Documentation**: Authored `docs/week4_refinement_evidence.md` documenting the problem-finding-refinement-retest lifecycle for pytest path resolution and API root enhancement.
7. **Reproducibility Evidence**: Authored `docs/week4_reproducibility_evidence.md` with step-by-step commands, dependency audits, and mathematical determinism verification.
8. **Governance & Limitations**: Created `docs/final_assumptions_limitations.md`.
9. **Final Documentation & Presentation**: Updated `README.md` and created `docs/final_video_outline.md`.

---

## 5. Final ML Workflow

The operational pipeline operates reliably end-to-end across both batch and real-time REST serving:

```text
┌────────────────────────────────────────────────────────┐
│  Client Ingestion (Batch CSV or Real-Time HTTP POST)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  1. Data Validation Gating (src/validation.py)         │
│  - Customer & Transaction schema conformity            │
│  - Foreign key referential integrity                   │
│  - Dynamic null tolerance (warning <= 5%, error > 5%)  │
│  - Regex ID patterns and domain categories             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  2. Preprocessing & Features (src/preprocessing.py)    │
│  - Relational merge on Customer_ID                     │
│  - Imputation: Device_Type ('Unknown'), Location (City)│
│  - Temporal: Hour, DayOfWeek, Is_Weekend (int 0/1)     │
│  - Location: Is_Local_Transaction (int 0/1)           │
│  - Categorical encoding: models/preprocessor.joblib    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  3. Model Interface & Serving (src/model_loader.py)    │
│  - Contract validation (predict / predict_proba)       │
│  - Automatic column reordering invariance              │
│  - Missing-feature drift detection                     │
│  - models/baseline_v1.joblib                           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  4. Structured Output                                  │
│  - Batch CSV: data/processed/FinTrust_Predictions.csv  │
│  - Real-Time JSON: { transaction_id, prediction, ... } │
│  - Traceability: model_version + prediction_timestamp  │
└────────────────────────────────────────────────────────┘
```

---

## 6. Model Integration Status

- **Status**: **CASE B — Documented Development Baseline Retained**
- **Artifacts Present**: `models/baseline_v1.joblib` (7,187 bytes), `models/preprocessor.joblib` (5,916 bytes).
- **Formal Statement**:
  > *"Final Data Science model was not available during Week 4. The existing documented model component was therefore retained to validate the final Machine Learning Engineering workflow."*
- **Operational Decoupling**: Fully verified through `ModelInterface`. When an external Data Science model is delivered, it can be dropped into `models/` and configured via `src/config.py` without code changes.

---

## 7. API / Service Status

- **Framework**: FastAPI `0.110+` running on Uvicorn.
- **Endpoints**:
  - `GET /`: Root discovery returning service overview, version, endpoint map, and responsible use disclaimer.
  - `GET /health`: Health diagnostic reporting component states, loaded model version (`baseline_v1`), preprocessor status, and feature count (`68`).
  - `POST /predict`: Single-transaction inference scoring with Pydantic schema validation, customer profile lookup fallback, referential integrity verification, and structured response formatting.
- **Error Handling**: Controlled HTTP error codes (400 Bad Request, 422 Unprocessable Entity, 503 Service Unavailable, 500 Internal Server Error) masking raw internal tracebacks and filesystem paths.

---

## 8. Testing Performed & 9. Test Results

### 8.1 Automated Pytest Regression Suite
```powershell
pytest -v
```
- **Tests Collected**: 65 items across 7 modules.
- **Passed**: **65 passed (100% PASS)** in **6.70 seconds**.
- **Failed**: 0.
- **Warnings**: 0.

| Test Module | Coverage Focus | Test Count | Status |
| :--- | :--- | :---: | :---: |
| `tests/test_validation.py` | Schemas, primary keys, null tolerance, integrity | 19 | **PASSED** |
| `tests/test_preprocessing.py` | Imputation, temporal features, encoding state | 6 | **PASSED** |
| `tests/test_model_loading.py` | ModelInterface contract, reordering invariance | 9 | **PASSED** |
| `tests/test_prediction.py` | Training, batch inference, output formatting | 7 | **PASSED** |
| `tests/test_reproducibility.py` | Deterministic weights, predictions, and metadata | 4 | **PASSED** |
| `tests/test_api.py` | Root, health, predict, fallback, error handling | 11 | **PASSED** |
| `tests/test_week4_e2e.py` | Automated wrapper for 9 Week 4 scenarios | 9 | **PASSED** |
| **Total Automated Suite** | **Comprehensive Regression Coverage** | **65** | **100% PASS** |

### 8.2 Technical Scenario Results (`tests/week4_e2e_scenario_runner.py`)
- **Scenarios Evaluated**: 9 / 9.
- **Passed**: **9 passed (100% PASS)**.
- Documented in `docs/week4_final_test_report.md`.

---

## 10. Validation Evidence

- **Batch Execution**: Executed `python src/prediction.py`:
  - 12,000 transactions processed in **0.99 seconds**.
  - Risk Flagged `Yes`: 4,878 (40.6%).
  - Risk Flagged `No`: 7,122 (59.4%).
  - Outputs persisted with 0 nulls to `data/processed/FinTrust_Predictions.csv`.
- **Validation Gating**: Executed `python src/validation.py`:
  - Successfully gates customer data, transaction data, and cross-table referential integrity.

---

## 11. Week 4 Refinement & 12. Re-test Results

1. **Pytest Path Configuration**:
   - Initial: `pytest` failed with `ModuleNotFoundError: No module named 'src'`.
   - Refinement: Added `pytest.ini` with `pythonpath = .` and `tests/__init__.py`.
   - Re-test: `pytest -v` collects cleanly and passes 65/65 tests.
2. **API Root Service Entrypoint**:
   - Initial: `GET /` returned `404 Not Found`.
   - Refinement: Added `RootResponse` schema and `GET /` in `src/api.py`.
   - Re-test: `test_root_endpoint` in `tests/test_api.py` passed with `200 OK`.
- Documented in `docs/week4_refinement_evidence.md`.

---

## 13. Reproducibility Evidence

- Fixed seeds (`random_state=42`) across train/test splitting and classifier fitting.
- Identical model weights across retrained instances (`np.allclose(atol=1e-9)`).
- Identical predictions across repeated runs (`np.allclose(atol=1e-7)`).
- Dynamic metadata (`prediction_timestamp`) verified to be isolated from predictions.
- Documented in `docs/week4_reproducibility_evidence.md`.

---

## 14. Major Technical Decisions

1. **Retain Case B Baseline Model**: Decided not to invent a synthetic or simulated Data Science model, maintaining strict engineering integrity and transparent role separation.
2. **Reuse Existing Modules in API**: `src/api.py` directly reuses `DataPreprocessor` and `ModelInterface`, preventing code duplication between offline batch pipelines and online services.
3. **Preserve Educational Risk Label**: Retained `Risk_Review_Flag` as an educational synthetic indicator with clear disclaimers, never representing it as authentic fraud detection.

---

## 15. Final Limitations & 16. Responsible-Use Considerations

- **Synthetic Target**: `Risk_Review_Flag` is an educational label, NOT authentic fraud determination.
- **Geographic Signal**: `Is_Local_Transaction` match rate (12.7%) is statistically consistent with random chance in this synthetic dataset.
- **In-Memory Cache**: Customer profile lookup uses an in-memory DataFrame, suitable for local serving but requiring Redis or SQL replica in production.
- Documented in `docs/final_assumptions_limitations.md`.

---

## 17. Final Deliverables Matrix

| Deliverable | Repository Evidence | Validation Evidence | Status |
| :--- | :--- | :--- | :---: |
| **Final Repository** | Modular structure: `src/`, `tests/`, `models/`, `data/`, `docs/` | Clean Git working tree on `main` branch | **COMPLETE** |
| **Final ML Workflow** | `src/validation.py`, `preprocessing.py`, `prediction.py` | Full batch pipeline runs in 0.99s across 12k rows | **COMPLETE** |
| **Pipeline Components** | Preprocessor, validator, model loader, API | 65 automated tests covering all components | **COMPLETE** |
| **Model Integration** | `src/model_loader.py`, `models/baseline_v1.joblib` | Case B formal determination in `docs/model_integration.md` | **COMPLETE** |
| **Test Results** | `tests/`, `docs/week4_final_test_report.md` | 65/65 pytest passed; 9/9 technical scenarios passed | **COMPLETE** |
| **API / Service** | `src/api.py` (`/`, `/health`, `/predict`) | 11 passing API tests via TestClient | **COMPLETE** |
| **README.md** | Comprehensive guide matching actual repository commands | Verified instructions and sample payloads | **COMPLETE** |
| **Requirements File** | `requirements.txt` with minimum version constraints | All 10 dependencies match codebase imports | **COMPLETE** |
| **Reproducibility Evidence**| `docs/week4_reproducibility_evidence.md`, `pytest.ini` | Automated determinism assertions pass (0.0 variance) | **COMPLETE** |
| **Technical Documentation**| Complete documentation suite in `docs/` | 8 comprehensive Markdown reports | **COMPLETE** |

---

## 18. Lessons Learned

### 3 Things That Worked Well:
1. **Decoupled Preprocessing State**: Serializing `OneHotEncoder` state into `models/preprocessor.joblib` cleanly bridged the gap between batch feature extraction and single-record real-time serving without code duplication.
2. **ModelInterface Contract Abstraction**: Creating an interface wrapper with column-order invariance and missing-feature detection provided total isolation between the serving pipeline and model estimators.
3. **Automated End-to-End Regression Testing**: Establishing a multi-tier test suite (unit tests, API tests, reproducibility tests, and technical scenario runners) caught regressions immediately and guaranteed operational determinism.

### 3 Challenges Encountered:
1. **Pytest Path Resolution Drift**: Standard `pytest` failed to import `src` without explicit environment configuration, which was solved by creating `pytest.ini`.
2. **Synthetic Data Feature Signal**: Evaluating engineered features like `Is_Local_Transaction` revealed a 12.7% match rate identical to random chance, requiring careful documentation so downstream consumers do not assume false predictive signal.
3. **Customer State Handling in Real-Time Serving**: Single incoming transactions rarely contain complete customer demographic profiles; implementing a customer lookup fallback in `src/api.py` resolved this while preserving referential integrity checks.

### 3 Lessons Learned:
1. **ML Engineering is System Engineering**: Building a production-grade ML solution is fundamentally about data contracts, validation gating, state persistence, error masking, and reproducibility, rather than merely fitting algorithms.
2. **Strict Track Boundaries Protect Project Quality**: Respecting the boundary between Machine Learning Engineering and Data Science ensures that engineering focuses on operational reliability while Data Science focuses on predictive optimization.
3. **Documentation Must Reflect Execution**: Every claim, metric, and command must be backed by real command executions rather than assumptions.

### Key Improvements Made in Week 4:
- Configured `pytest.ini` with `pythonpath = .` ensuring universal test runner compatibility.
- Implemented `GET /` root endpoint in FastAPI providing service discovery and governance.
- Expanded technical scenario evaluation to cover 9 full end-to-end workflow dimensions.
- Codified Case B model integration and created comprehensive reproducibility evidence.

### Recommendations for Future Work:
- Implement a distributed cache (e.g. Redis) for high-throughput customer profile lookups.
- Integrate Prometheus / OpenTelemetry middleware for API request latency and drift monitoring.
- Implement an automated Model Registry integration (e.g. MLflow) for model lifecycle tracking.

---

## 19. Final Project Status

**READY FOR SUBMISSION**
All Week 4 requirements are complete, verified by real execution, and fully documented.
