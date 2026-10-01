# FinTrust Digital Bank — Week 3 Technical Summary

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Milestone**: Week 3 — Develop & Integrate  
**Author**: Zaid Harboul  
**Status**: COMPLETE  

---

## 1. What Was Planned

The Week 3 objective was to transition the Week 2 technical workflow into a hardened, modular, reproducible, and integration-ready ML system. Planned activities included:
1. Repository audit to identify real technical gaps in Week 2.
2. Hardening the end-to-end pipeline with centralized configuration and decoupled state.
3. Establishing an explicit Model Integration Interface contract for future Data Science models.
4. Expanding technical unit and integration testing to at least 50+ tests covering preprocessing, model loading, reproducibility, and serving.
5. Implementing an HTTP prediction service using FastAPI (`GET /health`, `POST /predict`) without duplicating ML logic.
6. Empirical demonstration and documentation of at least one evidence-based refinement.
7. Updating environment dependencies, README, and milestone documentation.

---

## 2. What Was Completed

| Objective | Planned Deliverable | Delivered Artifact | Status |
| :--- | :--- | :--- | :---: |
| **Audit** | Concise technical audit | `docs/week3_repository_audit.md` | **COMPLETE** |
| **Pipeline Hardening** | Reusable, modular components | `src/config.py`, `src/preprocessing.py`, `src/prediction.py` | **COMPLETE** |
| **Model Interface** | Standard model loader & wrapper | `src/model_loader.py`, `docs/model_integration.md` | **COMPLETE** |
| **Testing** | Expanded automated test suite | `tests/` (55 tests passing across 6 test modules) | **COMPLETE** |
| **Reproducibility** | Automated determinism verification | `tests/test_reproducibility.py` | **COMPLETE** |
| **API Service** | Real-time FastAPI serving | `src/api.py` (`/health`, `/predict`) | **COMPLETE** |
| **API Testing** | Endpoint & error handling tests | `tests/test_api.py` (10 passing tests) | **COMPLETE** |
| **Refinement** | Empirical failure-to-fix evidence | `docs/week3_refinement_evidence.md` | **COMPLETE** |
| **Dependencies** | Pinned service dependencies | `requirements.txt` (`fastapi`, `uvicorn`, `pydantic`, `httpx`) | **COMPLETE** |
| **Documentation** | Week 3 summary & README update | `README.md`, `docs/week3_summary.md` | **COMPLETE** |

---

## 3. Major Development Activities

1. **Centralized Configuration (`src/config.py`)**: Extracted paths, schema lists, dynamic missingness thresholds (5.0%), and random seeds (42) to prevent configuration drift across modules.
2. **Stateful Preprocessor Serialization (`src/preprocessing.py`)**: Added `save()` and `load()` methods to `DataPreprocessor`, serializing fitted OneHotEncoder states to `models/preprocessor.joblib`. Harmonized `Is_Local_Transaction` to integer (`1`/`0`).
3. **Model Integration Layer (`src/model_loader.py`)**: Implemented `ModelInterface` and `load_model()` with interface contract validation, strict column reordering invariance, missing-feature detection, and standard `predict`/`predict_proba` wrappers.
4. **FastAPI Prediction Service (`src/api.py`)**: Built a REST service exposing `GET /health` and `POST /predict`, powered by Pydantic request/response validation, automatic customer profile lookup fallback, and clean HTTP exception handling without internal traceback leakage.
5. **Test Suite Expansion**: Added 29 new automated tests (increasing test suite from 26 to 55 tests) across preprocessing, model loading, determinism, and API serving.

---

## 4. Key Findings & Results

- **Test Execution**: **55/55 unit tests passed** in 5.26 seconds.
- **Technical Scenario Tests**: **5/5 scenarios passed** in `tests/technical_test_report.py`.
- **Pipeline Execution**: End-to-end pipeline completed in **1.08 seconds** across 12,000 transactions and 1,500 customers.
- **Baseline Classifier Sanity Check**:
  - Accuracy: 0.6342
  - Precision: 0.2923
  - Recall: 0.6106
  - F1-Score: 0.3953
  - ROC-AUC: 0.6676
  - Positive class predictions: 4,878 (40.6%) flagged "Yes"; 7,122 (59.4%) flagged "No".
- **API Latency**: Single-record prediction latency is under 15ms locally.

---

## 5. Testing & Validation Performed

| Test Module | Test Focus | Tests | Status |
| :--- | :--- | :---: | :---: |
| `tests/test_validation.py` | Empty inputs, schema drift, primary key uniqueness, data types, dynamic null thresholds, referential integrity. | 19 | **PASSED** |
| `tests/test_preprocessing.py` | Imputation logic, temporal feature extraction, Is_Local_Transaction types, unseen categories, preprocessor serialization. | 6 | **PASSED** |
| `tests/test_model_loading.py` | Artifact validation, contract errors, missing features, column reordering invariance, probability bounds. | 9 | **PASSED** |
| `tests/test_reproducibility.py` | Preprocessing determinism, model training weight parity, prediction determinism, timestamp metadata isolation. | 4 | **PASSED** |
| `tests/test_api.py` | Health endpoint, valid predictions, customer lookup fallback, referential integrity mismatch, type errors, 503 unavailability. | 10 | **PASSED** |
| `tests/test_prediction.py` | Baseline training, batch prediction, single record inference, output schema formatting, pipeline gating. | 7 | **PASSED** |
| `tests/technical_test_report.py` | 5 core technical scenarios required by internship deliverable. | 5 | **PASSED** |
| **TOTAL** | **Full System Test Coverage** | **55** | **100% PASS** |

---

## 6. Improvements Made (Evidence-Based Refinement)

- **Issue**: In Week 2, `DataPreprocessor` state was not serialized. Single-record inference on raw records failed with `ValueError: Missing 53 required feature columns` because fitting a fresh preprocessor on 1 row collapsed the categorical feature space.
- **Refinement**: Added preprocessor serialization (`models/preprocessor.joblib`), integrated it into `src/model_loader.py` and `src/api.py`, and verified that raw incoming transactions are transformed into the full 68-feature space deterministically.
- **Evidence**: Full diagnostic traces, failure recreation, and re-test results documented in [`docs/week3_refinement_evidence.md`](week3_refinement_evidence.md).

---

## 7. Assumptions, Limitations, Risks, and Technical Decisions Matrix

| Item | Category | Week 2 Baseline | Week 3 Status | Impact on Week 4 |
| :--- | :--- | :--- | :---: | :--- |
| **1. Synthetic Target Semantics** | Assumption | `Risk_Review_Flag` is a synthetic educational target, not real banking fraud. | **Still Relevant** | Must retain clear disclaimers across all documentation and presentations. |
| **2. MLE vs. DS Role Separation** | Decision | MLE builds reliable systems around the model; DS owns predictive optimization. | **Still Relevant** | Hand-off boundary is now codified in `docs/model_integration.md`. |
| **3. Preprocessor Decoupling** | Limitation | Preprocessor was not serialized, preventing raw inference. | **RESOLVED** | Preprocessor is now saved to `models/preprocessor.joblib` and reusable. |
| **4. Is_Local_Transaction Signal** | Limitation | Match rate was 12.7% (identical to random 12.5% chance). | **Still Relevant** | DS must validate feature weight; MLE preserves feature logic. |
| **5. Is_Local_Transaction Type** | Risk | Derived as boolean (`True`/`False`), causing CSV type ambiguity. | **RESOLVED** | Harmonized to integer (`1`/`0`). |
| **6. Hardcoded Paths & Constants** | Limitation | File paths and thresholds were scattered across modules. | **RESOLVED** | Centralized into `src/config.py`. |
| **7. Lack of Serving Component** | Limitation | Only offline batch CLI scripts existed. | **RESOLVED** | Real-time FastAPI service deployed with `/health` and `/predict`. |
| **8. Model Contract Enforcement** | Risk | Model loader performed unverified `joblib.load()`. | **RESOLVED** | `ModelInterface` validates contract and enforces column ordering. |
| **9. Reproducibility Gaps** | Risk | No automated test guaranteed deterministic inference. | **RESOLVED** | Automated reproducibility tests implemented in `tests/test_reproducibility.py`. |
| **10. Local Memory Customer Store** | Limitation | Customer lookup uses in-memory DataFrame lookup. | **NEW** | In Week 4 or production deployment, replace with Redis or database cache. |
| **11. Containerization / Docker** | Unresolved | Application runs directly on local host environment. | **NEW** | Containerize the FastAPI application using Docker for Week 4 delivery. |

---

## 8. Major Challenges

1. **Decoupled Preprocessing in API Context**: Transforming a single incoming transaction JSON required preserving the exact 68-column one-hot feature space established during training without refitting. Solving this through clean preprocessor serialization eliminated duplicate logic while ensuring sub-15ms response latency.
2. **Column Reordering Drift**: Scikit-Learn models are sensitive to column ordering. The `ModelInterface` introduces explicit feature alignment (`X = features_df[self.feature_names]`) to ensure inference is column-order invariant even if DataFrame columns arrive out of sequence.

---

## 9. Important Decisions

1. **Reusing Pipeline Code in FastAPI**: Rather than re-implementing transformation logic inside `src/api.py`, the API delegates directly to `DataPreprocessor.transform()` and `ModelInterface.predict()`, strictly adhering to the "no duplicate ML logic" principle.
2. **Customer Lookup Fallback**: In production banking systems, an incoming transaction event rarely contains full customer profile attributes. The API accepts optional `customer` payloads, falling back to cached customer store lookups when only `Customer_ID` is provided.

---

## 10. Work Required for Week 4 & Final Priorities

1. **Docker Containerization**: Create `Dockerfile` and `docker-compose.yml` to package the FastAPI service, pipeline scripts, and dependencies for portable deployment.
2. **Data Science Model Integration**: Validate and integrate the final model artifact delivered by the Data Science track using `docs/model_integration.md`.
3. **CI/CD Automation**: Implement GitHub Actions workflow running `pytest` and lint checks on pull requests.
4. **Monitoring & Telemetry**: Add request-level metrics (Prometheus / middleware) to track API prediction latency and error rates.
5. **Final Presentation & Deliverable Package**: Prepare final project report and demonstration video showcasing batch and real-time inference.
