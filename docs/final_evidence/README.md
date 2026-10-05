# FinTrust Digital Bank — Final Report Academic Evidence & Screenshot Index

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 4 — Test, Refine & Present  
**Document**: Visual Evidence Catalog for Final Academic Report & Presentation  
**Date**: October 2026  
**Status**: VERIFIED & AUDITED  

---

## 1. Overview & Verification Standards

This directory contains publication-ready, high-resolution evidence artifacts (2x device pixel ratio) generated from real executions of the FinTrust Machine Learning Engineering repository. 

All evidence complies with the strict project governance standards:
1. **Zero Fabrication**: Every visual artifact was generated directly from actual code execution, live Uvicorn serving, or interactive Playwright automation against the active codebase.
2. **Privacy & Security Masking**: No passwords, tokens, API keys, local username paths (`C:\Users\zaidh\...`), or private credentials appear in any artifact.
3. **Dual Access Paths**: Files are organized into thematic subdirectories (`01_repository/` through `06_outputs/`) and also maintained as flat copies in `docs/final_evidence/` for LaTeX and markdown compilation convenience.
4. **Governance Alignment**: FinTrust is documented throughout as a fictional institution with synthetic datasets; `Risk_Review_Flag` is explicitly maintained as an educational target, not real fraud determination.

---

## 2. Master Evidence Catalog (E01 – E13)

### Evidence E01: Repository Structure
- **Screenshot ID**: `E01`
- **Filename**: `01_repository_structure.png`
- **Categorized Location**: `docs/final_evidence/01_repository/01_repository_structure.png`
- **Flat Location**: `docs/final_evidence/01_repository_structure.png`
- **What it Demonstrates**: Clean, modular repository layout displaying the canonical directory structure (`data/`, `docs/`, `models/`, `src/`, `tests/`), configuration files (`pytest.ini`, `requirements.txt`), and documentation, with temporary cache files and IDE metadata excluded.
- **Command / Action Used**: `tree /F (Filtered Production Artifacts)`
- **Actual Result**: 100% verified production tree displaying 7 `src/` modules, 10 `tests/` files, 2 model artifacts, and 12 technical reports.
- **Recommended Report Chapter**: Chapter 2 — Requirements Analysis and System Design (or Chapter 3 — Implementation)
- **Recommended Figure Caption**: "Directory architecture and modular package layout of the FinTrust Machine Learning Engineering repository."
- **LaTeX Label**: `fig:repo_structure`

---

### Evidence E02: End-to-End Batch Pipeline Execution
- **Screenshot ID**: `E02`
- **Filename**: `02_pipeline_execution.png`
- **Categorized Location**: `docs/final_evidence/02_pipeline/02_pipeline_execution.png`
- **Flat Location**: `docs/final_evidence/02_pipeline_execution.png`
- **What it Demonstrates**: Sequential end-to-end execution of the operational ML batch pipeline across all 6 stages: Data Ingestion $\rightarrow$ Validation Gating $\rightarrow$ Preprocessing & Feature Extraction $\rightarrow$ Model Loading $\rightarrow$ Prediction Scoring $\rightarrow$ Structured Output Persistence.
- **Command / Action Used**: `python src/prediction.py`
- **Actual Result**: 12,000 transactions and 1,500 customers processed in 0.99 seconds. Validation gating passed 3/3 checks. Baseline model sanity metrics displayed (Accuracy: 0.6342, ROC-AUC: 0.6676). 12,000 predictions generated (4,878 flagged 'Yes' / 40.6%, 7,122 flagged 'No' / 59.4%). Status: `COMPLETED SUCCESSFULLY`.
- **Recommended Report Chapter**: Chapter 3 — Implementation of the Machine Learning Engineering Pipeline
- **Recommended Figure Caption**: "End-to-end batch pipeline execution log showing data validation gating, feature preparation, model inference, and output persistence."
- **LaTeX Label**: `fig:pipeline_execution`

---

### Evidence E03: Full Automated Pytest Regression Suite
- **Screenshot ID**: `E03`
- **Filename**: `03_full_pytest_results.png`
- **Categorized Location**: `docs/final_evidence/03_testing/03_full_pytest_results.png`
- **Flat Location**: `docs/final_evidence/03_full_pytest_results.png`
- **What it Demonstrates**: Flawless execution of the complete automated regression test suite covering validation rules, preprocessing state, model interface contracts, prediction determinism, API endpoints, and Week 4 end-to-end scenarios.
- **Command / Action Used**: `pytest -v`
- **Actual Result**: 65 tests collected across 7 test modules; **65 passed (100% PASS)** in 6.03 seconds with zero failures and zero warnings.
- **Recommended Report Chapter**: Chapter 4 — Testing, Validation and Technical Refinement
- **Recommended Figure Caption**: "Final automated regression test results verifying 65/65 passing test cases across all pipeline modules and services."
- **LaTeX Label**: `fig:pytest_results`

---

### Evidence E04: FastAPI Server Startup & Lifespan Initialization
- **Screenshot ID**: `E04`
- **Filename**: `04_fastapi_server.png`
- **Categorized Location**: `docs/final_evidence/04_api/04_fastapi_server.png`
- **Flat Location**: `docs/final_evidence/04_api_server.png`
- **What it Demonstrates**: Asynchronous application lifespan initialization: loading the serialized preprocessor (`models/preprocessor.joblib`), validating the baseline model artifact bundle (`models/baseline_v1.joblib`), caching customer lookup records (1,500 rows), and launching the HTTP worker on port 8000.
- **Command / Action Used**: `uvicorn src.api:app --host 127.0.0.1 --port 8000`
- **Actual Result**: Server initialized in under 1.2 seconds with active `ModelInterface` and `DataPreprocessor` instances loaded and ready for inference requests.
- **Recommended Report Chapter**: Chapter 3 — Implementation of the Machine Learning Engineering Pipeline
- **Recommended Figure Caption**: "FastAPI application startup log showing asynchronous lifespan loading of preprocessor state, model artifacts, and customer lookup cache."
- **LaTeX Label**: `fig:fastapi_startup`

---

### Evidence E05: Interactive Swagger / OpenAPI Documentation
- **Screenshot ID**: `E05`
- **Filename**: `05_fastapi_swagger.png`
- **Categorized Location**: `docs/final_evidence/04_api/05_fastapi_swagger.png`
- **Flat Location**: `docs/final_evidence/05_fastapi_swagger.png`
- **What it Demonstrates**: High-resolution browser rendering of the OpenAPI 3.1 Swagger documentation interface at `/docs`, displaying service metadata, educational governance disclaimers, and all 3 exposed REST endpoints: `GET /`, `GET /health`, and `POST /predict`.
- **Command / Action Used**: Browser navigation to `http://127.0.0.1:8000/docs` via Playwright
- **Actual Result**: Complete interactive documentation rendered with parameter specifications, Pydantic schemas, and execution buttons.
- **Recommended Report Chapter**: Chapter 3 — Implementation of the Machine Learning Engineering Pipeline
- **Recommended Figure Caption**: "FastAPI interactive OpenAPI/Swagger user interface displaying service metadata, endpoints, and request/response specifications."
- **LaTeX Label**: `fig:swagger_ui`

---

### Evidence E06: Health Diagnostic Endpoint (`GET /health`)
- **Screenshot ID**: `E06`
- **Filename**: `06_health_endpoint.png`
- **Categorized Location**: `docs/final_evidence/04_api/06_health_endpoint.png`
- **Flat Location**: `docs/final_evidence/06_health_endpoint.png`
- **What it Demonstrates**: Live execution of the health diagnostic endpoint returning real-time component availability, model versioning, feature dimensionality, and ISO 8601 UTC timestamp.
- **Command / Action Used**: Interactive execution of `GET /health` in Swagger UI via Playwright
- **Actual Result**: Status code `200 OK` returning `{"status": "healthy", "model_loaded": true, "model_version": "baseline_v1", "preprocessor_loaded": true, "features_count": 68}`.
- **Recommended Report Chapter**: Chapter 3 — Implementation of the Machine Learning Engineering Pipeline
- **Recommended Figure Caption**: "Diagnostic health check endpoint (`GET /health`) response confirming model and preprocessor availability."
- **LaTeX Label**: `fig:health_endpoint`

---

### Evidence E07: Successful Real-Time Single-Record Prediction (`POST /predict`)
- **Screenshot ID**: `E07`
- **Filename**: `07_successful_prediction.png`
- **Categorized Location**: `docs/final_evidence/04_api/07_successful_prediction.png`
- **Flat Location**: `docs/final_evidence/07_successful_prediction.png`
- **What it Demonstrates**: Real-time scoring of a valid transaction request payload (`FT-T000001`, `15,000 NGN`, Transfer) with associated customer profile (`FT-C00001`), demonstrating single-record feature transformation via persistent OneHotEncoder state and calibrated confidence output.
- **Command / Action Used**: Interactive execution of `POST /predict` in Swagger UI with JSON payload
- **Actual Result**: Status code `200 OK` returning structured JSON response: `transaction_id: "FT-T000001"`, `prediction: "No"`, `confidence: 0.4431`, `model_version: "baseline_v1"`, and ISO 8601 UTC timestamp. Response latency: 12ms.
- **Recommended Report Chapter**: Chapter 3 — Implementation (or Chapter 5 — Results)
- **Recommended Figure Caption**: "Successful real-time transaction scoring via `POST /predict` returning calibrated risk-review prediction and confidence score."
- **LaTeX Label**: `fig:successful_prediction`

---

### Evidence E08: Controlled API Error Handling & Validation
- **Screenshot ID**: `E08`
- **Filename**: `08_api_validation_error.png`
- **Categorized Location**: `docs/final_evidence/04_api/08_api_validation_error.png`
- **Flat Location**: `docs/final_evidence/08_api_validation_error.png`
- **What it Demonstrates**: Controlled rejection of invalid input data: submitting a transaction with `Customer_ID = "FT-C00001"` against an explicit customer profile payload with mismatched `Customer_ID = "FT-C00002"`, demonstrating referential integrity enforcement without internal traceback leakage.
- **Command / Action Used**: Interactive execution of `POST /predict` with mismatched customer identifiers
- **Actual Result**: Status code `400 Bad Request` returning controlled error detail: `"Referential integrity mismatch: transaction Customer_ID 'FT-C00001' does not match customer Customer_ID 'FT-C00002'."`
- **Recommended Report Chapter**: Chapter 4 — Testing, Validation and Technical Refinement
- **Recommended Figure Caption**: "Controlled error handling on `POST /predict` rejecting a cross-record referential integrity violation with an HTTP 400 Bad Request."
- **LaTeX Label**: `fig:api_error_handling`

---

### Evidence E09: Reproducibility & Determinism Tests
- **Screenshot ID**: `E09`
- **Filename**: `09_reproducibility.png`
- **Categorized Location**: `docs/final_evidence/05_reproducibility/09_reproducibility.png`
- **Flat Location**: `docs/final_evidence/09_reproducibility.png`
- **What it Demonstrates**: Automated assertions mathematically proving 100% deterministic preprocessing transformations, identical retrained model weights ($\Delta \mathbf{w} = 0.0$), identical predictions ($\Delta \mathbf{p} = 0.0$), and dynamic timestamp isolation.
- **Command / Action Used**: `pytest tests/test_reproducibility.py -v`
- **Actual Result**: 4/4 reproducibility tests passed in 3.44 seconds with zero variance.
- **Recommended Report Chapter**: Chapter 4 — Testing, Validation and Technical Refinement
- **Recommended Figure Caption**: "Automated reproducibility and determinism test results verifying identical model weights and prediction probabilities across repeated executions."
- **LaTeX Label**: `fig:reproducibility_tests`

---

### Evidence E10: Week 4 End-to-End Technical Scenarios
- **Screenshot ID**: `E10`
- **Filename**: `10_week4_e2e_tests.png`
- **Categorized Location**: `docs/final_evidence/03_testing/10_week4_e2e_tests.png`
- **Flat Location**: `docs/final_evidence/10_week4_e2e_tests.png`
- **What it Demonstrates**: Comprehensive scenario evaluation covering all 9 required workflow dimensions: Valid Input, Missing Values (<5%), Unexpected Categories, Invalid Data Types, Empty Input, Model Loading Contracts, Prediction Generation, Output Format, and Reproducibility.
- **Command / Action Used**: `python tests/week4_e2e_scenario_runner.py`
- **Actual Result**: Formatted ASCII table showing **9/9 technical scenarios PASSED (100%)** with markdown report written to `docs/week4_final_test_report.md`.
- **Recommended Report Chapter**: Chapter 4 — Testing, Validation and Technical Refinement
- **Recommended Figure Caption**: "Week 4 end-to-end technical scenario test execution table validating 9 core workflow dimensions."
- **LaTeX Label**: `fig:week4_scenarios`

---

### Evidence E11: Final Prediction Output Schema & Sample Records
- **Screenshot ID**: `E11`
- **Filename**: `11_prediction_output_sample.png`
- **Categorized Location**: `docs/final_evidence/06_outputs/11_prediction_output_sample.png`
- **Flat Location**: `docs/final_evidence/11_prediction_output_sample.png`
- **What it Demonstrates**: Tabular inspection of the persistent prediction output file showing the production audit schema (`Transaction_ID`, `predicted_Risk_Review_Flag`, `prediction_confidence`, `prediction_timestamp`, `model_version`) across 12,000 observations.
- **Command / Action Used**: Terminal inspection of `data/processed/FinTrust_Predictions.csv` via Pandas
- **Actual Result**: 12,000 records persisted with zero nulls; class balance: 4,878 'Yes' (40.6%), 7,122 'No' (59.4%); formatted sample of first 8 observations displayed.
- **Recommended Report Chapter**: Chapter 5 — Final Results, Limitations and Future Perspectives
- **Recommended Figure Caption**: "Sample records and audit schema from the final batch prediction output file (`FinTrust_Predictions.csv`)."
- **LaTeX Label**: `fig:prediction_output_sample`

---

### Evidence E12: Model & Preprocessor Artifact Verification
- **Screenshot ID**: `E12`
- **Filename**: `12_model_artifacts.png`
- **Categorized Location**: `docs/final_evidence/01_repository/12_model_artifacts.png`
- **Flat Location**: `docs/final_evidence/12_model_artifacts.png`
- **What it Demonstrates**: Programmatic inspection and integrity validation of the serialized `.joblib` artifacts on disk: verifying file sizes, estimator class types, feature names, dictionary bundle keys, and preprocessor state.
- **Command / Action Used**: Programmatic Python inspection of `models/baseline_v1.joblib` and `models/preprocessor.joblib`
- **Actual Result**: Both artifacts verified: model bundle (7,187 bytes, 68 features) and preprocessor (5,916 bytes, fitted OneHotEncoder with 13 categorical attributes). Status: `[PASS]`.
- **Recommended Report Chapter**: Chapter 3 — Implementation of the Machine Learning Engineering Pipeline
- **Recommended Figure Caption**: "Programmatic verification of serialized model artifact (`baseline_v1.joblib`) and preprocessor artifact (`preprocessor.joblib`)."
- **LaTeX Label**: `fig:model_artifacts`

---

### Evidence E13: Version Control Status & Git Audit Trail
- **Screenshot ID**: `E13`
- **Filename**: `13_git_repository_status.png`
- **Categorized Location**: `docs/final_evidence/01_repository/13_git_repository_status.png`
- **Flat Location**: `docs/final_evidence/13_git_repository_status.png`
- **What it Demonstrates**: Version control audit trail displaying active branch `main`, synchronized tracking status with `origin/main`, and concise linear commit history tracking milestone deliverables across Weeks 1, 2, 3, and 4.
- **Command / Action Used**: `git status; git log -n 6 --oneline`
- **Actual Result**: Clean commit history documenting sequential milestone deliveries with all Week 4 components verified and ready for staging.
- **Recommended Report Chapter**: Chapter 2 — Requirements Analysis and System Design (or Appendix)
- **Recommended Figure Caption**: "Git version control audit log displaying commit milestones across project phases."
- **LaTeX Label**: `fig:git_status`

---

## 3. Recommended Figure Selection for Academic Report

For a concise, high-impact final academic report (8–10 figures maximum), the following 8 core figures provide complete coverage without overloading the manuscript:

| Priority | ID | Filename | Chapter | Caption Summary |
| :---: | :---: | :--- | :--- | :--- |
| **1** | `E01` | `01_repository_structure.png` | Chapter 2 | Repository architecture and modular package layout. |
| **2** | `E02` | `02_pipeline_execution.png` | Chapter 3 | End-to-end batch pipeline execution across 6 stages. |
| **3** | `E05` | `05_fastapi_swagger.png` | Chapter 3 | FastAPI interactive OpenAPI/Swagger service interface. |
| **4** | `E07` | `07_successful_prediction.png` | Chapter 3 | Real-time transaction risk scoring via `POST /predict`. |
| **5** | `E03` | `03_full_pytest_results.png` | Chapter 4 | Automated regression test results (65/65 passing). |
| **6** | `E09` | `09_reproducibility.png` | Chapter 4 | Automated determinism assertions (0.0 variance). |
| **7** | `E10` | `10_week4_e2e_tests.png` | Chapter 4 | Week 4 end-to-end technical scenario evaluation (9/9 passed). |
| **8** | `E11` | `11_prediction_output_sample.png` | Chapter 5 | Structured prediction output schema and class distribution. |
