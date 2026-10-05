# FinTrust Digital Bank — Final Video Presentation Outline (5–10 Minutes)

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Presenter**: Zaid Harboul  
**Role**: Senior Machine Learning Engineer  
**Milestone**: Final Capstone Presentation (Week 4: Test, Refine & Present)  
**Target Duration**: 7–9 Minutes  

---

## Slide & Speaking Script Breakdown

### 1. Title & Introduction (0:00 – 0:45)
- **Visual**: Slide with Project Title, Track (Machine Learning Engineering), AnalystLab Africa Logo, Presenter Name.
- **Key Talking Points**:
  - Introduce yourself and your role as Senior Machine Learning Engineer on the FinTrust project.
  - State the overarching mission: *Operationalizing a reliable, reproducible, and robust ML serving system for FinTrust Digital Bank*.
  - State the mandatory governance disclaimer upfront: *FinTrust is a fictional entity, all datasets are synthetic, and the prediction target `Risk_Review_Flag` is an educational indicator, NOT an authentic fraud determination*.

### 2. The FinTrust Problem & Business Context (0:45 – 1:30)
- **Visual**: Digital banking ecosystem diagram; high-volume retail transactions across Nigeria.
- **Key Talking Points**:
  - FinTrust operates across 8 major Nigerian metropolitan hubs handling thousands of digital transactions daily.
  - Risk compliance analysts face high operational workloads reviewing flagged accounts.
  - The business need: An automated, resilient pipeline capable of ingesting raw transactional events, validating data contracts, and scoring risk probabilities with sub-15ms latency.

### 3. Machine Learning Engineering Mandate & Track Boundaries (1:30 – 2:15)
- **Visual**: Track boundary diagram distinguishing Data Analytics, Data Science, and Machine Learning Engineering.
- **Key Talking Points**:
  - Clearly delineate the MLE role: MLE does *not* invent experimental models to maximize F1-scores; that is the domain of the Data Science track.
  - Our responsibility: Building the production-grade engineering scaffolding around the model—input validation, relational feature extraction, preprocessor persistence, model contract interfaces, API serving, automated testing, and execution determinism.

### 4. What Was Personally Developed (2:15 – 3:15)
- **Visual**: Code structure diagram (`src/validation.py`, `src/preprocessing.py`, `src/model_loader.py`, `src/api.py`, `src/prediction.py`).
- **Key Talking Points**:
  - `DataValidator`: Enforces primary key uniqueness, foreign key referential integrity, and dynamic missingness thresholds (tolerated $\le 5\%$, blocking $> 5\%$).
  - `DataPreprocessor`: Relational join on `Customer_ID`, temporal feature engineering (hour, day of week, weekend), and persistent `OneHotEncoder` state.
  - `ModelInterface`: Decoupled abstraction ensuring column-order invariance and missing-feature drift detection.
  - FastAPI Prediction Service: Real-time REST endpoints (`GET /`, `GET /health`, `POST /predict`) with Pydantic validation and customer profile fallback.
  - End-to-End Test Suite: 65 automated pytest tests and 9 comprehensive technical scenario tests.

### 5. Technologies & Tools Used (3:15 – 3:45)
- **Visual**: Logos/icons of Python 3.12, Scikit-Learn, Pandas, FastAPI, Uvicorn, Pydantic, Joblib, Pytest, Git.
- **Key Talking Points**:
  - Python 3.12 as the runtime backbone.
  - Scikit-Learn and Joblib for ML transformations and model serialization.
  - FastAPI and Uvicorn for asynchronous high-performance serving.
  - Pytest for comprehensive multi-tier automated regression testing.

### 6. System Architecture & End-to-End Workflow (3:45 – 4:45)
- **Visual**: Architecture flow diagram:
  $$\text{Input} \longrightarrow \text{Validation Gating} \longrightarrow \text{Preprocessing} \longrightarrow \text{ModelInterface} \longrightarrow \text{Prediction} \longrightarrow \text{Output}$$
- **Key Talking Points**:
  - Walk through the data flow step-by-step from raw CSV ingestion or HTTP POST request.
  - Highlight validation gating: Corrupted or malicious inputs are blocked *before* reaching machine learning components.
  - Explain dual-mode serving: The exact same preprocessing and model interface logic serves both offline batch scoring (12,000 records in 0.99s) and real-time single-record API requests (<15ms).

### 7. Testing, Technical Scenarios & Main Results (4:45 – 5:45)
- **Visual**: Screenshots/tables of `pytest -v` (65/65 passed) and `docs/week4_final_test_report.md` (9/9 passed).
- **Key Talking Points**:
  - 65 automated tests passing with 0 failures across unit, integration, API, and reproducibility suites.
  - 9 technical scenario tests passing: Valid input, missing value tolerance, unexpected categories, invalid data types, empty inputs, model loading contracts, prediction calibration, output schema, and reproducibility.
  - Baseline model sanity check: Balanced Logistic Regression achieving 0.6342 accuracy and 0.6676 ROC-AUC across 2,400 test transactions.

### 8. Key Engineering Refinements (5:45 – 6:45)
- **Visual**: Before-and-after comparison slides illustrating Week 3 and Week 4 refinements.
- **Key Talking Points**:
  - *Week 3 Historical Context*: Resolved preprocessor decoupling where fresh preprocessor fitting on single records crashed due to 53 missing columns.
  - *Week 4 Refinements*:
    1. **Pytest Path Resolution**: Configured `pytest.ini` (`pythonpath = .`), eliminating import crashes when invoking standard `pytest` directly.
    2. **API Discovery & Governance**: Implemented `RootResponse` and `GET /` endpoint returning service metadata, endpoint routing, and responsible use notices.

### 9. Integration with the Wider FinTrust Solution (6:45 – 7:30)
- **Visual**: FinTrust Multi-Track Integration Diagram:
  $$\text{Data} \rightarrow \text{Analytics} \rightarrow \text{Data Science} \rightarrow \textbf{ML Engineering} \rightarrow \text{GenAI Support} \rightarrow \text{Solution}$$
- **Key Talking Points**:
  - Explain how the MLE track fits into the broader enterprise:
    - Ingests canonical customer and transaction data.
    - Bridges the Data Science predictive models to operational digital banking platforms.
    - Serves structured JSON outputs to downstream Knowledge-Based GenAI support agents and compliance dashboards.
  - State clearly: *Distinguish architectural integration readiness from actual artifact integration (Case B baseline retained)*.

### 10. Limitations, Responsible Use & Lessons Learned (7:30 – 8:30)
- **Visual**: Bulleted summary of limitations, ethical governance, and lessons learned.
- **Key Talking Points**:
  - Limitations: Synthetic data characteristics; `Is_Local_Transaction` match rate (12.7%) is statistically consistent with random chance.
  - Responsible Use: Must never represent predictions as authentic fraud determinations; model provides compliance decision-support alerts.
  - Lessons Learned:
    1. ML Engineering is system engineering—validation contracts and state persistence matter as much as algorithms.
    2. Respecting track boundaries ensures clean, modular software.
    3. Documentation must always be backed by real command executions.

### 11. Conclusion & Wrap-Up (8:30 – 9:00)
- **Visual**: Repository link, final checklist with all deliverables marked **COMPLETE**, contact info.
- **Key Talking Points**:
  - The FinTrust ML Engineering system is verified, tested, deterministic, and fully documented.
  - Status: **READY FOR SUBMISSION**.
  - Thank mentors, coordinators, and peers at AnalystLab Africa.
