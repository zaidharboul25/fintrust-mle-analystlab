# FinTrust Digital Bank — Final Assumptions, Limitations & Responsible Use

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 4 — Test, Refine & Present  
**Document**: Architectural Boundary, Technical Assumptions & Ethical Governance  
**Date**: October 2026  
**Status**: APPROVED & CODIFIED  

---

## 1. Technical & Architectural Assumptions

### 1.1 Dataset Assumptions
- **Canonical Sources**: The data consists of two relational tables placed in `data/raw/`: `FinTrust_Customer_Data.csv` (1,500 rows) and `FinTrust_Transaction_Data.csv` (12,000 rows).
- **Referential Integrity**: Every transaction record contains a `Customer_ID` that maps to an existing primary key in `Customer_Data`. The `DataValidator` enforces referential integrity, failing with a blocking error if orphaned transactions are detected.
- **Dynamic Missingness Policy**: Known missingness in `Device_Type` (0.8%) and `Location` (0.8%) is treated as normal data noise when $\le 5.0\%$, issuing non-blocking warnings. Any missingness $> 5.0\%$ is treated as structural corruption and blocks execution.
- **Supported Geographies**: All transactions and customer records originate within 8 predefined Nigerian cities (`Lagos`, `Kano`, `Port Harcourt`, `Benin City`, `Kaduna`, `Abuja`, `Ibadan`, `Enugu`).

### 1.2 Schema & Preprocessing Assumptions
- **Relational Merge**: Customer demographic features (`City`, `Customer_Segment`, `Account_Type`, `Monthly_Income_Band`, etc.) are joined on `Customer_ID` to provide contextual profile signals.
- **Categorical Feature Stability**: Categorical variables are encoded using a pre-fitted `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` serialized to `models/preprocessor.joblib`. This produces exactly 68 features (8 numerical/temporal features + 60 one-hot indicators) regardless of batch size ($N=1$ or $N=12,000$).
- **Derived Feature Types**: `Is_Local_Transaction` is derived as a binary integer (`1` or `0`), ensuring type parity with `Is_Weekend` and numerical stability across persistence layers.

### 1.3 Model Interface Assumptions
- **Contract Decoupling**: Models interact exclusively through `src.model_loader.ModelInterface`. Any candidate model artifact must be serialized via `joblib` as a dictionary bundle containing at minimum:
  - `model`: An estimator object implementing `.predict(X)` and `.predict_proba(X)`.
  - `feature_names`: Ordered list of expected feature strings.
  - `model_version`: Unique string identifier.
- **Column Order Invariance**: Downstream models assume strict feature index ordering. The `ModelInterface` automatically inspects incoming DataFrames, detects missing features, and aligns column order to match training order prior to calling estimator methods.

---

## 2. Technical Limitations

### 2.1 Synthetic Dataset Nature
- All records (names, account balances, transaction amounts, locations, and timestamps) are synthetically generated for educational training within the AnalystLab Africa internship.
- **Feature Signal Limitation**: Statistical analysis reveals that `Location` and `City` were generated independently in the synthetic data, resulting in a 12.7% match rate for `Is_Local_Transaction`—virtually identical to the random 12.5% uniform probability across 8 cities. While preserved for real-world conceptual completeness, the feature contains minimal genuine predictive signal.

### 2.2 Synthetic Prediction Target (`Risk_Review_Flag`)
- `Risk_Review_Flag` (`Yes` / `No`) is a synthetic educational target designed to demonstrate classification workflows.
- It **does not represent authentic banking fraud, financial crime, or regulatory suspicious activity**.

### 2.3 Baseline Development Model (Case B Status)
- In the absence of an external final Data Science model artifact, the ML Engineering pipeline retains `models/baseline_v1.joblib` (a balanced Logistic Regression classifier).
- **Sanity Performance**:
  - Accuracy: `0.6342`
  - Precision: `0.2923`
  - Recall: `0.6106`
  - F1-Score: `0.3953`
  - ROC-AUC: `0.6676`
- This model serves as an operational proof-of-concept pipeline placeholder to validate end-to-end data flows, API serving, and reproducibility, not as an optimized production classifier.

### 2.4 Local / Test Serving Environment
- The FastAPI service (`src/api.py`) is designed for local development, technical evaluation, and demonstration via Uvicorn.
- Customer lookup fallback queries an in-memory Pandas DataFrame indexed on `Customer_ID`. In a distributed high-throughput architecture, this lookup would be offloaded to an enterprise caching layer (e.g. Redis) or a read-replica database.

---

## 3. Responsible Use & Governance Principles

The following principles must govern the interpretation and presentation of this project:

1. **Non-Production Disclaimer**: FinTrust is a fictional entity. This software is an educational prototype and is **NOT a production banking system**.
2. **Prediction is NOT Fraud Determination**: Model outputs (`Yes` / `No`) and probabilities represent a synthetic "risk-review" flag. They must never be described or presented as:
   - *"fraud detected"*
   - *"confirmed financial crime"*
   - *"fraudulent transaction"*
3. **Mandatory Human Review**: In any real-world banking context, ML predictions must function strictly as decision-support alerts for human compliance analysts, never as autonomous transaction-declining mechanisms.
4. **No Financial or Legal Advice**: Outputs generated by this pipeline do not constitute financial, credit, or legal advice.
5. **Privacy & Data Security**: Synthetic identifiers adhere to pseudonymous patterns (`FT-C#####`, `FT-T######`). No real-world Personally Identifiable Information (PII) is processed or stored.

---

## 4. Final Technical Risks & Mitigation Strategies

| Technical Risk | Potential Impact | Implemented Engineering Mitigation |
| :--- | :--- | :--- |
| **Model / Feature Schema Drift** | Candidate model expects different feature names or column ordering. | `ModelInterface` strictly aligns columns, detects missing features, and raises explicit `ModelInterfaceError` before model evaluation. |
| **Out-of-Domain Categorical Values** | Unseen categories in production traffic cause encoding crashes. | `OneHotEncoder` initialized with `handle_unknown='ignore'`; unobserved levels encode as zero vectors without exception. |
| **Corrupted Numeric Types** | String characters in amount fields crash matrix math. | `DataValidator` and Pydantic schemas enforce type validation at ingestion, rejecting malformed records with HTTP 400 Bad Request. |
| **Decoupled Preprocessing State** | Real-time single record inference fails due to missing categorical levels. | Preprocessor fitted state serialized to `models/preprocessor.joblib` and loaded during API lifespan startup. |
| **Environment Inconsistency** | Pytest fails when invoked directly due to path resolution issues. | `pytest.ini` configures `pythonpath = .`, guaranteeing deterministic test execution across any OS or CI shell. |
| **Data Drift Over Time** | Input distribution drifts away from training distributions. | System designed with decoupled `ModelInterface` and versioned outputs (`model_version`, `prediction_timestamp`) for continuous auditability. |
