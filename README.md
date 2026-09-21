# FinTrust Digital Bank - ML Engineering Pipeline

## Week 2: Data Validation Component

This repository contains the Machine Learning Engineering pipeline for **FinTrust Digital Bank** (AnalystLab Africa Internship, Track ML Engineering).

### Project Overview & Directory Structure
```
.
├── data/
│   ├── raw/                 # Raw input datasets (FinTrust_Customer_Data & FinTrust_Transaction_Data)
│   └── processed/           # Processed and validated datasets
├── docs/                    # Technical documentation and architecture designs
├── models/                  # Serialized ML models and pipelines
├── notebooks/               # Exploratory data analysis and experimental notebooks
├── src/                     # Production source code
│   ├── __init__.py
│   └── validation.py        # Data validation module and schema definitions
└── tests/                   # Automated test suites
    └── test_validation.py   # Unit tests for schema and relational validation
```

---

### Data Validation Architecture (`src/validation.py`)

The validation component is designed around **testability, separation of concerns, and structured error reporting**. Rather than crashing unexpectedly or silently corrupting downstream joins, the validator inspects DataFrames and categorizes findings into **blocking errors** and **tolerated warnings**.

#### Core Modules & Capabilities:
1. **Schema Validation (`DataValidator`)**:
   - Checks for completely empty datasets (0 rows) — raises `EmptyDatasetError`.
   - Missing required columns vs unexpected extra columns (with configurable schema drift tolerance).
   - Duplicate primary key detection (`Customer_ID` in customer data, `Transaction_ID` in transaction data).
   - Logical data type enforcement (numeric, integer without lossy decimals, ISO datetime strings or timestamps).
   - Regular expression pattern enforcement (`FT-C\d{5}` for Customer IDs, `FT-T\d{6}` for Transaction IDs).
   - Categorical domain enforcement against permissible values (`Transaction_Status`, `Risk_Review_Flag`, `Account_Status`, etc.).
2. **Dynamic Missingness Tolerance**:
   - For strictly prohibited columns, any null value triggers a blocking `error`.
   - For tolerated columns (`Device_Type` and `Location`), missingness below a configurable threshold (default `5.0%`) produces a non-blocking `warning` (handling the intentional ~0.8% missingness in Week 2 raw data).
   - If missingness exceeds `5.0%`, the validator elevates the issue to a blocking `error` to guard against silent data corruption or distribution drift.
3. **Cross-Dataset Referential Integrity (`check_referential_integrity`)**:
   - Validates that every `Customer_ID` present in `Transaction_Data` exists in `Customer_Data`.
   - Flagged as a **blocking error** if orphan transactions are found, preventing silent failure during feature engineering joins.
   - Computes customer coverage and flags inactive customers as informative warnings.
4. **Structured Reporting (`ValidationReport`)**:
   - Dataclass returning `is_valid` boolean, `error_count`, `warning_count`, detailed issue lists, diagnostic breakdowns, and a formatted console `.summary()`.

---

### Running Validation

#### 1. Execute the Standalone Validator CLI
Validates both raw datasets and checks referential integrity:
```powershell
python src/validation.py
```

#### 2. Run the Automated Pytest Suite
Runs all 19 unit tests covering edge cases, schema violations, dynamic thresholds, and real data:
```powershell
python -m pytest tests/test_validation.py -v
```

---

### Technical Design Decisions & Known Limitations

1. **Logging vs Console `print()` Output**:
   - *Current Implementation*: The module configures standard library `logging` and formats human-readable summaries for the CLI demonstration via `print()`.
   - *Known Limitation & Future Evolution*: In Week 2, CLI summaries use standard output for visual inspection during local development. For future production pipeline steps (Part C/D: CI/CD & Airflow/Prefect orchestration), console printing will be replaced entirely with structured JSON loggers emitting metrics to an observability sink (e.g., Datadog, Prometheus, CloudWatch).

2. **CSV File Ingestion & Encodings**:
   - *Current Implementation*: Standard ingestion assumes UTF-8 encoded CSV or Excel formats as provided in `data/raw/`.
   - *Known Limitation & Future Evolution*: Files with alternate byte-order marks (BOM), custom delimiters (semicolons, tabs), or truncated lines are not automatically auto-detected. Future iterations will incorporate robust encoding detection (e.g. `charset-normalizer`) and CSV dialect sniffing.

3. **Dynamic Threshold Customization**:
   - *Current Implementation*: Default schema sets `max_warning_missing_pct = 5.0%` on `Device_Type` and `Location`.
   - *Future Evolution*: Dynamic thresholds will be decoupled into an external configuration file (YAML/JSON) to allow per-environment tuning (dev vs staging vs prod).
