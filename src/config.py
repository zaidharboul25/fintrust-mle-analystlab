"""
FinTrust Digital Bank - Pipeline Configuration Module
=====================================================
Week 3: ML Engineering Architecture Component

Centralizes repository paths, schema column specifications, dynamic quality thresholds,
and model training parameters to prevent configuration drift and duplicate constants.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# =====================================================================
# Filesystem Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Canonical Raw CSV Files
CUSTOMER_DATA_PATH = RAW_DATA_DIR / "FinTrust_Customer_Data.csv"
TRANSACTION_DATA_PATH = RAW_DATA_DIR / "FinTrust_Transaction_Data.csv"

# Canonical Processed Files
MODELLING_DATA_PATH = PROCESSED_DATA_DIR / "FinTrust_Modelling_Ready.csv"
PREDICTIONS_OUTPUT_PATH = PROCESSED_DATA_DIR / "FinTrust_Predictions.csv"

# Model Directories & Artifacts
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_VERSION = "baseline_v1"
DEFAULT_MODEL_ARTIFACT_PATH = MODEL_DIR / f"{DEFAULT_MODEL_VERSION}.joblib"
PREPROCESSOR_ARTIFACT_PATH = MODEL_DIR / "preprocessor.joblib"

# Documentation & Static Assets
DOCS_DIR = PROJECT_ROOT / "docs"
STATIC_DIR = PROJECT_ROOT / "src" / "static"

# =====================================================================
# Feature & Target Schema Definitions
# =====================================================================

ID_COLUMNS: List[str] = ["Transaction_ID", "Customer_ID"]
TARGET_COLUMN: str = "Risk_Review_Flag"

CATEGORICAL_FEATURES: List[str] = [
    "Transaction_Type",
    "Channel",
    "Device_Type",
    "Location",
    "International_Transaction",
    "Transaction_Status",
    "Gender",
    "City",
    "Customer_Segment",
    "Account_Type",
    "Monthly_Income_Band",
    "Preferred_Channel",
    "Account_Status",
]

NUMERICAL_FEATURES: List[str] = [
    "Amount_NGN",
    "Age",
    "Tenure_Months",
    "Digital_Engagement_Score",
    "Transaction_Hour",
    "Transaction_DayOfWeek",
    "Is_Weekend",
    "Is_Local_Transaction",
]

# =====================================================================
# Data Quality & Dynamic Tolerance Thresholds
# =====================================================================

# Known missingness in Device_Type and Location is ~0.8%.
# Any missingness <= 5.0% issues a non-blocking warning; > 5.0% triggers a blocking error.
MAX_TOLERATED_MISSING_PCT: float = 5.0

# Supported Nigerian Cities in Synthetic Dataset
VALID_CITIES: List[str] = [
    "Lagos",
    "Kano",
    "Port Harcourt",
    "Benin City",
    "Kaduna",
    "Abuja",
    "Ibadan",
    "Enugu",
]

# =====================================================================
# Reproducibility & Model Training Parameters
# =====================================================================

RANDOM_STATE: int = 42
TEST_SPLIT_RATIO: float = 0.20
DECISION_THRESHOLD: float = 0.50
