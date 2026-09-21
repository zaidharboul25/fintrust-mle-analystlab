"""
Unit tests for FinTrust Data Validation Module (src/validation.py).

Verifies all schema rules, primary key constraints, type checks, regex patterns,
categorical restrictions, dynamic missing value thresholds, and cross-dataset
referential integrity checks.
"""

import pytest
import pandas as pd
import numpy as np

from src.validation import (
    DataValidator,
    DatasetSchema,
    ColumnSchema,
    CUSTOMER_SCHEMA,
    TRANSACTION_SCHEMA,
    EmptyDatasetError,
    ReferentialIntegrityError,
    SchemaViolationError,
    check_referential_integrity,
    validate_customer_data,
    validate_transaction_data,
    validate_fintrust_pipeline,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Return a minimal valid Customer DataFrame."""
    return pd.DataFrame({
        "Customer_ID": ["FT-C00001", "FT-C00002", "FT-C00003"],
        "Customer_Name": ["Ibrahim Adeyemi", "Yusuf Mohammed", "Emeka Adeyemi"],
        "Age": [34, 45, 29],
        "Gender": ["Male", "Male", "Female"],
        "City": ["Lagos", "Kano", "Abuja"],
        "Customer_Segment": ["Premium", "Everyday", "Student"],
        "Account_Type": ["Savings", "Current", "Savings"],
        "Tenure_Months": [12, 36, 6],
        "Digital_Engagement_Score": [4.2, 3.8, 4.9],
        "Monthly_Income_Band": ["500k-999k", "100k-249k", "Below 100k"],
        "Preferred_Channel": ["Mobile App", "USSD", "Web"],
        "Account_Status": ["Active", "Active", "Dormant"],
    })


@pytest.fixture
def sample_transaction_df() -> pd.DataFrame:
    """Return a minimal valid Transaction DataFrame."""
    return pd.DataFrame({
        "Transaction_ID": ["FT-T000001", "FT-T000002", "FT-T000003"],
        "Customer_ID": ["FT-C00001", "FT-C00002", "FT-C00001"],
        "Transaction_DateTime": [
            "2024-01-15 10:30:00",
            "2024-01-15 11:15:00",
            "2024-01-16 09:00:00",
        ],
        "Transaction_Type": ["Transfer", "Card Purchase", "Airtime/Data"],
        "Amount_NGN": [15000.0, 3200.5, 500.0],
        "Channel": ["Mobile App", "POS", "Mobile App"],
        "Device_Type": ["Android", "POS Terminal", "iOS"],
        "Location": ["Lagos", "Kano", "Lagos"],
        "International_Transaction": ["No", "No", "No"],
        "Transaction_Status": ["Successful", "Successful", "Successful"],
        "Risk_Review_Flag": ["No", "No", "No"],
    })


# =====================================================================
# Requirement 4: Empty Dataset Handling
# =====================================================================

def test_empty_dataset_returns_invalid_report():
    """An empty DataFrame (0 rows) must produce is_valid=False and an empty_dataset error."""
    validator = DataValidator.for_customer_data()
    empty_df = pd.DataFrame(columns=list(CUSTOMER_SCHEMA.columns.keys()))
    
    report = validator.validate(empty_df, raise_on_error=False)
    
    assert not report.is_valid
    assert report.error_count == 1
    assert report.errors[0].check == "empty_dataset"


def test_empty_dataset_raises_when_requested():
    """An empty DataFrame must raise EmptyDatasetError when raise_on_error=True."""
    validator = DataValidator.for_customer_data()
    empty_df = pd.DataFrame(columns=list(CUSTOMER_SCHEMA.columns.keys()))
    
    with pytest.raises(EmptyDatasetError):
        validator.validate(empty_df, raise_on_error=True)


# =====================================================================
# Requirement 6: Column Presence & Schema Drift
# =====================================================================

def test_missing_required_column(sample_customer_df):
    """Missing required columns must flag a blocking error."""
    df = sample_customer_df.drop(columns=["Customer_Name", "Age"])
    report = validate_customer_data(df)

    assert not report.is_valid
    err = [e for e in report.errors if e.check == "missing_columns"]
    assert len(err) == 1
    assert "Customer_Name" in err[0].message
    assert "Age" in err[0].message


def test_unexpected_extra_columns_trigger_warning(sample_customer_df):
    """Unexpected extra columns should trigger a warning, not an error, when allowed."""
    df = sample_customer_df.copy()
    df["Extra_Metadata_Field"] = "test_value"
    report = validate_customer_data(df)

    # Extra columns should be a warning, not blocking overall pass
    assert report.is_valid
    warns = [w for w in report.warnings if w.check == "unexpected_columns"]
    assert len(warns) == 1
    assert "Extra_Metadata_Field" in warns[0].message


# =====================================================================
# Requirement 5: Duplicate Primary Keys
# =====================================================================

def test_duplicate_customer_id(sample_customer_df):
    """Duplicate Customer_ID in customer data must flag a blocking error."""
    df = sample_customer_df.copy()
    df.loc[1, "Customer_ID"] = "FT-C00001"  # Duplicate
    report = validate_customer_data(df)

    assert not report.is_valid
    dup_errs = [e for e in report.errors if e.check == "duplicate_primary_key"]
    assert len(dup_errs) == 1
    assert "Customer_ID" in dup_errs[0].column


def test_duplicate_transaction_id(sample_transaction_df):
    """Duplicate Transaction_ID in transaction data must flag a blocking error."""
    df = sample_transaction_df.copy()
    df.loc[1, "Transaction_ID"] = "FT-T000001"  # Duplicate
    report = validate_transaction_data(df)

    assert not report.is_valid
    dup_errs = [e for e in report.errors if e.check == "duplicate_primary_key"]
    assert len(dup_errs) == 1


# =====================================================================
# Requirement 2: Data Types and Regex Formatting
# =====================================================================

def test_invalid_id_regex_pattern(sample_customer_df):
    """IDs that violate regex pattern FT-C##### must be flagged as blocking errors."""
    df = sample_customer_df.copy()
    df.loc[0, "Customer_ID"] = "INVALID-123"
    report = validate_customer_data(df)

    assert not report.is_valid
    pattern_errs = [e for e in report.errors if e.check == "regex_format_mismatch"]
    assert len(pattern_errs) == 1


def test_invalid_numeric_type(sample_transaction_df):
    """Non-numeric values in Amount_NGN must flag a blocking error."""
    df = sample_transaction_df.copy()
    df["Amount_NGN"] = df["Amount_NGN"].astype(object)
    df.loc[0, "Amount_NGN"] = "fifty-thousand"
    report = validate_transaction_data(df)

    assert not report.is_valid
    type_errs = [e for e in report.errors if e.check == "invalid_data_type" and e.column == "Amount_NGN"]
    assert len(type_errs) == 1


def test_invalid_integer_type(sample_customer_df):
    """Floating point decimal values in integer columns (Age) must flag an error."""
    df = sample_customer_df.copy()
    df["Age"] = df["Age"].astype(float)
    df.loc[0, "Age"] = 34.7
    report = validate_customer_data(df)

    assert not report.is_valid
    type_errs = [e for e in report.errors if e.check == "invalid_data_type" and e.column == "Age"]
    assert len(type_errs) == 1


def test_invalid_datetime_type(sample_transaction_df):
    """Unparseable datetime strings must flag a blocking error."""
    df = sample_transaction_df.copy()
    df.loc[0, "Transaction_DateTime"] = "yesterday afternoon"
    report = validate_transaction_data(df)

    assert not report.is_valid
    dt_errs = [e for e in report.errors if e.check == "invalid_data_type" and e.column == "Transaction_DateTime"]
    assert len(dt_errs) == 1


# =====================================================================
# Requirement 3: Categorical Column Validation
# =====================================================================

def test_unexpected_categorical_value(sample_transaction_df):
    """Unexpected categories in Transaction_Status or Risk_Review_Flag must trigger errors."""
    df = sample_transaction_df.copy()
    df.loc[0, "Transaction_Status"] = "Under Investigation"  # Unknown status
    df.loc[1, "Risk_Review_Flag"] = "Maybe"  # Unknown flag

    report = validate_transaction_data(df)
    assert not report.is_valid

    cat_errs = [e for e in report.errors if e.check == "unexpected_categorical_value"]
    assert len(cat_errs) == 2
    cols = {e.column for e in cat_errs}
    assert cols == {"Transaction_Status", "Risk_Review_Flag"}


# =====================================================================
# Requirement 1 & 8: Dynamic Missing Value Thresholds
# =====================================================================

def test_strict_column_null_fails(sample_customer_df):
    """Missing value in a strictly non-null column must trigger a blocking error."""
    df = sample_customer_df.copy()
    df.loc[0, "Customer_Name"] = np.nan
    report = validate_customer_data(df)

    assert not report.is_valid
    strict_errs = [e for e in report.errors if e.check == "missing_values_strict"]
    assert len(strict_errs) == 1
    assert strict_errs[0].column == "Customer_Name"


def test_tolerated_missing_values_under_threshold(sample_transaction_df):
    """Missing values in Device_Type under threshold (5%) trigger a warning, NOT error."""
    # 1 out of 3 is 33%, which would exceed 5%. Let's create a 100-row dataframe with 1 missing (1%).
    rows = []
    for i in range(100):
        row = sample_transaction_df.iloc[0].to_dict()
        row["Transaction_ID"] = f"FT-T{i+1:06d}"
        if i == 0:
            row["Device_Type"] = np.nan  # 1% missing (under 5% threshold)
        rows.append(row)
    large_df = pd.DataFrame(rows)

    report = validate_transaction_data(large_df)

    assert report.is_valid  # Still valid!
    warns = [w for w in report.warnings if w.check == "missing_values_tolerated" and w.column == "Device_Type"]
    assert len(warns) == 1


def test_missing_values_exceeding_threshold_fails(sample_transaction_df):
    """Missing values in Device_Type exceeding 5% threshold must trigger a blocking error."""
    # 1 out of 3 is 33.3% missing (> 5% threshold)
    df = sample_transaction_df.copy()
    df.loc[0, "Device_Type"] = np.nan
    report = validate_transaction_data(df)

    assert not report.is_valid
    thresh_errs = [e for e in report.errors if e.check == "missing_values_threshold_exceeded"]
    assert len(thresh_errs) == 1
    assert thresh_errs[0].column == "Device_Type"


# =====================================================================
# Referential Integrity Verification
# =====================================================================

def test_referential_integrity_passes(sample_customer_df, sample_transaction_df):
    """All customer IDs in transactions exist in customer data."""
    report = check_referential_integrity(sample_customer_df, sample_transaction_df)

    assert report.is_valid
    assert report.error_count == 0
    assert report.check_details["orphan_customer_count"] == 0


def test_referential_integrity_fails_on_orphan_transactions(sample_customer_df, sample_transaction_df):
    """Transactions referencing non-existent Customer_IDs must trigger a blocking error."""
    df_tx = sample_transaction_df.copy()
    # Replace Customer_ID with non-existent customer
    df_tx.loc[0, "Customer_ID"] = "FT-C99999"

    report = check_referential_integrity(sample_customer_df, df_tx, raise_on_error=False)

    assert not report.is_valid
    assert report.error_count == 1
    err = report.errors[0]
    assert err.check == "referential_integrity_orphan_records"
    assert "FT-C99999" in str(err.details["sample_orphans"])


def test_referential_integrity_raises_on_error(sample_customer_df, sample_transaction_df):
    """check_referential_integrity raises ReferentialIntegrityError when requested."""
    df_tx = sample_transaction_df.copy()
    df_tx.loc[0, "Customer_ID"] = "FT-C99999"

    with pytest.raises(ReferentialIntegrityError):
        check_referential_integrity(sample_customer_df, df_tx, raise_on_error=True)


def test_referential_integrity_missing_key(sample_customer_df, sample_transaction_df):
    """Missing join key column flags an error."""
    df_cust = sample_customer_df.drop(columns=["Customer_ID"])
    report = check_referential_integrity(df_cust, sample_transaction_df)

    assert not report.is_valid
    assert report.errors[0].check == "missing_join_key"


# =====================================================================
# End-to-End Pipeline Validation on Actual Project CSV Files
# =====================================================================

def test_real_project_data_validation():
    """Verify that the actual project raw files pass validation."""
    import os

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    cust_path = os.path.join(raw_dir, "FinTrust_Customer_Data.csv")
    tx_path = os.path.join(raw_dir, "FinTrust_Transaction_Data.csv")

    if not os.path.exists(cust_path) or not os.path.exists(tx_path):
        pytest.skip("Raw CSV files not found")

    cust_df = pd.read_csv(cust_path)
    tx_df = pd.read_csv(tx_path)

    results = validate_fintrust_pipeline(cust_df, tx_df, raise_on_error=False)

    # Customer Data: Must pass with 0 errors
    assert results["customer_data"].is_valid
    assert results["customer_data"].error_count == 0

    # Transaction Data: Must pass with 0 errors and exactly 2 warnings (Device_Type & Location missing 96 rows)
    assert results["transaction_data"].is_valid
    assert results["transaction_data"].error_count == 0
    assert results["transaction_data"].warning_count == 2
    warn_cols = {w.column for w in results["transaction_data"].warnings}
    assert warn_cols == {"Device_Type", "Location"}

    # Referential Integrity: All 12,000 transactions must link to existing customers
    assert results["referential_integrity"].is_valid
    assert results["referential_integrity"].error_count == 0
    assert results["referential_integrity"].check_details["orphan_customer_count"] == 0
