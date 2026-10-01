"""
Unit tests for FinTrust Data Preprocessing Component (src/preprocessing.py).
===========================================================================
Week 3: ML Engineering Test Suite Expansion

Validates missing-value imputation, temporal feature extraction, categorical
encoding with unseen categories, derived signal logic, and preprocessor state
serialization/deserialization.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor, preprocess_pipeline


@pytest.fixture
def sample_raw_data():
    """Return minimal raw customer and transaction DataFrames for testing."""
    cust_df = pd.DataFrame({
        "Customer_ID": ["FT-C00001", "FT-C00002"],
        "Customer_Name": ["Ibrahim Adeyemi", "Yusuf Mohammed"],
        "Age": [34, 45],
        "Gender": ["Male", "Male"],
        "City": ["Lagos", "Kano"],
        "Customer_Segment": ["Premium", "Everyday"],
        "Account_Type": ["Savings", "Current"],
        "Tenure_Months": [12, 36],
        "Digital_Engagement_Score": [4.2, 3.8],
        "Monthly_Income_Band": ["500k-999k", "100k-249k"],
        "Preferred_Channel": ["Mobile App", "USSD"],
        "Account_Status": ["Active", "Active"],
    })

    tx_df = pd.DataFrame({
        "Transaction_ID": ["FT-T000001", "FT-T000002", "FT-T000003"],
        "Customer_ID": ["FT-C00001", "FT-C00002", "FT-C00001"],
        "Transaction_DateTime": [
            "2024-01-13 14:30:00",  # Saturday (Weekend)
            "2024-01-15 09:15:00",  # Monday (Weekday)
            "2024-01-16 23:45:00",  # Tuesday night
        ],
        "Transaction_Type": ["Transfer", "Card Purchase", "Airtime/Data"],
        "Amount_NGN": [15000.0, 3200.5, 500.0],
        "Channel": ["Mobile App", "POS", "Mobile App"],
        "Device_Type": ["Android", np.nan, "iOS"],             # Known missing value
        "Location": ["Lagos", "Abuja", np.nan],                # Known missing value
        "International_Transaction": ["No", "No", "No"],
        "Transaction_Status": ["Successful", "Successful", "Successful"],
        "Risk_Review_Flag": ["No", "Yes", "No"],
    })
    return cust_df, tx_df


def test_missing_value_imputation(sample_raw_data):
    """Verify that Device_Type and Location are imputed properly."""
    cust_df, tx_df = sample_raw_data
    preprocessor = DataPreprocessor()
    merged = preprocessor.prepare_raw_data(cust_df, tx_df)

    # Device_Type: NaN imputed with 'Unknown'
    assert merged.loc[1, "Device_Type"] == "Unknown"

    # Location: NaN imputed with customer's home City ('Lagos' for FT-C00001)
    assert merged.loc[2, "Location"] == "Lagos"
    assert merged["Device_Type"].isna().sum() == 0
    assert merged["Location"].isna().sum() == 0


def test_temporal_feature_extraction(sample_raw_data):
    """Verify hour, day of week, and weekend indicator calculations."""
    cust_df, tx_df = sample_raw_data
    preprocessor = DataPreprocessor()
    merged = preprocessor.prepare_raw_data(cust_df, tx_df)

    # Row 0: 2024-01-13 14:30:00 -> Saturday -> Hour=14, DayOfWeek=5, Is_Weekend=1
    assert merged.loc[0, "Transaction_Hour"] == 14
    assert merged.loc[0, "Transaction_DayOfWeek"] == 5
    assert merged.loc[0, "Is_Weekend"] == 1

    # Row 1: 2024-01-15 09:15:00 -> Monday -> Hour=9, DayOfWeek=0, Is_Weekend=0
    assert merged.loc[1, "Transaction_Hour"] == 9
    assert merged.loc[1, "Transaction_DayOfWeek"] == 0
    assert merged.loc[1, "Is_Weekend"] == 0


def test_is_local_transaction_derivation_and_type(sample_raw_data):
    """Verify Is_Local_Transaction derivation and integer data type."""
    cust_df, tx_df = sample_raw_data
    preprocessor = DataPreprocessor()
    merged = preprocessor.prepare_raw_data(cust_df, tx_df)

    # Row 0: Location='Lagos', Customer City='Lagos' -> Match -> 1
    assert merged.loc[0, "Is_Local_Transaction"] == 1
    # Row 1: Location='Abuja', Customer City='Kano' -> Mismatch -> 0
    assert merged.loc[1, "Is_Local_Transaction"] == 0
    # Must be integer dtype (0 or 1), not raw Python bool
    assert merged["Is_Local_Transaction"].dtype in [np.int64, np.int32, int]


def test_onehot_encoding_unseen_category(sample_raw_data):
    """Verify that unseen categories during inference are ignored without raising errors."""
    cust_df, tx_df = sample_raw_data
    preprocessor = DataPreprocessor()
    preprocessor.fit(cust_df, tx_df)

    # Create inference record with completely unseen category
    infer_tx = tx_df.iloc[[0]].copy()
    infer_tx["Transaction_Type"] = "Novel_Cryptocurrency_Swap"  # Unseen category
    infer_cust = cust_df.iloc[[0]].copy()

    # Transform should succeed without error due to handle_unknown='ignore'
    result_df = preprocessor.transform(infer_cust, infer_tx, include_target=False)

    assert len(result_df) == 1
    assert "Transaction_Type_Novel_Cryptocurrency_Swap" not in result_df.columns
    # Check that all features are non-null
    feature_cols = [c for c in result_df.columns if c not in ["Transaction_ID", "Customer_ID"]]
    assert result_df[feature_cols].isna().sum().sum() == 0


def test_preprocessor_serialization(sample_raw_data):
    """Verify that saving and reloading DataPreprocessor preserves fitted state and feature names."""
    cust_df, tx_df = sample_raw_data
    preprocessor = DataPreprocessor()
    preprocessor.fit(cust_df, tx_df)
    original_features = list(preprocessor.feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "preprocessor.joblib")
        preprocessor.save(save_path)
        assert os.path.exists(save_path)

        reloaded = DataPreprocessor.load(save_path)
        assert reloaded.is_fitted
        assert reloaded.feature_names == original_features

        # Verify transform on reloaded produces identical DataFrame
        df_orig = preprocessor.transform(cust_df, tx_df)
        df_reloaded = reloaded.transform(cust_df, tx_df)
        pd.testing.assert_frame_equal(df_orig, df_reloaded)


def test_unfitted_preprocessor_raises():
    """Verify that calling transform or save on an unfitted preprocessor raises RuntimeError."""
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({"col": [1]})

    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform(df, df)

    with pytest.raises(RuntimeError, match="Cannot save"):
        preprocessor.save("dummy_path.joblib")
