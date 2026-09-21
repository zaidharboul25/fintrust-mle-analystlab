"""
Unit tests for FinTrust Model & Prediction Pipeline (src/prediction.py).

Verifies model training, artifact serialization/deserialization, batch prediction,
single-record inference, output formatting, and end-to-end pipeline orchestration.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from src.preprocessing import DataPreprocessor, preprocess_pipeline
from src.prediction import (
    train_baseline_model,
    load_model_artifact,
    predict,
    format_and_save_predictions,
    run_pipeline,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_data():
    """Minimal customer and transaction data for testing."""
    cust_df = pd.DataFrame({
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

    # 10 synthetic transactions for testing
    tx_rows = []
    for i in range(20):
        tx_rows.append({
            "Transaction_ID": f"FT-T{i+1:06d}",
            "Customer_ID": f"FT-C0000{(i % 3) + 1}",
            "Transaction_DateTime": "2024-01-15 10:30:00",
            "Transaction_Type": "Transfer" if i % 2 == 0 else "Card Purchase",
            "Amount_NGN": float(1000 * (i + 1)),
            "Channel": "Mobile App" if i % 2 == 0 else "POS",
            "Device_Type": "Android" if i > 0 else np.nan,  # 1 missing value
            "Location": "Lagos" if i > 0 else np.nan,       # 1 missing value
            "International_Transaction": "No",
            "Transaction_Status": "Successful",
            "Risk_Review_Flag": "Yes" if i % 4 == 0 else "No",  # ~25% Yes
        })
    tx_df = pd.DataFrame(tx_rows)
    return cust_df, tx_df


# =====================================================================
# Preprocessing Tests
# =====================================================================

def test_preprocessing_pipeline(sample_data):
    """Test that preprocessing merges data, handles missing values, and produces features."""
    cust_df, tx_df = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "model_ready.csv")
        processed_df, preprocessor = preprocess_pipeline(cust_df, tx_df, output_path=out_csv)

        assert os.path.exists(out_csv)
        assert len(processed_df) == len(tx_df)
        assert "Transaction_ID" in processed_df.columns
        assert "Risk_Review_Flag" in processed_df.columns
        # Target must be binary integer 0 or 1
        assert set(processed_df["Risk_Review_Flag"].unique()).issubset({0, 1})
        # Check no NaN values in features
        feature_cols = [c for c in processed_df.columns if c not in ["Transaction_ID", "Customer_ID"]]
        assert processed_df[feature_cols].isna().sum().sum() == 0


# =====================================================================
# Model Training & Artifact Serialization Tests
# =====================================================================

def test_train_baseline_model(sample_data):
    """Test baseline model training, metrics calculation, and artifact creation."""
    cust_df, tx_df = sample_data
    processed_df, _ = preprocess_pipeline(cust_df, tx_df, output_path=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact, metrics = train_baseline_model(
            processed_df,
            model_version="test_v1",
            test_size=0.25,
            save_dir=tmpdir,
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

        # Verify artifact structure
        assert artifact["model_version"] == "test_v1"
        assert "model" in artifact
        assert "feature_names" in artifact
        assert len(artifact["feature_names"]) > 0

        # Verify joblib file existence and reloading
        saved_file = os.path.join(tmpdir, "test_v1.joblib")
        assert os.path.exists(saved_file)

        loaded = load_model_artifact(saved_file)
        assert loaded["model_version"] == "test_v1"


# =====================================================================
# Prediction Tests (Batch & Single-Record)
# =====================================================================

def test_batch_prediction(sample_data):
    """Test batch prediction produces expected decisions and confidence probabilities."""
    cust_df, tx_df = sample_data
    processed_df, _ = preprocess_pipeline(cust_df, tx_df, output_path=None)

    artifact, _ = train_baseline_model(processed_df, model_version="batch_test", save_dir=None)
    preds = predict(artifact, processed_df)

    assert len(preds) == len(processed_df)
    assert set(preds["prediction_decision"].unique()).issubset({"Yes", "No"})
    assert preds["prediction_confidence"].between(0.0, 1.0).all()
    assert set(preds["predicted_class"].unique()).issubset({0, 1})


def test_single_record_prediction(sample_data):
    """Test single-record inference simulating a real-time API or service endpoint."""
    cust_df, tx_df = sample_data
    processed_df, _ = preprocess_pipeline(cust_df, tx_df, output_path=None)

    artifact, _ = train_baseline_model(processed_df, model_version="single_test", save_dir=None)

    # Extract exactly 1 row
    single_record = processed_df.iloc[[0]]
    single_pred = predict(artifact, single_record)

    assert len(single_pred) == 1
    assert single_pred["prediction_decision"].iloc[0] in ["Yes", "No"]
    assert 0.0 <= single_pred["prediction_confidence"].iloc[0] <= 1.0


# =====================================================================
# Output Formatting Tests
# =====================================================================

def test_format_and_save_predictions(sample_data):
    """Test formatting output predictions with required schema."""
    cust_df, tx_df = sample_data
    processed_df, _ = preprocess_pipeline(cust_df, tx_df, output_path=None)
    artifact, _ = train_baseline_model(processed_df, model_version="fmt_test", save_dir=None)
    preds = predict(artifact, processed_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "predictions.csv")
        formatted_df = format_and_save_predictions(
            preds, source_df=processed_df, model_version="fmt_test", output_path=out_csv
        )

        assert os.path.exists(out_csv)
        assert list(formatted_df.columns) == [
            "Transaction_ID",
            "predicted_Risk_Review_Flag",
            "prediction_confidence",
            "prediction_timestamp",
            "model_version",
        ]
        assert len(formatted_df) == len(tx_df)
        assert formatted_df["Transaction_ID"].iloc[0] == tx_df["Transaction_ID"].iloc[0]


# =====================================================================
# End-to-End Orchestration Tests
# =====================================================================

def test_run_pipeline_end_to_end():
    """Test run_pipeline end-to-end with actual raw project files."""
    result = run_pipeline(
        raw_data_dir="data/raw",
        processed_dir="data/processed",
        model_dir="models",
        model_version="baseline_v1",
        retrain_model=True,
    )

    assert result["status"] == "SUCCESS"
    assert result["total_predictions"] == 12000
    assert result["flagged_yes"] + result["flagged_no"] == 12000
    assert os.path.exists(result["output_path"])
    assert os.path.exists(result["model_path"])


def test_run_pipeline_halts_on_blocking_validation_error():
    """Test that run_pipeline stops if input data contains blocking errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an empty customer CSV (blocking error: 0 rows)
        empty_cust = pd.DataFrame(columns=["Customer_ID", "Customer_Name", "Age"])
        cust_path = os.path.join(tmpdir, "FinTrust_Customer_Data.csv")
        empty_cust.to_csv(cust_path, index=False)

        # Copy valid transaction data
        tx_df = pd.read_csv("data/raw/FinTrust_Transaction_Data.csv")
        tx_path = os.path.join(tmpdir, "FinTrust_Transaction_Data.csv")
        tx_df.to_csv(tx_path, index=False)

        result = run_pipeline(
            raw_data_dir=tmpdir,
            processed_dir=tmpdir,
            model_dir=tmpdir,
            model_version="test_fail",
        )

        assert result["status"] == "FAILED_VALIDATION"
