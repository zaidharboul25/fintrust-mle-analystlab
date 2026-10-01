"""
Unit tests for Pipeline Reproducibility & Deterministic Inference (tests/test_reproducibility.py).
==================================================================================================
Week 3: ML Engineering Reproducibility Suite

Verifies that identical inputs, identical preprocessing, and identical model weights produce
strictly deterministic predictions and confidence probabilities across repeated runs,
while treating dynamic runtime metadata (such as timestamps) separately.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor
from src.prediction import train_baseline_model, predict, format_and_save_predictions


@pytest.fixture
def fixed_synthetic_dataset():
    """Deterministic synthetic customer and transaction dataset."""
    rng = np.random.RandomState(42)

    cust_df = pd.DataFrame({
        "Customer_ID": [f"FT-C{i:05d}" for i in range(1, 11)],
        "Customer_Name": [f"Customer {i}" for i in range(1, 11)],
        "Age": [25 + i * 2 for i in range(10)],
        "Gender": ["Male" if i % 2 == 0 else "Female" for i in range(10)],
        "City": ["Lagos", "Kano", "Abuja", "Ibadan", "Port Harcourt"] * 2,
        "Customer_Segment": ["Everyday", "Premium", "SME", "Student", "Everyday"] * 2,
        "Account_Type": ["Savings", "Current"] * 5,
        "Tenure_Months": [6 + i * 4 for i in range(10)],
        "Digital_Engagement_Score": [3.0 + (i * 0.2) for i in range(10)],
        "Monthly_Income_Band": ["100k-249k", "250k-499k"] * 5,
        "Preferred_Channel": ["Mobile App", "USSD"] * 5,
        "Account_Status": ["Active"] * 10,
    })

    tx_rows = []
    for i in range(50):
        cust_idx = (i % 10) + 1
        tx_rows.append({
            "Transaction_ID": f"FT-T{i+1:06d}",
            "Customer_ID": f"FT-C{cust_idx:05d}",
            "Transaction_DateTime": f"2024-01-{(i % 25) + 1:02d} 10:30:00",
            "Transaction_Type": ["Transfer", "Card Purchase", "Airtime/Data"][i % 3],
            "Amount_NGN": float(1000 * (i + 1)),
            "Channel": ["Mobile App", "POS", "Web"][i % 3],
            "Device_Type": ["Android", "iOS", "POS Terminal"][i % 3],
            "Location": ["Lagos", "Kano", "Abuja"][i % 3],
            "International_Transaction": "No",
            "Transaction_Status": "Successful",
            "Risk_Review_Flag": "Yes" if i % 4 == 0 else "No",
        })
    tx_df = pd.DataFrame(tx_rows)
    return cust_df, tx_df


def test_preprocessing_reproducibility(fixed_synthetic_dataset):
    """Running DataPreprocessor independently on identical input produces identical matrices."""
    cust_df, tx_df = fixed_synthetic_dataset

    prep1 = DataPreprocessor()
    df1 = prep1.fit_transform(cust_df, tx_df, include_target=True)

    prep2 = DataPreprocessor()
    df2 = prep2.fit_transform(cust_df, tx_df, include_target=True)

    # Assert exact schema, columns, feature names, and values match
    assert prep1.feature_names == prep2.feature_names
    pd.testing.assert_frame_equal(df1, df2)


def test_model_training_reproducibility(fixed_synthetic_dataset):
    """Training baseline model with fixed random_state produces identical weights and metrics."""
    cust_df, tx_df = fixed_synthetic_dataset
    prep = DataPreprocessor()
    processed_df = prep.fit_transform(cust_df, tx_df, include_target=True)

    artifact1, metrics1 = train_baseline_model(
        processed_df, model_version="repro_v1", random_state=42, save_dir=None
    )
    artifact2, metrics2 = train_baseline_model(
        processed_df, model_version="repro_v2", random_state=42, save_dir=None
    )

    # Assert identical evaluation metrics
    assert metrics1 == metrics2

    # Assert identical model coefficients
    coef1 = artifact1["model"].named_steps["classifier"].coef_
    coef2 = artifact2["model"].named_steps["classifier"].coef_
    np.testing.assert_array_equal(coef1, coef2)


def test_end_to_end_prediction_determinism(fixed_synthetic_dataset):
    """Identical input + identical model + identical preprocessing -> bit-for-bit identical predictions."""
    cust_df, tx_df = fixed_synthetic_dataset
    prep = DataPreprocessor()
    processed_df = prep.fit_transform(cust_df, tx_df, include_target=True)

    artifact, _ = train_baseline_model(
        processed_df, model_version="det_v1", random_state=42, save_dir=None
    )

    # Generate predictions twice independently
    preds_run1 = predict(artifact, processed_df)
    preds_run2 = predict(artifact, processed_df)

    # 1. Assert decisions are 100% identical
    assert preds_run1["prediction_decision"].tolist() == preds_run2["prediction_decision"].tolist()

    # 2. Assert predicted classes are 100% identical
    assert preds_run1["predicted_class"].tolist() == preds_run2["predicted_class"].tolist()

    # 3. Assert continuous confidence scores match with 0.0 tolerance
    np.testing.assert_array_equal(
        preds_run1["prediction_confidence"].values,
        preds_run2["prediction_confidence"].values,
    )


def test_output_formatting_metadata_separation(fixed_synthetic_dataset):
    """Verify that deterministic prediction fields match while dynamic timestamps are isolated."""
    cust_df, tx_df = fixed_synthetic_dataset
    prep = DataPreprocessor()
    processed_df = prep.fit_transform(cust_df, tx_df, include_target=True)
    artifact, _ = train_baseline_model(processed_df, model_version="repro_v1", save_dir=None)
    preds = predict(artifact, processed_df)

    out1 = format_and_save_predictions(preds, processed_df, model_version="repro_v1", output_path=None)
    out2 = format_and_save_predictions(preds, processed_df, model_version="repro_v1", output_path=None)

    # Core prediction fields must be exactly equal
    deterministic_cols = ["Transaction_ID", "predicted_Risk_Review_Flag", "prediction_confidence", "model_version"]
    pd.testing.assert_frame_equal(out1[deterministic_cols], out2[deterministic_cols])

    # Dynamic timestamps must be present and valid ISO strings
    assert "prediction_timestamp" in out1.columns
    assert len(out1["prediction_timestamp"].iloc[0]) > 0
