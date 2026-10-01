"""
FinTrust Digital Bank - Data Preprocessing Module
================================================
Week 3: ML Engineering Pipeline Component (Part C)

This module implements the Preprocessing and Feature Preparation workflow for the
FinTrust Digital Bank ML Engineering pipeline.

Pipeline Stage:
Data -> Validation -> [PREPROCESSING] -> [FEATURE PREPARATION] -> Model -> Prediction -> Output

Responsibilities:
1. Relational join: Merge Transaction_Data with Customer_Data on Customer_ID.
2. Missing-value handling: Impute known missing values (Device_Type and Location).
3. Feature engineering: Extract temporal features from Transaction_DateTime
   (Transaction_Hour, Transaction_DayOfWeek, Is_Weekend) and derived location match
   (Is_Local_Transaction as binary integer 0/1).
4. Feature encoding: Encode categorical variables using OneHotEncoder with
   handle_unknown='ignore' for robust single-record and batch inference.
5. Target encoding: Map Risk_Review_Flag ('Yes' -> 1, 'No' -> 0).
6. Preprocessor serialization: Persist and restore fitted preprocessor state
   to enable reproducible real-time and API inference without refitting.
7. Export: Save model-ready dataset to data/processed/FinTrust_Modelling_Ready.csv.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Centralized configuration imports with local fallback definitions
try:
    from src.config import (
        CATEGORICAL_FEATURES,
        ID_COLUMNS,
        MODELLING_DATA_PATH,
        NUMERICAL_FEATURES,
        PREPROCESSOR_ARTIFACT_PATH,
        TARGET_COLUMN,
    )
except ImportError:
    CATEGORICAL_FEATURES = [
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
    NUMERICAL_FEATURES = [
        "Amount_NGN",
        "Age",
        "Tenure_Months",
        "Digital_Engagement_Score",
        "Transaction_Hour",
        "Transaction_DayOfWeek",
        "Is_Weekend",
        "Is_Local_Transaction",
    ]
    TARGET_COLUMN = "Risk_Review_Flag"
    ID_COLUMNS = ["Transaction_ID", "Customer_ID"]
    MODELLING_DATA_PATH = Path("data/processed/FinTrust_Modelling_Ready.csv")
    PREPROCESSOR_ARTIFACT_PATH = Path("models/preprocessor.joblib")

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Stateful preprocessor for FinTrust transaction and customer data.

    Fits encoding transformations on training data, serializes state to disk,
    and applies consistent transformations during batch or single-record inference.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.encoded_cat_columns: List[str] = []

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle known missing values in transaction data.

        - Device_Type: Impute with 'Unknown'
        - Location: Impute with customer's City if available, else 'Unknown'
        """
        df = df.copy()
        if "Device_Type" in df.columns:
            df["Device_Type"] = df["Device_Type"].fillna("Unknown")

        if "Location" in df.columns:
            if "City" in df.columns:
                df["Location"] = df["Location"].fillna(df["City"])
            df["Location"] = df["Location"].fillna("Unknown")

        return df

    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour, day of week, and weekend indicator from Transaction_DateTime."""
        df = df.copy()
        if "Transaction_DateTime" in df.columns:
            dt_series = pd.to_datetime(df["Transaction_DateTime"], errors="coerce")
            df["Transaction_Hour"] = dt_series.dt.hour.fillna(12).astype(int)
            df["Transaction_DayOfWeek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
            df["Is_Weekend"] = df["Transaction_DayOfWeek"].isin([5, 6]).astype(int)
        else:
            # Fallback defaults if datetime is missing in single-record inference
            if "Transaction_Hour" not in df.columns:
                df["Transaction_Hour"] = 12
            if "Transaction_DayOfWeek" not in df.columns:
                df["Transaction_DayOfWeek"] = 0
            if "Is_Weekend" not in df.columns:
                df["Is_Weekend"] = 0

        return df

    def prepare_raw_data(
        self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge customer and transaction tables and apply cleaning and feature extraction.

        Args:
            customer_df: FinTrust customer DataFrame.
            transaction_df: FinTrust transaction DataFrame.

        Returns:
            Merged and cleaned DataFrame.
        """
        # Relational join on Customer_ID
        merged_df = transaction_df.merge(
            customer_df, on="Customer_ID", how="left", suffixes=("", "_cust")
        )

        # Is_Local_Transaction was engineered as a candidate feature capturing whether a transaction
        # occurred in the customer's home city (Location == City).
        # CRITICAL DATA ANALYSIS NOTE:
        # The observed 12.7% match rate is statistically consistent with Location and City having been
        # generated independently in this synthetic educational dataset (expected random match rate with
        # 8 uniform cities = 12.5%). Cross-tabulation confirms a uniform ~12.5% distribution across all cities.
        # This suggests the feature may not carry genuine behavioural signal in this particular synthetic dataset,
        # and its predictive value should be validated empirically by the Data Science track rather than assumed.
        if "Location" in merged_df.columns and "City" in merged_df.columns:
            merged_df["Is_Local_Transaction"] = (
                merged_df["Location"] == merged_df["City"]
            ).astype(int)
        else:
            merged_df["Is_Local_Transaction"] = 1

        # Impute missing values
        merged_df = self._impute_missing_values(merged_df)

        # Temporal feature extraction
        merged_df = self._engineer_temporal_features(merged_df)

        return merged_df

    def fit(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame) -> DataPreprocessor:
        """
        Fit categorical encoders on merged training data.

        Args:
            customer_df: Training customer DataFrame.
            transaction_df: Training transaction DataFrame.

        Returns:
            Fitted self.
        """
        merged_df = self.prepare_raw_data(customer_df, transaction_df)

        # Fit OneHotEncoder on categorical features
        cat_data = merged_df[CATEGORICAL_FEATURES].astype(str)
        self.encoder.fit(cat_data)
        self.encoded_cat_columns = list(
            self.encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        )

        # Full feature list: numericals + encoded categoricals
        self.feature_names = NUMERICAL_FEATURES + self.encoded_cat_columns
        self.is_fitted = True
        logger.info(
            f"DataPreprocessor fitted successfully. Total feature count: {len(self.feature_names)}"
        )
        return self

    def transform(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Transform raw transaction and customer records into modeling-ready features.

        Args:
            customer_df: Customer DataFrame.
            transaction_df: Transaction DataFrame.
            include_target: Whether to extract and append the target column if present.

        Returns:
            DataFrame containing Transaction_ID, engineered features, and optional target.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform().")

        merged_df = self.prepare_raw_data(customer_df, transaction_df)

        # 1. Numerical features
        num_df = merged_df[NUMERICAL_FEATURES].copy()

        # 2. Encoded categorical features
        cat_data = merged_df[CATEGORICAL_FEATURES].astype(str)
        encoded_cat = self.encoder.transform(cat_data)
        encoded_df = pd.DataFrame(
            encoded_cat, columns=self.encoded_cat_columns, index=merged_df.index
        )

        # 3. Combine features
        features_df = pd.concat([num_df, encoded_df], axis=1)

        # 4. Attach Transaction_ID and Customer_ID for traceability
        result_df = pd.DataFrame(index=merged_df.index)
        for id_col in ID_COLUMNS:
            if id_col in merged_df.columns:
                result_df[id_col] = merged_df[id_col]

        result_df = pd.concat([result_df, features_df], axis=1)

        # 5. Attach target if requested and present
        if include_target and TARGET_COLUMN in merged_df.columns:
            target_series = merged_df[TARGET_COLUMN].dropna()
            result_df[TARGET_COLUMN] = merged_df[TARGET_COLUMN].map(
                {"Yes": 1, "No": 0, 1: 1, 0: 0}
            )

        return result_df

    def fit_transform(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Fit preprocessor and transform dataset in a single call."""
        self.fit(customer_df, transaction_df)
        return self.transform(customer_df, transaction_df, include_target=include_target)

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Serialize fitted preprocessor state using joblib.

        Args:
            file_path: Destination file path for serialization (.joblib).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted DataPreprocessor.")
        dest_path = Path(file_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(dest_path))
        logger.info(f"Fitted DataPreprocessor serialized successfully to {dest_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> DataPreprocessor:
        """
        Deserialize a fitted DataPreprocessor from disk.

        Args:
            file_path: Path to serialized .joblib preprocessor artifact.

        Returns:
            Fitted DataPreprocessor instance.
        """
        src_path = Path(file_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {src_path}")
        preprocessor = joblib.load(str(src_path))
        if not isinstance(preprocessor, cls):
            raise TypeError(
                f"Loaded artifact is type {type(preprocessor)}, expected {cls.__name__}."
            )
        if not preprocessor.is_fitted:
            raise ValueError("Loaded DataPreprocessor artifact is not fitted.")
        logger.info(f"Loaded fitted DataPreprocessor from {src_path}")
        return preprocessor


# =====================================================================
# Functional Pipeline Runner
# =====================================================================

def preprocess_pipeline(
    customer_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "data/processed/FinTrust_Modelling_Ready.csv",
    save_preprocessor_path: Optional[Union[str, Path]] = "models/preprocessor.joblib",
) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Execute full preprocessing pipeline, save modeling-ready dataset, and serialize preprocessor.

    Args:
        customer_df: Customer DataFrame.
        transaction_df: Transaction DataFrame.
        output_path: Optional file path to save the modeling-ready CSV.
        save_preprocessor_path: Optional file path to serialize the fitted DataPreprocessor.

    Returns:
        Tuple of (model_ready_df, fitted_preprocessor).
    """
    preprocessor = DataPreprocessor()
    model_ready_df = preprocessor.fit_transform(customer_df, transaction_df)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        model_ready_df.to_csv(str(out_p), index=False)
        logger.info(
            f"Saved modeling-ready dataset ({model_ready_df.shape[0]} rows, "
            f"{model_ready_df.shape[1]} columns) to {out_p}"
        )

    if save_preprocessor_path:
        try:
            preprocessor.save(save_preprocessor_path)
        except Exception as e:
            logger.warning(f"Could not persist preprocessor to {save_preprocessor_path}: {e}")

    # Sanity check log on Is_Local_Transaction
    if "Is_Local_Transaction" in model_ready_df.columns:
        non_local_cnt = int((model_ready_df["Is_Local_Transaction"] == 0).sum())
        non_local_pct = (
            (non_local_cnt / len(model_ready_df)) * 100.0 if len(model_ready_df) > 0 else 0.0
        )
        logger.info(
            f"Sanity Check: {non_local_cnt:,} / {len(model_ready_df):,} rows ({non_local_pct:.1f}%) "
            f"have Is_Local_Transaction = 0 (out-of-town transactions)."
        )

    return model_ready_df, preprocessor


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    cust_path = os.path.join(raw_dir, "FinTrust_Customer_Data.csv")
    tx_path = os.path.join(raw_dir, "FinTrust_Transaction_Data.csv")

    if not os.path.exists(cust_path) or not os.path.exists(tx_path):
        print(f"ERROR: Raw CSV files missing from {raw_dir}")
        sys.exit(1)

    print("Loading raw datasets...")
    c_df = pd.read_csv(cust_path)
    t_df = pd.read_csv(tx_path)

    print(f"Running preprocessing on {len(t_df):,} transactions and {len(c_df):,} customers...")
    processed_df, _ = preprocess_pipeline(c_df, t_df)
    print("Preprocessing completed successfully!")
    print(f"Modelling dataset shape: {processed_df.shape}")
    print(f"Target distribution:\n{processed_df[TARGET_COLUMN].value_counts(normalize=True)}")
    if "Is_Local_Transaction" in processed_df.columns:
        non_local_count = int((processed_df["Is_Local_Transaction"] == 0).sum())
        non_local_pct = (non_local_count / len(processed_df)) * 100.0
        print(
            f"Is_Local_Transaction check: {non_local_count:,} / {len(processed_df):,} rows "
            f"({non_local_pct:.1f}%) have Is_Local_Transaction = 0."
        )
