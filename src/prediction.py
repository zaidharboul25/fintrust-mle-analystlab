"""
FinTrust Digital Bank - Model & Prediction Pipeline Module
=========================================================
Week 2: ML Engineering Pipeline Component (Part D)

This module implements the final stages of the FinTrust Digital Bank ML Engineering pipeline:
Model Training, Prediction Generation, and Output Formatting, orchestrating the full workflow:

Data -> Validation -> Preprocessing -> Feature Preparation -> [MODEL] -> [PREDICTION] -> [OUTPUT]

IMPORTANT ARCHITECTURAL CONTEXT:
--------------------------------
In accordance with the ML Engineering track specification for Week 2:
1. Proof-of-Concept Baseline: This module trains a simple baseline classifier
   (e.g., Logistic Regression with balanced class weights) purely for technical pipeline
   demonstration and verification. It is NOT a production fraud detection model.
   Production-grade feature selection, hyperparameter tuning, and advanced modeling
   belong to the Data Science track.
2. Synthetic Target: The target field 'Risk_Review_Flag' is an educational synthetic
   label ('Yes' / 'No'), not a real banking fraud determination.
3. End-to-End Orchestration: The pipeline ensures reproducible data flow from raw files,
   through validation gating, relational preprocessing, model training/loading, and
   structured output generation.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Internal pipeline dependencies
from src.preprocessing import (
    ID_COLUMNS,
    TARGET_COLUMN,
    DataPreprocessor,
    preprocess_pipeline,
)
from src.validation import validate_fintrust_pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "baseline_v1"
DEFAULT_MODEL_DIR = "models"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_RAW_DIR = "data/raw"


# =====================================================================
# Model Training & Serialization Stage
# =====================================================================

def train_baseline_model(
    model_ready_df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    model_version: str = DEFAULT_MODEL_VERSION,
    test_size: float = 0.20,
    random_state: int = 42,
    save_dir: Optional[str] = DEFAULT_MODEL_DIR,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Train a baseline classifier for technical pipeline verification.

    NOTE: This model serves as a proof-of-concept pipeline baseline to verify that data
    flows cleanly from features to predictions and serialized artifacts. Deep iterative
    modeling and feature optimization are the domain of the Data Science track.

    Args:
        model_ready_df: Preprocessed DataFrame containing features and target column.
        target_col: Name of the binary target column (1/0 or 'Yes'/'No'). Defaults to 'Risk_Review_Flag'.
        model_version: Version identifier string for artifact tracking.
        test_size: Ratio of the dataset allocated for test evaluation (default 0.20 = 20%).
        random_state: Random seed for reproducibility.
        save_dir: Optional directory where the trained artifact bundle will be saved.

    Returns:
        Tuple of (model_artifact_bundle, sanity_metrics_dict).
    """
    if target_col not in model_ready_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input DataFrame.")

    # 1. Separate features (X) and target (y)
    feature_cols = [
        col for col in model_ready_df.columns
        if col not in ID_COLUMNS and col != target_col
    ]
    X = model_ready_df[feature_cols].copy()
    y = model_ready_df[target_col].copy()

    # Ensure y is integer (1 for 'Yes'/1, 0 for 'No'/0)
    if y.dtype == object or isinstance(y.iloc[0], str):
        y = y.map({"Yes": 1, "No": 0})

    # 2. Stratified Train/Test Split
    # Stratification is critical due to the known ~80/20 class imbalance in Risk_Review_Flag
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"Train/Test split ({int((1-test_size)*100)}/{int(test_size*100)}) completed: "
        f"Train={len(X_train):,} rows, Test={len(X_test):,} rows (stratified on {target_col})."
    )

    # 3. Fit Simple Baseline Pipeline (StandardScaler + LogisticRegression with class_weight='balanced')
    # Logistic Regression is used here as a clean, transparent baseline classifier.
    # StandardScaler handles numerical scaling (e.g. Amount_NGN vs binary one-hot features) cleanly.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    # 4. Pipeline Sanity Check Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    sanity_metrics: Dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    _print_sanity_check_report(sanity_metrics, model_version)

    # 5. Construct Versioned Model Artifact Bundle
    artifact_bundle: Dict[str, Any] = {
        "model": model,
        "feature_names": feature_cols,
        "model_version": model_version,
        "target_mapping": {1: "Yes", 0: "No"},
        "sanity_metrics": sanity_metrics,
        "trained_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_ratio": f"{int((1-test_size)*100)}/{int(test_size*100)} (stratified)",
    }

    # 6. Save Artifact using joblib
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        artifact_path = os.path.join(save_dir, f"{model_version}.joblib")
        joblib.dump(artifact_bundle, artifact_path)
        logger.info(f"Model artifact bundle successfully saved to {artifact_path}")

    return artifact_bundle, sanity_metrics


def load_model_artifact(artifact_path: str) -> Dict[str, Any]:
    """
    Load a saved model artifact bundle from disk.

    Args:
        artifact_path: Path to the .joblib artifact file.

    Returns:
        The deserialized model artifact dictionary.
    """
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")
    artifact = joblib.load(artifact_path)
    logger.info(f"Loaded model artifact from {artifact_path}")
    return artifact


def _print_sanity_check_report(metrics: Dict[str, float], model_version: str) -> None:
    """Print a clean diagnostic summary of the baseline model sanity check."""
    sep = "-" * 65
    print("\n" + sep)
    print(f"PIPELINE SANITY CHECK METRICS (Model: {model_version})")
    print("NOTE: Technical baseline for pipeline verification only (not production DS model).")
    print(sep)
    print(f"  - Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  - Precision : {metrics['precision']:.4f}")
    print(f"  - Recall    : {metrics['recall']:.4f}")
    print(f"  - F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(sep + "\n")


# =====================================================================
# Prediction Stage
# =====================================================================

def predict(
    model_or_artifact: Union[Dict[str, Any], Any],
    features_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate predictions and risk probabilities for prepared feature records.

    Supports both batch prediction (full DataFrame) and single-record inference
    (1-row DataFrame, simulating an API or service endpoint).

    Args:
        model_or_artifact: Either a fitted scikit-learn model or an artifact bundle
            dictionary containing 'model' and 'feature_names'.
        features_df: Prepared features DataFrame. Extra columns (e.g. Transaction_ID,
            Customer_ID, Risk_Review_Flag) are automatically ignored.
        threshold: Decision threshold for the positive class ('Yes'). Defaults to 0.5.

    Returns:
        DataFrame containing:
          - 'prediction_decision': 'Yes' or 'No'
          - 'prediction_confidence': float probability (0.0 to 1.0)
          - 'predicted_class': integer (1 or 0)
    """
    # 1. Unpack model and required feature columns
    if isinstance(model_or_artifact, dict) and "model" in model_or_artifact:
        model = model_or_artifact["model"]
        feature_cols = model_or_artifact.get("feature_names")
    else:
        model = model_or_artifact
        feature_cols = None

    if feature_cols:
        # Verify feature presence and alignment
        missing_features = [col for col in feature_cols if col not in features_df.columns]
        if missing_features:
            raise ValueError(
                f"Missing {len(missing_features)} required feature columns for prediction: "
                f"{missing_features[:5]}"
            )
        X = features_df[feature_cols].copy()
    else:
        # Fallback: drop ID columns and target if present
        cols_to_drop = [col for col in ID_COLUMNS + [TARGET_COLUMN] if col in features_df.columns]
        X = features_df.drop(columns=cols_to_drop)

    # 2. Compute Probabilities & Decisions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba
        raw_preds = model.predict(X)
        probabilities = np.where(raw_preds == 1, 1.0, 0.0)

    decisions = ["Yes" if p >= threshold else "No" for p in probabilities]
    predicted_classes = [1 if p >= threshold else 0 for p in probabilities]

    result = pd.DataFrame(
        {
            "prediction_decision": decisions,
            "prediction_confidence": np.round(probabilities, 4),
            "predicted_class": predicted_classes,
        },
        index=features_df.index,
    )

    return result


# =====================================================================
# Output Formatting Stage
# =====================================================================

def format_and_save_predictions(
    predictions_df: pd.DataFrame,
    source_df: pd.DataFrame,
    model_version: str = DEFAULT_MODEL_VERSION,
    output_path: Optional[str] = os.path.join(DEFAULT_PROCESSED_DIR, "FinTrust_Predictions.csv"),
) -> pd.DataFrame:
    """
    Format predictions into a structured, production-ready schema and save to CSV.

    Output Schema:
      - Transaction_ID: Unique identifier for the transaction record
      - predicted_Risk_Review_Flag: 'Yes' or 'No' decision
      - prediction_confidence: Probability of Risk_Review_Flag='Yes'
      - prediction_timestamp: ISO 8601 UTC timestamp of inference
      - model_version: Identifier of model used for auditability

    Args:
        predictions_df: DataFrame returned by predict().
        source_df: Source DataFrame containing the 'Transaction_ID' column.
        model_version: Model version identifier string.
        output_path: Destination path for the predictions CSV file.

    Returns:
        Structured output DataFrame.
    """
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Extract or generate Transaction_ID
    if "Transaction_ID" in source_df.columns:
        tx_ids = source_df["Transaction_ID"].values
    else:
        tx_ids = [f"UNKNOWN_TX_{i:06d}" for i in range(len(predictions_df))]

    output_df = pd.DataFrame(
        {
            "Transaction_ID": tx_ids,
            "predicted_Risk_Review_Flag": predictions_df["prediction_decision"].values,
            "prediction_confidence": predictions_df["prediction_confidence"].values,
            "prediction_timestamp": timestamp_str,
            "model_version": model_version,
        }
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(
            f"Saved {len(output_df):,} formatted predictions to {output_path}"
        )

    return output_df


# =====================================================================
# End-to-End Pipeline Orchestrator
# =====================================================================

def run_pipeline(
    raw_data_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_version: str = DEFAULT_MODEL_VERSION,
    retrain_model: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full FinTrust Digital Bank ML Engineering pipeline end-to-end:

    Stage 1: Data Ingestion (Load raw CSVs)
    Stage 2: Validation (Enforce schemas, null thresholds, and referential integrity)
    Stage 3: Preprocessing & Feature Preparation (Join, impute, temporal feature engineering, encoding)
    Stage 4: Model Training / Loading (Train baseline classifier or load existing artifact)
    Stage 5: Prediction (Generate decisions and confidence scores)
    Stage 6: Output Formatting (Save structured predictions to CSV)

    Args:
        raw_data_dir: Path to raw datasets directory.
        processed_dir: Path to directory where processed CSVs will be saved.
        model_dir: Path to directory for serialized model artifacts.
        model_version: Model version tag to use for training or loading.
        retrain_model: Whether to train a new model artifact (True) or load existing (False).

    Returns:
        Dictionary containing pipeline execution status, reports, and summary statistics.
    """
    pipeline_start = datetime.datetime.now()
    sep = "=" * 70
    print("\n" + sep)
    print("FINTRUST DIGITAL BANK - ML ENGINEERING PIPELINE RUNNER (WEEK 2)")
    print("Flow: Data -> Validation -> Preprocessing -> Features -> Model -> Prediction -> Output")
    print(sep + "\n")

    # -------------------------------------------------------------
    # STAGE 1: DATA INGESTION
    # -------------------------------------------------------------
    print(">>> [STAGE 1/6] Ingesting Raw Datasets...")
    cust_path = os.path.join(raw_data_dir, "FinTrust_Customer_Data.csv")
    tx_path = os.path.join(raw_data_dir, "FinTrust_Transaction_Data.csv")

    if not os.path.exists(cust_path) or not os.path.exists(tx_path):
        raise FileNotFoundError(
            f"Required raw files missing in '{raw_data_dir}'. Expected both "
            f"'{cust_path}' and '{tx_path}'."
        )

    customer_df = pd.read_csv(cust_path)
    transaction_df = pd.read_csv(tx_path)
    print(f"    Loaded Customer Data   : {customer_df.shape[0]:,} rows, {customer_df.shape[1]} columns")
    print(f"    Loaded Transaction Data: {transaction_df.shape[0]:,} rows, {transaction_df.shape[1]} columns")

    # -------------------------------------------------------------
    # STAGE 2: DATA VALIDATION GATING
    # -------------------------------------------------------------
    print("\n>>> [STAGE 2/6] Executing Data Validation Gating...")
    validation_reports = validate_fintrust_pipeline(customer_df, transaction_df, raise_on_error=False)

    # Check for blocking errors
    has_blocking_errors = False
    for stage_name, report in validation_reports.items():
        status = "PASSED" if report.is_valid else "FAILED"
        print(f"    Validation '{stage_name:<25}': [{status}] "
              f"(Errors: {report.error_count}, Warnings: {report.warning_count})")
        if not report.is_valid:
            has_blocking_errors = True
            for err in report.errors:
                print(f"      [BLOCKING ERROR] {err.check}: {err.message}")

    if has_blocking_errors:
        print("\n[PIPELINE HALTED] Blocking data validation errors detected. Aborting pipeline.")
        return {
            "status": "FAILED_VALIDATION",
            "validation_reports": validation_reports,
        }
    print("    Validation Gating: ALL CHECKS PASSED. Proceeding to Preprocessing.")

    # -------------------------------------------------------------
    # STAGE 3 & 4: PREPROCESSING & FEATURE PREPARATION
    # -------------------------------------------------------------
    print("\n>>> [STAGE 3/6] Running Preprocessing & Feature Preparation...")
    modelling_csv_path = os.path.join(processed_dir, "FinTrust_Modelling_Ready.csv")
    model_ready_df, preprocessor = preprocess_pipeline(
        customer_df, transaction_df, output_path=modelling_csv_path
    )
    print(f"    Feature Preparation Complete: {model_ready_df.shape[0]:,} rows x {model_ready_df.shape[1]} features.")
    print(f"    Model-ready dataset saved to: {modelling_csv_path}")
    if "Is_Local_Transaction" in model_ready_df.columns:
        non_local_count = int((model_ready_df["Is_Local_Transaction"] == False).sum())
        non_local_pct = (non_local_count / len(model_ready_df)) * 100.0
        print(
            f"    Feature Sanity Check: {non_local_count:,} / {len(model_ready_df):,} rows "
            f"({non_local_pct:.1f}%) have Is_Local_Transaction = False (out-of-town transactions)."
        )

    # -------------------------------------------------------------
    # STAGE 4: MODEL TRAINING / LOADING
    # -------------------------------------------------------------
    artifact_path = os.path.join(model_dir, f"{model_version}.joblib")
    if retrain_model or not os.path.exists(artifact_path):
        print(f"\n>>> [STAGE 4/6] Training Baseline Classifier ('{model_version}')...")
        model_artifact, sanity_metrics = train_baseline_model(
            model_ready_df,
            model_version=model_version,
            save_dir=model_dir,
        )
    else:
        print(f"\n>>> [STAGE 4/6] Loading Existing Model Artifact ('{model_version}')...")
        model_artifact = load_model_artifact(artifact_path)
        sanity_metrics = model_artifact.get("sanity_metrics", {})

    # -------------------------------------------------------------
    # STAGE 5: PREDICTION
    # -------------------------------------------------------------
    print("\n>>> [STAGE 5/6] Generating Batch Predictions...")
    predictions_df = predict(model_artifact, model_ready_df)
    yes_count = int((predictions_df["prediction_decision"] == "Yes").sum())
    no_count = int((predictions_df["prediction_decision"] == "No").sum())
    print(f"    Generated {len(predictions_df):,} predictions.")
    print(f"    Decisions: {yes_count:,} flagged 'Yes' ({yes_count/len(predictions_df)*100:.1f}%), "
          f"{no_count:,} flagged 'No' ({no_count/len(predictions_df)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 6: OUTPUT FORMATTING & PERSISTENCE
    # -------------------------------------------------------------
    print("\n>>> [STAGE 6/6] Formatting and Saving Predictions...")
    predictions_output_path = os.path.join(processed_dir, "FinTrust_Predictions.csv")
    final_output_df = format_and_save_predictions(
        predictions_df,
        source_df=model_ready_df,
        model_version=model_version,
        output_path=predictions_output_path,
    )

    elapsed = (datetime.datetime.now() - pipeline_start).total_seconds()
    print(f"    Predictions saved successfully to: {predictions_output_path}")

    print("\n" + sep)
    print("PIPELINE EXECUTION SUMMARY")
    print(sep)
    print(f"  - Pipeline Status    : COMPLETED SUCCESSFULLY")
    print(f"  - Execution Time     : {elapsed:.2f} seconds")
    print(f"  - Total Predictions  : {len(final_output_df):,}")
    print(f"  - Risk Flagged 'Yes' : {yes_count:,} ({yes_count/len(final_output_df)*100:.1f}%)")
    print(f"  - Risk Flagged 'No'  : {no_count:,} ({no_count/len(final_output_df)*100:.1f}%)")
    print(f"  - Model Version      : {model_version}")
    print(f"  - Model Artifact     : {artifact_path}")
    print(f"  - Output File        : {predictions_output_path}")
    print(sep + "\n")

    return {
        "status": "SUCCESS",
        "total_predictions": len(final_output_df),
        "flagged_yes": yes_count,
        "flagged_no": no_count,
        "sanity_metrics": sanity_metrics,
        "output_path": predictions_output_path,
        "model_path": artifact_path,
        "execution_time_sec": elapsed,
    }


# =====================================================================
# Standalone CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Run end-to-end pipeline
    run_pipeline()
