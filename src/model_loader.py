"""
FinTrust Digital Bank - Model Loader & Interface Component
==========================================================
Week 3: ML Engineering Model Boundary Module

Defines a clean, standardized interface separating the ML Engineering serving pipeline
from underlying predictive models. Enables the Data Science track to deliver future
candidate models (e.g., LightGBM, XGBoost, Random Forest, or tuned ensembles) without
altering pipeline preprocessing, validation gating, or output persistence.

Standard Artifact Bundle Specification:
---------------------------------------
A valid model artifact serialized with joblib must be a dictionary with keys:
- 'model': An estimator implementing at least `.predict(X)` and optionally `.predict_proba(X)`.
- 'feature_names': List of expected feature column names in exact training order.
- 'model_version': Identifier string (e.g., 'baseline_v1', 'ds_lgbm_v2').
- 'trained_timestamp': ISO 8601 UTC timestamp string.
- 'sanity_metrics': Dictionary of evaluation metrics on test split.
- 'target_mapping': Dict mapping numeric predictions to labels, e.g. {1: 'Yes', 0: 'No'}.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACT_KEYS: List[str] = ["model", "feature_names", "model_version"]


class ModelInterfaceError(Exception):
    """Raised when a model artifact violates the interface contract."""
    pass


class ModelInterface:
    """
    Standardized wrapper around a loaded model artifact bundle.

    Guarantees consistent feature ordering, missing-column detection,
    and unified access to prediction decisions and confidence scores.
    """

    def __init__(self, artifact_bundle: Dict[str, Any], artifact_path: Optional[str] = None):
        """
        Initialize the ModelInterface.

        Args:
            artifact_bundle: Validated dictionary containing 'model', 'feature_names', etc.
            artifact_path: Optional path where artifact was loaded from.
        """
        self.artifact = artifact_bundle
        self.artifact_path = artifact_path
        self.model = artifact_bundle["model"]
        self.feature_names: List[str] = list(artifact_bundle["feature_names"])
        self.model_version: str = str(artifact_bundle.get("model_version", "unknown_version"))
        self.target_mapping: Dict[int, str] = artifact_bundle.get(
            "target_mapping", {1: "Yes", 0: "No"}
        )
        self.sanity_metrics: Dict[str, float] = artifact_bundle.get("sanity_metrics", {})
        self.trained_timestamp: str = artifact_bundle.get("trained_timestamp", "")

    def align_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and align incoming features against expected model feature schema and ordering.

        Args:
            features_df: Input features DataFrame.

        Returns:
            DataFrame with columns precisely matching self.feature_names in exact order.

        Raises:
            ModelInterfaceError: If required features are missing from features_df.
        """
        missing = [f for f in self.feature_names if f not in features_df.columns]
        if missing:
            sample_missing = missing[:5]
            raise ModelInterfaceError(
                f"Feature alignment failed: {len(missing)} required feature(s) missing from input: "
                f"{sample_missing} (total required: {len(self.feature_names)})"
            )
        # Reorder columns strictly to match training order
        return features_df[self.feature_names].copy()

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Compute predicted risk probability for the positive class (Risk_Review_Flag='Yes' / 1).

        Args:
            features_df: Input features DataFrame.

        Returns:
            1D numpy array of probabilities in range [0.0, 1.0].
        """
        X = self.align_features(features_df)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Binary classification: return probability of class 1
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        elif hasattr(self.model, "decision_function"):
            # Logistic sigmoid transform for models with decision_function only
            decision = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-decision))
        else:
            # Fallback for models only implementing predict
            raw_preds = self.model.predict(X)
            return np.where(raw_preds == 1, 1.0, 0.0)

    def predict(
        self, features_df: pd.DataFrame, threshold: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate binary predictions, confidence probabilities, and string decisions.

        Args:
            features_df: Input features DataFrame.
            threshold: Positive decision boundary threshold (default 0.50).

        Returns:
            Tuple of (binary_classes_array, confidence_scores_array, string_decisions_list).
        """
        probabilities = self.predict_proba(features_df)
        binary_classes = (probabilities >= threshold).astype(int)
        decisions = [
            self.target_mapping.get(cls, "Yes" if cls == 1 else "No")
            for cls in binary_classes
        ]
        return binary_classes, np.round(probabilities, 4), decisions


def validate_model_interface(artifact: Any) -> bool:
    """
    Validate that an artifact dictionary satisfies the model interface contract.

    Args:
        artifact: Deserialized artifact to inspect.

    Returns:
        True if valid.

    Raises:
        ModelInterfaceError: If contract is violated.
    """
    if not isinstance(artifact, dict):
        raise ModelInterfaceError(
            f"Invalid model artifact format: expected dictionary bundle, got {type(artifact)}."
        )

    for key in REQUIRED_ARTIFACT_KEYS:
        if key not in artifact:
            raise ModelInterfaceError(
                f"Model artifact violates interface contract: missing required key '{key}'."
            )

    model = artifact["model"]
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ModelInterfaceError(
            "Model artifact violates interface contract: 'model' object must have a callable 'predict' method."
        )

    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
        raise ModelInterfaceError(
            "Model artifact violates interface contract: 'feature_names' must be a non-empty list of strings."
        )

    return True


def load_model(artifact_path: Union[str, Path]) -> ModelInterface:
    """
    Load and validate a serialized model artifact bundle from disk.

    Args:
        artifact_path: File path to the .joblib artifact.

    Returns:
        Validated ModelInterface instance.

    Raises:
        FileNotFoundError: If artifact does not exist at specified path.
        ModelInterfaceError: If artifact fails interface validation.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    try:
        raw_artifact = joblib.load(str(path))
    except Exception as e:
        raise ModelInterfaceError(f"Failed to deserialize model artifact at {path}: {e}") from e

    validate_model_interface(raw_artifact)
    logger.info(
        f"Loaded and validated model artifact '{raw_artifact.get('model_version')}' "
        f"from {path} ({len(raw_artifact['feature_names'])} features)."
    )
    return ModelInterface(raw_artifact, artifact_path=str(path))
