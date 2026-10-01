"""
Unit tests for FinTrust Model Loader & Interface Component (src/model_loader.py).
================================================================================
Week 3: ML Engineering Test Suite Expansion

Validates artifact loading, interface contract enforcement, schema alignment,
missing feature detection, column reordering invariance, and probability scoring.
"""

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.model_loader import (
    ModelInterface,
    ModelInterfaceError,
    load_model,
    validate_model_interface,
)


@pytest.fixture
def dummy_artifact():
    """Create a minimal valid model artifact bundle."""
    # Simple model with 3 features
    feature_names = ["feature_a", "feature_b", "feature_c"]
    X = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        columns=feature_names,
    )
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "model_version": "test_bundle_v1",
        "target_mapping": {1: "Yes", 0: "No"},
        "sanity_metrics": {"accuracy": 1.0},
        "trained_timestamp": "2026-10-01T00:00:00Z",
    }
    return artifact


def test_validate_model_interface_valid(dummy_artifact):
    """A compliant artifact passes interface validation."""
    assert validate_model_interface(dummy_artifact) is True


def test_validate_model_interface_missing_keys():
    """Missing required keys trigger ModelInterfaceError."""
    # Missing 'model'
    with pytest.raises(ModelInterfaceError, match="missing required key 'model'"):
        validate_model_interface({"feature_names": ["f1"], "model_version": "v1"})

    # Missing 'feature_names'
    with pytest.raises(ModelInterfaceError, match="missing required key 'feature_names'"):
        validate_model_interface({"model": LogisticRegression(), "model_version": "v1"})

    # Non-dictionary object
    with pytest.raises(ModelInterfaceError, match="expected dictionary"):
        validate_model_interface("not_a_dict")


def test_validate_model_interface_invalid_model_object():
    """Model object without callable predict method triggers ModelInterfaceError."""
    artifact = {
        "model": "string_not_model",
        "feature_names": ["f1"],
        "model_version": "v1",
    }
    with pytest.raises(ModelInterfaceError, match="must have a callable 'predict' method"):
        validate_model_interface(artifact)


def test_validate_model_interface_empty_feature_names():
    """Empty or non-list feature_names triggers ModelInterfaceError."""
    artifact = {
        "model": LogisticRegression(),
        "feature_names": [],
        "model_version": "v1",
    }
    with pytest.raises(ModelInterfaceError, match="non-empty list of strings"):
        validate_model_interface(artifact)


def test_load_model_success(dummy_artifact):
    """Valid serialized artifact loads cleanly via load_model()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(dummy_artifact, artifact_path)

        interface = load_model(artifact_path)
        assert isinstance(interface, ModelInterface)
        assert interface.model_version == "test_bundle_v1"
        assert interface.feature_names == ["feature_a", "feature_b", "feature_c"]


def test_load_model_nonexistent_file():
    """Attempting to load a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_path/model.joblib")


def test_model_interface_missing_feature_raises(dummy_artifact):
    """Missing a required feature column raises ModelInterfaceError."""
    interface = ModelInterface(dummy_artifact)
    df_missing = pd.DataFrame({
        "feature_a": [1.0],
        # feature_b is missing!
        "feature_c": [3.0],
    })

    with pytest.raises(ModelInterfaceError, match="required feature.*missing"):
        interface.predict_proba(df_missing)


def test_model_interface_column_reordering_invariance(dummy_artifact):
    """Columns in different order are automatically aligned to match training order."""
    interface = ModelInterface(dummy_artifact)

    # Correct order: a, b, c
    df_correct = pd.DataFrame({"feature_a": [2.0], "feature_b": [3.0], "feature_c": [4.0]})
    # Shuffled order: c, a, b
    df_shuffled = pd.DataFrame({"feature_c": [4.0], "feature_a": [2.0], "feature_b": [3.0]})

    proba_correct = interface.predict_proba(df_correct)
    proba_shuffled = interface.predict_proba(df_shuffled)

    # Must produce identical probability
    np.testing.assert_allclose(proba_correct, proba_shuffled, atol=1e-6)


def test_model_interface_predictions_and_probabilities(dummy_artifact):
    """Verify predict() returns valid binary classes, confidences in [0, 1], and mapped decisions."""
    interface = ModelInterface(dummy_artifact)
    df = pd.DataFrame({"feature_a": [1.0, 10.0], "feature_b": [2.0, 11.0], "feature_c": [3.0, 12.0]})

    classes, confidences, decisions = interface.predict(df, threshold=0.50)

    assert len(classes) == 2
    assert set(classes).issubset({0, 1})
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert set(decisions).issubset({"Yes", "No"})
