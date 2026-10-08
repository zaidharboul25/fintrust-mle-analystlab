"""
Unit and Integration tests for FinTrust Prediction Service (src/api.py).
========================================================================
Week 3: ML Engineering API Testing Suite

Tests health endpoint, valid prediction requests, schema validation,
missing fields, invalid data types, referential integrity mismatches,
and response schemas using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, state
from src.config import DEFAULT_MODEL_ARTIFACT_PATH, PREPROCESSOR_ARTIFACT_PATH
from src.model_loader import load_model
from src.preprocessing import DataPreprocessor


@pytest.fixture(scope="module")
def client():
    """Create test client with active lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_request_payload():
    """Return a well-formed JSON prediction request payload."""
    return {
        "transaction": {
            "Transaction_ID": "FT-T000001",
            "Customer_ID": "FT-C00001",
            "Transaction_DateTime": "2024-01-15 10:30:00",
            "Transaction_Type": "Transfer",
            "Amount_NGN": 15000.0,
            "Channel": "Mobile App",
            "Device_Type": "Android",
            "Location": "Lagos",
            "International_Transaction": "No",
            "Transaction_Status": "Successful",
        },
        "customer": {
            "Customer_ID": "FT-C00001",
            "Customer_Name": "Ibrahim Adeyemi",
            "Age": 34,
            "Gender": "Male",
            "City": "Lagos",
            "Customer_Segment": "Premium",
            "Account_Type": "Savings",
            "Tenure_Months": 12,
            "Digital_Engagement_Score": 4.2,
            "Monthly_Income_Band": "500k-999k",
            "Preferred_Channel": "Mobile App",
            "Account_Status": "Active",
        },
    }


def test_root_endpoint(client):
    """GET / must return 200, service metadata, endpoints map, and responsible use notice."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "FinTrust" in data["service"]
    assert data["track"] == "Machine Learning Engineering"
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["predict"] == "/predict"
    assert "responsible_use_notice" in data
    assert "synthetic educational target" in data["responsible_use_notice"]


def test_health_endpoint(client):
    """GET /health must return 200, healthy status, and model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "model_version" in data
    assert "timestamp" in data
    assert data["service"] == "FinTrust Financial Intelligence Prediction Service"


def test_predict_valid_request(client, valid_request_payload):
    """POST /predict with valid request must return 200 and structured response schema."""
    response = client.post("/predict", json=valid_request_payload)
    assert response.status_code == 200
    data = response.json()

    # Verify structured response fields
    assert data["transaction_id"] == "FT-T000001"
    assert data["prediction"] in ["Yes", "No"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_version"] == "baseline_v1"
    assert "prediction_timestamp" in data


def test_predict_customer_lookup_fallback(client, valid_request_payload):
    """POST /predict without explicit customer payload looks up existing customer from database."""
    # FT-C00001 is a known customer in data/raw/FinTrust_Customer_Data.csv
    payload = {"transaction": valid_request_payload["transaction"]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "FT-T000001"
    assert data["prediction"] in ["Yes", "No"]


def test_predict_referential_integrity_mismatch(client, valid_request_payload):
    """POST /predict with mismatched Customer_IDs in transaction vs customer returns 400 Bad Request."""
    payload = valid_request_payload.copy()
    payload["customer"]["Customer_ID"] = "FT-C00002"  # Mismatch with FT-C00001 in transaction

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Referential integrity mismatch" in response.json()["detail"]


def test_predict_missing_required_field(client, valid_request_payload):
    """POST /predict missing a required field (e.g. Amount_NGN) returns 422 Unprocessable Entity."""
    payload = valid_request_payload.copy()
    del payload["transaction"]["Amount_NGN"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_data_type(client, valid_request_payload):
    """POST /predict with string in numeric field (Amount_NGN='fifty-thousand') returns 422."""
    payload = valid_request_payload.copy()
    payload["transaction"]["Amount_NGN"] = "fifty-thousand"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_regex_id(client, valid_request_payload):
    """POST /predict with invalid Transaction_ID regex pattern returns 422."""
    payload = valid_request_payload.copy()
    payload["transaction"]["Transaction_ID"] = "INVALID_ID_123"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_malformed_json(client):
    """POST /predict with malformed body returns 422."""
    response = client.post(
        "/predict",
        content="not a valid json string",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_unseen_category(client, valid_request_payload):
    """POST /predict with novel device type succeeds smoothly without crashing."""
    payload = valid_request_payload.copy()
    payload["transaction"]["Device_Type"] = "Novel_Smartwatch_OS"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Yes", "No"]


def test_service_unavailable_when_model_missing(client):
    """POST /predict returns 503 if model or preprocessor is uninitialized."""
    # Temporarily unset model in state after lifespan startup
    original_model = state.model
    try:
        state.model = None
        res = client.post(
            "/predict",
            json={
                "transaction": {
                    "Transaction_ID": "FT-T000001",
                    "Customer_ID": "FT-C00001",
                    "Transaction_DateTime": "2024-01-15 10:30:00",
                    "Transaction_Type": "Transfer",
                    "Amount_NGN": 1000.0,
                    "Channel": "Mobile App",
                }
            },
        )
        assert res.status_code == 503
        assert "unavailable" in res.json()["detail"].lower()
    finally:
        state.model = original_model


def test_ui_dashboard_endpoint(client):
    """GET /ui and GET /dashboard must return 200 with HTML content."""
    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "text/html" in res_ui.headers["content-type"]
    assert "FinTrust" in res_ui.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "text/html" in res_dash.headers["content-type"]


def test_static_assets_serving(client):
    """GET /static/style.css and /static/app.js must return valid assets."""
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "css" in res_css.headers["content-type"]

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
