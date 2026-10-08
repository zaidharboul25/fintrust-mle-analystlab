"""
FinTrust Digital Bank - Real-Time Prediction Service (FastAPI)
==============================================================
Week 3: ML Engineering Service Integration Component

Exposes the FinTrust ML Engineering pipeline as a high-performance HTTP service.
Reuses existing validation, preprocessing, and model interface modules without
duplicating ML logic.

Endpoints:
- GET  /health   : Diagnostic health check and loaded model metadata.
- POST /predict  : Single-transaction risk scoring returning structured prediction decisions.
"""

from __future__ import annotations

import datetime
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.config import (
    CUSTOMER_DATA_PATH,
    DEFAULT_MODEL_ARTIFACT_PATH,
    DEFAULT_MODEL_VERSION,
    PREPROCESSOR_ARTIFACT_PATH,
    STATIC_DIR,
    VALID_CITIES,
)
from src.model_loader import ModelInterface, load_model
from src.preprocessing import DataPreprocessor

# Configure logger
logger = logging.getLogger("fintrust.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =====================================================================
# Pydantic Request & Response Schemas
# =====================================================================

class CustomerPayload(BaseModel):
    """Schema for customer profile data."""
    Customer_ID: str = Field(..., pattern=r"^FT-C\d{5}$", description="Customer identifier FT-C#####")
    Customer_Name: str = Field(..., min_length=2, description="Customer full name")
    Age: int = Field(..., ge=18, le=120, description="Customer age in years")
    Gender: Literal["Male", "Female", "Prefer not to say"] = Field(..., description="Gender")
    City: str = Field(..., description="Customer home city")
    Customer_Segment: Literal["Premium", "Everyday", "SME", "Student"] = Field(...)
    Account_Type: Literal["Savings", "Current", "Premium"] = Field(...)
    Tenure_Months: int = Field(..., ge=0, description="Tenure in months")
    Digital_Engagement_Score: float = Field(..., ge=0.0, le=5.0, description="Score 0.0-5.0")
    Monthly_Income_Band: Literal["Below 100k", "100k-249k", "250k-499k", "500k-999k", "1m+"] = Field(...)
    Preferred_Channel: Literal["Mobile App", "USSD", "Web"] = Field(...)
    Account_Status: Literal["Active", "Dormant", "Restricted"] = Field(...)

    @field_validator("City")
    @classmethod
    def validate_city(cls, v: str) -> str:
        if v not in VALID_CITIES:
            raise ValueError(f"City '{v}' is not in the supported Nigerian cities: {VALID_CITIES}")
        return v


class TransactionPayload(BaseModel):
    """Schema for individual transaction details."""
    Transaction_ID: str = Field(..., pattern=r"^FT-T\d{6}$", description="Transaction identifier FT-T######")
    Customer_ID: str = Field(..., pattern=r"^FT-C\d{5}$", description="Customer identifier FT-C#####")
    Transaction_DateTime: str = Field(..., description="ISO datetime or 'YYYY-MM-DD HH:MM:SS'")
    Transaction_Type: Literal[
        "Card Purchase", "Cash Withdrawal", "Transfer", "Deposit", "Airtime/Data", "Bill Payment"
    ] = Field(...)
    Amount_NGN: float = Field(..., gt=0.0, description="Transaction amount in Nigerian Naira")
    Channel: Literal["Mobile App", "ATM", "POS", "Web", "USSD"] = Field(...)
    Device_Type: Optional[str] = Field(
        default="Unknown",
        description="Originating device type (Android, iOS, Web Browser, POS Terminal, ATM Terminal, Unknown)",
    )
    Location: Optional[str] = Field(
        default=None,
        description="Transaction location city (defaults to customer home city if omitted)",
    )
    International_Transaction: Literal["No", "Yes"] = Field(default="No")
    Transaction_Status: Literal["Successful", "Reversed", "Failed", "Pending"] = Field(default="Successful")


class PredictRequest(BaseModel):
    """
    Structured request schema for single-record risk prediction.
    Accepts explicit transaction and customer records.
    """
    transaction: TransactionPayload
    customer: Optional[CustomerPayload] = None


class PredictionResponse(BaseModel):
    """Structured, production-ready prediction response schema."""
    transaction_id: str
    prediction: Literal["Yes", "No"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prediction_timestamp: str


class HealthResponse(BaseModel):
    """Service health and diagnostic status schema."""
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_version: str
    preprocessor_loaded: bool
    features_count: int
    service: str = "FinTrust Financial Intelligence Prediction Service"
    timestamp: str


class RootResponse(BaseModel):
    """Root endpoint service overview and navigation schema."""
    service: str = "FinTrust Financial Intelligence & Digital Banking Support Solution"
    track: str = "Machine Learning Engineering"
    milestone: str = "Week 4 — Test, Refine & Present"
    version: str = "1.0.0"
    endpoints: Dict[str, str] = {
        "ui": "/ui",
        "health": "/health",
        "predict": "/predict",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
    }
    responsible_use_notice: str = (
        "FinTrust is a fictional organisation and Risk_Review_Flag is a synthetic educational target, "
        "NOT an authentic fraud determination. Predictions must not be used for real financial-crime decisions."
    )



# =====================================================================
# Global Service State & Lifespan Management
# =====================================================================

class ServiceState:
    """Manages loaded models, preprocessors, and customer lookup cache."""
    model: Optional[ModelInterface] = None
    preprocessor: Optional[DataPreprocessor] = None
    customer_store: Optional[pd.DataFrame] = None


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: load model and preprocessor artifacts upon startup."""
    logger.info("Initializing FinTrust Prediction Service resources...")

    # 1. Load Preprocessor
    try:
        if PREPROCESSOR_ARTIFACT_PATH.exists():
            state.preprocessor = DataPreprocessor.load(PREPROCESSOR_ARTIFACT_PATH)
            logger.info(f"Loaded DataPreprocessor from {PREPROCESSOR_ARTIFACT_PATH}")
        else:
            logger.warning(f"Preprocessor artifact not found at {PREPROCESSOR_ARTIFACT_PATH}.")
    except Exception as e:
        logger.error(f"Failed to load DataPreprocessor: {e}")

    # 2. Load Model Interface
    try:
        if DEFAULT_MODEL_ARTIFACT_PATH.exists():
            state.model = load_model(DEFAULT_MODEL_ARTIFACT_PATH)
            logger.info(f"Loaded ModelInterface from {DEFAULT_MODEL_ARTIFACT_PATH}")
        else:
            logger.warning(f"Model artifact not found at {DEFAULT_MODEL_ARTIFACT_PATH}.")
    except Exception as e:
        logger.error(f"Failed to load ModelInterface: {e}")

    # 3. Load Customer Data store for foreign key lookups if customer object is omitted
    try:
        if CUSTOMER_DATA_PATH.exists():
            state.customer_store = pd.read_csv(CUSTOMER_DATA_PATH).set_index("Customer_ID", drop=False)
            logger.info(f"Loaded {len(state.customer_store)} customer records for fast lookup.")
    except Exception as e:
        logger.warning(f"Could not load customer dataset cache: {e}")

    yield

    logger.info("Shutting down FinTrust Prediction Service resources.")


# =====================================================================
# FastAPI Application Initialization
# =====================================================================

app = FastAPI(
    title="FinTrust Financial Intelligence & Digital Banking Support Solution",
    description=(
        "Machine Learning Engineering Prediction API (Week 3: Develop & Integrate). "
        "Provides real-time risk-review predictions over standardized contracts. "
        "EDUCATIONAL DISCLAIMER: Risk_Review_Flag is a synthetic educational target and "
        "must never be used for real fraud detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =====================================================================
# Global Exception Handlers
# =====================================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Mask raw stack traces from client responses while logging internally."""
    logger.error(f"Unhandled exception during request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal service error occurred. The incident has been logged."},
    )


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/", summary="Root Service Information", response_model=None)
async def root(request: Request):
    """Return web dashboard if accessed via browser, otherwise root JSON service overview."""
    accept_header = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()
    index_path = STATIC_DIR / "index.html"
    if "text/html" in accept_header and "testclient" not in user_agent and index_path.exists():
        return FileResponse(index_path)
    return RootResponse()


@app.get("/ui", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Direct URL to the FinTrust frontend user interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frontend static files not found.")


@app.get("/health", response_model=HealthResponse, summary="Service Health Check")
async def health_check() -> HealthResponse:
    """Check service availability, loaded model version, and pipeline component status."""
    is_healthy = state.model is not None and state.preprocessor is not None
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=state.model is not None,
        model_version=state.model.model_version if state.model else "none",
        preprocessor_loaded=state.preprocessor is not None,
        features_count=len(state.model.feature_names) if state.model else 0,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Risk Review Flag")
async def predict_record(request_body: PredictRequest) -> PredictionResponse:
    """
    Generate a risk-review decision for a single transaction request.

    Reuses existing DataPreprocessor and ModelInterface without logic duplication.
    """
    # 1. Verify model and preprocessor are available
    if state.model is None or state.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model or preprocessor is currently unavailable.",
        )

    tx_dict = request_body.transaction.model_dump()
    tx_cust_id = tx_dict["Customer_ID"]

    # 2. Resolve Customer Profile
    if request_body.customer is not None:
        cust_dict = request_body.customer.model_dump()
        # Verify cross-record referential integrity
        if cust_dict["Customer_ID"] != tx_cust_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Referential integrity mismatch: transaction Customer_ID '{tx_cust_id}' "
                    f"does not match customer Customer_ID '{cust_dict['Customer_ID']}'."
                ),
            )
    else:
        # Lookup in preloaded customer database
        if state.customer_store is not None and tx_cust_id in state.customer_store.index:
            cust_dict = state.customer_store.loc[tx_cust_id].to_dict()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Customer profile for '{tx_cust_id}' not found in database. "
                    f"Please supply the explicit 'customer' payload object in the request."
                ),
            )

    # 3. Preprocess record using existing DataPreprocessor
    try:
        single_tx_df = pd.DataFrame([tx_dict])
        single_cust_df = pd.DataFrame([cust_dict])

        # Transform using pre-fitted preprocessor state
        features_df = state.preprocessor.transform(
            single_cust_df, single_tx_df, include_target=False
        )
    except Exception as e:
        logger.error(f"Preprocessing error for transaction {tx_dict['Transaction_ID']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing failed: {str(e)}",
        ) from e

    # 4. Predict using existing ModelInterface
    try:
        classes, confidences, decisions = state.model.predict(features_df, threshold=0.50)
    except Exception as e:
        logger.error(f"Model prediction error for transaction {tx_dict['Transaction_ID']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction evaluation failed.",
        ) from e

    # 5. Return structured prediction response
    return PredictionResponse(
        transaction_id=tx_dict["Transaction_ID"],
        prediction=decisions[0],
        confidence=float(confidences[0]),
        model_version=state.model.model_version,
        prediction_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
