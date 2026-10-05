# FinTrust Digital Bank — Week 4 Refinement Evidence

**Track**: Machine Learning Engineering (Track 3)  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 4 — Test, Refine & Present  
**Document**: Empirical Evidence of System Refinement  
**Date**: October 2026  
**Status**: REFINED & RE-TESTED  

---

## 1. Context & Objective

In Week 3, the critical technical refinement resolved the "53 missing feature columns" issue by serializing the fitted `DataPreprocessor` state to `models/preprocessor.joblib`.

For **Week 4 (*Test, Refine & Present*)**, the objective is conducting rigorous end-to-end testing across all operational touchpoints and identifying real architectural, execution, or serving gaps discovered during final validation. This document records two genuine Week 4 findings, their root cause analyses, implemented refinements, and re-test verification.

---

## 2. Refinement 1: Pytest Environment Path Resolution (`pytest.ini`)

### 2.1 Initial Output / Behavior
When initiating automated testing in a clean environment using the standard test command:
```powershell
pytest -v
```
The test runner immediately failed during the collection phase before executing a single test:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: C:\Users\zaidh\Desktop\Stage_number_2\Project folder
collecting ... 
ImportError while importing test module 'tests\test_validation.py'.
Traceback:
  tests\test_validation.py:13: in <module>
    from src.validation import DataValidator, ...
E   ModuleNotFoundError: No module named 'src'
=========================== short test summary info ===========================
ERROR tests/test_api.py
ERROR tests/test_model_loading.py
ERROR tests/test_prediction.py
ERROR tests/test_preprocessing.py
ERROR tests/test_reproducibility.py
ERROR tests/test_validation.py
!!!!!!!!!!!!!!!!!!! Interrupted: 6 errors during collection !!!!!!!!!!!!!!!!!!!
============================= 6 errors in 12.79s ==============================
```

### 2.2 Test Performed
Executed bare `pytest -v` across the `tests/` directory from the repository root.

### 2.3 Finding
All 6 test files failed to import modules from `src/`, halting the entire test suite. While invoking `python -m pytest` worked because the Python executable adds the current working directory to `sys.path`, invoking standard `pytest` directly (the default command used by many CI runners, IDE integrations, and developers) failed completely.

### 2.4 Root Cause Analysis
1. The repository lacked a `pytest.ini` or `pyproject.toml` file instructing pytest to configure `pythonpath`.
2. As a consequence, pytest did not register the workspace root directory on `sys.path` during standalone invocation, making `import src...` fail.

### 2.5 Technical Refinement
Created `pytest.ini` in the repository root configuring `pythonpath = .`:

```ini
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
python_functions = test_*
filterwarnings =
    ignore::DeprecationWarning
```

Also added `tests/__init__.py` to establish formal package boundaries.

### 2.6 Re-test Execution
Executed bare `pytest -v`:

```powershell
pytest -v
```

### 2.7 Final Result
- **Collection**: 55 items collected with **0 errors**.
- **Execution**: **55 passed in 6.26s (100% PASS)**.
- **Portability**: Both `pytest` and `python -m pytest` now execute deterministically across any developer shell or CI environment.

---

## 3. Refinement 2: FastAPI Root Discovery & Educational Governance Entrypoint (`GET /`)

### 3.1 Initial Output / Behavior
When a client, health monitor, or reviewer queried the root URL of the prediction service:
```powershell
curl -X GET "http://127.0.0.1:8000/"
```
The service returned a generic `404 Not Found` response:
```json
{
  "detail": "Not Found"
}
```

### 3.2 Test Performed
Queried root endpoint `/` via HTTP and automated `TestClient` in `tests/test_api.py`.

### 3.3 Finding
1. The API service only defined `/health` and `/predict`.
2. The root route provided zero service discovery, zero links to OpenAPI `/docs`, and zero immediate indication of the project's educational governance disclaimer.

### 3.4 Root Cause Analysis
No root handler was defined in `src/api.py`. In an operational banking or support service, the root entrypoint should provide service metadata, API version, available route endpoints, and mandatory responsible-use warnings.

### 3.5 Technical Refinement
1. Defined `RootResponse` schema in `src/api.py`:
```python
class RootResponse(BaseModel):
    """Root endpoint service overview and navigation schema."""
    service: str = "FinTrust Financial Intelligence & Digital Banking Support Solution"
    track: str = "Machine Learning Engineering"
    milestone: str = "Week 4 — Test, Refine & Present"
    version: str = "1.0.0"
    endpoints: Dict[str, str] = {
        "health": "/health",
        "predict": "/predict",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
    }
    responsible_use_notice: str = (
        "FinTrust is a fictional organisation and Risk_Review_Flag is a synthetic educational target, "
        "NOT an authentic fraud determination. Predictions must not be used for real financial-crime decisions."
    )
```
2. Implemented `GET /` endpoint in `src/api.py`:
```python
@app.get("/", response_model=RootResponse, summary="Root Service Information")
async def root() -> RootResponse:
    """Return basic project and service information, available endpoints, and responsible use notice."""
    return RootResponse()
```
3. Added automated unit test `test_root_endpoint` in `tests/test_api.py`.

### 3.6 Re-test Execution
Executed:
```powershell
pytest tests/test_api.py -v
```

### 3.7 Final Result
- **Status**: `test_root_endpoint` **PASSED**.
- **Response**: Calling `GET /` returns `200 OK` with full service overview, endpoint links, and educational disclaimer:
```json
{
  "service": "FinTrust Financial Intelligence & Digital Banking Support Solution",
  "track": "Machine Learning Engineering",
  "milestone": "Week 4 — Test, Refine & Present",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "documentation": "/docs",
    "openapi_schema": "/openapi.json"
  },
  "responsible_use_notice": "FinTrust is a fictional organisation and Risk_Review_Flag is a synthetic educational target, NOT an authentic fraud determination. Predictions must not be used for real financial-crime decisions."
}
```

---

## 4. Summary Table of Week 4 Refinements

| Refinement | Initial Output | Test Performed | Finding | Root Cause | Refinement Implemented | Re-test Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Pytest Path Configuration** | `pytest -v` failed with `ModuleNotFoundError: No module named 'src'`. | Executed bare `pytest` in PowerShell. | Collection aborted on all 6 test files. | Missing `pytest.ini` with `pythonpath = .`. | Added `pytest.ini` and `tests/__init__.py`. | **PASS**: 55/55 passed in 6.26s. |
| **2. API Root Entrypoint & Governance** | `GET /` returned `404 Not Found`. | Queried root URL `/` via `TestClient`. | No service discovery or disclaimer at root. | Absence of root route handler in `api.py`. | Implemented `RootResponse` and `GET /`; added unit test. | **PASS**: `test_root_endpoint` passed; 200 OK with metadata. |
