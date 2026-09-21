# FinTrust Digital Bank — Technical Test Report

**Week 2 Deliverable: Part E — Technical Testing**  
**Track**: Machine Learning Engineering  
**Status**: **5/5 Tests Passed**

---

## 1. Technical Test Summary Table

| Test | Scenario Description | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. Valid Input** | Well-formed synthetic batch with compliant IDs, types, and allowed categories. | Validation passes (is_valid=True), 0 blocking errors, 0 warnings. | is_valid=True, 0 error(s), 0 warning(s). | **`PASS`** |
| **2. Missing Values** | Tolerated missing values in Device_Type and Location within dynamic threshold (<5%). | Validation passes (is_valid=True), 0 blocking errors, 2 tolerated warnings (Device_Type, Location < 5%). | is_valid=True, 0 error(s), 2 warning(s) on ['Device_Type', 'Location']. | **`PASS`** |
| **3. Unexpected Category** | Transaction_Status = 'Cancelled' violates permissible domain defined in schema. | Validation fails (is_valid=False), 1 blocking error flagging unexpected category 'Cancelled'. | is_valid=False, 1 error(s). Detected: Column 'Transaction_Status' contains unexpected category value(s): ['Cancelled']. Affected rows: 1.. | **`PASS`** |
| **4. Incorrect Data Type** | Non-numeric string 'fifty-thousand' provided in numeric field Amount_NGN. | Validation fails (is_valid=False), 1 blocking error flagging non-numeric Amount_NGN. | is_valid=False, 1 error(s). Detected: Column 'Amount_NGN' expected numeric values, but contains 1 non-convertible value(s).. | **`PASS`** |
| **5. Empty Input** | DataFrame containing zero rows (0 observations). | Validation fails (is_valid=False), 1 blocking error flagging empty dataset (0 rows). | is_valid=False, 1 error(s). Detected: Dataset 'FinTrust Transaction Data' is completely empty (0 rows).. | **`PASS`** |

---

## 2. Test Execution Details

### Test 1: Valid Input
- **Purpose**: Verify that data matching all schema expectations passes without false positives.
- **Inputs**: Synthetic transaction batch with conforming IDs (`FT-T#####`), valid ISO datetimes, numeric amounts, and known categories.
- **Decision**: Pipeline continues with status `[PASSED]`. 0 blocking errors, 0 warnings.

### Test 2: Missing Values (Tolerated vs Blocking)
- **Purpose**: Verify the dynamic missingness threshold logic on `Device_Type` and `Location`.
- **Inputs**: 100 transaction records with 1 missing value each (1.0% missingness, well below the 5.0% threshold).
- **Decision**: Pipeline logs non-blocking warnings (`missing_values_tolerated`) and allows execution to proceed without failing.

### Test 3: Unexpected Category
- **Purpose**: Prevent invalid or corrupted categorical labels from entering downstream encoding.
- **Inputs**: Transaction with `Transaction_Status = 'Cancelled'` (unrecognized business state).
- **Decision**: Pipeline halts with a blocking error (`unexpected_categorical_value`), protecting feature encoding from out-of-vocabulary drift.

### Test 4: Incorrect Data Type
- **Purpose**: Catch corrupted string values in numerical fields before arithmetic or model ingestion.
- **Inputs**: String `'fifty-thousand'` passed to `Amount_NGN`.
- **Decision**: Pipeline halts with a blocking error (`invalid_data_type`).

### Test 5: Empty Input
- **Purpose**: Ensure that empty batches or corrupted 0-row data files fail immediately and explicitly.
- **Inputs**: `pd.DataFrame` with 0 rows.
- **Decision**: Pipeline detects `empty_dataset` and flags a blocking error (or raises `EmptyDatasetError` in strict mode).

---

## 3. Conclusion

**5/5 technical tests behaved as expected.**  
The validation component strictly enforces data quality boundaries while distinguishing between tolerated data noise and pipeline-breaking anomalies.
