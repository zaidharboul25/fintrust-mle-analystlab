"""
FinTrust Digital Bank - Technical Test Report Generator
======================================================
Week 2: ML Engineering Deliverable (Part E - Technical Testing)

This script executes exactly 5 technical test scenarios against DataValidator,
verifying and documenting:
  Test -> Expected Result -> Actual Result -> Status

Scenarios Tested:
1. Valid Input: Well-formed synthetic transactions pass with zero errors.
2. Missing Values: Known Device_Type/Location missingness (<5%) produces tolerated warnings.
3. Unexpected Category: Invalid category ('Cancelled') triggers a blocking error.
4. Incorrect Data Type: Non-numeric Amount_NGN ('fifty-thousand') triggers a blocking error.
5. Empty Input: 0-row DataFrame triggers a blocking empty_dataset error.

Outputs:
- Formatted console table
- docs/technical_test_report.md
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from src.validation import (
    DataValidator,
    TRANSACTION_SCHEMA,
    ValidationReport,
)


def create_valid_transaction_row(index: int = 1) -> Dict[str, Any]:
    """Helper creating a single valid transaction dictionary."""
    return {
        "Transaction_ID": f"FT-T{index:06d}",
        "Customer_ID": f"FT-C{index:05d}",
        "Transaction_DateTime": "2024-01-15 10:30:00",
        "Transaction_Type": "Card Purchase",
        "Amount_NGN": 5000.0,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful",
        "Risk_Review_Flag": "No",
    }


def run_technical_tests() -> List[Dict[str, str]]:
    """
    Execute the 5 required technical test scenarios and capture results.

    Returns:
        List of dictionaries with keys: 'Test', 'Expected Result', 'Actual Result', 'Status'.
    """
    validator = DataValidator.for_transaction_data()
    test_results: List[Dict[str, str]] = []

    # -------------------------------------------------------------
    # Test 1: Valid Input
    # -------------------------------------------------------------
    # Setup: 5 well-formed transaction rows matching schema expectations exactly
    valid_df = pd.DataFrame([create_valid_transaction_row(i) for i in range(1, 6)])
    report_1 = validator.validate(valid_df, raise_on_error=False)

    exp_1 = "Validation passes (is_valid=True), 0 blocking errors, 0 warnings."
    act_1 = f"is_valid={report_1.is_valid}, {report_1.error_count} error(s), {report_1.warning_count} warning(s)."
    status_1 = "PASS" if (report_1.is_valid and report_1.error_count == 0 and report_1.warning_count == 0) else "FAIL"

    test_results.append({
        "Test": "1. Valid Input",
        "Scenario Description": "Well-formed synthetic batch with compliant IDs, types, and allowed categories.",
        "Expected Result": exp_1,
        "Actual Result": act_1,
        "Status": status_1,
    })

    # -------------------------------------------------------------
    # Test 2: Missing Values (Known / Tolerated Missingness)
    # -------------------------------------------------------------
    # Setup: 100 rows where Device_Type and Location have 1 missing row each (1% missingness, < 5% threshold)
    rows_2 = [create_valid_transaction_row(i) for i in range(1, 101)]
    rows_2[0]["Device_Type"] = np.nan
    rows_2[0]["Location"] = np.nan
    missing_df = pd.DataFrame(rows_2)
    report_2 = validator.validate(missing_df, raise_on_error=False)

    exp_2 = "Validation passes (is_valid=True), 0 blocking errors, 2 tolerated warnings (Device_Type, Location < 5%)."
    warn_cols = [w.column for w in report_2.warnings]
    act_2 = f"is_valid={report_2.is_valid}, {report_2.error_count} error(s), {report_2.warning_count} warning(s) on {warn_cols}."
    status_2 = "PASS" if (report_2.is_valid and report_2.error_count == 0 and report_2.warning_count == 2) else "FAIL"

    test_results.append({
        "Test": "2. Missing Values",
        "Scenario Description": "Tolerated missing values in Device_Type and Location within dynamic threshold (<5%).",
        "Expected Result": exp_2,
        "Actual Result": act_2,
        "Status": status_2,
    })

    # -------------------------------------------------------------
    # Test 3: Unexpected Category
    # -------------------------------------------------------------
    # Setup: Row contains Transaction_Status = 'Cancelled' (not in allowed category list)
    rows_3 = [create_valid_transaction_row(i) for i in range(1, 4)]
    rows_3[0]["Transaction_Status"] = "Cancelled"
    cat_df = pd.DataFrame(rows_3)
    report_3 = validator.validate(cat_df, raise_on_error=False)

    cat_errors = [e for e in report_3.errors if e.check == "unexpected_categorical_value"]
    exp_3 = "Validation fails (is_valid=False), 1 blocking error flagging unexpected category 'Cancelled'."
    act_3 = (
        f"is_valid={report_3.is_valid}, {report_3.error_count} error(s). "
        f"Detected: {cat_errors[0].message if cat_errors else 'None'}."
    )
    status_3 = "PASS" if (not report_3.is_valid and len(cat_errors) == 1 and "Cancelled" in cat_errors[0].message) else "FAIL"

    test_results.append({
        "Test": "3. Unexpected Category",
        "Scenario Description": "Transaction_Status = 'Cancelled' violates permissible domain defined in schema.",
        "Expected Result": exp_3,
        "Actual Result": act_3,
        "Status": status_3,
    })

    # -------------------------------------------------------------
    # Test 4: Incorrect Data Type
    # -------------------------------------------------------------
    # Setup: Row contains a non-numeric string 'fifty-thousand' in Amount_NGN
    rows_4 = [create_valid_transaction_row(i) for i in range(1, 4)]
    rows_4[0]["Amount_NGN"] = "fifty-thousand"
    type_df = pd.DataFrame(rows_4)
    report_4 = validator.validate(type_df, raise_on_error=False)

    type_errors = [e for e in report_4.errors if e.check == "invalid_data_type" and e.column == "Amount_NGN"]
    exp_4 = "Validation fails (is_valid=False), 1 blocking error flagging non-numeric Amount_NGN."
    act_4 = (
        f"is_valid={report_4.is_valid}, {report_4.error_count} error(s). "
        f"Detected: {type_errors[0].message if type_errors else 'None'}."
    )
    status_4 = "PASS" if (not report_4.is_valid and len(type_errors) == 1) else "FAIL"

    test_results.append({
        "Test": "4. Incorrect Data Type",
        "Scenario Description": "Non-numeric string 'fifty-thousand' provided in numeric field Amount_NGN.",
        "Expected Result": exp_4,
        "Actual Result": act_4,
        "Status": status_4,
    })

    # -------------------------------------------------------------
    # Test 5: Empty Input
    # -------------------------------------------------------------
    # Setup: Empty DataFrame with zero rows
    empty_df = pd.DataFrame(columns=list(TRANSACTION_SCHEMA.columns.keys()))
    report_5 = validator.validate(empty_df, raise_on_error=False)

    empty_errors = [e for e in report_5.errors if e.check == "empty_dataset"]
    exp_5 = "Validation fails (is_valid=False), 1 blocking error flagging empty dataset (0 rows)."
    act_5 = (
        f"is_valid={report_5.is_valid}, {report_5.error_count} error(s). "
        f"Detected: {empty_errors[0].message if empty_errors else 'None'}."
    )
    status_5 = "PASS" if (not report_5.is_valid and len(empty_errors) == 1) else "FAIL"

    test_results.append({
        "Test": "5. Empty Input",
        "Scenario Description": "DataFrame containing zero rows (0 observations).",
        "Expected Result": exp_5,
        "Actual Result": act_5,
        "Status": status_5,
    })

    return test_results


def print_console_table(results: List[Dict[str, str]]) -> None:
    """Format and print results as an aligned ASCII table in the console."""
    sep = "=" * 125
    print("\n" + sep)
    print("FINTRUST DIGITAL BANK - TECHNICAL TEST REPORT (WEEK 2 - PART E)")
    print(sep)
    header = f"{'Test':<26} | {'Expected Result':<40} | {'Actual Result':<45} | {'Status':<6}"
    print(header)
    print("-" * 125)

    for r in results:
        test_col = r["Test"]
        exp_col = (r["Expected Result"][:37] + "...") if len(r["Expected Result"]) > 40 else r["Expected Result"]
        act_col = (r["Actual Result"][:42] + "...") if len(r["Actual Result"]) > 45 else r["Actual Result"]
        status_col = r["Status"]
        print(f"{test_col:<26} | {exp_col:<40} | {act_col:<45} | [{status_col}]")

    print(sep)


def generate_markdown_report(results: List[Dict[str, str]], output_path: str = "docs/technical_test_report.md") -> None:
    """Generate docs/technical_test_report.md formatted with GitHub markdown."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pass_count = sum(1 for r in results if r["Status"] == "PASS")
    total_tests = len(results)

    lines = [
        "# FinTrust Digital Bank — Technical Test Report",
        "",
        "**Week 2 Deliverable: Part E — Technical Testing**  ",
        "**Track**: Machine Learning Engineering  ",
        f"**Status**: **{pass_count}/{total_tests} Tests Passed**",
        "",
        "---",
        "",
        "## 1. Technical Test Summary Table",
        "",
        "| Test | Scenario Description | Expected Result | Actual Result | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]

    for r in results:
        lines.append(
            f"| **{r['Test']}** | {r['Scenario Description']} | {r['Expected Result']} | {r['Actual Result']} | **`{r['Status']}`** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Test Execution Details",
        "",
        "### Test 1: Valid Input",
        "- **Purpose**: Verify that data matching all schema expectations passes without false positives.",
        "- **Inputs**: Synthetic transaction batch with conforming IDs (`FT-T#####`), valid ISO datetimes, numeric amounts, and known categories.",
        "- **Decision**: Pipeline continues with status `[PASSED]`. 0 blocking errors, 0 warnings.",
        "",
        "### Test 2: Missing Values (Tolerated vs Blocking)",
        "- **Purpose**: Verify the dynamic missingness threshold logic on `Device_Type` and `Location`.",
        "- **Inputs**: 100 transaction records with 1 missing value each (1.0% missingness, well below the 5.0% threshold).",
        "- **Decision**: Pipeline logs non-blocking warnings (`missing_values_tolerated`) and allows execution to proceed without failing.",
        "",
        "### Test 3: Unexpected Category",
        "- **Purpose**: Prevent invalid or corrupted categorical labels from entering downstream encoding.",
        "- **Inputs**: Transaction with `Transaction_Status = 'Cancelled'` (unrecognized business state).",
        "- **Decision**: Pipeline halts with a blocking error (`unexpected_categorical_value`), protecting feature encoding from out-of-vocabulary drift.",
        "",
        "### Test 4: Incorrect Data Type",
        "- **Purpose**: Catch corrupted string values in numerical fields before arithmetic or model ingestion.",
        "- **Inputs**: String `'fifty-thousand'` passed to `Amount_NGN`.",
        "- **Decision**: Pipeline halts with a blocking error (`invalid_data_type`).",
        "",
        "### Test 5: Empty Input",
        "- **Purpose**: Ensure that empty batches or corrupted 0-row data files fail immediately and explicitly.",
        "- **Inputs**: `pd.DataFrame` with 0 rows.",
        "- **Decision**: Pipeline detects `empty_dataset` and flags a blocking error (or raises `EmptyDatasetError` in strict mode).",
        "",
        "---",
        "",
        "## 3. Conclusion",
        "",
        f"**{pass_count}/{total_tests} technical tests behaved as expected.**  ",
        "The validation component strictly enforces data quality boundaries while distinguishing between tolerated data noise and pipeline-breaking anomalies.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    results = run_technical_tests()
    print_console_table(results)

    report_path = os.path.join(project_root, "docs", "technical_test_report.md")
    generate_markdown_report(results, report_path)
    print(f"\nSaved Markdown technical test report to: {report_path}")

    pass_count = sum(1 for r in results if r["Status"] == "PASS")
    total_tests = len(results)
    print(f"\n{pass_count}/{total_tests} technical tests behaved as expected.\n")
