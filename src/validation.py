"""
FinTrust Digital Bank - Data Validation Module
=============================================
Week 2: ML Engineering Pipeline Component

This module implements a production-grade data validation system for the FinTrust
Digital Bank project. It validates customer and transaction datasets against predefined
schemas, enforces referential integrity across relational boundaries, and distinguishes
blocking errors from non-blocking warnings with configurable tolerance thresholds.

Key Features:
- Empty dataset detection (blocking error)
- Missing and unexpected column checks (schema drift)
- Primary key uniqueness checks (blocking error on duplicates)
- Data type compatibility and regex format verification (e.g. FT-C##### / FT-T######)
- Strict categorical value enforcement
- Missing value analysis with configurable dynamic thresholds (warning < 5%, error >= 5%)
- Cross-dataset referential integrity validation (Customer_ID foreign key check)
- Structured, non-crashing reporting (ValidationReport dataclass)
- Testable, pure DataFrame-oriented architecture
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)

ValidationSeverity = Literal["error", "warning"]


# =====================================================================
# Custom Exceptions
# =====================================================================

class DataValidationError(Exception):
    """Base exception for data validation failures."""
    pass


class EmptyDatasetError(DataValidationError):
    """Raised when an input DataFrame contains zero rows."""
    pass


class ReferentialIntegrityError(DataValidationError):
    """Raised when foreign key references do not match primary key entities."""
    pass


class SchemaViolationError(DataValidationError):
    """Raised when a DataFrame contains blocking schema violations."""
    pass


# =====================================================================
# Schema and Report Data Structures
# =====================================================================

@dataclass
class ColumnSchema:
    """
    Schema specification for a single DataFrame column.

    Attributes:
        name: Name of the column.
        expected_type: Expected data type ('numeric', 'datetime', 'string', 'integer').
        required: Whether the column must be present in the dataset.
        allowed_values: Optional list of acceptable categorical values.
        regex_pattern: Optional regex pattern the column values must match.
        allow_null: Whether missing/null values are permitted.
        max_warning_missing_pct: Maximum missingness percentage (0.0 to 100.0)
            tolerated as a non-blocking warning. If missingness exceeds this threshold
            (or if allow_null=False), it triggers a blocking error.
    """
    name: str
    expected_type: Literal["numeric", "datetime", "string", "integer"]
    required: bool = True
    allowed_values: Optional[List[str]] = None
    regex_pattern: Optional[str] = None
    allow_null: bool = False
    max_warning_missing_pct: float = 0.0


@dataclass
class DatasetSchema:
    """
    Schema specification for an entire dataset.

    Attributes:
        name: Name or label for the dataset.
        primary_key: Name of the primary key column (must be unique and non-null).
        columns: Dictionary mapping column names to ColumnSchema instances.
        allow_extra_columns: If True, extra unexpected columns trigger warnings;
            if False, they trigger blocking errors.
    """
    name: str
    primary_key: Optional[str] = None
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    allow_extra_columns: bool = True


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue or warning.

    Attributes:
        severity: Severity level ('error' for blocking issues, 'warning' for tolerated issues).
        check: Name of the check that detected the issue.
        message: Human-readable explanation of the issue.
        column: Name of the column involved, if applicable.
        details: Optional dictionary containing additional diagnostic context.
    """
    severity: ValidationSeverity
    check: str
    message: str
    column: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """
    Structured summary of a validation execution.

    Attributes:
        dataset_name: Name of the dataset or check evaluated.
        is_valid: True if no blocking errors were found; False otherwise.
        total_rows: Number of rows in the validated dataset.
        total_columns: Number of columns in the validated dataset.
        issues: List of all ValidationIssue objects detected.
        check_details: Diagnostic details and metrics computed during validation.
    """
    dataset_name: str
    is_valid: bool
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = field(default_factory=list)
    check_details: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Return all blocking errors."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Return all non-blocking warnings."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def error_count(self) -> int:
        """Count of blocking errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Count of non-blocking warnings."""
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary representation."""
        return asdict(self)

    def summary(self) -> str:
        """
        Generate a human-readable summary string suitable for console logs or reports.

        Returns:
            Formatted multiline string.
        """
        status_str = "PASSED" if self.is_valid else "FAILED"
        separator = "=" * 70
        lines = [
            separator,
            f"VALIDATION REPORT: {self.dataset_name}",
            f"Overall Status : [{status_str}]",
            f"Total Rows     : {self.total_rows:,}",
            f"Total Columns  : {self.total_columns}",
            f"Errors (Block) : {self.error_count}",
            f"Warnings       : {self.warning_count}",
            "-" * 70,
        ]

        if self.errors:
            lines.append("BLOCKING ERRORS:")
            for idx, err in enumerate(self.errors, 1):
                col_info = f" [{err.column}]" if err.column else ""
                lines.append(f"  {idx}. ({err.check}){col_info}: {err.message}")

        if self.warnings:
            lines.append("WARNINGS (NON-BLOCKING):")
            for idx, warn in enumerate(self.warnings, 1):
                col_info = f" [{warn.column}]" if warn.column else ""
                lines.append(f"  {idx}. ({warn.check}){col_info}: {warn.message}")

        if not self.errors and not self.warnings:
            lines.append("All validation checks passed with zero errors or warnings.")

        # Append missing values summary if available
        mv = self.check_details.get("missing_values")
        if mv:
            lines.append("-" * 70)
            lines.append("MISSING VALUES BREAKDOWN:")
            for col, stats in mv.items():
                if stats["count"] > 0:
                    lines.append(
                        f"  - {col:<26}: {stats['count']:>5} nulls "
                        f"({stats['percentage']:>5.2f}%) "
                        f"[Allowed: {stats['allow_null']}, Threshold: {stats['max_warning_pct']:.1f}%]"
                    )

        lines.append(separator)
        return "\n".join(lines)


# =====================================================================
# Predefined Schemas: Customer & Transaction Datasets
# =====================================================================

CUSTOMER_SCHEMA = DatasetSchema(
    name="FinTrust Customer Data",
    primary_key="Customer_ID",
    allow_extra_columns=True,
    columns={
        "Customer_ID": ColumnSchema(
            name="Customer_ID",
            expected_type="string",
            required=True,
            regex_pattern=r"^FT-C\d{5}$",
            allow_null=False,
        ),
        "Customer_Name": ColumnSchema(
            name="Customer_Name",
            expected_type="string",
            required=True,
            allow_null=False,
        ),
        "Age": ColumnSchema(
            name="Age",
            expected_type="integer",
            required=True,
            allow_null=False,
        ),
        "Gender": ColumnSchema(
            name="Gender",
            expected_type="string",
            required=True,
            allowed_values=["Male", "Female", "Prefer not to say"],
            allow_null=False,
        ),
        "City": ColumnSchema(
            name="City",
            expected_type="string",
            required=True,
            allowed_values=[
                "Lagos", "Kano", "Port Harcourt", "Benin City",
                "Kaduna", "Abuja", "Ibadan", "Enugu",
            ],
            allow_null=False,
        ),
        "Customer_Segment": ColumnSchema(
            name="Customer_Segment",
            expected_type="string",
            required=True,
            allowed_values=["Premium", "Everyday", "SME", "Student"],
            allow_null=False,
        ),
        "Account_Type": ColumnSchema(
            name="Account_Type",
            expected_type="string",
            required=True,
            allowed_values=["Savings", "Current", "Premium"],
            allow_null=False,
        ),
        "Tenure_Months": ColumnSchema(
            name="Tenure_Months",
            expected_type="integer",
            required=True,
            allow_null=False,
        ),
        "Digital_Engagement_Score": ColumnSchema(
            name="Digital_Engagement_Score",
            expected_type="numeric",
            required=True,
            allow_null=False,
        ),
        "Monthly_Income_Band": ColumnSchema(
            name="Monthly_Income_Band",
            expected_type="string",
            required=True,
            allowed_values=["Below 100k", "100k-249k", "250k-499k", "500k-999k", "1m+"],
            allow_null=False,
        ),
        "Preferred_Channel": ColumnSchema(
            name="Preferred_Channel",
            expected_type="string",
            required=True,
            allowed_values=["Mobile App", "USSD", "Web"],
            allow_null=False,
        ),
        "Account_Status": ColumnSchema(
            name="Account_Status",
            expected_type="string",
            required=True,
            allowed_values=["Active", "Dormant", "Restricted"],
            allow_null=False,
        ),
    },
)


TRANSACTION_SCHEMA = DatasetSchema(
    name="FinTrust Transaction Data",
    primary_key="Transaction_ID",
    allow_extra_columns=True,
    columns={
        "Transaction_ID": ColumnSchema(
            name="Transaction_ID",
            expected_type="string",
            required=True,
            regex_pattern=r"^FT-T\d{6}$",
            allow_null=False,
        ),
        "Customer_ID": ColumnSchema(
            name="Customer_ID",
            expected_type="string",
            required=True,
            regex_pattern=r"^FT-C\d{5}$",
            allow_null=False,
        ),
        "Transaction_DateTime": ColumnSchema(
            name="Transaction_DateTime",
            expected_type="datetime",
            required=True,
            allow_null=False,
        ),
        "Transaction_Type": ColumnSchema(
            name="Transaction_Type",
            expected_type="string",
            required=True,
            allowed_values=[
                "Card Purchase", "Cash Withdrawal", "Transfer",
                "Deposit", "Airtime/Data", "Bill Payment",
            ],
            allow_null=False,
        ),
        "Amount_NGN": ColumnSchema(
            name="Amount_NGN",
            expected_type="numeric",
            required=True,
            allow_null=False,
        ),
        "Channel": ColumnSchema(
            name="Channel",
            expected_type="string",
            required=True,
            allowed_values=["Mobile App", "ATM", "POS", "Web", "USSD"],
            allow_null=False,
        ),
        "Device_Type": ColumnSchema(
            name="Device_Type",
            expected_type="string",
            required=True,
            allowed_values=["Android", "POS Terminal", "Web Browser", "ATM Terminal", "iOS"],
            allow_null=True,
            max_warning_missing_pct=5.0,  # ~0.8% known missingness tolerated as warning; >5% is error
        ),
        "Location": ColumnSchema(
            name="Location",
            expected_type="string",
            required=True,
            allowed_values=[
                "Lagos", "Kano", "Port Harcourt", "Benin City",
                "Kaduna", "Abuja", "Ibadan", "Enugu",
            ],
            allow_null=True,
            max_warning_missing_pct=5.0,  # ~0.8% known missingness tolerated as warning; >5% is error
        ),
        "International_Transaction": ColumnSchema(
            name="International_Transaction",
            expected_type="string",
            required=True,
            allowed_values=["No", "Yes"],
            allow_null=False,
        ),
        "Transaction_Status": ColumnSchema(
            name="Transaction_Status",
            expected_type="string",
            required=True,
            allowed_values=["Reversed", "Successful", "Failed", "Pending"],
            allow_null=False,
        ),
        "Risk_Review_Flag": ColumnSchema(
            name="Risk_Review_Flag",
            expected_type="string",
            required=True,
            allowed_values=["Yes", "No"],
            allow_null=False,
        ),
    },
)


# =====================================================================
# Core DataValidator Class
# =====================================================================

class DataValidator:
    """
    Validates a pandas DataFrame against a defined DatasetSchema.

    Performs comprehensive quality, type, pattern, categorical, and nullability
    checks, categorizing findings into blocking errors and non-blocking warnings.
    """

    def __init__(self, schema: DatasetSchema):
        """
        Initialize the DataValidator with a schema.

        Args:
            schema: DatasetSchema defining expected columns, types, and constraints.
        """
        self.schema = schema

    @classmethod
    def for_customer_data(cls) -> DataValidator:
        """Create a DataValidator configured for FinTrust Customer Data."""
        return cls(CUSTOMER_SCHEMA)

    @classmethod
    def for_transaction_data(cls) -> DataValidator:
        """Create a DataValidator configured for FinTrust Transaction Data."""
        return cls(TRANSACTION_SCHEMA)

    def validate(self, df: pd.DataFrame, raise_on_error: bool = False) -> ValidationReport:
        """
        Validate a loaded pandas DataFrame against the configured schema.

        Args:
            df: The pandas DataFrame to validate.
            raise_on_error: If True, raises a DataValidationError when blocking
                errors or empty datasets are detected. Defaults to False.

        Returns:
            ValidationReport containing overall status, issues list, and metrics.

        Raises:
            EmptyDatasetError: If df has 0 rows and raise_on_error is True.
            SchemaViolationError: If blocking errors occur and raise_on_error is True.
        """
        issues: List[ValidationIssue] = []
        check_details: Dict[str, Any] = {}

        # 1. Empty Dataset Check
        empty_issues = self._check_empty(df)
        if empty_issues:
            issues.extend(empty_issues)
            if raise_on_error:
                raise EmptyDatasetError(empty_issues[0].message)
            return ValidationReport(
                dataset_name=self.schema.name,
                is_valid=False,
                total_rows=len(df),
                total_columns=len(df.columns),
                issues=issues,
                check_details={"empty": True},
            )

        # 2. Column Presence / Schema Drift Check
        col_issues, col_details = self._check_columns(df)
        issues.extend(col_issues)
        check_details["columns"] = col_details

        # 3. Primary Key Uniqueness Check
        if self.schema.primary_key:
            pk_issues = self._check_primary_key(df)
            issues.extend(pk_issues)

        # 4. Missing Values Check (with dynamic tolerance threshold)
        mv_issues, mv_details = self._check_missing_values(df)
        issues.extend(mv_issues)
        check_details["missing_values"] = mv_details

        # 5. Data Types Check
        dtype_issues, dtype_details = self._check_data_types(df)
        issues.extend(dtype_issues)
        check_details["data_types"] = dtype_details

        # 6. ID Regex Pattern Check
        pattern_issues = self._check_regex_patterns(df)
        issues.extend(pattern_issues)

        # 7. Categorical Allowed Values Check
        cat_issues = self._check_categorical_values(df)
        issues.extend(cat_issues)

        # Determine overall validity (valid if 0 blocking errors)
        has_errors = any(i.severity == "error" for i in issues)
        is_valid = not has_errors

        report = ValidationReport(
            dataset_name=self.schema.name,
            is_valid=is_valid,
            total_rows=len(df),
            total_columns=len(df.columns),
            issues=issues,
            check_details=check_details,
        )

        if raise_on_error and not is_valid:
            error_msgs = "; ".join(e.message for e in report.errors)
            raise SchemaViolationError(
                f"Validation failed for '{self.schema.name}' with {report.error_count} error(s): {error_msgs}"
            )

        return report

    # -----------------------------------------------------------------
    # Internal Check Methods
    # -----------------------------------------------------------------

    def _check_empty(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check if dataset contains zero rows."""
        if len(df) == 0:
            return [
                ValidationIssue(
                    severity="error",
                    check="empty_dataset",
                    message=f"Dataset '{self.schema.name}' is completely empty (0 rows).",
                )
            ]
        return []

    def _check_columns(self, df: pd.DataFrame) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Check for missing required columns and unexpected extra columns."""
        issues: List[ValidationIssue] = []
        df_cols = set(df.columns)
        expected_cols = set(self.schema.columns.keys())

        missing_required: List[str] = []
        for col_name, col_schema in self.schema.columns.items():
            if col_schema.required and col_name not in df_cols:
                missing_required.append(col_name)

        if missing_required:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="missing_columns",
                    message=f"Missing required column(s): {sorted(missing_required)}",
                    details={"missing": sorted(missing_required)},
                )
            )

        extra_cols = df_cols - expected_cols
        if extra_cols:
            severity: ValidationSeverity = (
                "warning" if self.schema.allow_extra_columns else "error"
            )
            issues.append(
                ValidationIssue(
                    severity=severity,
                    check="unexpected_columns",
                    message=f"Unexpected extra column(s) detected: {sorted(list(extra_cols))}",
                    details={"extra": sorted(list(extra_cols))},
                )
            )

        details = {
            "expected_count": len(expected_cols),
            "actual_count": len(df_cols),
            "missing_required": sorted(missing_required),
            "extra_columns": sorted(list(extra_cols)),
        }
        return issues, details

    def _check_primary_key(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check primary key existence and duplicate values."""
        issues: List[ValidationIssue] = []
        pk = self.schema.primary_key
        if not pk or pk not in df.columns:
            return issues

        duplicate_mask = df[pk].duplicated(keep=False)
        duplicate_count = df[pk].duplicated().sum()

        if duplicate_count > 0:
            sample_dups = df.loc[duplicate_mask, pk].dropna().unique()[:5].tolist()
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="duplicate_primary_key",
                    column=pk,
                    message=(
                        f"Primary key '{pk}' contains {duplicate_count:,} duplicate value(s). "
                        f"Sample duplicates: {sample_dups}"
                    ),
                    details={
                        "duplicate_count": int(duplicate_count),
                        "sample_duplicates": sample_dups,
                    },
                )
            )

        return issues

    def _check_missing_values(
        self, df: pd.DataFrame
    ) -> Tuple[List[ValidationIssue], Dict[str, Dict[str, Any]]]:
        """
        Inspect missing values per column and enforce dynamic thresholds.

        - If nulls are strictly disallowed (allow_null=False), any missing value triggers an error.
        - If allow_null=True and null_pct <= max_warning_missing_pct, triggers a warning.
        - If allow_null=True and null_pct > max_warning_missing_pct, triggers an error.
        """
        issues: List[ValidationIssue] = []
        details: Dict[str, Dict[str, Any]] = {}
        total_rows = len(df)

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = (null_count / total_rows) * 100.0 if total_rows > 0 else 0.0

            col_schema = self.schema.columns.get(col)
            allow_null = col_schema.allow_null if col_schema else False
            max_warning_pct = col_schema.max_warning_missing_pct if col_schema else 0.0

            details[col] = {
                "count": null_count,
                "percentage": round(null_pct, 4),
                "allow_null": allow_null,
                "max_warning_pct": max_warning_pct,
            }

            if null_count == 0:
                continue

            if not allow_null:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="missing_values_strict",
                        column=col,
                        message=(
                            f"Column '{col}' has {null_count:,} ({null_pct:.2f}%) missing values "
                            f"(nulls strictly prohibited)."
                        ),
                        details={"null_count": null_count, "null_pct": null_pct},
                    )
                )
            elif null_pct > max_warning_pct:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="missing_values_threshold_exceeded",
                        column=col,
                        message=(
                            f"Column '{col}' missingness of {null_pct:.2f}% ({null_count:,} rows) "
                            f"exceeds tolerance threshold of {max_warning_pct:.2f}%."
                        ),
                        details={
                            "null_count": null_count,
                            "null_pct": null_pct,
                            "threshold": max_warning_pct,
                        },
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        check="missing_values_tolerated",
                        column=col,
                        message=(
                            f"Column '{col}' has {null_count:,} ({null_pct:.2f}%) missing values "
                            f"(tolerated non-blocking warning, below {max_warning_pct:.2f}% limit)."
                        ),
                        details={
                            "null_count": null_count,
                            "null_pct": null_pct,
                            "threshold": max_warning_pct,
                        },
                    )
                )

        return issues, details

    def _check_data_types(
        self, df: pd.DataFrame
    ) -> Tuple[List[ValidationIssue], Dict[str, str]]:
        """Verify that column values can conform to the expected logical data type."""
        issues: List[ValidationIssue] = []
        details: Dict[str, str] = {}

        for col_name, col_schema in self.schema.columns.items():
            if col_name not in df.columns:
                continue

            series = df[col_name]
            expected = col_schema.expected_type
            details[col_name] = str(series.dtype)

            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            if expected == "numeric":
                coerced = pd.to_numeric(non_null, errors="coerce")
                invalid_count = coerced.isna().sum()
                if invalid_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            check="invalid_data_type",
                            column=col_name,
                            message=(
                                f"Column '{col_name}' expected numeric values, but contains "
                                f"{invalid_count:,} non-convertible value(s)."
                            ),
                            details={"invalid_count": int(invalid_count)},
                        )
                    )

            elif expected == "integer":
                coerced = pd.to_numeric(non_null, errors="coerce")
                invalid_count = coerced.isna().sum()
                if invalid_count > 0:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            check="invalid_data_type",
                            column=col_name,
                            message=(
                                f"Column '{col_name}' expected integer values, but contains "
                                f"{invalid_count:,} non-numeric value(s)."
                            ),
                            details={"invalid_count": int(invalid_count)},
                        )
                    )
                else:
                    # Check if floating point numbers with decimals exist
                    non_int_count = (coerced % 1 != 0).sum()
                    if non_int_count > 0:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                check="invalid_data_type",
                                column=col_name,
                                message=(
                                    f"Column '{col_name}' expected integer values, but contains "
                                    f"{non_int_count:,} non-integer decimal value(s)."
                                ),
                                details={"non_integer_count": int(non_int_count)},
                            )
                        )

            elif expected == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(series):
                    coerced = pd.to_datetime(non_null, errors="coerce", format="mixed")
                    invalid_count = coerced.isna().sum()
                    if invalid_count > 0:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                check="invalid_data_type",
                                column=col_name,
                                message=(
                                    f"Column '{col_name}' expected datetime values, but contains "
                                    f"{invalid_count:,} unparseable datetime string(s)."
                                ),
                                details={"invalid_count": int(invalid_count)},
                            )
                        )

        return issues, details

    def _check_regex_patterns(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate non-null string values against specified regular expressions."""
        issues: List[ValidationIssue] = []

        for col_name, col_schema in self.schema.columns.items():
            if col_name not in df.columns or not col_schema.regex_pattern:
                continue

            non_null = df[col_name].dropna().astype(str)
            if len(non_null) == 0:
                continue

            pattern = re.compile(col_schema.regex_pattern)
            mismatches = non_null[~non_null.str.match(pattern)]
            mismatch_count = len(mismatches)

            if mismatch_count > 0:
                sample_mismatches = mismatches.head(5).tolist()
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="regex_format_mismatch",
                        column=col_name,
                        message=(
                            f"Column '{col_name}' has {mismatch_count:,} value(s) violating regex "
                            f"'{col_schema.regex_pattern}'. Sample: {sample_mismatches}"
                        ),
                        details={
                            "pattern": col_schema.regex_pattern,
                            "mismatch_count": mismatch_count,
                            "sample_mismatches": sample_mismatches,
                        },
                    )
                )

        return issues

    def _check_categorical_values(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Verify that categorical columns only contain values from their allowed list."""
        issues: List[ValidationIssue] = []

        for col_name, col_schema in self.schema.columns.items():
            if col_name not in df.columns or not col_schema.allowed_values:
                continue

            non_null = df[col_name].dropna().astype(str)
            if len(non_null) == 0:
                continue

            actual_categories = set(non_null.unique())
            expected_categories = set(col_schema.allowed_values)
            unexpected = actual_categories - expected_categories

            if unexpected:
                unexpected_list = sorted(list(unexpected))
                # Count affected rows
                affected_count = int(non_null.isin(unexpected).sum())
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="unexpected_categorical_value",
                        column=col_name,
                        message=(
                            f"Column '{col_name}' contains unexpected category value(s): {unexpected_list}. "
                            f"Affected rows: {affected_count:,}."
                        ),
                        details={
                            "unexpected_categories": unexpected_list,
                            "expected_categories": sorted(list(expected_categories)),
                            "affected_rows": affected_count,
                        },
                    )
                )

        return issues


# =====================================================================
# Cross-Dataset Referential Integrity Validation
# =====================================================================

def check_referential_integrity(
    customer_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    join_key: str = "Customer_ID",
    raise_on_error: bool = False,
) -> ValidationReport:
    """
    Validate relational referential integrity between Customer and Transaction Data.

    Enforces that every transaction's foreign key (`join_key`) exists in the parent
    customer dataset. Also calculates customer coverage metrics (e.g. inactive customers).

    Args:
        customer_df: Parent Customer DataFrame.
        transaction_df: Child Transaction DataFrame.
        join_key: Foreign key column name present in both datasets. Defaults to 'Customer_ID'.
        raise_on_error: If True, raises ReferentialIntegrityError when orphan transactions
            are detected. Defaults to False.

    Returns:
        ValidationReport with referential integrity status, issue details, and coverage metrics.

    Raises:
        ReferentialIntegrityError: If orphan transactions exist and raise_on_error=True.
        EmptyDatasetError: If either DataFrame is empty and raise_on_error=True.
    """
    issues: List[ValidationIssue] = []
    check_details: Dict[str, Any] = {}

    # Check for empty datasets
    if len(customer_df) == 0 or len(transaction_df) == 0:
        msg = f"Cannot perform referential integrity check: Customer_Data has {len(customer_df)} rows, Transaction_Data has {len(transaction_df)} rows."
        issues.append(
            ValidationIssue(
                severity="error",
                check="referential_integrity_empty_dataset",
                message=msg,
            )
        )
        if raise_on_error:
            raise EmptyDatasetError(msg)
        return ValidationReport(
            dataset_name="Referential Integrity Check",
            is_valid=False,
            total_rows=len(transaction_df),
            total_columns=0,
            issues=issues,
            check_details={"error": "Empty dataset"},
        )

    # Check join key presence in both
    for name, df in [("Customer_Data", customer_df), ("Transaction_Data", transaction_df)]:
        if join_key not in df.columns:
            msg = f"Join key '{join_key}' is missing from {name}."
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="missing_join_key",
                    column=join_key,
                    message=msg,
                )
            )

    if issues:
        if raise_on_error:
            raise ReferentialIntegrityError(issues[0].message)
        return ValidationReport(
            dataset_name="Referential Integrity Check",
            is_valid=False,
            total_rows=len(transaction_df),
            total_columns=0,
            issues=issues,
        )

    # Distinct IDs extraction
    customer_ids: Set[str] = set(customer_df[join_key].dropna().astype(str).unique())
    transaction_customer_ids: Set[str] = set(
        transaction_df[join_key].dropna().astype(str).unique()
    )

    # 1. Orphan Check: Transactions referencing non-existent customers (BLOCKING ERROR)
    orphan_ids = transaction_customer_ids - customer_ids
    if orphan_ids:
        orphan_sample = sorted(list(orphan_ids))[:5]
        orphan_mask = transaction_df[join_key].astype(str).isin(orphan_ids)
        orphan_tx_count = int(orphan_mask.sum())

        issues.append(
            ValidationIssue(
                severity="error",
                check="referential_integrity_orphan_records",
                column=join_key,
                message=(
                    f"Referential integrity failure: {len(orphan_ids):,} unique '{join_key}'s in "
                    f"Transaction_Data do not exist in Customer_Data. "
                    f"Affects {orphan_tx_count:,} transaction rows. Sample orphans: {orphan_sample}"
                ),
                details={
                    "orphan_id_count": len(orphan_ids),
                    "orphan_transaction_rows": orphan_tx_count,
                    "sample_orphans": orphan_sample,
                },
            )
        )

    # 2. Coverage Check: Customers with no transactions (INFORMATIONAL WARNING)
    inactive_cust_ids = customer_ids - transaction_customer_ids
    inactive_count = len(inactive_cust_ids)
    inactive_pct = (inactive_count / len(customer_ids)) * 100.0 if customer_ids else 0.0

    if inactive_count > 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                check="customer_transaction_coverage",
                column=join_key,
                message=(
                    f"{inactive_count:,} out of {len(customer_ids):,} customers ({inactive_pct:.2f}%) "
                    f"have zero transaction records in the transaction dataset."
                ),
                details={
                    "inactive_customer_count": inactive_count,
                    "inactive_customer_pct": round(inactive_pct, 2),
                    "sample_inactive_customers": sorted(list(inactive_cust_ids))[:5],
                },
            )
        )

    check_details["distinct_customers"] = len(customer_ids)
    check_details["distinct_tx_customers"] = len(transaction_customer_ids)
    check_details["orphan_customer_count"] = len(orphan_ids)
    check_details["inactive_customer_count"] = inactive_count
    check_details["active_customer_coverage_pct"] = round(
        ((len(customer_ids) - inactive_count) / len(customer_ids)) * 100.0, 2
    )

    is_valid = len(orphan_ids) == 0

    report = ValidationReport(
        dataset_name="Cross-Dataset Referential Integrity (Customer <-> Transaction)",
        is_valid=is_valid,
        total_rows=len(transaction_df),
        total_columns=len(customer_df.columns) + len(transaction_df.columns),
        issues=issues,
        check_details=check_details,
    )

    if raise_on_error and not is_valid:
        raise ReferentialIntegrityError(report.errors[0].message)

    return report


# =====================================================================
# Functional Wrappers & Pipeline Integration
# =====================================================================

def validate_customer_data(
    df: pd.DataFrame, raise_on_error: bool = False
) -> ValidationReport:
    """
    Validate a FinTrust Customer DataFrame.

    Args:
        df: Customer pandas DataFrame.
        raise_on_error: Whether to raise an exception on blocking errors.

    Returns:
        ValidationReport with validation findings.
    """
    return DataValidator.for_customer_data().validate(df, raise_on_error=raise_on_error)


def validate_transaction_data(
    df: pd.DataFrame, raise_on_error: bool = False
) -> ValidationReport:
    """
    Validate a FinTrust Transaction DataFrame.

    Args:
        df: Transaction pandas DataFrame.
        raise_on_error: Whether to raise an exception on blocking errors.

    Returns:
        ValidationReport with validation findings.
    """
    return DataValidator.for_transaction_data().validate(df, raise_on_error=raise_on_error)


def validate_fintrust_pipeline(
    customer_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    raise_on_error: bool = False,
) -> Dict[str, ValidationReport]:
    """
    Execute full pipeline data validation covering:
      1. Customer Data schema & quality validation
      2. Transaction Data schema & quality validation
      3. Cross-dataset referential integrity validation

    Args:
        customer_df: Customer pandas DataFrame.
        transaction_df: Transaction pandas DataFrame.
        raise_on_error: Whether to abort on blocking error.

    Returns:
        Dictionary mapping stage names to their respective ValidationReport objects.
    """
    reports: Dict[str, ValidationReport] = {}

    reports["customer_data"] = validate_customer_data(
        customer_df, raise_on_error=raise_on_error
    )
    reports["transaction_data"] = validate_transaction_data(
        transaction_df, raise_on_error=raise_on_error
    )
    reports["referential_integrity"] = check_referential_integrity(
        customer_df, transaction_df, raise_on_error=raise_on_error
    )

    return reports


# =====================================================================
# Standalone CLI / Verification Usage Example
# =====================================================================

if __name__ == "__main__":
    import os
    import sys

    # Configure readable console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("FINTRUST DIGITAL BANK - DATA VALIDATION RUNNER (WEEK 2)")
    print("=" * 70 + "\n")

    # Locate input files (CSV preferred, fallback to XLSX)
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

    cust_csv = os.path.join(raw_dir, "FinTrust_Customer_Data.csv")
    cust_xlsx = os.path.join(raw_dir, "FinTrust_Customer_Data.xlsx")
    tx_csv = os.path.join(raw_dir, "FinTrust_Transaction_Data.csv")
    tx_xlsx = os.path.join(raw_dir, "FinTrust_Transaction_Data.xlsx")

    # Load Customer Data
    if os.path.exists(cust_csv):
        print(f"Loading Customer Data from CSV: {cust_csv}")
        cust_df = pd.read_csv(cust_csv)
    elif os.path.exists(cust_xlsx):
        print(f"Loading Customer Data from XLSX: {cust_xlsx}")
        cust_df = pd.read_excel(cust_xlsx)
    else:
        print(f"ERROR: Could not find Customer Data in {raw_dir}")
        sys.exit(1)

    # Load Transaction Data
    if os.path.exists(tx_csv):
        print(f"Loading Transaction Data from CSV: {tx_csv}")
        tx_df = pd.read_csv(tx_csv)
    elif os.path.exists(tx_xlsx):
        print(f"Loading Transaction Data from XLSX: {tx_xlsx}")
        tx_df = pd.read_excel(tx_xlsx)
    else:
        print(f"ERROR: Could not find Transaction Data in {raw_dir}")
        sys.exit(1)

    # Run full pipeline validation
    results = validate_fintrust_pipeline(cust_df, tx_df, raise_on_error=False)

    all_passed = True
    for report_name, report in results.items():
        print("\n" + report.summary())
        if not report.is_valid:
            all_passed = False

    # Print Final Pipeline Readiness Assessment
    print("\n" + "#" * 70)
    if all_passed:
        print("PIPELINE GATE DECISION: [PASS] - Datasets are valid for ML Feature Pipeline.")
        print("Notes: Device_Type and Location missingness were identified as tolerated warnings.")
    else:
        print("PIPELINE GATE DECISION: [BLOCK] - Blocking errors found. Resolve before proceeding.")
    print("#" * 70 + "\n")
