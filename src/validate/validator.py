"""Data validation module."""

import pandas as pd
import logging
from typing import List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    rule_name: str
    status: ValidationStatus
    message: str
    actual_value: Any = None
    expected_value: Any = None

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED


@dataclass
class ValidationRule:
    """A validation rule definition."""
    name: str
    check_func: Callable[[pd.DataFrame], bool]
    error_message: str
    is_critical: bool = True  # If True, pipeline stops on failure


class DataValidator:
    """
    Validate DataFrames against defined rules.

    Supports both built-in and custom validation rules.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.results: List[ValidationResult] = []

    def add_rule(self, rule: ValidationRule) -> "DataValidator":
        """Add a validation rule."""
        self.rules.append(rule)
        return self

    def add_uniqueness_check(
        self,
        column: str,
        is_critical: bool = True
    ) -> "DataValidator":
        """Add a uniqueness validation rule."""
        rule = ValidationRule(
            name=f"uniqueness_{column}",
            check_func=lambda df, col=column: len(df) == df[col].nunique(),
            error_message=f"Column '{column}' contains duplicate values",
            is_critical=is_critical
        )
        return self.add_rule(rule)

    def add_not_null_check(
        self,
        column: str,
        is_critical: bool = True
    ) -> "DataValidator":
        """Add a not-null validation rule."""
        rule = ValidationRule(
            name=f"not_null_{column}",
            check_func=lambda df, col=column: df[col].isna().sum() == 0,
            error_message=f"Column '{column}' contains null values",
            is_critical=is_critical
        )
        return self.add_rule(rule)

    def add_distinct_count_check(
        self,
        column: str,
        max_count: int,
        is_critical: bool = True
    ) -> "DataValidator":
        """Add a distinct count validation rule."""
        rule = ValidationRule(
            name=f"distinct_count_{column}",
            check_func=lambda df, col=column, max_c=max_count: df[col].nunique() < max_c,
            error_message=f"Column '{column}' has too many distinct values (max: {max_count})",
            is_critical=is_critical
        )
        return self.add_rule(rule)

    def add_row_count_check(
        self,
        min_rows: int = 0,
        max_rows: Optional[int] = None,
        is_critical: bool = True
    ) -> "DataValidator":
        """Add a row count validation rule."""
        def check(df):
            count = len(df)
            if count < min_rows:
                return False
            if max_rows and count > max_rows:
                return False
            return True

        rule = ValidationRule(
            name="row_count",
            check_func=check,
            error_message=f"Row count out of range (min: {min_rows}, max: {max_rows})",
            is_critical=is_critical
        )
        return self.add_rule(rule)

    def add_custom_check(
        self,
        name: str,
        check_func: Callable[[pd.DataFrame], bool],
        error_message: str,
        is_critical: bool = True
    ) -> "DataValidator":
        """Add a custom validation rule."""
        rule = ValidationRule(
            name=name,
            check_func=check_func,
            error_message=error_message,
            is_critical=is_critical
        )
        return self.add_rule(rule)

    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Run all validation rules against the DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            List of ValidationResult objects
        """
        logger.info(f"Running {len(self.rules)} validation rules")
        self.results = []

        for rule in self.rules:
            try:
                passed = rule.check_func(df)
                status = ValidationStatus.PASSED if passed else ValidationStatus.FAILED

                result = ValidationResult(
                    rule_name=rule.name,
                    status=status,
                    message="Validation passed" if passed else rule.error_message
                )

                if passed:
                    logger.info(f"✓ {rule.name}: PASSED")
                else:
                    log_func = logger.error if rule.is_critical else logger.warning
                    log_func(f"✗ {rule.name}: FAILED - {rule.error_message}")

                self.results.append(result)

            except Exception as e:
                logger.error(f"✗ {rule.name}: ERROR - {str(e)}")
                self.results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    message=f"Validation error: {str(e)}"
                ))

        return self.results

    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self.results)

    def critical_passed(self) -> bool:
        """Check if all critical validations passed."""
        critical_rules = {r.name for r in self.rules if r.is_critical}
        return all(
            r.passed for r in self.results
            if r.rule_name in critical_rules
        )

    def get_summary(self) -> dict:
        """Get validation summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        return {
            "total_rules": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0
        }
