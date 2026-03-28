"""Data quality check framework for PySpark DataFrames.

Provides a composable, Great-Expectations-inspired quality gate system
that runs between medallion layers. Each check returns a QualityResult;
critical failures halt the pipeline.

Usage:
    checker = (
        DataQualityChecker("silver_suppliers")
        .expect_column_not_null("supplier_id")
        .expect_column_unique("supplier_id")
        .expect_column_values_in_set("nation_name", {"USA", "BRAZIL", ...})
        .expect_row_count_between(min_rows=1)
    )
    results = checker.run(df)
    if not checker.all_passed():
        raise RuntimeError("Quality gate failed")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class QualityStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"


@dataclass
class QualityResult:
    check_name: str
    status: QualityStatus
    message: str
    actual_value: object = None
    expected_value: object = None
    is_critical: bool = True

    @property
    def passed(self) -> bool:
        return self.status == QualityStatus.PASSED


@dataclass
class _QualityCheck:
    name: str
    fn: Callable[[DataFrame], QualityResult]
    is_critical: bool = True


class DataQualityChecker:
    """Composable data quality checker with fluent API."""

    def __init__(self, checkpoint_name: str):
        self.checkpoint_name = checkpoint_name
        self._checks: list[_QualityCheck] = []
        self._results: list[QualityResult] = []

    # ── Expectation builders ──────────────────────────────────────────

    def expect_column_not_null(
        self, column: str, *, critical: bool = True
    ) -> "DataQualityChecker":
        """Assert that a column contains zero null values."""
        def check(df: DataFrame) -> QualityResult:
            null_count = df.filter(F.col(column).isNull()).count()
            passed = null_count == 0
            return QualityResult(
                check_name=f"not_null_{column}",
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message=f"{column}: {null_count} nulls found" if not passed else "OK",
                actual_value=null_count,
                expected_value=0,
                is_critical=critical,
            )

        self._checks.append(_QualityCheck(f"not_null_{column}", check, critical))
        return self

    def expect_column_unique(
        self, column: str, *, critical: bool = True
    ) -> "DataQualityChecker":
        """Assert that all values in a column are unique."""
        def check(df: DataFrame) -> QualityResult:
            total = df.count()
            distinct = df.select(column).distinct().count()
            passed = total == distinct
            return QualityResult(
                check_name=f"unique_{column}",
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message=f"{column}: {total - distinct} duplicates" if not passed else "OK",
                actual_value=distinct,
                expected_value=total,
                is_critical=critical,
            )

        self._checks.append(_QualityCheck(f"unique_{column}", check, critical))
        return self

    def expect_row_count_between(
        self,
        *,
        min_rows: int = 0,
        max_rows: Optional[int] = None,
        critical: bool = True,
    ) -> "DataQualityChecker":
        """Assert that row count falls within the expected range."""
        def check(df: DataFrame) -> QualityResult:
            count = df.count()
            passed = count >= min_rows and (max_rows is None or count <= max_rows)
            return QualityResult(
                check_name="row_count_range",
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message=f"row_count={count}, expected [{min_rows}, {max_rows}]",
                actual_value=count,
                expected_value=f"[{min_rows}, {max_rows}]",
                is_critical=critical,
            )

        self._checks.append(_QualityCheck("row_count_range", check, critical))
        return self

    def expect_column_values_in_set(
        self, column: str, allowed_values: set, *, critical: bool = True
    ) -> "DataQualityChecker":
        """Assert that all column values are within an allowed set."""
        def check(df: DataFrame) -> QualityResult:
            distinct_vals = {row[0] for row in df.select(column).distinct().collect()}
            invalid = distinct_vals - allowed_values
            passed = len(invalid) == 0
            return QualityResult(
                check_name=f"values_in_set_{column}",
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message=f"{column}: invalid values {invalid}" if not passed else "OK",
                actual_value=invalid if not passed else set(),
                expected_value=allowed_values,
                is_critical=critical,
            )

        self._checks.append(_QualityCheck(f"values_in_set_{column}", check, critical))
        return self

    def expect_column_max_below(
        self, column: str, max_value: float, *, critical: bool = True
    ) -> "DataQualityChecker":
        """Assert that the maximum value of a numeric column is below a threshold."""
        def check(df: DataFrame) -> QualityResult:
            actual_max = df.agg(F.max(column)).collect()[0][0]
            passed = actual_max is not None and actual_max <= max_value
            return QualityResult(
                check_name=f"max_below_{column}",
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message=f"{column}: max={actual_max}, threshold={max_value}",
                actual_value=actual_max,
                expected_value=max_value,
                is_critical=critical,
            )

        self._checks.append(_QualityCheck(f"max_below_{column}", check, critical))
        return self

    def expect_distinct_count_between(
        self,
        column: str,
        *,
        min_count: int = 1,
        max_count: Optional[int] = None,
        critical: bool = True,
    ) -> "DataQualityChecker":
        """Assert that the distinct count of a column is within range."""
        def check(df: DataFrame) -> QualityResult:
            distinct = df.select(column).distinct().count()
            passed = distinct >= min_count and (max_count is None or distinct <= max_count)
            return QualityResult(
                check_name=f"distinct_count_{column}",
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message=f"{column}: {distinct} distinct, expected [{min_count}, {max_count}]",
                actual_value=distinct,
                expected_value=f"[{min_count}, {max_count}]",
                is_critical=critical,
            )

        self._checks.append(
            _QualityCheck(f"distinct_count_{column}", check, critical)
        )
        return self

    def add_custom_check(
        self,
        name: str,
        check_fn: Callable[[DataFrame], bool],
        error_message: str = "Custom check failed",
        *,
        critical: bool = True,
    ) -> "DataQualityChecker":
        """Add an arbitrary custom quality check."""
        def check(df: DataFrame) -> QualityResult:
            passed = check_fn(df)
            return QualityResult(
                check_name=name,
                status=QualityStatus.PASSED if passed else QualityStatus.FAILED,
                message="OK" if passed else error_message,
                is_critical=critical,
            )

        self._checks.append(_QualityCheck(name, check, critical))
        return self

    # ── Execution ─────────────────────────────────────────────────────

    def run(self, df: DataFrame) -> list[QualityResult]:
        """Execute all registered checks against the DataFrame."""
        logger.info("Running %d quality checks [%s]", len(self._checks), self.checkpoint_name)
        self._results = []

        for qc in self._checks:
            try:
                result = qc.fn(df)
            except Exception as exc:
                result = QualityResult(
                    check_name=qc.name,
                    status=QualityStatus.ERROR,
                    message=str(exc),
                    is_critical=qc.is_critical,
                )

            self._results.append(result)
            symbol = "PASS" if result.passed else "FAIL"
            log_fn = logger.info if result.passed else logger.error
            log_fn("[%s] %s — %s", symbol, result.check_name, result.message)

        return self._results

    def all_passed(self) -> bool:
        return all(r.passed for r in self._results)

    def critical_passed(self) -> bool:
        return all(r.passed for r in self._results if r.is_critical)

    def summary(self) -> dict:
        passed = sum(1 for r in self._results if r.passed)
        failed = len(self._results) - passed
        return {
            "checkpoint": self.checkpoint_name,
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(len(self._results), 1), 4),
        }
