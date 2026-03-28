"""Unit tests for the data quality framework."""

import pytest
from pyspark.sql import functions as F

from pipeline.quality.checks import DataQualityChecker, QualityStatus


@pytest.fixture
def clean_df(spark):
    """A clean DataFrame with no quality issues."""
    return spark.createDataFrame(
        [(1, "Alice", 100.0), (2, "Bob", 200.0), (3, "Charlie", 300.0)],
        schema=["id", "name", "balance"],
    )


@pytest.fixture
def dirty_df(spark):
    """A DataFrame with nulls and duplicates."""
    return spark.createDataFrame(
        [(1, "Alice", 100.0), (1, "Alice", 100.0), (2, None, 200.0), (3, "Charlie", None)],
        schema=["id", "name", "balance"],
    )


class TestNotNullCheck:

    def test_passes_on_clean_data(self, clean_df):
        checker = DataQualityChecker("test").expect_column_not_null("name")
        results = checker.run(clean_df)
        assert results[0].passed

    def test_fails_on_null_data(self, dirty_df):
        checker = DataQualityChecker("test").expect_column_not_null("name")
        results = checker.run(dirty_df)
        assert not results[0].passed
        assert results[0].actual_value == 1  # one null


class TestUniqueCheck:

    def test_passes_on_unique_data(self, clean_df):
        checker = DataQualityChecker("test").expect_column_unique("id")
        results = checker.run(clean_df)
        assert results[0].passed

    def test_fails_on_duplicates(self, dirty_df):
        checker = DataQualityChecker("test").expect_column_unique("id")
        results = checker.run(dirty_df)
        assert not results[0].passed


class TestRowCountCheck:

    def test_passes_within_range(self, clean_df):
        checker = DataQualityChecker("test").expect_row_count_between(min_rows=1, max_rows=10)
        results = checker.run(clean_df)
        assert results[0].passed

    def test_fails_below_minimum(self, clean_df):
        checker = DataQualityChecker("test").expect_row_count_between(min_rows=100)
        results = checker.run(clean_df)
        assert not results[0].passed

    def test_fails_above_maximum(self, clean_df):
        checker = DataQualityChecker("test").expect_row_count_between(max_rows=1)
        results = checker.run(clean_df)
        assert not results[0].passed


class TestValuesInSetCheck:

    def test_passes_when_all_in_set(self, clean_df):
        checker = DataQualityChecker("test").expect_column_values_in_set(
            "name", {"Alice", "Bob", "Charlie"}
        )
        results = checker.run(clean_df)
        assert results[0].passed

    def test_fails_with_unexpected_value(self, clean_df):
        checker = DataQualityChecker("test").expect_column_values_in_set(
            "name", {"Alice", "Bob"}
        )
        results = checker.run(clean_df)
        assert not results[0].passed


class TestDistinctCountCheck:

    def test_passes_within_range(self, clean_df):
        checker = DataQualityChecker("test").expect_distinct_count_between(
            "name", min_count=1, max_count=5
        )
        results = checker.run(clean_df)
        assert results[0].passed

    def test_fails_above_max(self, clean_df):
        checker = DataQualityChecker("test").expect_distinct_count_between(
            "name", max_count=1
        )
        results = checker.run(clean_df)
        assert not results[0].passed


class TestMaxBelowCheck:

    def test_passes_below_threshold(self, clean_df):
        checker = DataQualityChecker("test").expect_column_max_below("balance", 500.0)
        results = checker.run(clean_df)
        assert results[0].passed

    def test_fails_above_threshold(self, clean_df):
        checker = DataQualityChecker("test").expect_column_max_below("balance", 100.0)
        results = checker.run(clean_df)
        assert not results[0].passed


class TestCustomCheck:

    def test_custom_check_passes(self, clean_df):
        checker = DataQualityChecker("test").add_custom_check(
            "positive_balance",
            lambda df: df.filter(F.col("balance") <= 0).count() == 0,
            "Negative balances found",
        )
        results = checker.run(clean_df)
        assert results[0].passed


class TestCheckerComposition:

    def test_multiple_checks(self, clean_df):
        checker = (
            DataQualityChecker("multi")
            .expect_column_not_null("id")
            .expect_column_unique("id")
            .expect_row_count_between(min_rows=1)
        )
        results = checker.run(clean_df)
        assert len(results) == 3
        assert checker.all_passed()

    def test_critical_vs_non_critical(self, dirty_df):
        checker = (
            DataQualityChecker("mixed")
            .expect_column_not_null("name", critical=True)
            .expect_column_unique("id", critical=False)
        )
        checker.run(dirty_df)
        assert not checker.all_passed()
        assert not checker.critical_passed()

    def test_summary(self, clean_df):
        checker = (
            DataQualityChecker("summary_test")
            .expect_column_not_null("id")
            .expect_column_unique("id")
        )
        checker.run(clean_df)
        summary = checker.summary()
        assert summary["passed"] == 2
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 1.0
