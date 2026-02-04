"""Tests for the validate module."""

import pytest
import pandas as pd

from src.validate.validator import (
    DataValidator,
    ValidationResult,
    ValidationRule,
    ValidationStatus,
)


class TestDataValidator:
    """Tests for DataValidator."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "country": ["US", "US", "UK"],
        })

    def test_uniqueness_check_pass(self, sample_df):
        validator = DataValidator().add_uniqueness_check("id")
        results = validator.validate(sample_df)
        assert len(results) == 1
        assert results[0].passed

    def test_uniqueness_check_fail(self, sample_df):
        validator = DataValidator().add_uniqueness_check("country")
        results = validator.validate(sample_df)
        assert len(results) == 1
        assert not results[0].passed

    def test_not_null_check_pass(self, sample_df):
        validator = DataValidator().add_not_null_check("name")
        results = validator.validate(sample_df)
        assert results[0].passed

    def test_not_null_check_fail(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        validator = DataValidator().add_not_null_check("a")
        results = validator.validate(df)
        assert not results[0].passed

    def test_distinct_count_check_pass(self, sample_df):
        validator = DataValidator().add_distinct_count_check("country", max_count=5)
        results = validator.validate(sample_df)
        assert results[0].passed

    def test_distinct_count_check_fail(self, sample_df):
        validator = DataValidator().add_distinct_count_check("country", max_count=1)
        results = validator.validate(sample_df)
        assert not results[0].passed

    def test_row_count_check_pass(self, sample_df):
        validator = DataValidator().add_row_count_check(min_rows=1, max_rows=10)
        results = validator.validate(sample_df)
        assert results[0].passed

    def test_row_count_check_fail_too_few(self, sample_df):
        validator = DataValidator().add_row_count_check(min_rows=10)
        results = validator.validate(sample_df)
        assert not results[0].passed

    def test_row_count_check_fail_too_many(self, sample_df):
        validator = DataValidator().add_row_count_check(min_rows=0, max_rows=1)
        results = validator.validate(sample_df)
        assert not results[0].passed

    def test_custom_check_pass(self, sample_df):
        validator = DataValidator().add_custom_check(
            name="has_id_column",
            check_func=lambda df: "id" in df.columns,
            error_message="Missing id column",
        )
        results = validator.validate(sample_df)
        assert results[0].passed

    def test_custom_check_fail(self, sample_df):
        validator = DataValidator().add_custom_check(
            name="has_email_column",
            check_func=lambda df: "email" in df.columns,
            error_message="Missing email column",
        )
        results = validator.validate(sample_df)
        assert not results[0].passed

    def test_all_passed_true(self, sample_df):
        validator = (
            DataValidator()
            .add_uniqueness_check("id")
            .add_not_null_check("name")
        )
        validator.validate(sample_df)
        assert validator.all_passed()

    def test_all_passed_false(self, sample_df):
        validator = (
            DataValidator()
            .add_uniqueness_check("country")
            .add_not_null_check("name")
        )
        validator.validate(sample_df)
        assert not validator.all_passed()

    def test_critical_passed(self, sample_df):
        validator = (
            DataValidator()
            .add_uniqueness_check("id", is_critical=True)
            .add_uniqueness_check("country", is_critical=False)
        )
        validator.validate(sample_df)
        assert validator.critical_passed()

    def test_multiple_rules_chained(self, sample_df):
        validator = (
            DataValidator()
            .add_uniqueness_check("id")
            .add_not_null_check("name")
            .add_row_count_check(min_rows=1)
            .add_distinct_count_check("country", max_count=10)
        )
        results = validator.validate(sample_df)
        assert len(results) == 4
        assert all(r.passed for r in results)

    def test_get_summary(self, sample_df):
        validator = (
            DataValidator()
            .add_uniqueness_check("id")
            .add_uniqueness_check("country")
        )
        validator.validate(sample_df)
        summary = validator.get_summary()
        assert summary["total_rules"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5

    def test_validation_result_passed_property(self):
        result = ValidationResult(
            rule_name="test",
            status=ValidationStatus.PASSED,
            message="ok",
        )
        assert result.passed is True

        result_fail = ValidationResult(
            rule_name="test",
            status=ValidationStatus.FAILED,
            message="fail",
        )
        assert result_fail.passed is False
