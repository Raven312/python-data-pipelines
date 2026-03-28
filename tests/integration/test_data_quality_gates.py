"""Integration tests for data quality gates within the pipeline."""

import pytest
from pyspark.sql import functions as F

from pipeline.bronze.ingestion import BronzeIngestion, create_bronze_schemas
from pipeline.quality.checks import DataQualityChecker
from pipeline.silver.transformations import SilverTransformer


@pytest.fixture
def silver_df(spark, sample_supplier_data, sample_nation_data):
    schemas = create_bronze_schemas()
    suppliers = BronzeIngestion(spark, "s").ingest_from_dict(
        sample_supplier_data, schema=schemas["supplier"]
    )
    nations = BronzeIngestion(spark, "n").ingest_from_dict(
        sample_nation_data, schema=schemas["nation"]
    )
    return SilverTransformer().transform(suppliers, nations)


class TestSilverQualityGate:
    """Quality gates that must pass for Silver data."""

    def test_supplier_id_not_null(self, silver_df):
        checker = DataQualityChecker("silver").expect_column_not_null("supplier_id")
        checker.run(silver_df)
        assert checker.all_passed()

    def test_supplier_name_not_null(self, silver_df):
        checker = DataQualityChecker("silver").expect_column_not_null("supplier_name")
        checker.run(silver_df)
        assert checker.all_passed()

    def test_supplier_id_unique(self, silver_df):
        checker = DataQualityChecker("silver").expect_column_unique("supplier_id")
        checker.run(silver_df)
        assert checker.all_passed()

    def test_nation_count_within_bounds(self, silver_df):
        checker = DataQualityChecker("silver").expect_distinct_count_between(
            "nation_name", min_count=1, max_count=25
        )
        checker.run(silver_df)
        assert checker.all_passed()

    def test_full_silver_gate(self, silver_df):
        checker = (
            DataQualityChecker("silver_full")
            .expect_column_not_null("supplier_id")
            .expect_column_not_null("supplier_name")
            .expect_column_unique("supplier_id")
            .expect_row_count_between(min_rows=1)
            .expect_distinct_count_between("nation_name", min_count=1, max_count=25)
        )
        checker.run(silver_df)
        assert checker.all_passed()
        assert checker.summary()["pass_rate"] == 1.0


class TestBronzeQualityGate:
    """Quality gates that must pass for Bronze data."""

    def test_bronze_audit_columns_present(self, spark, sample_supplier_data):
        schemas = create_bronze_schemas()
        bronze = BronzeIngestion(spark, "test")
        df = bronze.ingest_from_dict(sample_supplier_data, schema=schemas["supplier"])

        checker = (
            DataQualityChecker("bronze")
            .expect_column_not_null("_batch_id")
            .expect_column_not_null("_source")
            .expect_column_not_null("_ingested_at")
            .expect_row_count_between(min_rows=1)
        )
        checker.run(df)
        assert checker.all_passed()
