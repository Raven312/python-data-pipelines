"""Unit tests for the Silver transformation layer."""

import pytest
from pyspark.sql import functions as F

from pipeline.bronze.ingestion import BronzeIngestion, create_bronze_schemas
from pipeline.silver.transformations import SilverTransformer


@pytest.fixture
def bronze_suppliers(spark, sample_supplier_data):
    schemas = create_bronze_schemas()
    return BronzeIngestion(spark, "test").ingest_from_dict(
        sample_supplier_data, schema=schemas["supplier"]
    )


@pytest.fixture
def bronze_nations(spark, sample_nation_data):
    schemas = create_bronze_schemas()
    return BronzeIngestion(spark, "test").ingest_from_dict(
        sample_nation_data, schema=schemas["nation"]
    )


@pytest.fixture
def silver_df(bronze_suppliers, bronze_nations):
    return SilverTransformer().transform(bronze_suppliers, bronze_nations)


class TestSilverTransformer:
    """Tests for SilverTransformer.transform."""

    def test_output_has_canonical_columns(self, silver_df):
        expected_cols = {
            "supplier_id", "supplier_name", "supplier_address",
            "supplier_phone", "account_balance", "nation_name", "nation_key",
            "_valid_from", "_valid_to", "_is_current",
        }
        assert expected_cols == set(silver_df.columns)

    def test_row_count_after_join(self, silver_df):
        # All 10 suppliers have a matching nation in the sample data
        assert silver_df.count() == 10

    def test_supplier_names_uppercased(self, silver_df):
        names = [row.supplier_name for row in silver_df.select("supplier_name").collect()]
        for name in names:
            assert name == name.upper()

    def test_scd2_columns_present(self, silver_df):
        row = silver_df.first()
        assert row._valid_from is not None
        assert row._valid_to is None
        assert row._is_current is True

    def test_no_nulls_in_supplier_id(self, silver_df):
        null_count = silver_df.filter(F.col("supplier_id").isNull()).count()
        assert null_count == 0

    def test_no_nulls_in_supplier_name(self, silver_df):
        null_count = silver_df.filter(F.col("supplier_name").isNull()).count()
        assert null_count == 0

    def test_unique_supplier_ids(self, silver_df):
        total = silver_df.count()
        distinct = silver_df.select("supplier_id").distinct().count()
        assert total == distinct


class TestSilverDeduplication:
    """Tests for deduplication logic."""

    def test_dedup_removes_duplicates(self, spark, sample_supplier_data):
        schemas = create_bronze_schemas()
        bronze = BronzeIngestion(spark, "test", batch_id="batch1")
        df1 = bronze.ingest_from_dict(sample_supplier_data, schema=schemas["supplier"])

        bronze2 = BronzeIngestion(spark, "test", batch_id="batch2")
        df2 = bronze2.ingest_from_dict(sample_supplier_data, schema=schemas["supplier"])

        # Union to simulate duplicate ingestion
        combined = df1.union(df2)
        assert combined.count() == 20

        deduped = SilverTransformer._deduplicate(combined, "s_suppkey", "_ingested_at")
        assert deduped.count() == 10

    def test_drop_nulls_removes_null_rows(self, spark):
        data = spark.createDataFrame(
            [(1, "A"), (2, None), (3, "C")],
            schema=["id", "name"],
        )
        cleaned = SilverTransformer._drop_nulls(data, ["name"])
        assert cleaned.count() == 2

    def test_standardize_strings(self, spark):
        data = spark.createDataFrame(
            [(1, "  hello "), (2, "world  ")],
            schema=["id", "name"],
        )
        result = SilverTransformer._standardize_strings(data, ["name"])
        names = [row.name for row in result.select("name").collect()]
        assert names == ["HELLO", "WORLD"]
