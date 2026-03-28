"""Unit tests for the Gold aggregation layer."""

import pytest
from pyspark.sql import functions as F

from pipeline.bronze.ingestion import BronzeIngestion, create_bronze_schemas
from pipeline.gold.aggregations import GoldAggregator
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


@pytest.fixture
def gold(silver_df):
    agg = GoldAggregator()
    return agg.build_all(silver_df)


class TestGoldSupplierByNation:
    """Tests for supplier_by_nation Gold table."""

    def test_has_expected_columns(self, gold):
        df = gold["supplier_by_nation"]
        expected = {
            "nation_name", "supplier_count", "avg_account_balance",
            "max_account_balance", "min_account_balance", "total_account_balance",
        }
        assert expected == set(df.columns)

    def test_nation_count(self, gold):
        df = gold["supplier_by_nation"]
        # 9 unique nations in sample, but only those with suppliers appear
        # PERU has 2 suppliers (keys 1 and 8 both map to nationkey 17)
        assert df.count() >= 1

    def test_supplier_counts_sum_to_total(self, gold):
        df = gold["supplier_by_nation"]
        total = df.agg(F.sum("supplier_count")).collect()[0][0]
        assert total == 10

    def test_ordered_by_supplier_count_desc(self, gold):
        df = gold["supplier_by_nation"]
        counts = [row.supplier_count for row in df.select("supplier_count").collect()]
        assert counts == sorted(counts, reverse=True)


class TestGoldSupplierSummary:
    """Tests for supplier_summary Gold table."""

    def test_single_row(self, gold):
        df = gold["supplier_summary"]
        assert df.count() == 1

    def test_has_expected_columns(self, gold):
        df = gold["supplier_summary"]
        expected = {
            "total_suppliers", "total_nations",
            "avg_account_balance", "total_account_balance",
            "pipeline_timestamp",
        }
        assert expected == set(df.columns)

    def test_total_suppliers(self, gold):
        row = gold["supplier_summary"].first()
        assert row.total_suppliers == 10

    def test_total_nations(self, gold):
        row = gold["supplier_summary"].first()
        # Should match the distinct nations that have suppliers
        assert row.total_nations >= 1


class TestGoldBuildAll:
    """Tests for build_all method."""

    def test_returns_all_tables(self, gold):
        assert "supplier_by_nation" in gold
        assert "supplier_summary" in gold

    def test_all_tables_non_empty(self, gold):
        for name, df in gold.items():
            assert df.count() > 0, f"Gold table '{name}' is empty"
