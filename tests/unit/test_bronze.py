"""Unit tests for the Bronze ingestion layer."""

import pytest
from pyspark.sql.types import IntegerType, StringType

from pipeline.bronze.ingestion import BronzeIngestion, create_bronze_schemas


class TestBronzeIngestion:
    """Tests for BronzeIngestion.ingest_from_dict."""

    def test_ingest_adds_audit_columns(self, spark, sample_supplier_data):
        bronze = BronzeIngestion(spark, "test_source", batch_id="test_batch_001")
        df = bronze.ingest_from_dict(sample_supplier_data)

        assert "_ingested_at" in df.columns
        assert "_source" in df.columns
        assert "_batch_id" in df.columns

    def test_ingest_preserves_row_count(self, spark, sample_supplier_data):
        bronze = BronzeIngestion(spark, "test_source")
        df = bronze.ingest_from_dict(sample_supplier_data)

        assert df.count() == 10

    def test_ingest_source_tag(self, spark, sample_supplier_data):
        bronze = BronzeIngestion(spark, "my_source")
        df = bronze.ingest_from_dict(sample_supplier_data)

        sources = [row._source for row in df.select("_source").collect()]
        assert all(s == "my_source" for s in sources)

    def test_ingest_batch_id(self, spark, sample_supplier_data):
        bronze = BronzeIngestion(spark, "src", batch_id="B42")
        df = bronze.ingest_from_dict(sample_supplier_data)

        batch_ids = [row._batch_id for row in df.select("_batch_id").collect()]
        assert all(b == "B42" for b in batch_ids)

    def test_ingest_with_schema_enforcement(self, spark, sample_supplier_data, supplier_schema):
        bronze = BronzeIngestion(spark, "test_source")
        df = bronze.ingest_from_dict(sample_supplier_data, schema=supplier_schema)

        # Original columns should match schema types (audit cols are always string)
        field_map = {f.name: f.dataType for f in df.schema.fields}
        assert isinstance(field_map["s_suppkey"], IntegerType)
        assert isinstance(field_map["s_name"], StringType)

    def test_ingest_nation_data(self, spark, sample_nation_data, nation_schema):
        bronze = BronzeIngestion(spark, "nations")
        df = bronze.ingest_from_dict(sample_nation_data, schema=nation_schema)

        assert df.count() == 9
        assert "n_nationkey" in df.columns
        assert "n_name" in df.columns


class TestBronzeSchemas:
    """Tests for create_bronze_schemas."""

    def test_schemas_contain_expected_keys(self):
        schemas = create_bronze_schemas()
        assert "supplier" in schemas
        assert "nation" in schemas

    def test_supplier_schema_fields(self):
        schemas = create_bronze_schemas()
        field_names = [f.name for f in schemas["supplier"].fields]
        assert "s_suppkey" in field_names
        assert "s_name" in field_names
        assert "s_nationkey" in field_names

    def test_nation_schema_fields(self):
        schemas = create_bronze_schemas()
        field_names = [f.name for f in schemas["nation"].fields]
        assert "n_nationkey" in field_names
        assert "n_name" in field_names
