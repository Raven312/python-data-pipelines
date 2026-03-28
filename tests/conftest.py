"""Shared pytest fixtures for the PySpark test suite.

Provides a session-scoped SparkSession and reusable sample data fixtures
used across unit and integration tests.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from pipeline.utils.spark_factory import SparkSessionFactory


# ── SparkSession ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def spark():
    """Session-scoped SparkSession for all tests."""
    session = SparkSessionFactory.get_or_create(
        app_name="MedallionETL_Tests",
        environment="test",
    )
    yield session
    SparkSessionFactory.stop()


# ── Sample data (raw dicts) ──────────────────────────────────────────


@pytest.fixture
def sample_supplier_data():
    """Raw supplier data as column-oriented dict."""
    return {
        "s_suppkey": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "s_name": [
            "Supplier#001", "Supplier#002", "Supplier#003", "Supplier#004",
            "Supplier#005", "Supplier#006", "Supplier#007", "Supplier#008",
            "Supplier#009", "Supplier#010",
        ],
        "s_address": [
            "Address1", "Address2", "Address3", "Address4", "Address5",
            "Address6", "Address7", "Address8", "Address9", "Address10",
        ],
        "s_nationkey": [17, 5, 24, 2, 19, 0, 23, 17, 10, 8],
        "s_phone": [
            "27-918-335-1736", "15-679-861-2259", "34-546-815-5376",
            "12-314-759-5682", "29-650-264-3518", "10-514-483-8935",
            "33-484-637-4851", "27-217-225-2336", "20-403-398-8662",
            "18-688-445-3302",
        ],
        "s_acctbal": [
            5755.94, 4032.68, 9694.28, 4656.24, 9653.32,
            2963.17, 8142.56, 9862.18, 5733.73, 1048.88,
        ],
    }


@pytest.fixture
def sample_nation_data():
    """Raw nation data as column-oriented dict."""
    return {
        "n_nationkey": [0, 2, 5, 8, 10, 17, 19, 23, 24],
        "n_name": [
            "ALGERIA", "BRAZIL", "ETHIOPIA", "INDIA", "IRAN",
            "PERU", "ROMANIA", "UNITED KINGDOM", "UNITED STATES",
        ],
        "n_regionkey": [0, 1, 0, 2, 4, 1, 3, 3, 1],
    }


# ── Spark DataFrames ─────────────────────────────────────────────────


@pytest.fixture
def supplier_schema():
    return StructType([
        StructField("s_suppkey", IntegerType(), False),
        StructField("s_name", StringType(), False),
        StructField("s_address", StringType(), True),
        StructField("s_nationkey", IntegerType(), False),
        StructField("s_phone", StringType(), True),
        StructField("s_acctbal", DoubleType(), True),
    ])


@pytest.fixture
def nation_schema():
    return StructType([
        StructField("n_nationkey", IntegerType(), False),
        StructField("n_name", StringType(), False),
        StructField("n_regionkey", IntegerType(), True),
    ])


@pytest.fixture
def supplier_df(spark, sample_supplier_data, supplier_schema):
    """Spark DataFrame of raw supplier data."""
    rows = list(zip(*sample_supplier_data.values()))
    return spark.createDataFrame(rows, schema=supplier_schema)


@pytest.fixture
def nation_df(spark, sample_nation_data, nation_schema):
    """Spark DataFrame of raw nation data."""
    rows = list(zip(*sample_nation_data.values()))
    return spark.createDataFrame(rows, schema=nation_schema)
