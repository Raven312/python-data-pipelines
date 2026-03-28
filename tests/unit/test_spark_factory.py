"""Unit tests for SparkSession factory."""

import pytest

from pipeline.utils.spark_factory import SparkSessionFactory


class TestSparkSessionFactory:

    def test_creates_session(self, spark):
        assert spark is not None
        assert spark.sparkContext is not None

    def test_test_environment_config(self, spark):
        conf = spark.sparkContext.getConf()
        assert conf.get("spark.sql.shuffle.partitions") == "2"
        assert conf.get("spark.ui.enabled") == "false"

    def test_get_or_create_returns_same_instance(self, spark):
        same = SparkSessionFactory.get_or_create(
            app_name="MedallionETL_Tests", environment="test"
        )
        assert spark is same

    def test_base_config_local(self):
        config = SparkSessionFactory._base_config("local")
        assert config["spark.master"] == "local[*]"
        assert config["spark.sql.session.timeZone"] == "UTC"

    def test_base_config_databricks(self):
        config = SparkSessionFactory._base_config("databricks")
        assert "spark.databricks.delta.optimizeWrite.enabled" in config

    def test_base_config_test(self):
        config = SparkSessionFactory._base_config("test")
        assert config["spark.master"] == "local[1]"
        assert config["spark.ui.enabled"] == "false"
