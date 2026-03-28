"""SparkSession factory with environment-aware configuration."""

import logging
from typing import Optional

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


class SparkSessionFactory:
    """Creates and configures SparkSession instances for different environments.

    Supports local development, Databricks, and test configurations with
    appropriate defaults for each environment.
    """

    _instance: Optional[SparkSession] = None

    @classmethod
    def get_or_create(
        cls,
        app_name: str = "MedallionETL",
        environment: str = "local",
        extra_config: Optional[dict] = None,
    ) -> SparkSession:
        """Get existing or create new SparkSession.

        Args:
            app_name: Spark application name.
            environment: One of 'local', 'databricks', 'test'.
            extra_config: Additional Spark configuration overrides.

        Returns:
            Configured SparkSession.
        """
        if cls._instance and cls._instance.sparkContext._jsc.sc().isStopped():
            cls._instance = None

        if cls._instance is not None:
            return cls._instance

        builder = SparkSession.builder.appName(app_name)

        configs = cls._base_config(environment)
        if extra_config:
            configs.update(extra_config)

        for key, value in configs.items():
            builder = builder.config(key, value)

        if environment == "databricks":
            builder = builder.enableHiveSupport()

        cls._instance = builder.getOrCreate()
        logger.info("SparkSession created [env=%s, app=%s]", environment, app_name)
        return cls._instance

    @classmethod
    def _base_config(cls, environment: str) -> dict:
        """Return base Spark config for the given environment."""
        common = {
            "spark.sql.session.timeZone": "UTC",
            "spark.sql.sources.partitionOverwriteMode": "dynamic",
        }

        env_configs = {
            "local": {
                "spark.master": "local[*]",
                "spark.driver.memory": "2g",
                "spark.sql.shuffle.partitions": "8",
                "spark.sql.adaptive.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            },
            "databricks": {
                "spark.sql.shuffle.partitions": "200",
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.databricks.delta.optimizeWrite.enabled": "true",
                "spark.databricks.delta.autoCompact.enabled": "true",
            },
            "test": {
                "spark.master": "local[1]",
                "spark.driver.memory": "1g",
                "spark.sql.shuffle.partitions": "2",
                "spark.ui.enabled": "false",
                "spark.sql.adaptive.enabled": "false",
            },
        }

        config = {**common, **env_configs.get(environment, env_configs["local"])}
        return config

    @classmethod
    def stop(cls) -> None:
        """Stop the active SparkSession."""
        if cls._instance:
            cls._instance.stop()
            cls._instance = None
            logger.info("SparkSession stopped")
