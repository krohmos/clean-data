"""Pytest configuration for Spark testing."""

from typing import Iterator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """
    Creates a SparkSession for PySpark testing.

    Returns:
        Iterator[SparkSession]: A generator that yields a SparkSession object.

    Example:
        >>> for spark_session in spark():
        ...     # Use the spark_session for testing
        ...     df = spark.createDataFrame([(1, 'Alice'), (2, 'Bob')], ['id', 'name'])
        ...     df.show()
        ...
    """
    spark_session = (
        SparkSession.builder.master("local[*]")
        .appName("PySparkTesting")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()
