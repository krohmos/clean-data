"""Tests for the transformation module."""

import json
from datetime import datetime

import pytest as _
from pyspark.sql import Row, SparkSession

from cleandata.transformation import from_json, to_json


def row_to_dict(row: Row) -> dict:
    """
    Converts a PySpark Row to a dictionary, including nested rows.

    Args:
        row (Row): The PySpark Row to convert.

    Returns:
        dict: A dictionary representation of the Row.
    """

    def convert(value):
        if isinstance(value, Row):
            return {k: convert(v) for k, v in value.asDict().items()}
        elif isinstance(value, list):
            return [convert(v) for v in value]
        else:
            return value

    return {k: convert(v) for k, v in row.asDict().items()}


def test_from_json_basic(spark: SparkSession):
    """Test the from_json function in basic terms."""
    data = [
        {"name": "Alice", "age": 34},
        {"name": "Bob", "age": 36},
        {"name": "Bob", "age": 45},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()
    schema = "name string, age int"

    result = result.withColumn("info", from_json("raw", schema))

    assert data == [row_to_dict(row["info"]) for row in result.collect()]


def test_from_json_nested(spark: SparkSession):
    """Test the from_json function with nested structures."""
    data = [
        {"name": "Alice", "age": 34, "details": {"city": "Springfield"}},
        {"name": "Bob", "age": 45, "details": {"city": "Shelbyville"}},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()
    schema = "name string, age int, details struct<city: string>"

    result = result.withColumn("info", from_json("raw", schema))

    assert data == [row_to_dict(row["info"]) for row in result.collect()]


def test_from_json_deep_nested(spark: SparkSession):
    """Test the from_json function with deep nested structures."""
    data = [
        {"A": {"B": {"C": 1}}},
        {"A": {"B": {"C": 2}}},
        {"A": {"B": {"C": 3}}},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()
    schema = "A struct<B: struct<C: int>>"

    result = result.withColumn("info", from_json("raw", schema))

    assert data == [row_to_dict(row["info"]) for row in result.collect()]


def test_from_json_binary(spark: SparkSession):
    """Test the from_json function with binary data."""
    data = [
        {"name": "Alice", "age": 34, "details": b"Springfield"},
        {"name": "Bob", "age": 45, "details": b"Shelbyville"},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()
    schema = "name string, age int, details binary"

    result = result.withColumn("info", from_json("raw", schema))

    assert data == [row_to_dict(row["info"]) for row in result.collect()]


def test_from_json_timestamp(spark: SparkSession):
    """Test the from_json function with timestamps."""
    data = [
        {"name": "Alice", "age": 34, "details": datetime(2021, 1, 1, 12, 0, 0)},
        {"name": "Bob", "age": 45, "details": datetime(2021, 1, 2, 12, 0, 0)},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()
    schema = "name string, age int, details timestamp"

    result = result.withColumn("info", from_json("raw", schema))

    assert data == [row_to_dict(row["info"]) for row in result.collect()]


def test_to_json_basic(spark: SparkSession):
    """Test the to_json function in basic terms."""
    data = [
        {"name": "Alice", "age": 34},
        {"name": "Bob", "age": 45},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()

    assert [json.loads(row["raw"]) for row in result.collect()] == [
        {"age": "34", "name": "Alice"},
        {"age": "45", "name": "Bob"},
    ]


def test_to_json_nested(spark: SparkSession):
    """Test the to_json function with nested structures."""
    data = [
        {"name": "Alice", "age": 34, "details": {"city": "Springfield"}},
        {"name": "Bob", "age": 45, "details": {"city": "Shelbyville"}},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()

    assert [json.loads(row["raw"]) for row in result.collect()] == [
        {"age": "34", "details": {"city": "Springfield"}, "name": "Alice"},
        {"age": "45", "details": {"city": "Shelbyville"}, "name": "Bob"},
    ]


def test_to_json_deep_nested(spark: SparkSession):
    """Test the to_json function with deep nested structures."""

    data = [{"A": {"B": {"C": 1}}}, {"A": {"B": {"C": 2}}}, {"A": {"B": {"C": 3}}}]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()

    assert [json.loads(row["raw"]) for row in result.collect()] == [
        {"A": {"B": {"C": "1"}}},
        {"A": {"B": {"C": "2"}}},
        {"A": {"B": {"C": "3"}}},
    ]


def test_to_json_binary(spark: SparkSession):
    """Test the to_json function with binary data."""
    data = [
        {"name": "Alice", "age": 34, "details": b"Springfield"},
        {"name": "Bob", "age": 45, "details": b"Shelbyville"},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()

    assert [json.loads(row["raw"]) for row in result.collect()] == [
        {"age": "34", "details": "U3ByaW5nZmllbGQ=", "name": "Alice"},
        {"age": "45", "details": "U2hlbGJ5dmlsbGU=", "name": "Bob"},
    ]


def test_to_json_timestamp(spark: SparkSession):
    """Test the to_json function with timestamps."""
    data = [
        {"name": "Alice", "age": 34, "details": datetime(2021, 1, 1, 12, 0, 0)},
        {"name": "Bob", "age": 45, "details": datetime(2021, 1, 2, 12, 0, 0)},
    ]
    df = spark.createDataFrame(data)
    result = df.rdd.map(to_json).toDF()

    assert [json.loads(row["raw"]) for row in result.collect()] == [
        {"age": "34", "details": "2021-01-01 12:00:00.000000", "name": "Alice"},
        {"age": "45", "details": "2021-01-02 12:00:00.000000", "name": "Bob"},
    ]
