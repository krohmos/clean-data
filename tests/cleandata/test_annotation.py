from datetime import datetime

import pytest as _
from pyspark.sql import SparkSession

from cleandata.annotation import DataAnnotator
from cleandata.transformation import from_json


def test_annotation_basic(spark: SparkSession):
    """Test the DataAnnotator with basic data types."""
    df = spark.createDataFrame(
        data=[("a", 1, 2.2), ("b", 3, 4.4)],
        schema="A string, B int, C double",
    )

    annotator = DataAnnotator("test")
    annotated_df = annotator(df)

    # Deconvert.
    deannotated_df = annotated_df.withColumn("x", from_json("raw", df.schema)).select(
        "x.*"
    )

    assert deannotated_df.schema == df.schema
    assert deannotated_df.collect() == df.collect()


def test_annotation_nested(spark: SparkSession):
    """Test the DataAnnotator with nested data types."""
    df = spark.createDataFrame(
        data=[("a", 1, ("b", 2, ("c", 3)))],
        schema="A string, B int, C struct<D string, E int, F struct<G string, H int>>",
    )

    annotator = DataAnnotator("test")
    annotated_df = annotator(df)

    # Deconvert.
    deannotated_df = annotated_df.withColumn("x", from_json("raw", df.schema)).select(
        "x.*"
    )

    assert deannotated_df.schema == df.schema
    assert deannotated_df.collect() == df.collect()


def test_annotation_binary(spark: SparkSession):
    """Test the DataAnnotator with binary data types."""
    df = spark.createDataFrame(
        data=[(1, "a", b"2xww"), (2, "b", b"234")],
        schema="id int, value string, b binary",
    )

    annotator = DataAnnotator("test")
    annotated_df = annotator(df)

    # Deconvert.
    deannotated_df = annotated_df.withColumn("x", from_json("raw", df.schema)).select(
        "x.*"
    )

    assert deannotated_df.schema == df.schema
    assert deannotated_df.collect() == df.collect()
