"""Tests for the parsing module."""

import pytest as _
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cleandata.annotation import DataAnnotator
from cleandata.parsing import DataParser, SchemaParser


def test_schema_parser(spark: SparkSession):
    """Test the SchemaParser with basic data types."""
    df = spark.createDataFrame(
        data=[("a", 1, 2.2), ("b", 3, 4.4)],
        schema="A string, B int, C double",
    )

    annotated_df = DataAnnotator(df, "test")

    rules = "B is not null and C > 4"
    parser = SchemaParser(df.schema, F.expr(rules), F.lit(rules))
    parsed_df = parser(annotated_df)

    assert parsed_df.count() == 2
    assert parsed_df.where("__meta__.status = 'OK'").count() == 1
    assert (
        parsed_df.select("origin.raw").collect() == annotated_df.select("raw").collect()
    )
    assert parsed_df.select("A", "B", "C").collect() == df.collect()


def test_data_parser(spark: SparkSession):
    """Test the DataParser with basic data types."""
    df = spark.createDataFrame(
        data=[("a", 1, 2.2), ("b", 3, 4.4), ("c", 0, 5.1)],
        schema="A string, B int, C double",
    )

    annotated_df = DataAnnotator(df, "test")

    rules = "B is not null and C > 4"
    parser = SchemaParser(df.schema, F.expr(rules), F.lit(rules))
    parsed_df = parser(annotated_df)

    def func(src: DataFrame) -> DataFrame:
        return src.withColumn("D", F.col("origin.B") * 3)

    rules = "D > 4"
    parser = DataParser(func, F.expr(rules), F.lit(rules))
    transformed_df = parser(parsed_df.where("__meta__.status = 'OK'"))

    assert transformed_df.count() == 2
    assert transformed_df.where("__meta__.status = 'OK'").count() == 1
