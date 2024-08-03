"""Annotation module for SparkFlow."""

from __future__ import annotations

import re
from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from cleandata.transformation import to_json


class DataAnnotator:
    """
    Annotates data with additional details.

    Args:
        event (str, optional): The event associated with the data. Defaults to "n/a".

    Returns:
        DataFrame: The annotated data.

    Example:
        annotator = DataAnnotator(event="animal")
        annotated_data = annotator(data)
    """

    def __new__(cls, src: DataFrame, event: str = "n/a") -> DataFrame:
        """
        Annotates the source DataFrame with additional details.

        Args:
            src (DataFrame): The source DataFrame to annotate.
            event (str, optional): The event associated with the data. Defaults to "n/a".

        Returns:
            DataFrame: The annotated DataFrame.
        """
        obj = super(DataAnnotator, cls).__new__(cls)
        obj.__init__(event)
        return obj(src)

    def __init__(self, event: str = "n/a") -> None:
        self.event = event

    def __call__(self, src: DataFrame) -> DataFrame:
        """
        Annotates the source DataFrame with additional details.

        Args:
            src (DataFrame): The source DataFrame to annotate.

        Returns:
            DataFrame: The annotated DataFrame.
        """
        return src.transform(self.prepare).transform(self.complete)

    def complete(self, src: DataFrame) -> DataFrame:
        """
        Completes the annotation by adding details to the DataFrame.

        Args:
            src (DataFrame): The DataFrame to complete the annotation.

        Returns:
            DataFrame: The completed DataFrame with added details.
        """
        return src.withColumn(
            "__meta__",
            F.struct(
                F.expr("uuid()").alias("uuid"),
                F.when(F.input_file_name() != "", F.input_file_name())
                .otherwise(F.lit("n/a"))
                .alias("source"),
                F.lit(self.event).alias("event"),
                F.lit(datetime.now()).alias("at"),
            ),
        )

    def prepare(self, src: DataFrame) -> DataFrame:
        """
        Prepares the DataFrame for annotation.

        Args:
            src (DataFrame): The DataFrame to prepare.

        Returns:
            DataFrame: The prepared DataFrame.
        """

        rdd = src.rdd.map(to_json)

        return rdd.toDF()

    def _format(self, s: str) -> str:
        """
        Formats a string by replacing non-alphanumeric characters with underscores.

        Args:
            s (str): The string to format.

        Returns:
            str: The formatted string.
        """
        formatted = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        formatted = re.sub(r"__+", "_", formatted).lower()

        return formatted or "empty"
