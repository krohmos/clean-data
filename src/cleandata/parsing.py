"""Parsing classes for the CleanData module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pyspark.sql import Column, DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from cleandata.transformation import from_json


class Parser(ABC):
    """Abstract base class for parsers.

    Attributes:
        validation_expr (Column): The validation expression column.
        details_expr (Column, optional): The details expression column.

    Methods:
        __call__(self, src: DataFrame) -> DataFrame: Applies the parsing operations on the source DataFrame.
        _set_parsing_details(self, src: DataFrame) -> DataFrame: Sets the parsing details on the source DataFrame.
        complete(self, src: DataFrame) -> DataFrame: Completes the parsing process on the source DataFrame.
        prepare(self, src: DataFrame) -> DataFrame: Prepares the source DataFrame for parsing.
        parse(self, src: DataFrame) -> DataFrame: Abstract method to be implemented by subclasses for parsing.

    """

    def __init__(self, validation_expr: Column, details_expr: Column = None) -> None:
        """Initializes a Parser instance.

        Args:
            validation_expr (Column): The validation expression column.
            details_expr (Column, optional): The details expression column.

        """
        self.validation_expr = validation_expr
        self.details_expr = details_expr

    def __call__(self, src: DataFrame) -> DataFrame:
        """Applies the parsing operations on the source DataFrame.

        Args:
            src (DataFrame): The source DataFrame.

        Returns:
            DataFrame: The parsed DataFrame.

        """
        return (
            src.transform(self.prepare)
            .select(F.struct("*").alias("origin"))
            .transform(self.parse)
            .transform(self._set_parsing_details)
            .transform(self.complete)
        )

    def _set_parsing_details(self, src: DataFrame) -> DataFrame:
        """Sets the parsing details on the source DataFrame.

        Args:
            src (DataFrame): The source DataFrame.

        Returns:
            DataFrame: The DataFrame with parsing details.

        """
        return src.withColumn(
            "__meta__",
            F.struct(
                F.col("origin.__meta__.uuid").alias("uuid"),
                F.when(self.validation_expr, F.lit("OK"))
                .otherwise(F.lit("INVALID"))
                .alias("status"),
                self.details_expr.alias("details"),
                F.lit(datetime.now()).alias("at"),
            ),
        )

    def complete(self, src: DataFrame) -> DataFrame:
        """Completes the parsing process on the source DataFrame.

        Args:
            src (DataFrame): The source DataFrame.

        Returns:
            DataFrame: The completed DataFrame.

        """
        return src

    def prepare(self, src: DataFrame) -> DataFrame:
        """Prepares the source DataFrame for parsing.

        Args:
            src (DataFrame): The source DataFrame.

        Returns:
            DataFrame: The prepared DataFrame.

        """
        return src

    @abstractmethod
    def parse(self, src: DataFrame) -> DataFrame:
        """Abstract method to be implemented by subclasses for parsing.

        Args:
            src (DataFrame): The source DataFrame.

        Returns:
            DataFrame: The parsed DataFrame.

        """
        pass


class DataParser(Parser):
    """A class for parsing data using a transform function.

    Args:
        transform_func (callable): The transform function to be applied to the data.
        validation_expr (Column): The validation expression to be used for data validation.
        details_expr (Column, optional): The details expression for additional data details. Defaults to None.

    Returns:
        DataParser: An instance of the DataParser class.

    """

    def __init__(
        self,
        transform_func: callable,
        validation_expr: Column,
        details_expr: Column = None,
    ) -> DataParser:

        self.transform_func = transform_func
        self.validation_expr = validation_expr
        self.details_expr = details_expr

    def parse(self, src: DataFrame) -> DataFrame:
        """Parse the source DataFrame using the transform function.

        Args:
            src (DataFrame): The source DataFrame to be parsed.

        Returns:
            DataFrame: The parsed DataFrame.

        """
        return src.transform(self.transform_func)

    def prepare(self, src: DataFrame) -> DataFrame:
        """Prepare the source DataFrame by dropping the 'origin' column.

        Args:
            src (DataFrame): The source DataFrame to be prepared.

        Returns:
            DataFrame: The prepared DataFrame.

        """
        return src.drop("origin")

    def complete(self, src: DataFrame) -> DataFrame:
        """Complete the source DataFrame by adding the 'origin' and '__meta__' columns.

        Args:
            src (DataFrame): The source DataFrame to be completed.

        Returns:
            DataFrame: The completed DataFrame.

        """
        return src.withColumn(
            "origin",
            F.struct(
                F.expr(
                    "to_json(named_struct("
                    + ", ".join(
                        [
                            f"'{field}', origin.{field}"
                            for field in src.schema["origin"].dataType.names
                            if field != "__meta__"
                        ]
                    )
                    + "))"
                ).alias("raw"),
                "origin.__meta__",
            ),
        ).withColumn(
            "__meta__",
            F.col("__meta__").withField("event", F.col("origin.__meta__.event")),
        )


class SchemaParser(Parser):
    """
    A class for parsing data using a schema.

    Args:
        schema (pyspark.sql.types.StructType or str): The schema to be used for parsing.
        validation_expr (pyspark.sql.Column): The validation expression for the parsed data.
        details_expr (pyspark.sql.Column, optional): The details expression for the parsed data. Defaults to None.

    Methods:
        complete(src: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
            Completes the parsing process by adding a "__meta__" column to the source DataFrame.

        parse(src: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
            Parses the source DataFrame using the specified schema and returns the parsed DataFrame.
    """

    def __init__(
        self,
        schema: StructType | str,
        validation_expr: Column,
        details_expr: Column = None,
    ) -> None:

        self.validation_expr = validation_expr
        self.details_expr = details_expr
        self.schema = schema

    def complete(self, src: DataFrame) -> DataFrame:
        """
        Completes the parsing process by adding a "__meta__" column to the source DataFrame.

        Args:
            src (pyspark.sql.DataFrame): The source DataFrame to be parsed.

        Returns:
            pyspark.sql.DataFrame: The parsed DataFrame with the "__meta__" column added.
        """
        return src.withColumn(
            "__meta__",
            F.col("__meta__").withField("event", F.col("origin.__meta__.event")),
        )

    def parse(self, src: DataFrame) -> DataFrame:
        """
        Parses the source DataFrame using the specified schema and returns the parsed DataFrame.

        Args:
            src (pyspark.sql.DataFrame): The source DataFrame to be parsed.

        Returns:
            pyspark.sql.DataFrame: The parsed DataFrame.
        """
        return src.withColumn("_parsed", from_json("origin.raw", self.schema)).select(
            "origin", "_parsed.*"
        )
