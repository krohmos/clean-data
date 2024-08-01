"""Row transformations."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime

from pyspark.sql import Column, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    _parse_datatype_string,
)


def from_json(col: Column, schema: StructType) -> Column:
    """
    Converts a column of JSON strings to a struct column according to the specified schema.

    Args:
        col (Column): The column to convert. Can be either a column name or a Column object.
        schema (StructType or str): The schema to use for the conversion. Can be either a StructType object or a string representation of the schema.

    Returns:
        Column: The converted struct column.

    Raises:
        ValueError: If the schema is not a valid StructType or string representation of a schema.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.functions import col
        >>> from pyspark.sql.types import StructType, StringType
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = spark.createDataFrame([(1, '{"name": "John", "age": 30}'), (2, '{"name": "Jane", "age": 25}')], ["id", "json"])
        >>> schema = StructType().add("name", StringType()).add("age", StringType())
        >>> df.withColumn("parsed_json", from_json(col("json"), schema)).show(truncate=False)
        +---+-------------------+----------------+
        |id |json               |parsed_json     |
        +---+-------------------+----------------+
        |1  |{"name": "John", "age": 30} |[John, 30]      |
        |2  |{"name": "Jane", "age": 25} |[Jane, 25]      |
        +---+-------------------+----------------+
    """
    if isinstance(schema, str):
        schema = _parse_datatype_string(schema)

    def set_fields_to_string(schema):
        new_fields = []
        for field in schema.fields:
            if isinstance(field.dataType, StructType):
                new_fields.append(
                    StructField(field.name, set_fields_to_string(field.dataType), True)
                )
            else:
                new_fields.append(StructField(field.name, StringType(), True))
        return StructType(new_fields)

    str_schema = set_fields_to_string(schema)

    if isinstance(col, str):
        col = F.col(col)
    col = F.from_json(col, str_schema)

    def convert_fields(schema, col):
        for field in schema.fields:
            if isinstance(field.dataType, StructType):
                col = col.withField(
                    field.name, convert_fields(field.dataType, col[field.name])
                )
            else:
                if isinstance(field.dataType, BinaryType):
                    col = col.withField(
                        field.name, F.unbase64(col[field.name]).cast(BinaryType())
                    )
                elif isinstance(field.dataType, TimestampType):
                    col = col.withField(field.name, toTimestamp(col[field.name]))
                else:
                    col = col.withField(
                        field.name, col[field.name].cast(field.dataType)
                    )
        return col

    col = convert_fields(schema, col)

    return col


def to_json(row: Row) -> Row:
    """
    Converts a Row object to a JSON string representation.

    Args:
        row (pyspark.sql.Row): The Row object to be converted.

    Returns:
        pyspark.sql.Row: A new Row object with the JSON string representation.

    """
    raw = row.asDict()

    def convert(value):
        """
        Recursively converts a value to its string representation.

        Args:
            value: The value to be converted.

        Returns:
            str: The string representation of the value.

        """
        if value is None:
            return ""
        if isinstance(value, Row):
            return {k: convert(v) for k, v in value.asDict().items()}
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, bytearray):
            return base64.b64encode(value).decode("utf-8")
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        if isinstance(value, set):
            return {convert(v) for v in value}
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S.%f")

        return str(value)

    json_string = json.dumps(convert(raw))

    return Row(raw=json_string)


def _to_timestamp(ts: str = None) -> datetime | None:
    """Convert a string into a datetime object by inference.

    Args:
        ts (str, optional): A string version of a timestamp. Defaults to None.

    Returns:
        Optional[datetime.datetime]: Datetime object parsed from string.
            Will return None if it is not possible to infer.
    """
    if ts is None:
        return None
    formatted_ts = re.sub(r"[^0-9]", " ", ts).strip()

    patterns = [
        "%Y %m %d",
        "%Y %m %d %H %M %S",
        "%Y %m %d %H %M %S %f",
        "%m %d %Y",
        "%m %d %Y %H %M %S",
        "%m %d %Y %H %M %S %f",
        "%d %m %Y",
        "%d %m %Y %H %M %S",
        "%d %m %Y %H %M %S %f",
    ]

    for pattern in patterns:
        try:
            return datetime.strptime(formatted_ts, pattern)
        except ValueError:
            continue

    return None


toTimestamp = F.udf(_to_timestamp, TimestampType())
