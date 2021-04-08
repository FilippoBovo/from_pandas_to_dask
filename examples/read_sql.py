import dask.dataframe as dd
import pandas as pd
import sqlalchemy

# SQLAlchemy connection string.
# See https://docs.sqlalchemy.org/en/14/core/engines.html.
sql_connection_string = "dialect+driver://username:password@host:port/database"

sql_connection = sqlalchemy.create_engine(sql_connection_string)
metadata = sqlalchemy.MetaData(schema="schema_name")
table = sqlalchemy.Table(
    "table_name", metadata, autoload=True, autoload_with=sql_connection
)

# SQLAlchemy query API.
# See https://docs.sqlalchemy.org/en/14/orm/query.html.
query = (
    sqlalchemy.select(
        [
            table.c.id,  # Type: BIGINT
            table.c.a,  # Type: INT, no nulls
            sqlalchemy.cast(table.c.b, sqlalchemy.Float()).label(
                "c"
            ),  # Type: INT, with nulls
            table.c.c,  # Type: VARCHAR
            table.c.d,  # Type: DATETIME
        ]
    )
    .where(
        sqlalchemy.and_(
            table.c.a > 0,
            table.c.d.isnot(None),  # Drop null values
        )
    )
    .limit(10_000)
    .alias("my_data")  # Any alias name would work
)

# Dask metadata
dtypes = {
    "id": "int64",
    "a": "int32",
    "b": "float32",
    "c": "object",
    "d": "datetime64[ns]",
}
metadata = (
    pd.DataFrame(columns=dtypes.keys())
    .astype(dtypes)
    .set_index("id", drop=True)
)

# Query
dask_partition_size = 268_435_456  # Bytes == 256 MiB
ddf = dd.read_sql_table(
    table=query,
    uri=sql_connection_string,
    index_col="id",
    bytes_per_chunk=dask_partition_size,
    meta=metadata,
)
