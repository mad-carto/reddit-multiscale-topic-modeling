"""
DuckDB connection, schema creation, and read/write helpers.
All intermediate pipeline data is stored here instead of intermediate CSVs.
"""

import duckdb
import pandas as pd
from pathlib import Path


def get_connection(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the file if it doesn't exist."""
    return duckdb.connect(str(db_path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all pipeline tables if they don't already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            origin_id       VARCHAR,
            post_guid       VARCHAR,
            park_name       VARCHAR,
            topic_group     VARCHAR,
            user_guid       VARCHAR,
            publish_date    VARCHAR,
            post_thumbnail_url VARCHAR,
            like_count      VARCHAR,
            post_comment_count VARCHAR,
            post_url        VARCHAR,
            tags            VARCHAR,
            emoji           VARCHAR,
            post_title      VARCHAR,
            body            VARCHAR,
            post_filter     VARCHAR,
            reaction_guid   VARCHAR,
            reaction_type   VARCHAR,
            referencedpostreaction_guid VARCHAR,
            text            VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS normalized_posts (
            origin_id       VARCHAR,
            post_guid       VARCHAR,
            park_name       VARCHAR,
            user_guid       VARCHAR,
            publish_date    VARCHAR,
            post_thumbnail_url VARCHAR,
            like_count      VARCHAR,
            post_comment_count VARCHAR,
            post_url        VARCHAR,
            tags            VARCHAR,
            emoji           VARCHAR,
            post_title      VARCHAR,
            body            VARCHAR,
            text            VARCHAR,
            tokens          VARCHAR,
            loc_entities    VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS global_classified (
            origin_id           VARCHAR,
            post_guid           VARCHAR,
            park_name           VARCHAR,
            user_guid           VARCHAR,
            publish_date        VARCHAR,
            post_thumbnail_url  VARCHAR,
            like_count          VARCHAR,
            post_comment_count  VARCHAR,
            post_url            VARCHAR,
            tags                VARCHAR,
            emoji               VARCHAR,
            post_title          VARCHAR,
            body                VARCHAR,
            text                VARCHAR,
            tokens              VARCHAR,
            loc_entities        VARCHAR,
            Global_Topic_ID     INTEGER,
            Global_Topic        VARCHAR,
            Global_Repr         VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS regional_classified (
            origin_id           VARCHAR,
            post_guid           VARCHAR,
            park_name           VARCHAR,
            user_guid           VARCHAR,
            publish_date        VARCHAR,
            post_thumbnail_url  VARCHAR,
            like_count          VARCHAR,
            post_comment_count  VARCHAR,
            post_url            VARCHAR,
            tags                VARCHAR,
            emoji               VARCHAR,
            post_title          VARCHAR,
            body                VARCHAR,
            text                VARCHAR,
            tokens              VARCHAR,
            loc_entities        VARCHAR,
            Global_Topic_ID     INTEGER,
            Global_Topic        VARCHAR,
            Global_Repr         VARCHAR,
            Regional_Topic_ID   INTEGER,
            Regional_Topic      VARCHAR,
            Regional_Repr       VARCHAR,
            Regional_Topic_Prob DOUBLE
        )
    """)


def write_dataframe(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table: str,
    mode: str = "append",
) -> None:
    """
    Write a DataFrame to a DuckDB table.

    Args:
        conn: DuckDB connection
        df: DataFrame to write
        table: target table name
        mode: 'append' (default) or 'replace' to overwrite the table
    """
    if mode == "replace":
        conn.execute(f"DELETE FROM {table}")
    conn.execute(f"INSERT INTO {table} SELECT * FROM df")


def read_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    park_name: str | None = None,
) -> pd.DataFrame:
    """
    Read a full table or filter by park_name.

    Args:
        conn: DuckDB connection
        table: table name to read from
        park_name: optional park filter
    """
    if park_name:
        return conn.execute(
            f"SELECT * FROM {table} WHERE park_name = ?", [park_name]
        ).df()
    return conn.execute(f"SELECT * FROM {table}").df()


def list_parks(conn: duckdb.DuckDBPyConnection, table: str = "normalized_posts") -> list[str]:
    """Return a sorted list of unique park names in the given table."""
    result = conn.execute(f"SELECT DISTINCT park_name FROM {table} ORDER BY park_name").fetchall()
    return [row[0] for row in result]


def table_row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Return the number of rows in a table."""
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]