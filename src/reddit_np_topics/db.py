"""
DuckDB connection, schema creation, and read/write helpers.
All pipeline stages read from and write to the database via these functions.
"""

import duckdb
import pandas as pd
from pathlib import Path


DB_PATH = Path("data/reddit_np.duckdb")


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the file if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def init_schema(db_path: Path = DB_PATH) -> None:
    """Create all pipeline tables if they don't already exist."""
    con = get_connection(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            origin_id                   VARCHAR,
            post_guid                   VARCHAR,
            park_name                   VARCHAR,
            topic_group                 VARCHAR,
            user_guid                   VARCHAR,
            publish_date                VARCHAR,
            post_thumbnail_url          VARCHAR,
            like_count                  VARCHAR,
            post_comment_count          VARCHAR,
            post_url                    VARCHAR,
            tags                        VARCHAR,
            emoji                       VARCHAR,
            post_title                  VARCHAR,
            body                        VARCHAR,
            post_filter                 VARCHAR,
            reaction_guid               VARCHAR,
            reaction_type               VARCHAR,
            referencedpostreaction_guid VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS normalized_posts (
            origin_id          VARCHAR,
            post_guid          VARCHAR,
            park_name          VARCHAR,
            user_guid          VARCHAR,
            publish_date       VARCHAR,
            post_thumbnail_url VARCHAR,
            like_count         VARCHAR,
            post_comment_count VARCHAR,
            post_url           VARCHAR,
            tags               VARCHAR,
            emoji              VARCHAR,
            post_title         VARCHAR,
            body               VARCHAR,
            text               VARCHAR,
            tokens             VARCHAR,
            loc_entities       VARCHAR,
            sentiment          VARCHAR,
            sentiment_score    DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS global_classified (
            origin_id          VARCHAR,
            post_guid          VARCHAR,
            park_name          VARCHAR,
            user_guid          VARCHAR,
            publish_date       VARCHAR,
            post_title         VARCHAR,
            body               VARCHAR,
            text               VARCHAR,
            tokens             VARCHAR,
            loc_entities       VARCHAR,
            sentiment          VARCHAR,
            sentiment_score    DOUBLE,
            Global_Topic_ID    INTEGER,
            Global_Topic       VARCHAR,
            Global_Repr        VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS regional_classified (
            origin_id           VARCHAR,
            post_guid           VARCHAR,
            park_name           VARCHAR,
            user_guid           VARCHAR,
            publish_date        VARCHAR,
            post_title          VARCHAR,
            body                VARCHAR,
            text                VARCHAR,
            tokens              VARCHAR,
            loc_entities        VARCHAR,
            sentiment           VARCHAR,
            sentiment_score     DOUBLE,
            Global_Topic_ID     INTEGER,
            Global_Topic        VARCHAR,
            Global_Repr         VARCHAR,
            Regional_Topic_ID   INTEGER,
            Regional_Topic      VARCHAR,
            Regional_Repr       VARCHAR,
            Regional_Topic_Prob DOUBLE
        )
    """)

    con.close()
    print(f"Schema initialised at {db_path}")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_raw_posts(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    con = get_connection(db_path)
    con.execute("INSERT INTO raw_posts SELECT * FROM df")
    con.close()


def write_normalized_posts(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    con = get_connection(db_path)
    if not df.empty:
        parks = df["park_name"].unique().tolist()
        placeholders = ", ".join(["?" for _ in parks])
        con.execute(
            f"DELETE FROM normalized_posts WHERE park_name IN ({placeholders})",
            parks,
        )
        con.execute("INSERT INTO normalized_posts SELECT * FROM df")
    con.close()


def write_global_classified(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    con = get_connection(db_path)
    con.execute("DELETE FROM global_classified")
    con.execute("INSERT INTO global_classified SELECT * FROM df")
    con.close()


def write_regional_classified(
    df: pd.DataFrame, park_name: str, db_path: Path = DB_PATH
) -> None:
    con = get_connection(db_path)
    con.execute(
        "DELETE FROM regional_classified WHERE park_name = ?", [park_name]
    )
    con.execute("INSERT INTO regional_classified SELECT * FROM df")
    con.close()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_table(table: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    con = get_connection(db_path)
    df = con.execute(f"SELECT * FROM {table}").df()
    con.close()
    return df


def read_normalized_posts(db_path: Path = DB_PATH) -> pd.DataFrame:
    return read_table("normalized_posts", db_path)


def read_global_classified(db_path: Path = DB_PATH) -> pd.DataFrame:
    return read_table("global_classified", db_path)


def read_regional_classified(db_path: Path = DB_PATH) -> pd.DataFrame:
    return read_table("regional_classified", db_path)


def read_park(
    park_name: str, table: str = "global_classified", db_path: Path = DB_PATH
) -> pd.DataFrame:
    con = get_connection(db_path)
    df = con.execute(
        f"SELECT * FROM {table} WHERE park_name = ?", [park_name]
    ).df()
    con.close()
    return df


def list_parks(
    table: str = "normalized_posts", db_path: Path = DB_PATH
) -> list[str]:
    con = get_connection(db_path)
    result = con.execute(
        f"SELECT DISTINCT park_name FROM {table} ORDER BY park_name"
    ).fetchall()
    con.close()
    return [r[0] for r in result]