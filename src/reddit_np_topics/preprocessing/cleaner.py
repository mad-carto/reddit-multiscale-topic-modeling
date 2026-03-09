"""
Raw data cleaning for Reddit National Parks CSV files.
Handles loading, deduplication, removal of deleted/spam posts,
and combining post title + body into a single text field.
"""

import os
import pandas as pd
from pathlib import Path


RAW_COLUMNS = {
    "origin_id", "post_guid", "topic_group", "user_guid", "publish_date",
    "post_thumbnail_url", "like_count", "post_comment_count", "post_url",
    "tags", "emoji", "post_title", "body", "post_filter",
    "reaction_guid", "reaction_type", "referencedpostreaction_guid",
}

RAW_DTYPES = {col: str for col in RAW_COLUMNS}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a single raw Reddit CSV file."""
    return pd.read_csv(path, usecols=RAW_COLUMNS, dtype=RAW_DTYPES, encoding="utf-8")


def extract_park_name(filename: str) -> str:
    """Extract park name from CSV filename convention: *_<ParkName>.csv"""
    return Path(filename).stem.split("_")[-1]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a raw Reddit DataFrame.

    Steps:
    - Drop rows where both post_title and body are NaN
    - Remove submissions marked as [removed] or [deleted]
    - Remove associated comments of removed submissions
    - Remove deleted comments
    - Combine post_title + body into a single 'text' column
    - Drop rows with empty text
    """
    # Drop entries where both post_title and body are NaN
    df = df.dropna(subset=["post_title", "body"], how="all").copy()

    # Drop removed/deleted submissions and their associated comments
    removed_guids = df[
        (df["body"].isin(["[removed]", "[deleted]"])) &
        (df["reaction_type"] != "comment")
    ]["post_guid"].tolist()
    df = df[~df["post_guid"].isin(removed_guids)]

    # Drop deleted/removed comments
    df = df[~(
        df["body"].isin(["[removed]", "[deleted]"]) &
        (df["reaction_type"] == "comment")
    )]

    # Fill remaining NaN in text fields
    df["body"] = df["body"].fillna("")
    df["post_title"] = df["post_title"].fillna("")

    # Combine title and body
    df["text"] = df["post_title"] + " " + df["body"]
    df["text"] = df["text"].str.strip()

    # Drop empty text
    df = df[df["text"] != ""]

    return df.reset_index(drop=True)


def load_and_clean_all(raw_folder: str | Path) -> pd.DataFrame:
    """
    Load and clean all CSV files in a folder.
    Adds a 'park_name' column derived from each filename.

    Returns a single concatenated DataFrame of all parks.
    """
    raw_folder = Path(raw_folder)
    csv_files = sorted(raw_folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_folder}")

    frames = []
    for path in csv_files:
        park_name = extract_park_name(path.name)
        print(f"Loading: {path.name} → park_name='{park_name}'")

        df = load_csv(path)
        df.insert(1, "park_name", park_name)
        df = clean(df)

        print(f"  {len(df)} rows after cleaning")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows across all parks: {len(combined)}")
    return combined