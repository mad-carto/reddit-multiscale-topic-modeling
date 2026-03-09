"""
Utility functions for topic model post-processing.
Includes cosine similarity analysis and topic merge helpers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic


def compute_topic_distances(model: BERTopic) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between all topic embeddings.

    Returns a long-format DataFrame with columns:
        topic1, topic2, similarity
    Excludes outlier topic (-1) and duplicate pairs.
    """
    distance_matrix = cosine_similarity(np.array(model.topic_embeddings_))
    labels = list(model.topic_labels_.values())

    dist_df = pd.DataFrame(distance_matrix, columns=labels, index=labels)

    records = []
    for rec in dist_df.reset_index().to_dict("records"):
        t1 = rec["index"]
        for t2, score in rec.items():
            if t2 == "index":
                continue
            records.append({"topic1": t1, "topic2": t2, "similarity": score})

    pair_df = pd.DataFrame(records)

    # Exclude outlier topic and keep only unique pairs (upper triangle)
    pair_df = pair_df[
        ~pair_df["topic1"].str.startswith("-1") &
        ~pair_df["topic2"].str.startswith("-1")
    ]
    pair_df = pair_df[pair_df["topic1"] < pair_df["topic2"]]

    return pair_df.sort_values("similarity", ascending=False).reset_index(drop=True)


def get_merge_candidates(model: BERTopic, threshold: float = 0.85, top_n: int = 30) -> pd.DataFrame:
    """
    Return topic pairs with cosine similarity above a threshold.
    Useful for identifying candidates for manual merging.

    Args:
        model: fitted BERTopic model
        threshold: minimum similarity score to flag as candidate
        top_n: number of top pairs to return

    Returns:
        DataFrame of (topic1, topic2, similarity) sorted by similarity desc
    """
    pair_df = compute_topic_distances(model)
    candidates = pair_df[pair_df["similarity"] >= threshold].head(top_n)
    return candidates


def reduce_outliers(
    model: BERTopic,
    docs: list[str],
    topics: list[int],
    strategy: str = "distributions",
    threshold: float = 0.5,
) -> list[int]:
    """
    Reduce outlier documents (topic -1) using the specified strategy.

    Args:
        model: fitted BERTopic model
        docs: list of document strings
        topics: current topic assignments
        strategy: 'distributions' or 'probabilities'
        threshold: probability threshold (used for 'probabilities' strategy)

    Returns:
        Updated topic list with fewer -1 assignments
    """
    try:
        if strategy == "probabilities":
            new_topics = model.reduce_outliers(docs, topics, strategy=strategy, threshold=threshold)
        else:
            new_topics = model.reduce_outliers(docs, topics, strategy=strategy)
        return new_topics
    except ValueError as e:
        print(f"Outlier reduction failed ({strategy}): {e}. Returning original topics.")
        return topics


def attach_topic_info(
    df: pd.DataFrame,
    model: BERTopic,
    topics: list[int],
    id_col: str = "Global_Topic_ID",
    name_col: str = "Global_Topic",
    repr_col: str = "Global_Repr",
) -> pd.DataFrame:
    """
    Merge topic metadata (name, representation) back onto the documents DataFrame.

    Args:
        df: documents DataFrame
        model: fitted BERTopic model with custom labels set
        topics: topic assignments aligned with df rows
        id_col: column name for topic ID
        name_col: column name for topic custom label
        repr_col: column name for topic keyword representation

    Returns:
        df with topic columns added
    """
    model_info = model.get_topic_info()
    topics_df = pd.DataFrame({"Topic": topics})
    result = pd.concat([df.reset_index(drop=True), topics_df], axis=1)
    result["Topic"] = result["Topic"].astype(int)
    result = pd.merge(result, model_info, on="Topic", how="left")

    label_col = "CustomName" if "CustomName" in result.columns else "Name"
    result = result.rename(columns={
        "Topic": id_col,
        label_col: name_col,
        "Representation": repr_col,
    })

    drop_cols = [c for c in ["Count", "Representative_Docs", "Top_n_words",
                              "Representative_document", "Name", "CustomName"] if c in result.columns]
    result = result.drop(columns=drop_cols)

    return result