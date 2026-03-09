"""
Global incremental BERTopic model training.

The full dataset is split into N batches and a BERTopic model is trained
on each batch. Models are then merged to produce a single global model,
mitigating bias from unequal data volumes across National Parks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import stop_words
from umap import UMAP


def build_model_components(cfg: dict) -> dict:
    """
    Instantiate all BERTopic sub-components from config.

    Args:
        cfg: the 'modeling.global' section of config.yaml

    Returns:
        dict with keys: embedding_model, umap_model, hdbscan_model,
                        vectorizer_model, representation_model
    """
    embedding_model = SentenceTransformer(cfg["embedding_model"])

    umap_cfg = cfg["umap"]
    umap_model = UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        n_components=umap_cfg["n_components"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
    )

    hdbscan_cfg = cfg["hdbscan"]
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg["cluster_size"],
        metric=hdbscan_cfg["metric"],
        cluster_selection_method=hdbscan_cfg["cluster_selection_method"],
        prediction_data=True,
    )

    vec_cfg = cfg["vectorizer"]
    stop_words_list = list(stop_words.STOP_WORDS)
    vectorizer_model = CountVectorizer(
        stop_words=stop_words_list,
        min_df=vec_cfg["min_df"],
    )

    repr_cfg = cfg["representation"]
    representation_model = [
        KeyBERTInspired(top_n_words=repr_cfg["keybert_top_n_words"]),
        MaximalMarginalRelevance(diversity=repr_cfg["mmr_diversity"]),
    ]

    return {
        "embedding_model": embedding_model,
        "umap_model": umap_model,
        "hdbscan_model": hdbscan_model,
        "vectorizer_model": vectorizer_model,
        "representation_model": representation_model,
    }


def train_incremental(
    docs: pd.Series,
    cfg: dict,
    n_batches: int = 5,
) -> BERTopic:
    """
    Train a global BERTopic model incrementally across N shuffled batches.

    Args:
        docs: Series of normalized document strings (tokens column)
        cfg: full modeling config dict
        n_batches: number of batches to split data into

    Returns:
        Merged BERTopic base model
    """
    global_cfg = cfg["global"]
    global_cfg["embedding_model"] = cfg["embedding_model"]
    components = build_model_components(global_cfg)

    batches = np.array_split(docs.reset_index(drop=True), n_batches)
    base_model = None

    for i, batch in enumerate(batches):
        print(f"\nBatch {i + 1}/{n_batches} — {len(batch)} documents")
        batch_docs = batch.tolist()

        model = BERTopic(
            umap_model=components["umap_model"],
            hdbscan_model=components["hdbscan_model"],
            representation_model=components["representation_model"],
            top_n_words=global_cfg["top_n_words"],
            verbose=True,
        )
        topics, _ = model.fit_transform(batch_docs)

        try:
            new_topics = model.reduce_outliers(batch_docs, topics, strategy="distributions")
            model.update_topics(batch_docs, topics=new_topics)
        except ValueError as e:
            print(f"  Outlier reduction skipped: {e}")

        if base_model is None:
            base_model = model
        else:
            base_model = BERTopic.merge_models([base_model, model])
            print(f"  Merged → {len(base_model.get_topic_info())} topics")

    return base_model


def classify_with_base_model(
    docs: pd.Series,
    base_model: BERTopic,
    cfg: dict,
) -> tuple[BERTopic, list[int], list[float]]:
    """
    Use the merged base model's topic assignments to fit a clean final model
    via the manual classification approach. This preserves topic structure
    while applying the full vectorizer pipeline.

    Args:
        docs: Series of normalized document strings
        base_model: merged incremental model
        cfg: full modeling config dict

    Returns:
        (final_model, topics, probabilities)
    """
    global_cfg = cfg["global"]
    global_cfg["embedding_model"] = cfg["embedding_model"]
    components = build_model_components(global_cfg)

    embedding_model = components["embedding_model"]
    embeddings = embedding_model.encode(docs.tolist(), show_progress_bar=True)

    y = base_model.topics_

    final_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        representation_model=components["representation_model"],
        top_n_words=global_cfg["top_n_words"],
        verbose=True,
        calculate_probabilities=True,
    )

    topics, probs = final_model.fit_transform(docs.tolist(), embeddings=embeddings, y=y)
    return final_model, topics, probs


def save_model(model: BERTopic, output_dir: str | Path, name: str) -> Path:
    """Save a BERTopic model using safetensors serialization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    model.save(str(path), serialization="safetensors")
    print(f"Model saved to {path}")
    return path


def load_model(path: str | Path, embedding_model_name: str) -> BERTopic:
    """Load a saved BERTopic model."""
    embedding_model = SentenceTransformer(embedding_model_name)
    return BERTopic.load(str(path), embedding_model=embedding_model)