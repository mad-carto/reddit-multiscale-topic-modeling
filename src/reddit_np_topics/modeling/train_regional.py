"""
Regional BERTopic model training — one model per National Park.

Each park's subset of globally-classified documents is used to train
a park-specific topic model, revealing local discussions that would
be invisible in a global model.
"""

import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import stop_words
from umap import UMAP

from reddit_np_topics.modeling.utils import reduce_outliers, attach_topic_info


def build_regional_components(cfg: dict, n_docs: int) -> dict:
    """
    Build BERTopic sub-components for a regional (per-park) model.
    Cluster size is dynamically set as a fraction of the park's document count.

    Args:
        cfg: full modeling config dict
        n_docs: number of documents for this park

    Returns:
        dict of model components
    """
    regional_cfg = cfg["regional"]

    embedding_model = SentenceTransformer(cfg["embedding_model"])

    cluster_size = max(
        int(regional_cfg["cluster_size_ratio"] * n_docs),
        regional_cfg["cluster_size_min"],
    )

    umap_cfg = regional_cfg["umap"]
    umap_model = UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        n_components=umap_cfg["n_components"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
    )

    hdbscan_cfg = regional_cfg["hdbscan"]
    hdbscan_model = HDBSCAN(
        min_cluster_size=cluster_size,
        metric=hdbscan_cfg["metric"],
        cluster_selection_method=hdbscan_cfg["cluster_selection_method"],
        prediction_data=True,
    )

    vec_cfg = regional_cfg["vectorizer"]
    stop_words_list = list(stop_words.STOP_WORDS)
    vectorizer_model = CountVectorizer(
        stop_words=stop_words_list,
        max_df=vec_cfg["max_df"],
    )

    repr_cfg = regional_cfg["representation"]
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
        "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True),
        "cluster_size": cluster_size,
    }


def train_regional_model(
    park_df: pd.DataFrame,
    park_name: str,
    cfg: dict,
) -> tuple[BERTopic, pd.DataFrame]:
    """
    Train a BERTopic model for a single National Park.

    Args:
        park_df: DataFrame of globally-classified documents for this park
        park_name: name of the park (used for logging)
        cfg: full modeling config dict

    Returns:
        (fitted BERTopic model, park_df with regional topic columns added)
    """
    regional_cfg = cfg["regional"]
    docs = park_df["tokens"].tolist()
    n_docs = len(docs)
    print(f"\n[{park_name}] Training regional model on {n_docs} documents")

    components = build_regional_components(cfg, n_docs)
    print(f"[{park_name}] Cluster size: {components['cluster_size']}")

    embeddings = components["embedding_model"].encode(docs, show_progress_bar=True)

    model = BERTopic(
        embedding_model=components["embedding_model"],
        umap_model=components["umap_model"],
        hdbscan_model=components["hdbscan_model"],
        vectorizer_model=components["vectorizer_model"],
        ctfidf_model=components["ctfidf_model"],
        representation_model=components["representation_model"],
        top_n_words=regional_cfg["top_n_words"],
        verbose=True,
        calculate_probabilities=True,
    )

    topics, probs = model.fit_transform(docs, embeddings=embeddings)

    # Outlier reduction
    outlier_cfg = regional_cfg["outlier_reduction"]
    topics = reduce_outliers(
        model, docs, topics,
        strategy=outlier_cfg["strategy"],
        threshold=outlier_cfg["threshold"],
    )
    model.update_topics(docs, topics=topics)

    print(f"[{park_name}] {len(model.get_topic_info())} topics found")

    # Attach topic metadata to DataFrame
    result_df = attach_topic_info(
        park_df,
        model,
        topics,
        id_col="Regional_Topic_ID",
        name_col="Regional_Topic",
        repr_col="Regional_Repr",
    )

    # Add topic probability
    topic_probs = [
        float(probs[i][t]) if t != -1 else 0.0
        for i, t in enumerate(topics)
    ]
    result_df["Regional_Topic_Prob"] = topic_probs

    return model, result_df


def train_all_regional_models(
    global_df: pd.DataFrame,
    cfg: dict,
    models_dir: str | Path,
    parks: list[str] | None = None,
) -> pd.DataFrame:
    """
    Train a regional model for each National Park and collect results.

    Args:
        global_df: globally-classified DataFrame with 'park_name' column
        cfg: full modeling config dict
        models_dir: directory to save regional models
        parks: optional list of park names to process (default: all)

    Returns:
        Combined DataFrame with both global and regional topic columns
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if parks is None:
        parks = sorted(global_df["park_name"].unique())

    all_results = []

    for park_name in parks:
        park_df = global_df[global_df["park_name"] == park_name].reset_index(drop=True)

        if len(park_df) < 10:
            print(f"[{park_name}] Skipping — only {len(park_df)} documents (minimum 10 required)")
            continue

        model, result_df = train_regional_model(park_df, park_name, cfg)

        model_path = models_dir / park_name
        model.save(str(model_path), serialization="safetensors")
        print(f"[{park_name}] Model saved to {model_path}")

        all_results.append(result_df)

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nRegional classification complete — {len(combined)} documents across {len(all_results)} parks")
    return combined