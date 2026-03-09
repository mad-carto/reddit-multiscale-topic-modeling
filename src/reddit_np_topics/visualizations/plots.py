"""
BERTopic visualization wrappers.
All figures are saved as interactive HTML files to outputs/visualizations/.
"""

import pandas as pd
from pathlib import Path
from bertopic import BERTopic


def save_fig(fig, output_dir: Path, filename: str) -> Path:
    """Save a Plotly figure as an HTML file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.write_html(str(path))
    print(f"Saved: {path}")
    return path


def plot_topic_overview(
    model: BERTopic,
    output_dir: str | Path,
    prefix: str = "global",
    top_n_topics: int = 36,
) -> None:
    """
    Save the main topic overview visualizations:
    - Topic barchart (top keywords per topic)
    - 2D topic map

    Args:
        model: fitted BERTopic model with custom labels set
        output_dir: directory to write HTML files
        prefix: filename prefix (e.g. 'global' or park name)
        top_n_topics: number of topics to include in barchart
    """
    output_dir = Path(output_dir)

    fig_bar = model.visualize_barchart(top_n_topics=top_n_topics, custom_labels=True)
    save_fig(fig_bar, output_dir, f"{prefix}_topic_barchart.html")

    fig_map = model.visualize_topics(custom_labels=True)
    save_fig(fig_map, output_dir, f"{prefix}_topic_map.html")


def plot_hierarchy(
    model: BERTopic,
    docs: list[str],
    output_dir: str | Path,
    prefix: str = "global",
) -> None:
    """
    Save a hierarchical topic dendrogram.

    Args:
        model: fitted BERTopic model
        docs: list of document strings used for training
        output_dir: directory to write HTML file
        prefix: filename prefix
    """
    output_dir = Path(output_dir)
    hierarchical_topics = model.hierarchical_topics(docs)
    fig = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels=True)
    save_fig(fig, output_dir, f"{prefix}_hierarchy.html")


def plot_topics_per_park(
    model: BERTopic,
    docs: list[str],
    park_names: list[str],
    output_dir: str | Path,
    prefix: str = "global",
    top_n_topics: int = 36,
) -> None:
    """
    Save a normalized topics-per-class chart showing topic distribution
    across National Parks.

    Args:
        model: fitted BERTopic model
        docs: list of document strings
        park_names: list of class labels aligned with docs
        output_dir: directory to write HTML file
        prefix: filename prefix
        top_n_topics: number of topics to include
    """
    output_dir = Path(output_dir)
    topics_per_class = model.topics_per_class(docs, classes=park_names)
    fig = model.visualize_topics_per_class(
        topics_per_class,
        title="Topic Distribution per National Park",
        top_n_topics=top_n_topics,
        custom_labels=True,
        normalize_frequency=True,
    )
    save_fig(fig, output_dir, f"{prefix}_topics_per_park.html")


def plot_all_global(
    model: BERTopic,
    docs: list[str],
    park_names: list[str],
    output_dir: str | Path,
    top_n_topics: int = 36,
) -> None:
    """
    Convenience function: generate all global-level visualizations at once.

    Args:
        model: fitted global BERTopic model
        docs: list of document strings
        park_names: park name label per document
        output_dir: directory to write HTML files
        top_n_topics: number of topics for barchart and per-park chart
    """
    plot_topic_overview(model, output_dir, prefix="global", top_n_topics=top_n_topics)
    plot_hierarchy(model, docs, output_dir, prefix="global")
    plot_topics_per_park(model, docs, park_names, output_dir, prefix="global", top_n_topics=top_n_topics)


def plot_regional(
    model: BERTopic,
    docs: list[str],
    park_name: str,
    output_dir: str | Path,
) -> None:
    """
    Generate visualizations for a single regional (per-park) model.

    Args:
        model: fitted regional BERTopic model
        docs: list of document strings for this park
        park_name: name of the park (used in filenames)
        output_dir: directory to write HTML files
    """
    output_dir = Path(output_dir) / park_name
    plot_topic_overview(model, output_dir, prefix=park_name)
    plot_hierarchy(model, docs, output_dir, prefix=park_name)