# Multiscale Topic Modeling of Reddit Discussions on US National Parks

An end-to-end NLP pipeline for extracting and analyzing discussion topics from Reddit data across all US National Parks. The project applies a two-level topic modeling approach — identifying both broad themes shared across all parks and park-specific topics — while actively mitigating geographic and volumetric data bias. Document-level sentiment analysis is integrated into the pipeline to capture how visitors feel about each topic.

---

## What this project demonstrates

- **NLP pipeline design** — building a modular, reproducible text processing pipeline from raw social media data to structured topic classifications
- **Named Entity Recognition** — using GLiNER to detect and mask location mentions, preventing high-traffic parks from dominating topic representations
- **Sentiment analysis** — applying a RoBERTa model fine-tuned on social media text to classify visitor sentiment per document
- **Topic modeling at scale** — incremental BERTopic training across shuffled data batches to handle class imbalance across 60+ National Parks
- **Analytical database integration** — replacing intermediate CSV files with DuckDB for efficient storage and querying of all pipeline stages
- **Clean Python packaging** — refactored from notebooks into an installable `src`-layout package with a clear separation of concerns

---

## Pipeline

```
data/raw/ (CSVs per park)
    │
    ▼
Preprocessing
  ├── Data cleaning (remove deleted/spam posts)
  ├── GLiNER NER → toponym masking
  ├── Text normalization (lemmatization, stopwords, regex)
  └── Sentiment analysis (per document) → label + confidence score
    │
    ▼
Global Topic Model
  ├── Dataset shuffled and split into 5 batches
  ├── BERTopic trained per batch, models merged incrementally
  └── Full corpus classified → global topics stored in DuckDB
    │
    ▼
Regional Topic Models
  ├── Per-park BERTopic model trained on globally-classified subset
  └── Park-specific topics stored in DuckDB
    │
    ▼
Analysis & Visualization
  └── Interactive HTML plots (topic maps, hierarchies, per-park distributions,
      sentiment distributions per topic and park)
```

All intermediate data is stored in a local **DuckDB** database instead of intermediate CSVs.

---

## Tech Stack

| Area | Tools |
|---|---|
| Topic Modeling | `bertopic`, `umap-learn`, `hdbscan` |
| NLP / Embeddings | `sentence-transformers`, `gliner`, `nltk` |
| Sentiment Analysis | `transformers` (cardiffnlp/twitter-roberta-base-sentiment) |
| Data & Storage | `duckdb`, `pandas` |
| Visualization | `plotly` |
| Environment | `uv`, `pyproject.toml` |

---

## Setup

```bash
git clone https://github.com/mad-carto/reddit-multiscale-topic-modeling.git
cd reddit-multiscale-topic-modeling
uv sync
```

Place raw Reddit CSVs in `data/raw/`, then run the notebooks in order:

```
notebooks/01_preprocessing.ipynb
notebooks/02_train_global_model.ipynb
notebooks/03_train_regional_models.ipynb
notebooks/04_analysis_visualization.ipynb
```

---

## Citations

**Dataset**
Reddit National Parks data provided by Alexander Dunkel.

**BERTopic**
Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*. https://arxiv.org/abs/2203.05794

**GLiNER**
Zaratiana, A., Tomeh, N., Holat, P., & Charnois, T. (2023). GLiNER: Generalist model for named entity recognition using bidirectional transformer. *arXiv preprint arXiv:2311.08526*. https://arxiv.org/abs/2311.08526

**Sentiment Model**
Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *arXiv preprint arXiv:2010.12421*. https://arxiv.org/abs/2010.12421

**Cite this project**
```
@misc{reddit_np_topic_modeling,
  author = {Madalina Gugulica},
  title  = {Multiscale Topic Modeling of Reddit Discussions on US National Parks},
  year   = {2024},
  url    = {https://github.com/mad-carto/reddit-multiscale-topic-modeling}
}
```