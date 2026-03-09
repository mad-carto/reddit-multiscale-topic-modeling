# Multiscale Topic Modeling of Reddit Discussions on US National Parks

An end-to-end NLP pipeline for extracting and analyzing discussion topics from Reddit data across all US National Parks. The project applies a two-level topic modeling approach — identifying both broad themes shared across all parks and park-specific topics — while actively mitigating geographic and volumetric data bias.

---

## What this project demonstrates

- **NLP pipeline design** — building a modular, reproducible text processing pipeline from raw social media data to structured topic classifications
- **Named Entity Recognition** — using GLiNER to detect and mask location mentions, preventing high-traffic parks from dominating topic representations
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
  ├── NER with GLiNER → toponym masking
  └── Text normalization (lemmatization, stopwords, regex)
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
  └── Interactive HTML plots (topic maps, hierarchies, per-park distributions)
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Topic Modeling | `bertopic`, `umap-learn`, `hdbscan` |
| NLP / Embeddings | `sentence-transformers`, `gliner`, `nltk` |
| Data & Storage | `duckdb`, `pandas`, `swifter` |
| Visualization | `plotly` |
| Environment | `uv`, `pyproject.toml` |

---

## Project Structure

```
src/reddit_np_topics/
├── db.py                     # DuckDB schema + read/write helpers
├── preprocessing/
│   ├── cleaner.py            # Raw data cleaning
│   ├── ner.py                # GLiNER NER + toponym masking
│   └── normalizer.py         # Text normalization
├── modeling/
│   ├── train_global.py       # Incremental global model training
│   ├── train_regional.py     # Per-park model training
│   └── utils.py              # Topic merging, coherence scoring
└── visualization/
    └── plots.py              # BERTopic visualization wrappers
```

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

**Cite this project**
```
@misc{reddit_np_topic_modeling,
  author = {Your Name},
  title  = {Multiscale Topic Modeling of Reddit Discussions on US National Parks},
  year   = {2024},
  url    = {https://github.com/mad-carto/reddit-multiscale-topic-modeling}
}
```