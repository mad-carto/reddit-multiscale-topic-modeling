# Reddit National Parks — Multiscale Topic Modeling

Analyzing Reddit discussions about US National Parks using a two-level BERTopic pipeline. Topics are extracted at a **global** level (across all parks) and a **regional** level (per individual park).

Location bias is mitigated by detecting place names with GLiNER and replacing them with a `TOPONYM` token before modeling. Incremental training across shuffled batches addresses data imbalance between parks.

---

## Pipeline

```
Raw CSVs → Cleaning → NER + Toponym Masking → Text Normalization
       → Global BERTopic Model (incremental, 5 batches)
       → Regional BERTopic Models (per park)
       → Analysis & Visualization
```

All intermediate data is stored in a local **DuckDB** database 
---

## Setup

```bash
git clone https://github.com/yourusername/reddit-multiscale-topic-modeling.git
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
# Reddit National Parks — Multiscale Topic Modeling

Analyzing Reddit discussions about US National Parks using a two-level BERTopic pipeline. Topics are extracted at a **global** level (across all parks) and a **regional** level (per individual park).

Location bias is mitigated by detecting place names with GLiNER and replacing them with a `TOPONYM` token before modeling. Incremental training across shuffled batches addresses data imbalance between parks.

---

## Pipeline

```
Raw CSVs → Cleaning → NER + Toponym Masking → Text Normalization
       → Global BERTopic Model (incremental, 5 batches)
       → Regional BERTopic Models (per park)
       → Analysis & Visualization
```

All intermediate data is stored in a local **DuckDB** database instead of intermediate CSVs.

---

## Setup

```bash
git clone https://github.com/yourusername/reddit-multiscale-topic-modeling.git
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

## Stack

`bertopic` · `gliner` · `sentence-transformers` · `duckdb` · `umap-learn` · `hdbscan`

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
## Stack

`bertopic` · `gliner` · `sentence-transformers` · `duckdb` · `umap-learn` · `hdbscan`