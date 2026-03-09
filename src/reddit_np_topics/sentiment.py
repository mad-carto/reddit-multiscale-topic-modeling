"""
Document-level sentiment analysis using a RoBERTa model fine-tuned on social media text.
Adds sentiment label and score to each document in the pipeline.
"""

import pandas as pd
from transformers import pipeline
from torch.utils.data import Dataset

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = {
    "positive": "positive",
    "neutral": "neutral", 
    "negative": "negative",
}
MAX_LENGTH = 512


class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        self.batch_size = batch_size
        print(f"Loading sentiment model: {model_name}")
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            max_length=MAX_LENGTH,
            truncation=True,
            device=-1,  # CPU; set to 0 for GPU
        )

    def analyze_series(self, texts: pd.Series) -> pd.DataFrame:
        """
        Run sentiment analysis on a Series of strings.

        Returns a DataFrame with two columns:
            sentiment       : 'positive', 'neutral', or 'negative'
            sentiment_score : confidence score (0-1)
        """
        # Replace NaN/None with empty string to avoid pipeline errors
        texts = texts.fillna("").astype(str).tolist()

        print(f"Running sentiment analysis on {len(texts)} documents...")
        results = self.pipe(texts, batch_size=self.batch_size)

        sentiments = [r["label"].lower() for r in results]
        scores = [round(r["score"], 4) for r in results]

        print("Sentiment analysis complete.")
        return pd.DataFrame({
            "sentiment": sentiments,
            "sentiment_score": scores,
        })

    def analyze_dataframe(
        self, df: pd.DataFrame, text_col: str = "text"
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame in place.

        Args:
            df       : DataFrame containing the text column
            text_col : column to run sentiment on (use 'text' for raw text,
                       or 'tokens' for normalized text)

        Returns:
            DataFrame with 'sentiment' and 'sentiment_score' columns added.
        """
        df = df.copy()
        sentiment_df = self.analyze_series(df[text_col])
        df["sentiment"] = sentiment_df["sentiment"].values
        df["sentiment_score"] = sentiment_df["sentiment_score"].values
        return df