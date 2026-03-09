"""
Named Entity Recognition using GLiNER.
Detects location entities in text and replaces them with TOPONYM token
to prevent geographic bias in topic modeling.
"""

import pandas as pd
from gliner import GLiNER
from nltk.tokenize import TweetTokenizer

TOPONYM_TOKEN = "TOPONYM"
_tweet_tokenizer = TweetTokenizer()


class NERProcessor:
    def __init__(
        self,
        model_name: str = "urchade/gliner_base",
        labels: list[str] = None,
        threshold: float = 0.5,
        toponym_token: str = TOPONYM_TOKEN,
    ):
        self.model_name = model_name
        self.labels = labels or ["location"]
        self.threshold = threshold
        self.toponym_token = toponym_token
        print(f"Loading NER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)

    def process_text(self, text) -> tuple[list[str], str]:
        """
        Tokenize a single text string and mask location entities.
        Returns (token_list, loc_entities_string).
        """
        # Guard against None, NaN, or non-string values
        if not isinstance(text, str) or not text.strip():
            return [], ""

        try:
            entities = self.model.predict_entities(
                text, self.labels, threshold=self.threshold
            )
            tokenized = []
            loc_entities = []
            offset = 0

            for entity in entities:
                start, end = entity["start"], entity["end"]
                tokenized.extend(_tweet_tokenizer.tokenize(text[offset:start]))
                tokenized.append(self.toponym_token)
                loc_entities.append(entity["text"])
                offset = end + 1

            tokenized.extend(_tweet_tokenizer.tokenize(text[offset:]))
            return tokenized, "; ".join(loc_entities)

        except Exception as e:
            print(f"NER error: {e} — on: '{str(text)[:60]}'")
            try:
                return _tweet_tokenizer.tokenize(str(text)), ""
            except Exception:
                return [], ""

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Apply NER to a DataFrame column.
        Adds 'tokens' (list) and 'loc_entities' (str) columns.
        """
        print(f"Running NER on {len(df)} documents...")
        df = df.copy()

        # Use regular apply — swifter causes issues with class methods
        results = df[text_col].apply(self.process_text)
        df["tokens"], df["loc_entities"] = zip(*results)

        print("NER complete.")
        return df