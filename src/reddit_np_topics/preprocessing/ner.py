"""
Named Entity Recognition using GLiNER and toponym masking.

Location entities detected in Reddit posts are replaced with the
TOPONYM token to prevent popular park names from dominating topic
representations due to unequal data volumes across National Parks.
"""

import pandas as pd
import swifter  # noqa: F401 — enables .swifter.apply()
from gliner import GLiNER
from nltk.tokenize import TweetTokenizer

TOPONYM_TOKEN = "TOPONYM"
DEFAULT_MODEL = "urchade/gliner_base"
DEFAULT_LABELS = ["location"]
DEFAULT_THRESHOLD = 0.5


class NERProcessor:
    """
    Wraps GLiNER for batch NER processing of Reddit text.

    For each document:
    - Detects location entities
    - Replaces them with TOPONYM in the token stream
    - Returns the masked token list and a semicolon-separated string of extracted locations
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        labels: list[str] = DEFAULT_LABELS,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        print(f"Loading GLiNER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)
        self.labels = labels
        self.threshold = threshold
        self.tokenizer = TweetTokenizer()

    def process_text(self, text: str) -> tuple[list[str], str]:
        """
        Tokenize a single text, masking location entities as TOPONYM.

        Returns:
            tokens: list of tokens with locations replaced by TOPONYM
            loc_str: semicolon-separated string of detected location entities
        """
        try:
            if not text or not text.strip():
                return [], ""

            entities = self.model.predict_entities(text, self.labels, threshold=self.threshold)

            tokenized = []
            loc_entities = []
            offset = 0

            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                tokenized.extend(self.tokenizer.tokenize(text[offset:start]))
                tokenized.append(TOPONYM_TOKEN)
                loc_entities.append(entity["text"])
                offset = end + 1

            tokenized.extend(self.tokenizer.tokenize(text[offset:]))
            return tokenized, "; ".join(loc_entities)

        except Exception as e:
            print(f"NER error on text snippet '{text[:80]}...': {e}")
            return self.tokenizer.tokenize(text), ""

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Apply NER processing to a DataFrame column in parallel using swifter.

        Adds two new columns:
        - 'tokens': list of tokens with toponyms masked
        - 'loc_entities': semicolon-separated detected locations

        Returns the modified DataFrame.
        """
        print(f"Running NER on {len(df)} documents...")
        df = df.copy()
        df["tokens"], df["loc_entities"] = zip(
            *df[text_col].swifter.apply(self.process_text)
        )
        print("NER complete.")
        return df