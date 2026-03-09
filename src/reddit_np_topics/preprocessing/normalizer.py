"""
Text normalization for Reddit post tokens.
Handles lowercasing, stopword removal, lemmatization,
URL/emoji/punctuation stripping, and more.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first use
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


EXTRA_STOPWORDS = [
    "toponym", "id", "youd", "im", "ive", "shes", "weve",
    "youre", "youve", "theyve", "youll", "u", "dont", "didnt",
]

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)

HTML_PATTERN = re.compile("<.*?>")
MENTION_HASHTAG_PATTERN = re.compile(r"(@\S+|#\S+)")
DATE_PATTERN = re.compile(r"\d{1,2}(st|nd|rd|th)?[-./]\d{1,2}[-./]\d{2,4}")
URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^\w\s]")


class TextNormalizer:
    """
    Normalizes a list of string tokens through a series of cleaning steps.
    Designed to be applied after tokenization and NER (toponym masking).
    """

    def __init__(self, extra_stopwords: list[str] | None = None):
        base_stopwords = stopwords.words("english")
        self.stopwords = set(base_stopwords + EXTRA_STOPWORDS + (extra_stopwords or []))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def normalize_tokens(self, tokens: list[str]) -> str:
        """
        Apply full normalization pipeline to a list of tokens.
        Returns a single normalized string.
        """
        tokens = [t.lower() for t in tokens]
        tokens = [NUMBER_PATTERN.sub("", t) for t in tokens]
        tokens = [URL_PATTERN.sub("", t) for t in tokens]
        tokens = [EMAIL_PATTERN.sub("", t) for t in tokens]
        tokens = [DATE_PATTERN.sub("", t) for t in tokens]
        tokens = [HTML_PATTERN.sub("", t) for t in tokens]
        tokens = [EMOJI_PATTERN.sub("", t) for t in tokens]
        tokens = [MENTION_HASHTAG_PATTERN.sub("", t) for t in tokens]
        tokens = [t.translate(self.punctuation_table) for t in tokens]
        tokens = [NON_ALPHANUMERIC_PATTERN.sub("", t) for t in tokens]
        tokens = [WHITESPACE_PATTERN.sub(" ", t).strip() for t in tokens]
        tokens = [t for t in tokens if t and t not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t]

        return " ".join(tokens)

    def normalize_series(self, token_lists: "pd.Series") -> "pd.Series":
        """Apply normalization to a pandas Series of token lists."""
        return token_lists.apply(self.normalize_tokens)