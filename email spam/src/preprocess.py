"""Text preprocessing and data loading utilities for SMS/Email Spam Classifier.

This module handles:
- Checking/downloading required NLTK datasets (stopwords, punkt)
- Data ingestion with latin-1 encoding
- Column cleanup and label mapping (ham -> 0, spam -> 1)
- Deduplication and text cleaning (punctuation, digits, stopwords, stemming)
"""

import os
import re
import socket
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Set socket timeout to prevent indefinite network hangs
socket.setdefaulttimeout(5.0)

_NLTK_INITIALIZED = False
_STOPWORDS_SET = None
_STEMMER = PorterStemmer()


def download_nltk_resources() -> None:
    """Download required NLTK resources safely with timeout protection."""
    global _NLTK_INITIALIZED, _STOPWORDS_SET
    if _NLTK_INITIALIZED:
        return

    # Check and download essential resources
    for resource_path, download_name in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(f"[Warning] Could not download '{download_name}': {e}", flush=True)

    try:
        _STOPWORDS_SET = set(stopwords.words("english"))
    except Exception:
        _STOPWORDS_SET = set()

    _NLTK_INITIALIZED = True


def clean_text(
    text: str,
    use_stemming: bool = True,
    remove_stopwords_flag: bool = True
) -> str:
    """Preprocess a single text message.

    Steps:
    1. Lowercase text
    2. Remove URLs, email addresses
    3. Remove numbers / digits
    4. Remove punctuation
    5. Tokenize
    6. Remove English stopwords
    7. Apply PorterStemmer
    8. Rejoin into clean string

    Args:
        text: Input string message.
        use_stemming: Whether to apply PorterStemmer.
        remove_stopwords_flag: Whether to remove stopwords.

    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""

    if not _NLTK_INITIALIZED:
        download_nltk_resources()

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and email patterns
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # 3. Remove digits and numbers
    text = re.sub(r"\d+", " ", text)

    # 4. Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # 5. Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r"\b[a-z]+\b", text)

    # 6. Stopwords removal
    if remove_stopwords_flag and _STOPWORDS_SET:
        tokens = [w for w in tokens if w not in _STOPWORDS_SET and len(w) > 1]
    else:
        tokens = [w for w in tokens if len(w) > 1]

    # 7. Stemming
    if use_stemming:
        tokens = [_STEMMER.stem(w) for w in tokens]

    return " ".join(tokens)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load raw spam.csv dataset, clean columns, drop duplicates, and preprocess text.

    Args:
        filepath: Path to spam.csv dataset.

    Returns:
        Cleaned pandas DataFrame with columns:
        ['label', 'text', 'cleaned_text', 'char_length', 'word_count']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print(f"[Data Loader] Reading dataset from: {filepath}", flush=True)
    # Read with latin-1 encoding
    df = pd.read_csv(filepath, encoding="latin-1")

    # Drop unnamed extra columns if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:") or "Unnamed" in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Ensure v1 and v2 exist, rename to label and text
    rename_dict = {}
    if "v1" in df.columns:
        rename_dict["v1"] = "label"
    if "v2" in df.columns:
        rename_dict["v2"] = "text"

    df = df.rename(columns=rename_dict)

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Columns 'label' and 'text' not found in dataset. Found: {list(df.columns)}")

    # Keep only label and text
    df = df[["label", "text"]].copy()

    # Drop NaN rows
    df = df.dropna(subset=["label", "text"])

    # Map ham -> 0, spam -> 1
    label_map = {"ham": 0, "spam": 1}
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    total_before = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    total_after = len(df)
    duplicates_removed = total_before - total_after

    print(f"[Data Loader] Loaded {total_before} records. Dropped {duplicates_removed} duplicates. Remaining: {total_after}", flush=True)
    ham_count = (df["label"] == 0).sum()
    spam_count = (df["label"] == 1).sum()
    print(f"[Data Loader] Class distribution: Ham (0) = {ham_count} ({ham_count/total_after*100:.1f}%), Spam (1) = {spam_count} ({spam_count/total_after*100:.1f}%)", flush=True)

    # Feature engineering for EDA
    df["char_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda s: len(s.split()))

    # Apply text cleaning
    print("[Data Loader] Cleaning text messages...", flush=True)
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["cleaned_text"] = df["cleaned_text"].fillna("")

    print("[Data Loader] Preprocessing completed successfully.", flush=True)
    return df


if __name__ == "__main__":
    download_nltk_resources()
    sample = "WINNER!! As a valued customer you have won $1000 cash prize! Text CLAIM to 88088 now."
    print("Sample original:", sample)
    print("Sample cleaned :", clean_text(sample))
