from pathlib import Path

import pandas as pd

from src.q3_pipeline.normalizer import normalize_word


def load_words(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Unique words CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"Unique words CSV is empty: {csv_path}")

    word_column = "word" if "word" in frame.columns else frame.columns[0]
    words = frame[[word_column]].copy()
    words.columns = ["word"]
    words["word"] = words["word"].fillna("").astype(str)
    words = words[words["word"].str.strip().astype(bool)].reset_index(drop=True)
    words["normalized_word"] = words["word"].map(normalize_word)
    return words
