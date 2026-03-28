import csv
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


LOGGER_NAME = "q4_pipeline"

MOJIBAKE_MARKERS = ("\u00e0\u00a4", "\u00e0\u00a5", "\u00ef\u00bf\u00bd")
NON_TOKEN_RE = re.compile(r"[^0-9A-Za-z\u0900-\u097F]+")
NUKTA_RE = re.compile("\u093c")
MATRA_RE = re.compile(r"[\u093e-\u094c\u0955-\u0957\u0962\u0963]")
MULTISPACE_RE = re.compile(r"\s+")
DEVANAGARI_DIGIT_TRANS = str.maketrans("०१२३४५६७८९", "0123456789")

NUMBER_WORDS = {
    "शून्य": "0",
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पांच": "5",
    "पाँच": "5",
    "छह": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
    "दस": "10",
    "ग्यारह": "11",
    "बारह": "12",
    "तेरह": "13",
    "चौदह": "14",
    "पंद्रह": "15",
    "पन्द्रह": "15",
    "सोलह": "16",
    "सत्रह": "17",
    "अठारह": "18",
    "उन्नीस": "19",
    "बीस": "20",
    "तीस": "30",
    "चालीस": "40",
    "पचास": "50",
    "साठ": "60",
    "सत्तर": "70",
    "अस्सी": "80",
    "नब्बे": "90",
    "सौ": "100",
    "हजार": "1000",
    "हज़ार": "1000",
}


@dataclass
class TaskRecord:
    utterance_id: str
    segment_url: str
    reference: str
    model_outputs: Dict[str, str]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def repair_mojibake(text: str) -> str:
    text = str(text or "")
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def normalize_text(text: str) -> str:
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", " ").replace("\u200d", " ").replace("\ufeff", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_token(token: str) -> str:
    token = normalize_text(token)
    token = NON_TOKEN_RE.sub("", token)
    return token.lower()


def normalize_numeric_token(token: str) -> str:
    token = token.translate(DEVANAGARI_DIGIT_TRANS)
    return NUMBER_WORDS.get(token, token)


def comparison_key(token: str) -> str:
    token = normalize_token(token)
    token = normalize_numeric_token(token)
    token = NUKTA_RE.sub("", token)
    token = token.replace("ँ", "ं")
    return token


def phonetic_key(token: str) -> str:
    token = comparison_key(token)
    token = MATRA_RE.sub("", token)
    token = re.sub(r"(.)\1+", r"\1", token)
    return token


def tokenize_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = []
    for raw_token in normalized.split():
        token = normalize_token(raw_token)
        if token:
            tokens.append(token)
    return tokens


def edit_distance(left: str, right: str) -> int:
    rows = len(left) + 1
    cols = len(right) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if left[row - 1] == right[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + substitution_cost,
            )
    return matrix[-1][-1]


def tokens_are_similar(left: str, right: str, max_edit_distance: int = 2) -> bool:
    if not left or not right:
        return False

    left_key = comparison_key(left)
    right_key = comparison_key(right)
    if left_key == right_key:
        return True

    if phonetic_key(left_key) == phonetic_key(right_key):
        return True

    if min(len(left_key), len(right_key)) <= 2:
        return left_key == right_key

    return edit_distance(left_key, right_key) <= max_edit_distance


def merge_split_pairs(sequence_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    inventory = {comparison_key(token) for tokens in sequence_map.values() for token in tokens if token}
    merged_sequences: Dict[str, List[str]] = {}

    for name, tokens in sequence_map.items():
        merged: List[str] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens):
                current = tokens[index]
                next_token = tokens[index + 1]
                merged_token = normalize_token(current + next_token)
                if merged_token and merged_token in inventory and min(len(current), len(next_token)) <= 2:
                    merged.append(merged_token)
                    index += 2
                    continue

            merged.append(tokens[index])
            index += 1

        merged_sequences[name] = merged

    return merged_sequences


def load_task_records(csv_path: str | Path) -> List[TaskRecord]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Question 4 CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"Question 4 CSV has no headers: {csv_path}")

        fieldnames = [field.strip() for field in reader.fieldnames if field and field.strip()]
        segment_column = next((field for field in fieldnames if "segment_url" in field.lower()), None)
        reference_column = next((field for field in fieldnames if field.lower() == "human"), None)
        model_columns = [field for field in fieldnames if field.lower().startswith("model")]

        if segment_column is None or reference_column is None or not model_columns:
            raise ValueError(
                "Question 4 CSV must contain a segment URL column, a Human reference column, "
                "and at least one Model column."
            )

        records: List[TaskRecord] = []
        for index, row in enumerate(reader, start=1):
            segment_url = normalize_text(row.get(segment_column, ""))
            reference = normalize_text(row.get(reference_column, ""))
            model_outputs = {
                column: normalize_text(row.get(column, ""))
                for column in model_columns
            }

            records.append(
                TaskRecord(
                    utterance_id=f"utt_{index:04d}",
                    segment_url=segment_url,
                    reference=reference,
                    model_outputs=model_outputs,
                )
            )

    return records


def serialize_sequence_map(sequence_map: Dict[str, Iterable[str | None]]) -> Dict[str, List[str | None]]:
    return {name: list(values) for name, values in sequence_map.items()}

