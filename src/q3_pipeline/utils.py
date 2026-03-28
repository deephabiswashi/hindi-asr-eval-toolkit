import json
import logging
import random
from pathlib import Path


try:
    import editdistance as _editdistance
except ImportError:  # pragma: no cover - fallback for environments without editdistance
    _editdistance = None


LOGGER_NAME = "q3_pipeline"


def setup_logging():
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def edit_distance(left, right):
    if _editdistance is not None:
        return _editdistance.eval(left, right)

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


def deterministic_sample(items, sample_size, seed=42):
    items = list(items)
    if len(items) <= sample_size:
        return items

    rng = random.Random(seed)
    return rng.sample(items, sample_size)

