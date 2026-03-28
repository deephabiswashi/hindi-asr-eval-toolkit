from dataclasses import dataclass
from typing import Iterable, List, Sequence

from src.q4_pipeline.utils import comparison_key, tokens_are_similar


@dataclass
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_tokens: int = 0

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_tokens == 0:
            return 0.0
        return self.total_errors / self.reference_tokens

    def add(self, other: "ErrorCounts") -> None:
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.reference_tokens += other.reference_tokens


def _strict_match(left: str, right: str) -> bool:
    return comparison_key(left) == comparison_key(right)


def _bin_match(bin_payload: dict, token: str) -> bool:
    accepted_tokens = bin_payload.get("accepted_tokens", [])
    return any(tokens_are_similar(token, candidate) for candidate in accepted_tokens)


def _dp_counts(reference: Sequence, hypothesis: Sequence[str], matcher) -> ErrorCounts:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    matrix = [[0] * cols for _ in range(rows)]
    backpointer = [[None] * cols for _ in range(rows)]

    for row in range(1, rows):
        matrix[row][0] = row
        backpointer[row][0] = (row - 1, 0, "delete")
    for col in range(1, cols):
        matrix[0][col] = col
        backpointer[0][col] = (0, col - 1, "insert")

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if matcher(reference[row - 1], hypothesis[col - 1]) else 1
            candidates = [
                (matrix[row - 1][col - 1] + substitution_cost, row - 1, col - 1, "match"),
                (matrix[row - 1][col] + 1, row - 1, col, "delete"),
                (matrix[row][col - 1] + 1, row, col - 1, "insert"),
            ]
            best_cost, prev_row, prev_col, op = min(candidates, key=lambda item: item[0])
            matrix[row][col] = best_cost
            backpointer[row][col] = (prev_row, prev_col, op)

    row = len(reference)
    col = len(hypothesis)
    counts = ErrorCounts(reference_tokens=len(reference))

    while row > 0 or col > 0:
        prev_row, prev_col, op = backpointer[row][col]
        if op == "match":
            if matrix[row][col] != matrix[prev_row][prev_col]:
                counts.substitutions += 1
        elif op == "delete":
            counts.deletions += 1
        else:
            counts.insertions += 1
        row, col = prev_row, prev_col

    return counts


def compute_baseline_wer(reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str]) -> ErrorCounts:
    return _dp_counts(reference_tokens, hypothesis_tokens, _strict_match)


def compute_lattice_wer(lattice_bins: Sequence[dict], hypothesis_tokens: Sequence[str]) -> ErrorCounts:
    return _dp_counts(lattice_bins, hypothesis_tokens, _bin_match)


def aggregate_counts(items: Iterable[ErrorCounts]) -> ErrorCounts:
    total = ErrorCounts()
    for item in items:
        total.add(item)
    return total
