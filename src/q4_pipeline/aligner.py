from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.q4_pipeline.utils import tokens_are_similar


@dataclass
class AlignmentResult:
    bins: List[Set[str]]
    aligned_sequences: Dict[str, List[Optional[str]]]


def _token_matches_bin(token: str, bin_tokens: Set[str]) -> bool:
    return any(tokens_are_similar(token, candidate) for candidate in bin_tokens if candidate)


def _align_bins_to_sequence(bins: List[Set[str]], tokens: List[str]) -> List[Tuple[str, Optional[str]]]:
    rows = len(bins) + 1
    cols = len(tokens) + 1
    matrix = [[0] * cols for _ in range(rows)]
    backpointer: List[List[Tuple[int, int, str] | None]] = [[None] * cols for _ in range(rows)]

    for row in range(1, rows):
        matrix[row][0] = row
        backpointer[row][0] = (row - 1, 0, "delete")
    for col in range(1, cols):
        matrix[0][col] = col
        backpointer[0][col] = (0, col - 1, "insert")

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if _token_matches_bin(tokens[col - 1], bins[row - 1]) else 1
            candidates = [
                (matrix[row - 1][col - 1] + substitution_cost, row - 1, col - 1, "match"),
                (matrix[row - 1][col] + 1, row - 1, col, "delete"),
                (matrix[row][col - 1] + 1, row, col - 1, "insert"),
            ]
            best_cost, prev_row, prev_col, op = min(candidates, key=lambda item: item[0])
            matrix[row][col] = best_cost
            backpointer[row][col] = (prev_row, prev_col, op)

    operations: List[Tuple[str, Optional[str]]] = []
    row = len(bins)
    col = len(tokens)

    while row > 0 or col > 0:
        prev_row, prev_col, op = backpointer[row][col]
        token = tokens[col - 1] if col > 0 and op in {"match", "insert"} else None
        operations.append((op, token))
        row, col = prev_row, prev_col

    operations.reverse()
    return operations


def _apply_alignment(
    existing: AlignmentResult,
    sequence_name: str,
    tokens: List[str],
    operations: List[Tuple[str, Optional[str]]],
) -> AlignmentResult:
    new_bins: List[Set[str]] = []
    new_aligned: Dict[str, List[Optional[str]]] = {
        name: []
        for name in existing.aligned_sequences
    }
    new_aligned[sequence_name] = []

    bin_index = 0
    token_index = 0

    for op, token in operations:
        if op == "match":
            updated_bin = set(existing.bins[bin_index])
            if token:
                updated_bin.add(token)
            new_bins.append(updated_bin)

            for name, aligned_tokens in existing.aligned_sequences.items():
                new_aligned[name].append(aligned_tokens[bin_index])
            new_aligned[sequence_name].append(token)

            bin_index += 1
            token_index += 1
            continue

        if op == "delete":
            new_bins.append(set(existing.bins[bin_index]))
            for name, aligned_tokens in existing.aligned_sequences.items():
                new_aligned[name].append(aligned_tokens[bin_index])
            new_aligned[sequence_name].append(None)
            bin_index += 1
            continue

        if op == "insert":
            new_bins.append({token} if token else set())
            for name in existing.aligned_sequences:
                new_aligned[name].append(None)
            new_aligned[sequence_name].append(token)
            token_index += 1

    return AlignmentResult(bins=new_bins, aligned_sequences=new_aligned)


def progressive_align(sequence_map: Dict[str, List[str]], anchor_name: str = "Human") -> AlignmentResult:
    if anchor_name not in sequence_map:
        raise KeyError(f"Anchor sequence '{anchor_name}' not found.")

    anchor_tokens = sequence_map[anchor_name]
    result = AlignmentResult(
        bins=[{token} for token in anchor_tokens],
        aligned_sequences={anchor_name: list(anchor_tokens)},
    )

    for sequence_name, tokens in sequence_map.items():
        if sequence_name == anchor_name:
            continue

        operations = _align_bins_to_sequence(result.bins, tokens)
        result = _apply_alignment(result, sequence_name, tokens, operations)

    return result

