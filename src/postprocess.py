import re
import unicodedata
from dataclasses import dataclass


MOJIBAKE_MARKERS = ("\u00e0\u00a4", "\u00e0\u00a5", "\u00ef\u00bf\u00bd")


@dataclass(frozen=True)
class RepetitionMetrics:
    repeated_token_ratio: float
    max_token_run: int
    max_ngram_repeat: int


def repair_mojibake(text):
    text = str(text or "")
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def normalize_spaces(text):
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0964", " ").replace("\u0965", " ").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def whitespace_tokens(text):
    text = normalize_spaces(text)
    if not text:
        return []
    return text.split(" ")


def _max_token_run(tokens):
    if not tokens:
        return 0

    best = 1
    current = 1
    for previous, current_token in zip(tokens, tokens[1:]):
        if current_token == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _find_repeat_blocks(tokens, max_ngram=8, min_repeat=3):
    blocks = []
    index = 0

    while index < len(tokens):
        best_block = None
        max_size = min(max_ngram, (len(tokens) - index) // min_repeat)

        for ngram_size in range(max_size, 0, -1):
            phrase = tokens[index : index + ngram_size]
            repeat_count = 1

            while index + (repeat_count + 1) * ngram_size <= len(tokens):
                start = index + repeat_count * ngram_size
                end = start + ngram_size
                if tokens[start:end] != phrase:
                    break
                repeat_count += 1

            if repeat_count >= min_repeat:
                best_block = (index, ngram_size, repeat_count, phrase)
                break

        if best_block is None:
            index += 1
            continue

        blocks.append(best_block)
        index += best_block[1] * best_block[2]

    return blocks


def repetition_metrics(text, max_ngram=8, min_repeat=3):
    tokens = whitespace_tokens(text)
    if not tokens:
        return RepetitionMetrics(0.0, 0, 0)

    blocks = _find_repeat_blocks(tokens, max_ngram=max_ngram, min_repeat=min_repeat)
    repeated_tokens = sum(ngram_size * repeat_count for _, ngram_size, repeat_count, _ in blocks)
    max_ngram_repeat = max((repeat_count for _, _, repeat_count, _ in blocks), default=1)

    return RepetitionMetrics(
        repeated_token_ratio=repeated_tokens / len(tokens),
        max_token_run=_max_token_run(tokens),
        max_ngram_repeat=max_ngram_repeat,
    )


def collapse_repetition_loops(text, max_ngram=8, min_repeat=3, keep_repeats=1, max_token_repeat=2):
    tokens = whitespace_tokens(text)
    if not tokens:
        return ""

    for _ in range(3):
        blocks = _find_repeat_blocks(tokens, max_ngram=max_ngram, min_repeat=min_repeat)
        if not blocks:
            break

        rebuilt = []
        cursor = 0
        for start, ngram_size, repeat_count, phrase in blocks:
            rebuilt.extend(tokens[cursor:start])
            rebuilt.extend(phrase * keep_repeats)
            cursor = start + ngram_size * repeat_count
        rebuilt.extend(tokens[cursor:])
        tokens = rebuilt

    compact_tokens = []
    for token in tokens:
        if compact_tokens and compact_tokens[-1] == token:
            run_length = 1
            for previous in reversed(compact_tokens[:-1]):
                if previous != token:
                    break
                run_length += 1
            if run_length >= max_token_repeat:
                continue
        compact_tokens.append(token)

    return " ".join(compact_tokens)


def _trim_trailing_loop(tokens, max_ngram=4, min_repeat=3):
    trimmed_tokens = list(tokens)

    while len(trimmed_tokens) >= min_repeat:
        changed = False
        max_size = min(max_ngram, len(trimmed_tokens) // min_repeat)

        for ngram_size in range(max_size, 0, -1):
            phrase = trimmed_tokens[-ngram_size:]
            repeat_count = 1

            while repeat_count * ngram_size < len(trimmed_tokens):
                start = len(trimmed_tokens) - (repeat_count + 1) * ngram_size
                end = start + ngram_size
                if start < 0 or trimmed_tokens[start:end] != phrase:
                    break
                repeat_count += 1

            if repeat_count >= min_repeat:
                keep_from = len(trimmed_tokens) - repeat_count * ngram_size
                trimmed_tokens = trimmed_tokens[:keep_from] + phrase
                changed = True
                break

        if not changed:
            break

    return trimmed_tokens


def clean_prediction(text):
    text = normalize_spaces(text)
    if not text:
        return ""

    tokens = whitespace_tokens(text)
    if not tokens:
        return ""

    # Collapse only clear loops so valid emphasis like "बहुत बहुत" is preserved.
    cleaned_text = collapse_repetition_loops(
        text,
        max_ngram=4,
        min_repeat=3,
        keep_repeats=1,
        max_token_repeat=2,
    )
    cleaned_tokens = whitespace_tokens(cleaned_text)
    cleaned_tokens = _trim_trailing_loop(cleaned_tokens, max_ngram=4, min_repeat=3)
    return normalize_spaces(" ".join(cleaned_tokens))
