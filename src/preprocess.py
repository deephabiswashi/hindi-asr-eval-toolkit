import re
import unicodedata
from collections import Counter

import numpy as np

from src.config import (
    MAX_AUDIO_LEN,
    MAX_CHARS_PER_SECOND,
    MAX_CLIPPING_RATIO,
    MAX_REPEATED_CHAR_RUN,
    MAX_REPEATED_TOKEN_FRACTION,
    MIN_ACTIVE_SPEECH_RATIO,
    MIN_AUDIO_LEN,
    MIN_CHARS_PER_SECOND,
    SAMPLE_RATE,
    SHORT_UTTERANCE_MAX_DURATION,
    SHORT_UTTERANCE_MAX_WORDS,
)


ALLOWED_TEXT_RE = re.compile(r"[^0-9A-Za-z\u0900-\u097F\s]")


def _content_character_count(text):
    return sum(1 for char in text if char.isalnum() or ("\u0900" <= char <= "\u097F"))


def _dominant_token_fraction(tokens):
    if not tokens:
        return 0.0

    token_counts = Counter(tokens)
    return token_counts.most_common(1)[0][1] / len(tokens)


def _longest_character_run(text):
    longest_run = 0
    current_run = 0
    previous_char = None

    for char in text:
        if char.isspace():
            previous_char = None
            current_run = 0
            continue

        if char == previous_char:
            current_run += 1
        else:
            previous_char = char
            current_run = 1

        longest_run = max(longest_run, current_run)

    return longest_run


def _audio_activity_metrics(audio_array):
    audio_array = np.asarray(audio_array, dtype=np.float32)
    if audio_array.size == 0:
        return 0.0, 0.0, 0.0

    abs_audio = np.abs(audio_array)
    peak = float(abs_audio.max())
    if peak <= 1e-6:
        return 0.0, 0.0, peak

    energy_floor = float(np.quantile(abs_audio, 0.60))
    activity_threshold = max(0.008, energy_floor * 2.5, peak * 0.08)
    active_ratio = float(np.mean(abs_audio >= activity_threshold))
    clipping_ratio = float(np.mean(abs_audio >= 0.995))
    return active_ratio, clipping_ratio, peak


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\u200c", " ").replace("\u200d", " ").replace("\ufeff", " ")
    text = text.replace("\u0964", " ").replace("\u0965", " ").replace("|", " ")

    # Keep Hindi, Latin, and digits; strip punctuation and stray symbols.
    text = ALLOWED_TEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_sample(sample):
    audio_array = np.asarray(sample.get("audio_array", []), dtype=np.float32)
    sampling_rate = int(sample.get("sampling_rate", SAMPLE_RATE))

    if audio_array.size == 0:
        return False, "empty_audio"

    duration_seconds = audio_array.size / max(sampling_rate, 1)
    if duration_seconds < MIN_AUDIO_LEN:
        return False, "audio_too_short"

    if duration_seconds > MAX_AUDIO_LEN:
        return False, "audio_too_long"

    normalized_text = normalize_text(sample.get("text", ""))
    if len(normalized_text) == 0:
        return False, "empty_text"

    tokens = normalized_text.split()
    content_characters = _content_character_count(normalized_text)
    if content_characters == 0:
        return False, "no_content_characters"

    if _longest_character_run(normalized_text) > MAX_REPEATED_CHAR_RUN:
        return False, "repeated_character_noise"

    if len(tokens) >= 4 and _dominant_token_fraction(tokens) > MAX_REPEATED_TOKEN_FRACTION:
        return False, "repeated_token_noise"

    chars_per_second = content_characters / max(duration_seconds, 1e-6)
    short_utterance = (
        duration_seconds <= SHORT_UTTERANCE_MAX_DURATION
        and len(tokens) <= SHORT_UTTERANCE_MAX_WORDS
    )

    if chars_per_second < MIN_CHARS_PER_SECOND and not short_utterance:
        return False, "transcript_too_sparse"

    if chars_per_second > MAX_CHARS_PER_SECOND:
        return False, "transcript_too_dense"

    active_ratio, clipping_ratio, peak = _audio_activity_metrics(audio_array)
    if peak < 0.003:
        return False, "near_silent_audio"

    if clipping_ratio > MAX_CLIPPING_RATIO:
        return False, "clipped_audio"

    if active_ratio < MIN_ACTIVE_SPEECH_RATIO and not short_utterance:
        return False, "low_speech_activity"

    filtered_sample = dict(sample)
    filtered_sample["audio_array"] = audio_array
    filtered_sample["sampling_rate"] = sampling_rate
    filtered_sample["text"] = normalized_text
    return True, filtered_sample


def preprocess(samples, return_stats=False):
    processed = []
    dropped_reasons = Counter()

    for sample in samples:
        keep_sample, result = filter_sample(sample)
        if not keep_sample:
            dropped_reasons[result] += 1
            continue

        processed.append(result)

    if return_stats:
        return processed, {
            "kept": len(processed),
            "dropped": sum(dropped_reasons.values()),
            "drop_reasons": dict(dropped_reasons),
        }

    return processed
