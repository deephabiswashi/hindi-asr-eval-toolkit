from functools import lru_cache

from src.q3_pipeline.confidence_scorer import build_reason, score_confidence
from src.q3_pipeline.english_detector import EnglishDetector
from src.q3_pipeline.hindi_dictionary import HindiDictionary
from src.q3_pipeline.matra_rules import has_matra_error
from src.q3_pipeline.normalizer import HARD_INVALID_ISSUES, analyze_orthography, is_plausible_hindi_word, normalize_word
from src.q3_pipeline.phonetic_similarity import is_phonetically_similar


class SpellClassifier:
    def __init__(self, hindi_dictionary=None, english_detector=None):
        self.hindi_dictionary = hindi_dictionary or HindiDictionary()
        self.english_detector = english_detector or EnglishDetector(self.hindi_dictionary)

    @lru_cache(maxsize=250000)
    def classify_normalized(self, normalized_word):
        issues = analyze_orthography(normalized_word)
        english_info = self.english_detector.detect(normalized_word)

        if not normalized_word:
            return self._finalize(
                normalized_word,
                label="incorrect_spelling",
                source="invalid_pattern",
                issues=["empty_after_normalization"],
            )

        if english_info["is_english"]:
            source = "roman_english" if english_info["strength"] == "strong" and normalized_word.isascii() else (
                "devanagari_english_strong" if english_info["strength"] == "strong" else "devanagari_english_medium"
            )
            return self._finalize(
                normalized_word,
                label="correct_spelling",
                source=source,
                issues=issues,
            )

        if self.hindi_dictionary.contains(normalized_word):
            return self._finalize(
                normalized_word,
                label="correct_spelling",
                source="exact_dictionary",
                issues=issues,
            )

        stem = self.hindi_dictionary.inflected_stem(normalized_word)
        if stem:
            return self._finalize(
                normalized_word,
                label="correct_spelling",
                source="inflected_dictionary",
                issues=issues,
                stem=stem,
            )

        suggestion = self.hindi_dictionary.nearest_match(normalized_word, max_distance=2)
        phonetic_suggestion = self.hindi_dictionary.phonetic_match(normalized_word, max_distance=1)
        hard_invalid = bool(set(issues) & HARD_INVALID_ISSUES)

        if hard_invalid:
            return self._finalize(
                normalized_word,
                label="incorrect_spelling",
                source="invalid_pattern",
                issues=issues,
                suggestion=suggestion,
            )

        if suggestion and has_matra_error(normalized_word, suggestion["candidate"]):
            return self._finalize(
                normalized_word,
                label="incorrect_spelling",
                source="matra_match",
                issues=issues,
                suggestion=suggestion,
            )

        if suggestion and suggestion["distance"] <= 2:
            return self._finalize(
                normalized_word,
                label="incorrect_spelling",
                source="near_dictionary_match",
                issues=issues,
                suggestion=suggestion,
            )

        if phonetic_suggestion and is_phonetically_similar(normalized_word, phonetic_suggestion["candidate"], max_distance=1):
            return self._finalize(
                normalized_word,
                label="incorrect_spelling",
                source="phonetic_match",
                issues=issues,
                suggestion=phonetic_suggestion,
            )

        if is_plausible_hindi_word(normalized_word):
            return self._finalize(
                normalized_word,
                label="correct_spelling",
                source="plausible_unknown",
                issues=issues,
                suggestion=suggestion,
            )

        return self._finalize(
            normalized_word,
            label="incorrect_spelling",
            source="unknown_suspicious",
            issues=issues,
            suggestion=suggestion,
        )

    def classify(self, word, normalized_word=None):
        normalized_word = normalized_word if normalized_word is not None else normalize_word(word)
        decision = dict(self.classify_normalized(normalized_word))
        if self._has_raw_noise(word):
            decision.update(
                self._finalize(
                    normalized_word,
                    label="incorrect_spelling",
                    source="invalid_pattern",
                    issues=["punctuation_contamination"],
                )
            )
        decision["word"] = word
        return decision

    def _finalize(self, normalized_word, label, source, issues=None, suggestion=None, stem=None):
        signals = {
            "source": source,
            "issues": issues or [],
            "suggestion": suggestion,
            "stem": stem,
        }
        return {
            "normalized_word": normalized_word,
            "label": label,
            "confidence": score_confidence(signals),
            "reason": build_reason(signals),
            "source": source,
            "suggestion": suggestion["candidate"] if suggestion else None,
        }

    @staticmethod
    def _has_raw_noise(word):
        raw_word = str(word or "")
        return any(char in raw_word for char in {".", ",", ";", ":", "!", "?", "/", "\\", "|", "*", "_"})
