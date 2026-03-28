import re
import unicodedata


MOJIBAKE_MARKERS = ("\u00e0\u00a4", "\u00e0\u00a5", "\u00ef\u00bf\u00bd")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z\u0900-\u097F]+")
LATIN_RE = re.compile(r"[A-Za-z]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
DIGIT_RE = re.compile(r"\d")
DEPENDENT_MARKS = "\u093a\u093b\u093c\u093e-\u094d\u094e\u094f\u0955-\u0957\u0962\u0963"
LEADING_DEPENDENT_RE = re.compile(rf"^[{DEPENDENT_MARKS}]")
TRAILING_HALANT_RE = re.compile(r"\u094d$")
DOUBLE_HALANT_RE = re.compile(r"\u094d{2,}")
CONSECUTIVE_MATRA_RE = re.compile(r"[\u093e-\u094c]{2,}")
REPEATED_MARK_RE = re.compile(r"([\u093a-\u094d\u094e-\u0957\u0962\u0963])\1+")
EXCESSIVE_REPEAT_RE = re.compile(r"(.)\1{2,}")
ONLY_MARKS_RE = re.compile(rf"^[{DEPENDENT_MARKS}]+$")
NUKTA_RE = re.compile(r"\u093c")
STRIP_MARKS_RE = re.compile(rf"[{DEPENDENT_MARKS}]")

HARD_INVALID_ISSUES = {
    "empty_after_normalization",
    "only_digits",
    "no_letters",
    "leading_dependent_mark",
    "double_halant",
    "consecutive_matras",
    "repeated_diacritic",
    "repeated_character_noise",
    "marks_only_token",
    "mixed_script_noise",
}


def repair_mojibake(text):
    text = str(text or "")
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def normalize_word(text):
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.strip()
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    text = NON_WORD_RE.sub("", text)
    return text.lower()


def script_profile(word):
    return {
        "has_devanagari": bool(DEVANAGARI_RE.search(word)),
        "has_latin": bool(LATIN_RE.search(word)),
        "has_digits": bool(DIGIT_RE.search(word)),
    }


def is_devanagari_token(word):
    profile = script_profile(word)
    return profile["has_devanagari"] and not profile["has_latin"] and not profile["has_digits"]


def is_roman_token(word):
    profile = script_profile(word)
    return profile["has_latin"] and not profile["has_devanagari"]


def consonant_skeleton(word):
    return STRIP_MARKS_RE.sub("", word)


def analyze_orthography(word):
    issues = []
    normalized = normalize_word(word)
    if not normalized:
        return ["empty_after_normalization"]

    profile = script_profile(normalized)
    if profile["has_digits"] and not profile["has_devanagari"] and not profile["has_latin"]:
        issues.append("only_digits")

    if profile["has_devanagari"] and profile["has_latin"]:
        issues.append("mixed_script_noise")

    if not profile["has_devanagari"] and not profile["has_latin"]:
        issues.append("no_letters")

    if ONLY_MARKS_RE.match(normalized):
        issues.append("marks_only_token")

    if LEADING_DEPENDENT_RE.search(normalized):
        issues.append("leading_dependent_mark")

    if TRAILING_HALANT_RE.search(normalized):
        issues.append("trailing_halant")

    if DOUBLE_HALANT_RE.search(normalized):
        issues.append("double_halant")

    if CONSECUTIVE_MATRA_RE.search(normalized):
        issues.append("consecutive_matras")

    if REPEATED_MARK_RE.search(normalized):
        issues.append("repeated_diacritic")

    if EXCESSIVE_REPEAT_RE.search(normalized):
        issues.append("repeated_character_noise")

    if normalized.count("\u093c") > 1 or NUKTA_RE.match(normalized):
        issues.append("suspicious_nukta_usage")

    return issues


def is_plausible_hindi_word(word):
    normalized = normalize_word(word)
    if not normalized:
        return False

    if not is_devanagari_token(normalized):
        return False

    issues = set(analyze_orthography(normalized))
    return not (issues & HARD_INVALID_ISSUES)

