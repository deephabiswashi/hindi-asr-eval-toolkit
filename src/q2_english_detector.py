import csv
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from src.postprocess import repair_mojibake


logger = logging.getLogger(__name__)


ROMAN_WORD_RE = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
LEADING_PUNCTUATION_RE = re.compile(r'^[\(\[\{"\'“‘]+')
TRAILING_PUNCTUATION_RE = re.compile(r'[\)\]\}"\'”’।,!?;:]+$')

BASE_HINDI_WORDS = {
    "है",
    "हैं",
    "और",
    "में",
    "का",
    "की",
    "के",
    "को",
    "से",
    "पर",
    "यह",
    "वह",
    "मैं",
    "हम",
    "आप",
    "कि",
    "तो",
    "भी",
    "नहीं",
    "एक",
    "दो",
    "तीन",
    "दिन",
    "बात",
    "बातें",
    "सवाल",
}

KNOWN_DEVANAGARI_ENGLISH = {
    "इंटरव्यू",
    "जॉब",
    "कंप्यूटर",
    "कम्प्यूटर",
    "फाइल",
    "फाइलें",
    "प्रॉब्लम",
    "ऑफिस",
    "फोन",
    "लैपटॉप",
    "ईमेल",
    "मेल",
    "मीटिंग",
    "मैनेजर",
    "टीम",
    "डेटा",
    "सिस्टम",
    "प्रोजेक्ट",
    "कोड",
    "लॉगिन",
    "पासवर्ड",
    "सॉफ्टवेयर",
    "हार्डवेयर",
    "क्लास",
    "फॉर्म",
    "नेटवर्क",
}

DEVANAGARI_ENGLISH_HINTS = (
    "इंटर",
    "जॉब",
    "कंप",
    "कम्प",
    "फाइल",
    "प्रॉब्ल",
    "ऑफ",
    "फोन",
    "लैप",
    "मेल",
    "मीट",
    "मैनेज",
    "टीम",
    "डेटा",
    "सिस्टम",
    "प्रोजेक्ट",
    "कोड",
    "लॉग",
    "पास",
    "सॉफ्ट",
    "हार्ड",
    "फॉर्म",
    "नेट",
    "ट्र",
    "ड्र",
    "कॉ",
    "ऑ",
)


def _normalize_token(token):
    token = unicodedata.normalize("NFKC", str(token or ""))
    token = repair_mojibake(token)
    token = token.replace("़", "")
    return token


def _split_affixes(token):
    leading_match = LEADING_PUNCTUATION_RE.match(token)
    trailing_match = TRAILING_PUNCTUATION_RE.search(token)

    leading = leading_match.group(0) if leading_match else ""
    trailing = trailing_match.group(0) if trailing_match else ""

    core_start = len(leading)
    core_end = len(token) - len(trailing) if trailing else len(token)
    core = token[core_start:core_end]
    return leading, core, trailing


@lru_cache(maxsize=1)
def load_hindi_lexicon():
    lexicon = set(BASE_HINDI_WORDS)
    candidate_path = Path("Unique Words Data - Sheet1.csv")
    if not candidate_path.exists():
        return lexicon

    try:
        with candidate_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                word = repair_mojibake(row.get("word", "")).strip()
                if word:
                    lexicon.add(_normalize_token(word))
    except Exception as exc:
        logger.warning("Unable to load Hindi lexicon from %s: %s", candidate_path, exc)

    return lexicon


def _is_devanagari_english(core_token):
    token = _normalize_token(core_token)
    if not token:
        return False

    if token in KNOWN_DEVANAGARI_ENGLISH:
        return True

    if token in load_hindi_lexicon():
        return False

    if any(hint in token for hint in DEVANAGARI_ENGLISH_HINTS):
        return True

    return False


def _tag_core_token(core_token):
    if core_token.startswith("[EN]") and core_token.endswith("[/EN]"):
        return core_token

    normalized = _normalize_token(core_token)
    if ROMAN_WORD_RE.match(core_token):
        return f"[EN]{core_token}[/EN]"

    if _is_devanagari_english(normalized):
        return f"[EN]{core_token}[/EN]"

    return core_token


def tag_english_words(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    if not text.strip():
        return text

    tagged_tokens = []
    for token in text.split():
        leading, core, trailing = _split_affixes(token)
        if not core:
            tagged_tokens.append(token)
            continue

        tagged_core = _tag_core_token(core)
        tagged_tokens.append(f"{leading}{tagged_core}{trailing}")

    return " ".join(tagged_tokens)
