import logging
import re
import unicodedata


logger = logging.getLogger(__name__)


NUMBER_WORDS = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "छह": 6,
    "छः": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "पन्द्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "अट्ठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "इक्कीस": 21,
    "बाईस": 22,
    "बाइस": 22,
    "तेईस": 23,
    "तेइस": 23,
    "चौबीस": 24,
    "पच्चीस": 25,
    "छब्बीस": 26,
    "सत्ताईस": 27,
    "अट्ठाईस": 28,
    "उनतीस": 29,
    "उन्तीस": 29,
    "तीस": 30,
    "इकतीस": 31,
    "बत्तीस": 32,
    "तैंतीस": 33,
    "तेतीस": 33,
    "चौंतीस": 34,
    "चौतीस": 34,
    "पैंतीस": 35,
    "छत्तीस": 36,
    "सैंतीस": 37,
    "अड़तीस": 38,
    "अडतीस": 38,
    "उनतालीस": 39,
    "चालीस": 40,
    "इकतालीस": 41,
    "बयालीस": 42,
    "तैंतालीस": 43,
    "तैतालीस": 43,
    "चवालीस": 44,
    "पैंतालीस": 45,
    "छियालीस": 46,
    "सैंतालीस": 47,
    "अड़तालीस": 48,
    "अडतालीस": 48,
    "उनचास": 49,
    "पचास": 50,
    "इक्यावन": 51,
    "बावन": 52,
    "तिरेपन": 53,
    "चौवन": 54,
    "पचपन": 55,
    "छप्पन": 56,
    "सत्तावन": 57,
    "अट्ठावन": 58,
    "उनसठ": 59,
    "साठ": 60,
    "इकसठ": 61,
    "बासठ": 62,
    "तिरसठ": 63,
    "चौंसठ": 64,
    "चौसठ": 64,
    "पैंसठ": 65,
    "छियासठ": 66,
    "सड़सठ": 67,
    "सड़सठ": 67,
    "अड़सठ": 68,
    "अडसठ": 68,
    "उनहत्तर": 69,
    "सत्तर": 70,
    "इकहत्तर": 71,
    "बहत्तर": 72,
    "तिहत्तर": 73,
    "चौहत्तर": 74,
    "पचहत्तर": 75,
    "छिहत्तर": 76,
    "सतहत्तर": 77,
    "अठहत्तर": 78,
    "उन्यासी": 79,
    "अस्सी": 80,
    "इक्यासी": 81,
    "बयासी": 82,
    "तिरासी": 83,
    "चौरासी": 84,
    "पचासी": 85,
    "छियासी": 86,
    "सत्तासी": 87,
    "अट्ठासी": 88,
    "नवासी": 89,
    "नब्बे": 90,
    "इक्यानवे": 91,
    "बानवे": 92,
    "तिरानवे": 93,
    "चौरानवे": 94,
    "पचानवे": 95,
    "छियानवे": 96,
    "सत्तानवे": 97,
    "अट्ठानवे": 98,
    "निन्यानवे": 99,
}

SCALE_WORDS = {
    "सौ": 100,
    "हजार": 1000,
    "हज़ार": 1000,
}

LEADING_PUNCTUATION_RE = re.compile(r'^[\(\[\{"\'“‘]+')
TRAILING_PUNCTUATION_RE = re.compile(r'[\)\]\}"\'”’।,!?;:]+$')

IDIOMATIC_PATTERNS = {
    ("एक", "दो"),
    ("दो", "चार"),
}

AMBIGUOUS_STANDALONE_UNITS = {word for word, value in NUMBER_WORDS.items() if 0 <= value <= 9}


def _normalize_token(token):
    token = unicodedata.normalize("NFKC", str(token or ""))
    token = token.replace("ज़", "ज").replace("ड़", "ड").replace("ढ़", "ढ")
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


def _is_number_word(word):
    normalized = _normalize_token(word)
    return normalized in NUMBER_WORDS or normalized in SCALE_WORDS


def _should_skip_sequence(sequence_words, next_word):
    if not sequence_words:
        return True

    if any("-" in word for word in sequence_words):
        return True

    normalized_sequence = tuple(_normalize_token(word) for word in sequence_words)
    if len(normalized_sequence) >= 2 and normalized_sequence[:2] in IDIOMATIC_PATTERNS:
        return True

    if len(normalized_sequence) == 1 and normalized_sequence[0] in AMBIGUOUS_STANDALONE_UNITS:
        return True

    if len(normalized_sequence) == 2 and normalized_sequence in IDIOMATIC_PATTERNS and next_word:
        return True

    return False


def _parse_number_words(sequence_words):
    total = 0
    current = 0

    for raw_word in sequence_words:
        word = _normalize_token(raw_word)
        if word in SCALE_WORDS:
            scale = SCALE_WORDS[word]
            if scale == 100:
                current = (current or 1) * scale
            else:
                total += (current or 1) * scale
                current = 0
            continue

        current += NUMBER_WORDS[word]

    return total + current


def normalize_numbers(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    if not text.strip():
        return text

    tokens = text.split()
    normalized_tokens = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        leading, core, trailing = _split_affixes(token)
        normalized_core = _normalize_token(core)

        if not core or not _is_number_word(normalized_core):
            normalized_tokens.append(token)
            index += 1
            continue

        sequence_tokens = []
        sequence_words = []
        cursor = index

        while cursor < len(tokens):
            current_token = tokens[cursor]
            current_leading, current_core, current_trailing = _split_affixes(current_token)
            current_normalized_core = _normalize_token(current_core)

            if not current_core or current_leading or current_trailing and cursor != index:
                # Preserve punctuation-boundary safety for downstream reconstruction.
                if cursor != index and current_trailing and _is_number_word(current_normalized_core):
                    sequence_tokens.append((current_leading, current_core, current_trailing))
                    sequence_words.append(current_core)
                    cursor += 1
                break

            if not _is_number_word(current_normalized_core):
                break

            sequence_tokens.append((current_leading, current_core, current_trailing))
            sequence_words.append(current_core)
            cursor += 1

        next_word = ""
        if cursor < len(tokens):
            _, next_core, _ = _split_affixes(tokens[cursor])
            next_word = _normalize_token(next_core)

        if _should_skip_sequence(sequence_words, next_word):
            normalized_tokens.extend(tokens[index:cursor])
            index = cursor
            continue

        numeric_value = _parse_number_words(sequence_words)
        first_leading = sequence_tokens[0][0]
        last_trailing = sequence_tokens[-1][2]
        normalized_tokens.append(f"{first_leading}{numeric_value}{last_trailing}")
        logger.debug("Converted Hindi number phrase '%s' -> %s", " ".join(sequence_words), numeric_value)
        index = cursor

    return " ".join(normalized_tokens)


def edge_case_examples():
    examples = [
        {
            "text": "दो-चार बातें करनी हैं",
            "decision": normalize_numbers("दो-चार बातें करनी हैं"),
            "reason": "Skipped because hyphenated expressions are treated as idiomatic ranges, not literal numeric quantities.",
        },
        {
            "text": "एक दो दिन और रुकते हैं",
            "decision": normalize_numbers("एक दो दिन और रुकते हैं"),
            "reason": "Skipped because 'एक दो' is a common approximate-duration phrase and converting it would change the meaning.",
        },
        {
            "text": "उसने एक सवाल पूछा",
            "decision": normalize_numbers("उसने एक सवाल पूछा"),
            "reason": "Skipped because a standalone unit word such as 'एक' is often a determiner rather than an explicit numeric entity.",
        },
    ]
    return examples
