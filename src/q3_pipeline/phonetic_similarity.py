from src.q3_pipeline.normalizer import normalize_word
from src.q3_pipeline.utils import edit_distance


PHONETIC_MAP = {
    "क": "K", "ख": "K", "ग": "G", "घ": "G",
    "च": "C", "छ": "C", "ज": "J", "झ": "J",
    "ट": "T", "ठ": "T", "ड": "D", "ढ": "D",
    "त": "T", "थ": "T", "द": "D", "ध": "D",
    "प": "P", "फ": "P", "ब": "B", "भ": "B",
    "य": "Y", "र": "R", "ल": "L", "व": "V",
    "श": "S", "ष": "S", "स": "S", "ह": "H",
    "म": "M", "न": "N", "ण": "N", "ञ": "N", "ङ": "N",
    "ँ": "N", "ं": "N",
    "अ": "A", "आ": "A", "इ": "I", "ई": "I", "उ": "U", "ऊ": "U",
    "ए": "E", "ऐ": "E", "ओ": "O", "औ": "O", "ऋ": "R",
}

STRIP_SIGNS = set("ािीुूृेैोौंः़्")


def phonetic_key(word):
    normalized = normalize_word(word)
    key = []
    for char in normalized:
        if char in STRIP_SIGNS:
            continue
        key.append(PHONETIC_MAP.get(char, char))
    return "".join(key)


def phonetic_distance(left, right):
    return edit_distance(phonetic_key(left), phonetic_key(right))


def is_phonetically_similar(left, right, max_distance=1):
    left_key = phonetic_key(left)
    right_key = phonetic_key(right)
    if not left_key or not right_key:
        return False
    return edit_distance(left_key, right_key) <= max_distance

