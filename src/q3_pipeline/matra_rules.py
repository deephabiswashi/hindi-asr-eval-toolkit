from src.q3_pipeline.normalizer import consonant_skeleton, normalize_word


MATRAS = set("ािीुूृेैोौंँः")


def matra_signature(word):
    normalized = normalize_word(word)
    return "".join(char for char in normalized if char in MATRAS)


def has_matra_error(word, candidate):
    word = normalize_word(word)
    candidate = normalize_word(candidate)
    if not word or not candidate or word == candidate:
        return False

    if consonant_skeleton(word) != consonant_skeleton(candidate):
        return False

    return matra_signature(word) != matra_signature(candidate)


def matra_reason(word, candidate):
    if has_matra_error(word, candidate):
        return f"Likely matra/diacritic error relative to '{candidate}'."
    return None
