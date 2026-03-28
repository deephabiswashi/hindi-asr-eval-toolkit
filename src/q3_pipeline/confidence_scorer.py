def score_confidence(signals):
    source = signals["source"]
    suggestion = signals.get("suggestion")

    if source in {"exact_dictionary", "roman_english", "devanagari_english_strong"}:
        return "high"

    if source in {"inflected_dictionary", "devanagari_english_medium", "phonetic_match", "matra_match"}:
        return "medium"

    if source == "near_dictionary_match" and suggestion and suggestion.get("distance", 99) <= 1:
        return "medium"

    return "low"


def build_reason(signals):
    source = signals["source"]
    issues = signals.get("issues", [])
    suggestion = signals.get("suggestion")

    if source == "exact_dictionary":
        return "Exact match in curated Hindi lexicon."

    if source == "inflected_dictionary":
        return f"Matches a common Hindi stem: {signals['stem']}."

    if source == "roman_english":
        return "Roman-script English token, treated as correctly spelled."

    if source == "devanagari_english_strong":
        return "Recognized Devanagari spelling of an English borrowing."

    if source == "devanagari_english_medium":
        return "Likely Devanagari form of an English word."

    if source == "near_dictionary_match" and suggestion:
        return f"Edit distance match with common Hindi word '{suggestion['candidate']}'."

    if source == "phonetic_match" and suggestion:
        return f"Phonetically similar to common Hindi word '{suggestion['candidate']}'."

    if source == "matra_match" and suggestion:
        return f"Likely matra/diacritic variant of '{suggestion['candidate']}'."

    if source == "invalid_pattern":
        issue_text = ", ".join(issues[:2]) if issues else "invalid orthographic pattern"
        return f"Suspicious Hindi character pattern: {issue_text}."

    if source == "plausible_unknown":
        return "Ambiguous form: plausible Hindi spelling but outside curated lexicon."

    issue_text = ", ".join(issues[:2]) if issues else "unknown form"
    return f"Unknown token with ambiguous or noisy spelling: {issue_text}."
