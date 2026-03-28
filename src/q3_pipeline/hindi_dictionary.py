from collections import defaultdict

from src.q3_pipeline.phonetic_similarity import is_phonetically_similar, phonetic_key
from src.q3_pipeline.normalizer import consonant_skeleton, normalize_word
from src.q3_pipeline.utils import edit_distance


CURATED_HINDI_WORDS = {
    "है", "हैं", "था", "थी", "थे", "हो", "हूँ", "रहा", "रही", "रहे",
    "मैं", "हम", "तुम", "आप", "वह", "यह", "ये", "वे", "वो", "जो", "तो",
    "का", "की", "के", "को", "से", "में", "पर", "और", "भी", "नहीं", "हाँ",
    "क्यों", "क्या", "कैसे", "कहाँ", "कब", "कौन", "किस", "किसी", "किसका",
    "मेरा", "मेरी", "मेरे", "हमारा", "हमारी", "हमारे", "आपका", "आपकी", "आपके",
    "उसका", "उसकी", "उसके", "उनका", "उनकी", "उनके", "इसका", "इसकी", "इसके",
    "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "दिन", "रात", "समय", "साल", "महीना", "मिनट", "घंटा", "बार", "बाद", "पहले",
    "बात", "बातें", "सवाल", "जवाब", "समस्या", "ज़रूरत", "मदद", "काम", "घर", "बाहर",
    "लोग", "आदमी", "औरत", "बच्चा", "बच्चे", "माता", "पिता", "भाई", "बहन", "दोस्त",
    "स्कूल", "कॉलेज", "पढ़ाई", "लिखाई", "किताब", "कहानी", "भाषा", "शब्द", "नाम", "जगह",
    "देश", "राज्य", "शहर", "गाँव", "सड़क", "कमरा", "दफ्तर", "बाज़ार", "समाज", "जीवन",
    "दुनिया", "वजह", "तरीका", "सोच", "विश्वास", "अनुभव", "मौका", "ज़रूरी", "सही", "गलत",
    "अच्छा", "अच्छी", "अच्छे", "बुरा", "बुरी", "बड़े", "बड़ी", "छोटा", "छोटी", "छोटे",
    "नया", "नई", "पुराना", "पुरानी", "आसान", "मुश्किल", "ज़रूरी", "खुश", "दुखी", "तेज़",
    "धीरे", "साफ", "पूरा", "खाली", "कम", "ज्यादा", "बहुत", "थोड़ा", "सिर्फ", "लगभग",
    "चलना", "आना", "जाना", "करना", "होना", "बनना", "देना", "लेना", "रखना", "देखना",
    "सुनना", "बोलना", "कहना", "पूछना", "समझना", "सोचना", "पढ़ना", "लिखना", "खाना", "पीना",
    "बैठना", "उठना", "मिलना", "सीखना", "जीना", "चाहना", "लगना", "रुकना", "खेलना", "हँसना",
    "रोना", "मिलकर", "साथ", "अंदर", "ऊपर", "नीचे", "आगे", "पीछे", "पास", "दूर",
    "फिर", "अब", "आज", "कल", "अभी", "हमेशा", "कभी", "कहीं", "वहीं", "यहीं",
    "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर",
    "नवंबर", "दिसंबर", "सुबह", "शाम", "दोपहर", "रविवार", "सोमवार", "मंगलवार", "बुधवार", "गुरुवार",
    "शुक्रवार", "शनिवार", "जी", "जीवन", "दिल", "दिमाग", "हाथ", "पैर", "आँख", "कान",
    "जैसे", "क्योंकि", "हालांकि", "संस्कृत", "लेकिन", "अगर", "मगर", "फिरभी", "इसलिए", "वजह",
}

COMMON_SUFFIXES = (
    "ों", "ें", "याँ", "यों", "ाएँ", "ाए", "ियाँ", "ियों", "ाना", "ाने", "ाती",
    "ाता", "ाते", "ती", "ता", "ते", "ना", "नी", "कर", "पन", "वाला", "वाली", "वाले",
)


class HindiDictionary:
    def __init__(self, extra_words=None):
        self.words = {normalize_word(word) for word in CURATED_HINDI_WORDS if normalize_word(word)}
        if extra_words:
            self.words.update(normalize_word(word) for word in extra_words if normalize_word(word))

        self.prefix_index = defaultdict(list)
        self.skeleton_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.length_index = defaultdict(list)

        for word in self.words:
            self.prefix_index[word[:1]].append(word)
            self.skeleton_index[consonant_skeleton(word)[:3]].append(word)
            self.phonetic_index[phonetic_key(word)].append(word)
            self.length_index[len(word)].append(word)

    def contains(self, word):
        return normalize_word(word) in self.words

    def inflected_stem(self, word):
        normalized = normalize_word(word)
        if normalized in self.words:
            return normalized

        for suffix in COMMON_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                stem = normalized[: -len(suffix)]
                if stem in self.words:
                    return stem

        return None

    def nearest_match(self, word, max_distance=2):
        normalized = normalize_word(word)
        if not normalized:
            return None

        candidates = set()
        candidates.update(self.prefix_index.get(normalized[:1], []))
        candidates.update(self.skeleton_index.get(consonant_skeleton(normalized)[:3], []))
        for length in range(max(1, len(normalized) - 2), len(normalized) + 3):
            candidates.update(self.length_index.get(length, []))

        best_candidate = None
        best_distance = max_distance + 1

        for candidate in candidates:
            if abs(len(candidate) - len(normalized)) > max_distance:
                continue

            distance = edit_distance(normalized, candidate)
            if distance < best_distance:
                best_candidate = candidate
                best_distance = distance
                if best_distance == 0:
                    break

        if best_candidate is None or best_distance > max_distance:
            return None

        return {
            "candidate": best_candidate,
            "distance": best_distance,
        }

    def phonetic_match(self, word, max_distance=1):
        normalized = normalize_word(word)
        if not normalized:
            return None

        phonetic_candidates = set(self.phonetic_index.get(phonetic_key(normalized), []))
        phonetic_candidates.update(self.prefix_index.get(normalized[:1], []))

        best_candidate = None
        best_distance = max_distance + 1
        for candidate in phonetic_candidates:
            if not is_phonetically_similar(normalized, candidate, max_distance=max_distance):
                continue

            distance = edit_distance(normalized, candidate)
            if distance < best_distance:
                best_candidate = candidate
                best_distance = distance

        if best_candidate is None:
            return None

        return {
            "candidate": best_candidate,
            "distance": best_distance,
        }
