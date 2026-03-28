import re

from src.q3_pipeline.normalizer import normalize_word


ROMAN_WORD_RE = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")

KNOWN_DEVANAGARI_ENGLISH = {
    "कंप्यूटर", "कम्प्यूटर", "इंटरव्यू", "मैनेजर", "मैनेजमेंट", "फाइल", "फॉर्म", "ऑफिस",
    "फोन", "मोबाइल", "लैपटॉप", "प्रोजेक्ट", "सिस्टम", "डेटा", "कोड", "लॉगिन", "पासवर्ड",
    "जॉब", "टीम", "मीटिंग", "ड्राइवर", "ट्रेनर", "क्लास", "कोर्स", "मार्केट", "बैंक",
    "डॉक्टर", "पुलिस", "मैसेज", "मेल", "ईमेल", "वीडियो", "ऑडियो", "सॉफ्टवेयर", "हार्डवेयर",
    "नेटवर्क", "फीडबैक", "सेल्स", "मार्केटिंग", "प्रोडक्ट", "सर्विस", "डिजाइन", "मैप", "टिकट",
}

DEVANAGARI_HINTS = (
    "इंटर", "कंप", "कम्प", "मैने", "प्रोजे", "सिस्टम", "डेटा", "कोड", "लॉग", "पास",
    "जॉब", "मीट", "फाइल", "फॉर्म", "ऑफ", "फोन", "मोबा", "लैप", "ड्राइ", "ट्रे",
    "क्ला", "कोर्स", "बैंक", "डॉक", "मेल", "वीडि", "ऑडि", "सॉफ्ट", "हार्ड", "नेट",
)

FOREIGN_CLUSTER_HINTS = ("ट्र", "ड्र", "फ्र", "ग्र", "क्ल", "प्ल", "स्क", "स्प", "स्ट", "कॉ", "ऑ")


class EnglishDetector:
    def __init__(self, hindi_dictionary):
        self.hindi_dictionary = hindi_dictionary

    def detect(self, word):
        normalized = normalize_word(word)
        if not normalized:
            return {
                "is_english": False,
                "strength": None,
                "reason": None,
            }

        if ROMAN_WORD_RE.match(word):
            return {
                "is_english": True,
                "strength": "strong",
                "reason": "Roman-script English token.",
            }

        if normalized in KNOWN_DEVANAGARI_ENGLISH:
            return {
                "is_english": True,
                "strength": "strong",
                "reason": "Known Devanagari rendering of an English borrowing.",
            }

        if self.hindi_dictionary.contains(normalized):
            return {
                "is_english": False,
                "strength": None,
                "reason": None,
            }

        if any(hint in normalized for hint in DEVANAGARI_HINTS) or any(cluster in normalized for cluster in FOREIGN_CLUSTER_HINTS):
            return {
                "is_english": True,
                "strength": "medium",
                "reason": "Looks like a Devanagari spelling of an English borrowed word.",
            }

        return {
            "is_english": False,
            "strength": None,
            "reason": None,
        }
