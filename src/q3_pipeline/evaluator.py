import argparse
from collections import defaultdict
from pathlib import Path
import re

from src.q3_pipeline.utils import load_json, save_json, setup_logging


def parse_manual_label(raw_value):
    value = str(raw_value or "").strip().lower()
    if value in {"y", "yes", "correct", "correct_spelling", "c"}:
        return "correct_spelling"
    if value in {"n", "no", "incorrect", "incorrect_spelling", "i"}:
        return "incorrect_spelling"
    return None


def parse_manual_response(raw_value):
    value = str(raw_value or "").strip()
    if not value:
        return None, "", True

    match = re.match(r"^\s*([ynsYNScCiI])(?:[\s:,\-]+(.*))?$", value)
    if not match:
        return None, "", False

    label_token = match.group(1).lower()
    reviewer_note = (match.group(2) or "").strip()

    if label_token == "s":
        return None, reviewer_note, True

    manual_label = parse_manual_label(label_token)
    return manual_label, reviewer_note, False


def interactive_annotations(samples, max_annotations=50):
    annotations = []
    skipped = []
    for sample in samples:
        if len(annotations) >= max_annotations:
            break

        print(f"\nReview progress: {len(annotations) + 1}/{max_annotations}")
        print(f"\nWord: {sample['word']}")
        print(f"Predicted label: {sample['label']}")
        print(f"Confidence: {sample['confidence']}")
        print(f"Reason: {sample['reason']}")
        answer = input("Manual label [y=correct / n=incorrect / s=skip] + optional note: ").strip()
        manual_label, reviewer_note, is_skip = parse_manual_response(answer)

        if is_skip:
            skipped.append(
                {
                    "word": sample["word"],
                    "predicted_label": sample["label"],
                    "confidence": sample["confidence"],
                    "reason": sample["reason"],
                    "reviewer_note": reviewer_note,
                }
            )
            continue

        if manual_label is None:
            print("Skipping due to invalid input.")
            continue

        annotations.append(
            {
                "review_index": len(annotations) + 1,
                "word": sample["word"],
                "predicted_label": sample["label"],
                "manual_label": manual_label,
                "confidence": sample["confidence"],
                "reason": sample["reason"],
                "reviewer_note": reviewer_note,
                "source": sample.get("source"),
                "suggestion": sample.get("suggestion"),
            }
        )
    return annotations, skipped


def evaluate_annotations(annotations):
    total = len(annotations)
    correct = sum(1 for row in annotations if row["predicted_label"] == row["manual_label"])
    incorrect = total - correct
    predicted_correct = sum(1 for row in annotations if row["predicted_label"] == "correct_spelling")
    predicted_incorrect = total - predicted_correct
    actual_correct = sum(1 for row in annotations if row["manual_label"] == "correct_spelling")
    actual_incorrect = total - actual_correct

    summary = {
        "annotated_samples": total,
        "system_right": correct,
        "system_wrong": incorrect,
        "low_confidence_accuracy": round(correct / total, 4) if total else None,
        "predicted_correct_count": predicted_correct,
        "predicted_incorrect_count": predicted_incorrect,
        "manual_correct_count": actual_correct,
        "manual_incorrect_count": actual_incorrect,
    }
    summary["q3c_takeaway"] = build_q3c_takeaway(summary)
    return summary


def build_q3c_takeaway(summary):
    if not summary["annotated_samples"]:
        return "No low-confidence words were reviewed, so the breakdown point cannot be estimated yet."

    accuracy = summary["low_confidence_accuracy"]
    if accuracy is None:
        return "No low-confidence words were reviewed, so the breakdown point cannot be estimated yet."
    if accuracy >= 0.75:
        return (
            "The low-confidence bucket is still reasonably useful, but the system becomes less reliable on borrowed words, "
            "rare valid Hindi forms, and punctuation-contaminated tokens."
        )
    if accuracy >= 0.5:
        return (
            "The low-confidence bucket captures many genuinely ambiguous cases. The heuristic system starts to break down on "
            "tokenization noise, borrowed English words in Devanagari, and rare but valid vocabulary outside the curated lexicon."
        )
    return (
        "The low-confidence bucket is highly unreliable, which means the heuristic rules are overfitting to dictionary proximity "
        "and orthographic patterns instead of robustly handling real conversational vocabulary."
    )


def categorize_failure(annotation):
    source = annotation.get("source") or ""
    reason = (annotation.get("reason") or "").lower()
    reviewer_note = (annotation.get("reviewer_note") or "").lower()
    word = annotation.get("word") or ""

    if any(symbol in word for symbol in [",", ".", ";", ":", "|"]):
        return "punctuation_or_tokenization_noise"

    if "punctuation" in reason or "token" in reviewer_note:
        return "punctuation_or_tokenization_noise"

    if "english" in reason or source in {"roman_english", "devanagari_english_strong", "devanagari_english_medium"}:
        return "code_mixed_or_borrowed_words"

    if annotation.get("manual_label") == "correct_spelling" and annotation.get("predicted_label") == "incorrect_spelling":
        if "edit distance" in reason or "match" in reason:
            return "over_aggressive_near_match_rules"
        return "rare_valid_words_outside_lexicon"

    if annotation.get("manual_label") == "incorrect_spelling" and annotation.get("predicted_label") == "correct_spelling":
        return "plausible_but_wrong_forms"

    if source == "matra_match" or "matra" in reason or "diacritic" in reason:
        return "matra_or_orthographic_variation"

    return "other_ambiguous_forms"


def category_explanation(category):
    explanations = {
        "punctuation_or_tokenization_noise": "Multiple valid words are merged with punctuation, so word-level validation sees a noisy token instead of clean lexical items.",
        "code_mixed_or_borrowed_words": "Borrowed English words written in Devanagari often look rare or non-standard, so dictionary-only rules underperform.",
        "over_aggressive_near_match_rules": "Nearest-word matching can force a rare valid word toward a common but wrong dictionary neighbor.",
        "rare_valid_words_outside_lexicon": "Conversational datasets contain names, regional forms, and rare valid words that are not covered by a compact curated lexicon.",
        "plausible_but_wrong_forms": "Some misspellings still look orthographically plausible, so they slip through rule-based checks as false positives.",
        "matra_or_orthographic_variation": "Matra variation, optional orthographic alternants, and colloquial spellings can confuse deterministic spelling rules.",
        "other_ambiguous_forms": "These are genuinely ambiguous forms where context would help more than isolated word-level heuristics.",
    }
    return explanations[category]


def build_failure_analysis(annotations):
    wrong_predictions = [row for row in annotations if row["predicted_label"] != row["manual_label"]]
    grouped = defaultdict(list)
    for row in wrong_predictions:
        grouped[categorize_failure(row)].append(
            {
                "word": row["word"],
                "predicted_label": row["predicted_label"],
                "manual_label": row["manual_label"],
                "reason": row.get("reason"),
                "source": row.get("source"),
                "suggestion": row.get("suggestion"),
                "reviewer_note": row.get("reviewer_note"),
            }
        )

    ordered_categories = sorted(grouped.keys(), key=lambda key: len(grouped[key]), reverse=True)

    q3d_unreliable_categories = [
        {
            "category": category,
            "count": len(grouped[category]),
            "why_unreliable": category_explanation(category),
            "example_words": [item["word"] for item in grouped[category][:5]],
        }
        for category in ordered_categories[:2]
    ]

    return {
        "total_failures": len(wrong_predictions),
        "category_counts": {category: len(grouped.get(category, [])) for category in ordered_categories},
        "categories": {
            category: grouped.get(category, [])[:10]
            for category in ordered_categories
        },
        "q3d_summary": q3d_unreliable_categories,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate low-confidence Q3 samples with manual annotations.")
    parser.add_argument(
        "--samples",
        default="outputs/q3_low_confidence_samples.json",
        help="Path to the saved low-confidence sample JSON.",
    )
    parser.add_argument(
        "--annotations-output",
        default="outputs/q3_low_confidence_annotations.json",
        help="Where to save manual annotations collected in interactive mode.",
    )
    parser.add_argument(
        "--evaluation-output",
        default="outputs/q3_evaluation.json",
        help="Where to save the low-confidence evaluation summary.",
    )
    parser.add_argument(
        "--failure-analysis-output",
        default="outputs/q3_failure_analysis.json",
        help="Where to save categorized failure analysis for reviewed low-confidence samples.",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=50,
        help="Maximum number of manually labeled low-confidence words to collect in interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Collect manual labels interactively in the terminal.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    samples = load_json(args.samples)
    skipped_reviews = []

    if args.interactive:
        annotations, skipped_reviews = interactive_annotations(samples, max_annotations=args.max_annotations)
        save_json(args.annotations_output, annotations)
    else:
        annotations_path = Path(args.annotations_output)
        if not annotations_path.exists():
            raise FileNotFoundError(
                "No annotation file found. Run with --interactive first or provide "
                f"{annotations_path}."
            )
        annotations = load_json(annotations_path)

    summary = evaluate_annotations(annotations)
    summary["max_annotations"] = args.max_annotations
    summary["skipped_samples"] = len(skipped_reviews)
    failure_analysis = build_failure_analysis(annotations)
    summary["q3d_unreliable_categories"] = failure_analysis["q3d_summary"]
    save_json(args.evaluation_output, summary)
    save_json(args.failure_analysis_output, failure_analysis)
    logger.info("Low-confidence evaluation complete: %s", summary)
    logger.info("Failure analysis summary: %s", failure_analysis["category_counts"])


if __name__ == "__main__":
    main()
