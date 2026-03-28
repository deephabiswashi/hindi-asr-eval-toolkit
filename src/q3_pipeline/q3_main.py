import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.q3_pipeline.spell_classifier import SpellClassifier
from src.q3_pipeline.utils import deterministic_sample, ensure_output_dir, save_json, setup_logging
from src.q3_pipeline.word_loader import load_words


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Hindi word validation pipeline for Q3.")
    parser.add_argument(
        "--input-csv",
        default=r"C:\Users\admin\Desktop\JoshTalks\Unique Words Data - Sheet1.csv",
        help="Path to the unique words CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where Q3 outputs will be written.",
    )
    parser.add_argument(
        "--low-confidence-sample-size",
        type=int,
        default=50,
        help="Number of low-confidence examples to save for review.",
    )
    return parser.parse_args()


def run_q3_pipeline(input_csv, output_dir, low_confidence_sample_size=50):
    logger = setup_logging()
    output_dir = ensure_output_dir(output_dir)

    words = load_words(input_csv)
    classifier = SpellClassifier()

    results = []
    label_counter = Counter()
    confidence_counter = Counter()
    source_counter = Counter()

    for row in tqdm(words.itertuples(index=False), total=len(words), desc="Q3 word validation"):
        decision = classifier.classify(row.word, normalized_word=row.normalized_word)
        results.append(
            {
                "word": row.word,
                "label": decision["label"],
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "normalized_word": decision["normalized_word"],
                "source": decision.get("source"),
                "suggestion": decision.get("suggestion"),
            }
        )
        label_counter[decision["label"]] += 1
        confidence_counter[decision["confidence"]] += 1
        source_counter[decision.get("source")] += 1

    results_frame = pd.DataFrame(results)
    results_frame[["word", "label", "confidence", "reason"]].to_csv(
        output_dir / "q3_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    results_frame[["word", "label"]].assign(
        label=lambda frame: frame["label"].map(
            {
                "correct_spelling": "correct spelling",
                "incorrect_spelling": "incorrect spelling",
            }
        )
    ).to_csv(
        output_dir / "q3_word_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    low_confidence_rows = results_frame[results_frame["confidence"] == "low"].to_dict("records")
    low_confidence_sample = deterministic_sample(low_confidence_rows, low_confidence_sample_size, seed=42)
    save_json(output_dir / "q3_low_confidence_samples.json", low_confidence_sample)

    pd.DataFrame(
        [
            {"metric": "total_words", "value": int(len(results_frame))},
            {"metric": "correct_spelling_count", "value": int(label_counter["correct_spelling"])},
            {"metric": "incorrect_spelling_count", "value": int(label_counter["incorrect_spelling"])},
        ]
    ).to_csv(
        output_dir / "q3_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "input_csv": str(Path(input_csv)),
        "total_words": int(len(results_frame)),
        "correct_spelling_count": int(label_counter["correct_spelling"]),
        "incorrect_spelling_count": int(label_counter["incorrect_spelling"]),
        "confidence_distribution": {
            "high": int(confidence_counter["high"]),
            "medium": int(confidence_counter["medium"]),
            "low": int(confidence_counter["low"]),
        },
        "classification_source_distribution": {
            key: int(value) for key, value in sorted(source_counter.items()) if key
        },
        "low_confidence_sample_count": int(len(low_confidence_sample)),
        "approach_summary": [
            "Normalize each token with unicode cleanup, whitespace stripping, and punctuation removal.",
            "Mark exact Hindi dictionary matches and common inflected stems as likely correct.",
            "Treat Roman English and Devanagari English borrowings as correctly spelled when they match lexical or phonetic heuristics.",
            "Use orthographic rule checks plus near-dictionary edit distance to catch likely spelling errors.",
            "Assign high, medium, or low confidence based on how direct or ambiguous the evidence is.",
        ],
        "deliverables": {
            "detailed_results_csv": str(output_dir / "q3_results.csv"),
            "word_labels_csv": str(output_dir / "q3_word_labels.csv"),
            "counts_csv": str(output_dir / "q3_counts.csv"),
            "low_confidence_samples_json": str(output_dir / "q3_low_confidence_samples.json"),
        },
    }
    save_json(output_dir / "q3_summary.json", summary)

    logger.info("Total words: %s", summary["total_words"])
    logger.info("Correct spelling count: %s", summary["correct_spelling_count"])
    logger.info("Incorrect spelling count: %s", summary["incorrect_spelling_count"])
    logger.info("Confidence distribution: %s", summary["confidence_distribution"])
    return summary


def main():
    args = parse_args()
    run_q3_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        low_confidence_sample_size=args.low_confidence_sample_size,
    )


if __name__ == "__main__":
    main()
