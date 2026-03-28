import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.config import (
    DEBUG_MODE,
    DEBUG_RECORD_POOL_TARGET,
    FULL_SAMPLE_COUNT,
    MODEL_DIR,
    RANDOM_SEED,
)
from src.data_loader import extract_segments
from src.dataset import create_dataset
from src.download_dataset import download_dataset
from src.error_analysis import sample_errors
from src.evaluate import FIXED_GENERATION_KWARGS, run_baseline
from src.fleurs_evaluation import eval_fleurs
from src.postprocess import clean_prediction
from src.preprocess import preprocess
from src.q2_pipeline import run_q2_cleanup_pipeline
from src.train import load_saved_model, saved_model_available, train_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DATASET_MANIFEST_CANDIDATES = (
    Path("FT Data.xlsx"),
    Path("FT Data - data.csv"),
)
Q1G_DEBUG_SAMPLE_COUNT = 100
Q1G_DEBUG_FLEURS_SAMPLE_COUNT = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Josh Talks ASR pipeline. By default this runs the full Q1 + Q2 flow."
    )
    parser.add_argument(
        "--q2-only",
        action="store_true",
        help="Skip Q1 baseline, training, evaluation, and error analysis; only build the shared dataset and run the Q2 cleanup pipeline.",
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Reuse saved baseline/evaluation artifacts and the saved fine-tuned model when available to avoid rerunning expensive stages during debugging.",
    )
    return parser.parse_args()


def resolve_dataset_manifest():
    for candidate in DATASET_MANIFEST_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    candidate_list = ", ".join(str(path) for path in DATASET_MANIFEST_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a dataset manifest file. Expected one of: "
        f"{candidate_list}"
    )


def collect_local_records(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    wav_files = sorted(raw_dir.glob("*.wav"))
    records = []

    for wav_path in wav_files:
        json_path = wav_path.with_suffix(".json")
        if not json_path.exists():
            continue

        records.append(
            {
                "audio_path": str(wav_path),
                "json_path": str(json_path),
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No paired .wav/.json files were found in {raw_dir}."
        )

    return records


def log_stage(message):
    print(f"\n=== {message} ===")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def cache_file(output_dir, name):
    return Path(output_dir) / "cache" / f"{name}.json"


def stratified_sample(samples, target_size, seed=RANDOM_SEED, strata=4):
    if target_size is None or len(samples) <= target_size:
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    rng = random.Random(seed)
    ordered = sorted(samples, key=lambda sample: len(sample["audio_array"]))
    ordered_with_index = list(enumerate(ordered))
    buckets = [[] for _ in range(strata)]

    for index, sample in ordered_with_index:
        bucket_index = min(strata - 1, (index * strata) // len(ordered))
        buckets[bucket_index].append((index, sample))

    selected = []
    selected_indices = set()
    remaining = target_size

    for bucket_index, bucket in enumerate(buckets):
        if not bucket:
            continue

        proportional_take = round(target_size * len(bucket) / len(ordered))
        min_reserved = max(0, sum(1 for future_bucket in buckets[bucket_index + 1:] if future_bucket))
        take = max(1, proportional_take)
        take = min(len(bucket), max(1, remaining - min_reserved), take)

        chosen = rng.sample(bucket, take)
        selected.extend(sample for _, sample in chosen)
        selected_indices.update(index for index, _ in chosen)
        remaining = target_size - len(selected)

    if len(selected) < target_size:
        leftovers = [sample for index, sample in ordered_with_index if index not in selected_indices]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_size - len(selected)])

    rng.shuffle(selected)
    return selected[:target_size]


def collect_processed_samples(records, debug_mode):
    rng = random.Random(RANDOM_SEED)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    processed_samples = []
    extracted_segments = 0
    filtered_segments = 0
    drop_reasons = Counter()

    for record_index, record in enumerate(shuffled_records, start=1):
        segments = extract_segments(record["audio_path"], record["json_path"])
        extracted_segments += len(segments)
        processed_batch, batch_stats = preprocess(segments, return_stats=True)
        processed_samples.extend(processed_batch)
        filtered_segments += batch_stats["dropped"]
        drop_reasons.update(batch_stats["drop_reasons"])

        if debug_mode and len(processed_samples) >= DEBUG_RECORD_POOL_TARGET:
            print(
                "Debug mode: stopping early after "
                f"{record_index} recordings with {len(processed_samples)} valid segments."
            )
            break

    print("Extracted segments:", extracted_segments)
    print("Processed segments:", len(processed_samples))
    print("Filtered segments:", filtered_segments)
    if drop_reasons:
        print("Top filter reasons:", dict(drop_reasons.most_common(5)))
    return processed_samples


def build_training_dataset(records, debug_mode=DEBUG_MODE):
    processed_samples = collect_processed_samples(records, debug_mode=debug_mode)
    sample_target = Q1G_DEBUG_SAMPLE_COUNT if debug_mode else FULL_SAMPLE_COUNT
    selected_samples = stratified_sample(processed_samples, sample_target)
    dataset = create_dataset(selected_samples)

    print("Selected training samples:", len(selected_samples))
    print("Dataset rows:", len(dataset))
    return dataset, selected_samples


def run_q2_stage(dataset, output_dir, stage_label):
    log_stage(stage_label)
    q2_payload = run_q2_cleanup_pipeline(dataset, output_dir=output_dir)
    print("Q2 cleaned samples:", len(q2_payload["results"]))
    print("Q2 number examples:", len(q2_payload["number_examples"]))
    print("Q2 English tagging examples:", len(q2_payload["english_tagging_examples"]))
    return q2_payload


def build_comparison_examples(before_records, after_records, limit=10):
    changed_examples = []
    fallback_examples = []

    for before_item, after_item in zip(before_records, after_records):
        example = {
            "reference": after_item["reference"],
            "before": before_item["prediction"],
            "after": after_item["prediction"],
        }
        if example["before"] != example["after"]:
            changed_examples.append(example)
        else:
            fallback_examples.append(example)

    combined = changed_examples[:limit]
    if len(combined) < limit:
        combined.extend(fallback_examples[: limit - len(combined)])
    return combined


def main():
    args = parse_args()

    if args.q2_only:
        log_stage("Stage 1: Load Existing Raw Dataset")
        records = collect_local_records("data/raw")
        print("Local recordings available:", len(records))
    else:
        log_stage("Stage 1: Download / Verify Dataset")
        dataset_manifest = resolve_dataset_manifest()
        records = download_dataset(dataset_manifest, "data/raw")
        print("Recordings available:", len(records))

    log_stage("Stage 2: Build Training Dataset")
    dataset, selected_samples = build_training_dataset(records, debug_mode=DEBUG_MODE)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.q2_only:
        q2_payload = run_q2_stage(
            dataset,
            output_dir=output_dir,
            stage_label="Stage 3: Q2 Cleanup Pipeline",
        )

        q2_summary = {
            "mode": "q2_only",
            "debug_mode": DEBUG_MODE,
            "training_samples": len(selected_samples),
            "q2_cleaned_samples": len(q2_payload["results"]),
            "q2_number_examples": len(q2_payload["number_examples"]),
            "q2_english_tagging_examples": len(q2_payload["english_tagging_examples"]),
        }

        save_json(output_dir / "q2_run_summary.json", q2_summary)

        print("\n===== Q2 ONLY RESULTS =====")
        print("Training samples prepared:", len(selected_samples))
        print("Q2 cleaned samples:", len(q2_payload["results"]))
        return

    baseline_wer = None
    baseline_preds = []
    baseline_cache_path = cache_file(output_dir, "training_baseline")

    if DEBUG_MODE:
        print("DEBUG_MODE is enabled: skipping baseline evaluation.")
    else:
        if args.reuse_artifacts and baseline_cache_path.exists():
            log_stage("Stage 3: Reuse Cached Baseline on Training Segments")
            baseline_cache = load_json(baseline_cache_path)
            baseline_wer = baseline_cache["wer"]
            baseline_preds = baseline_cache["preds"]
            print("Loaded cached baseline predictions:", len(baseline_preds))
            print("Baseline segment WER:", baseline_wer)
        else:
            log_stage("Stage 3: Baseline on Training Segments")
            baseline_wer, baseline_preds, baseline_refs = run_baseline(dataset)
            print("Valid predictions:", len(baseline_preds))
            print("Baseline segment WER:", baseline_wer)
            save_json(
                baseline_cache_path,
                {
                    "wer": baseline_wer,
                    "preds": baseline_preds,
                    "refs": baseline_refs,
                },
            )

    log_stage("Stage 4: Fine-Tune Whisper-small")
    if DEBUG_MODE:
        print("DEBUG_MODE is enabled: using 100 training samples and limiting FLEURS evaluation to 100 samples.")
    print("Training mode: fresh fine-tune from openai/whisper-small")
    print("Model output directory:", MODEL_DIR)
    train_max_steps = None
    if args.reuse_artifacts and saved_model_available(MODEL_DIR):
        print("Reusing saved fine-tuned model from:", MODEL_DIR)
        model, processor = load_saved_model(MODEL_DIR)
    else:
        model, processor = train_model(dataset, max_steps=train_max_steps)

    fleurs_limit = Q1G_DEBUG_FLEURS_SAMPLE_COUNT if DEBUG_MODE else None
    fleurs_baseline_wer = None
    fleurs_baseline_preds = []
    fleurs_baseline_cache_path = cache_file(output_dir, "fleurs_baseline")

    if not DEBUG_MODE:
        if args.reuse_artifacts and fleurs_baseline_cache_path.exists():
            log_stage("Stage 5: Reuse Cached Baseline on FLEURS")
            fleurs_baseline_cache = load_json(fleurs_baseline_cache_path)
            fleurs_baseline_wer = fleurs_baseline_cache["wer"]
            fleurs_baseline_preds = fleurs_baseline_cache["preds"]
            print("Loaded cached FLEURS baseline predictions:", len(fleurs_baseline_preds))
            print("FLEURS baseline WER:", fleurs_baseline_wer)
        else:
            log_stage("Stage 5: Baseline on FLEURS")
            fleurs_baseline_wer, fleurs_baseline_preds, fleurs_baseline_refs = eval_fleurs()
            print("Valid predictions:", len(fleurs_baseline_preds))
            print("FLEURS baseline WER:", fleurs_baseline_wer)
            save_json(
                fleurs_baseline_cache_path,
                {
                    "wer": fleurs_baseline_wer,
                    "preds": fleurs_baseline_preds,
                    "refs": fleurs_baseline_refs,
                },
            )

    before_cache_path = cache_file(output_dir, "fine_tuned_eval_before")

    if args.reuse_artifacts and before_cache_path.exists():
        log_stage("Stage 6: Reuse Cached Fine-Tuned Evaluation on FLEURS (Legacy Decoding)")
        before_cache = load_json(before_cache_path)
        before_wer = before_cache["wer"]
        before_preds = before_cache["preds"]
        before_refs = before_cache["refs"]
        before_records = before_cache["records"]
        before_stats = before_cache["stats"]
    else:
        log_stage("Stage 6: Fine-Tuned Evaluation on FLEURS (Legacy Decoding)")
        before_wer, before_preds, before_refs, before_records, before_stats = eval_fleurs(
            model,
            processor,
            max_samples=fleurs_limit,
            return_records=True,
        )
        save_json(
            before_cache_path,
            {
                "wer": before_wer,
                "preds": before_preds,
                "refs": before_refs,
                "records": before_records,
                "stats": before_stats,
            },
        )
    print("Predictions generated:", before_stats["predictions_generated"])
    print("Skipped samples:", before_stats["skipped_samples"])
    print("Fine-tuned FLEURS WER (before fixes):", before_wer)

    if len(before_preds) == 0 or len(before_refs) == 0:
        raise ValueError("Fine-tuned FLEURS evaluation produced no valid predictions before fixes.")

    save_json(output_dir / "before_preds.json", before_records)

    after_cache_path = cache_file(output_dir, "fine_tuned_eval_after")

    if args.reuse_artifacts and after_cache_path.exists():
        log_stage("Stage 7: Reuse Cached Fine-Tuned Evaluation on FLEURS (Fixed Decoding + Cleanup)")
        after_cache = load_json(after_cache_path)
        after_wer = after_cache["wer"]
        after_preds = after_cache["preds"]
        after_refs = after_cache["refs"]
        after_records = after_cache["records"]
        after_stats = after_cache["stats"]
    else:
        log_stage("Stage 7: Fine-Tuned Evaluation on FLEURS (Fixed Decoding + Cleanup)")
        after_wer, after_preds, after_refs, after_records, after_stats = eval_fleurs(
            model,
            processor,
            max_samples=fleurs_limit,
            generation_kwargs=FIXED_GENERATION_KWARGS,
            postprocess_fn=clean_prediction,
            return_records=True,
        )
        save_json(
            after_cache_path,
            {
                "wer": after_wer,
                "preds": after_preds,
                "refs": after_refs,
                "records": after_records,
                "stats": after_stats,
            },
        )
    print("Predictions generated:", after_stats["predictions_generated"])
    print("Skipped samples:", after_stats["skipped_samples"])
    print("Fine-tuned FLEURS WER (after fixes):", after_wer)

    if len(after_preds) == 0 or len(after_refs) == 0:
        raise ValueError("Fine-tuned FLEURS evaluation produced no valid predictions after fixes.")

    save_json(output_dir / "after_preds.json", after_records)

    comparison_payload = {
        "baseline_wer": before_wer,
        "fixed_wer": after_wer,
        "improvement": before_wer - after_wer,
    }
    save_json(output_dir / "final_comparison.json", comparison_payload)

    comparison_examples = build_comparison_examples(before_records, after_records, limit=10)
    save_json(output_dir / "final_examples.json", comparison_examples)

    log_stage("Stage 8: Error Sampling")
    errors = sample_errors(after_preds, after_refs)
    print("Sampled error examples:", len(errors))
    q2_payload = run_q2_stage(
        dataset,
        output_dir=output_dir,
        stage_label="Stage 9: Q2 Cleanup Pipeline",
    )

    results = {
        "debug_mode": DEBUG_MODE,
        "reuse_artifacts_enabled": args.reuse_artifacts,
        "training_mode": "fresh_fine_tune_from_base",
        "model_output_dir": str(MODEL_DIR),
        "training_samples": len(selected_samples),
        "baseline_segment_wer": baseline_wer,
        "fleurs_baseline_wer": fleurs_baseline_wer,
        "fleurs_fine_tuned_wer": after_wer,
        "fleurs_fine_tuned_wer_before_fix": before_wer,
        "fleurs_fine_tuned_wer_after_fix": after_wer,
        "valid_training_baseline_predictions": len(baseline_preds),
        "valid_fleurs_baseline_predictions": len(fleurs_baseline_preds),
        "valid_fleurs_fine_tuned_predictions": len(after_preds),
        "valid_fleurs_fine_tuned_predictions_before_fix": len(before_preds),
        "valid_fleurs_fine_tuned_predictions_after_fix": len(after_preds),
        "before_skipped_samples": before_stats["skipped_samples"],
        "after_skipped_samples": after_stats["skipped_samples"],
        "before_vs_after_improvement": before_wer - after_wer,
        "q2_cleaned_samples": len(q2_payload["results"]),
        "q2_number_examples": len(q2_payload["number_examples"]),
        "q2_english_tagging_examples": len(q2_payload["english_tagging_examples"]),
    }

    save_json(output_dir / "results.json", results)
    save_json(output_dir / "error_samples.json", errors)

    print("\n===== RESULTS =====")
    print("Training samples:", len(selected_samples))
    print("FLEURS baseline WER:", fleurs_baseline_wer)
    print("FLEURS fine-tuned WER (before fixes):", before_wer)
    print("FLEURS fine-tuned WER (after fixes):", after_wer)
    print("Improvement:", before_wer - after_wer)


if __name__ == "__main__":
    main()
