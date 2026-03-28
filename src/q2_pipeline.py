import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import BATCH_SIZE, DEBUG_MODE, LANGUAGE, SAMPLE_RATE, TASK
from src.evaluate import _load_audio, _load_model_and_processor
from src.q2_english_detector import tag_english_words
from src.q2_number_normalizer import edge_case_examples, normalize_numbers


logger = logging.getLogger(__name__)

Q2_DEBUG_SAMPLE_COUNT = 100
FALLBACK_NUMBER_TEXTS = [
    "मुझे तीन सौ चौवन रुपये चाहिए",
    "आज एक हज़ार दो सौ लोग आए",
    "उसने चौवन किताबें पढ़ीं",
    "यह उन्नीस सौ निन्यानवे की बात है",
    "करीब बीस हजार लोग मौजूद थे",
]
FALLBACK_ENGLISH_TEXTS = [
    "मेरा interview अच्छा गया",
    "मैंने कंप्यूटर और file दोनों भेजे",
    "आज team meeting जल्दी खत्म हुई",
    "यह job problem नहीं है",
    "डेटा सिस्टम update करना है",
]


def _subset_dataset(dataset, debug_mode=DEBUG_MODE):
    if not debug_mode:
        return dataset

    if hasattr(dataset, "select"):
        limit = min(Q2_DEBUG_SAMPLE_COUNT, len(dataset))
        logger.info("Q2 debug mode enabled: using %s samples.", limit)
        return dataset.select(range(limit))

    return dataset[:Q2_DEBUG_SAMPLE_COUNT]


def generate_raw_asr(dataset, batch_size=BATCH_SIZE):
    dataset = _subset_dataset(dataset)
    model, processor, device = _load_model_and_processor()
    logger.info("Generating raw ASR outputs with pretrained Whisper-small.")

    raw_outputs = []
    batch_audio = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_audio, batch_meta
        if not batch_audio:
            return

        inputs = processor.feature_extractor(
            batch_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language=LANGUAGE,
                task=TASK,
            )

        batch_predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for meta, prediction in zip(batch_meta, batch_predictions):
            raw_outputs.append(
                {
                    "audio_path": meta["audio_path"],
                    "reference": meta["reference"],
                    "prediction": prediction.strip(),
                }
            )

        batch_audio = []
        batch_meta = []

    progress = tqdm(dataset, desc="Q2 Raw ASR")
    for sample in progress:
        try:
            batch_audio.append(_load_audio(sample["audio_path"]))
            batch_meta.append(
                {
                    "audio_path": sample["audio_path"],
                    "reference": sample["text"],
                }
            )
        except Exception as exc:
            tqdm.write(f"Skipping corrupted Q2 sample {sample.get('audio_path', '<missing>')}: {exc}")
            continue

        if len(batch_audio) >= batch_size:
            flush_batch()

    flush_batch()
    logger.info("Generated %s raw ASR predictions for Q2.", len(raw_outputs))
    return raw_outputs


def _collect_number_examples(results, limit=5):
    examples = []
    for item in results:
        if item["raw_prediction"] == item["normalized"]:
            continue

        examples.append(
            {
                "reference": item["reference"],
                "raw_prediction": item["raw_prediction"],
                "normalized": item["normalized"],
            }
        )
        if len(examples) >= limit:
            break

    for text in FALLBACK_NUMBER_TEXTS:
        if len(examples) >= limit:
            break
        normalized = normalize_numbers(text)
        if normalized == text:
            continue
        examples.append(
            {
                "reference": text,
                "raw_prediction": text,
                "normalized": normalized,
            }
        )

    return examples


def _collect_english_examples(results, limit=5):
    examples = []
    for item in results:
        if item["normalized"] == item["tagged"]:
            continue

        examples.append(
            {
                "reference": item["reference"],
                "normalized": item["normalized"],
                "tagged": item["tagged"],
            }
        )
        if len(examples) >= limit:
            break

    for text in FALLBACK_ENGLISH_TEXTS:
        if len(examples) >= limit:
            break
        tagged = tag_english_words(text)
        if tagged == text:
            continue
        examples.append(
            {
                "reference": text,
                "normalized": text,
                "tagged": tagged,
            }
        )

    return examples


def _save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def run_q2_cleanup_pipeline(dataset, output_dir="outputs"):
    logger.info("Starting Q2 cleanup pipeline.")
    raw_outputs = generate_raw_asr(dataset)

    cleaned_results = []
    for item in raw_outputs:
        normalized = normalize_numbers(item["prediction"])
        tagged = tag_english_words(normalized)
        cleaned_results.append(
            {
                "audio_path": item["audio_path"],
                "reference": item["reference"],
                "raw_prediction": item["prediction"],
                "normalized": normalized,
                "tagged": tagged,
            }
        )

    number_examples = _collect_number_examples(cleaned_results, limit=5)
    english_examples = _collect_english_examples(cleaned_results, limit=5)
    edge_cases = edge_case_examples()

    examples_payload = {
        "number_examples": number_examples,
        "edge_cases": edge_cases,
        "english_tagging_examples": english_examples,
    }
    report_payload = {
        "number_examples": number_examples,
        "edge_cases": edge_cases,
        "english_tagging_examples": english_examples,
    }

    output_dir = Path(output_dir)
    _save_json(output_dir / "q2_results.json", cleaned_results)
    _save_json(output_dir / "q2_examples.json", examples_payload)
    _save_json(output_dir / "q2_report.json", report_payload)

    logger.info(
        "Q2 cleanup pipeline completed with %s samples, %s number examples, and %s English tagging examples.",
        len(cleaned_results),
        len(number_examples),
        len(english_examples),
    )

    return {
        "results": cleaned_results,
        "number_examples": number_examples,
        "edge_cases": edge_cases,
        "english_tagging_examples": english_examples,
    }
