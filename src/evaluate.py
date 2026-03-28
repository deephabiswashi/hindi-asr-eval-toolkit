import evaluate
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import BATCH_SIZE, LANGUAGE, SAMPLE_RATE, TASK, WHISPER_MODEL

wer_metric = evaluate.load("wer")

FIXED_GENERATION_KWARGS = {
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "length_penalty": 1.0,
    "max_new_tokens": 128,
    "early_stopping": True,
}


def _load_audio(audio_path):
    audio_array, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_array = np.asarray(audio_array, dtype=np.float32)

    if audio_array.size == 0:
        raise ValueError(f"Empty audio loaded from: {audio_path}")

    return audio_array


def _load_model_and_processor():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE,
        task=TASK,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device


def compute_wer(preds, refs):
    if not preds or not refs:
        raise ValueError("No valid predictions generated, so WER cannot be computed.")

    return wer_metric.compute(predictions=preds, references=refs)


def transcribe_dataset(
    dataset,
    model=None,
    processor=None,
    batch_size=BATCH_SIZE,
    desc="Running ASR",
    generation_kwargs=None,
    postprocess_fn=None,
    return_records=False,
):
    if model is None or processor is None:
        model, processor, device = _load_model_and_processor()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANGUAGE,
            task=TASK,
        )
        model.to(device)
        model.eval()

    generation_kwargs = dict(generation_kwargs or {})
    preds, refs = [], []
    batch_audio = []
    batch_refs = []
    batch_paths = []
    prediction_records = []
    skipped_samples = 0

    def flush_batch():
        nonlocal batch_audio, batch_refs, batch_paths

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
                **generation_kwargs,
            )

        batch_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for audio_path, reference, prediction in zip(batch_paths, batch_refs, batch_preds):
            raw_prediction = prediction.strip()
            final_prediction = postprocess_fn(raw_prediction) if postprocess_fn else raw_prediction
            preds.append(final_prediction)
            refs.append(reference.strip())
            prediction_records.append(
                {
                    "audio_path": audio_path,
                    "reference": reference.strip(),
                    "raw_prediction": raw_prediction,
                    "prediction": final_prediction,
                }
            )

        batch_audio = []
        batch_refs = []
        batch_paths = []

    progress = tqdm(dataset, desc=desc)

    for sample in progress:
        try:
            batch_audio.append(_load_audio(sample["audio_path"]))
            batch_refs.append(sample["text"])
            batch_paths.append(sample["audio_path"])
        except Exception as exc:
            skipped_samples += 1
            tqdm.write(f"Skipping corrupted sample {sample.get('audio_path', '<missing>')}: {exc}")
            continue

        if len(batch_audio) >= batch_size:
            flush_batch()

    flush_batch()

    wer = compute_wer(preds, refs)
    if return_records:
        stats = {
            "total_samples": len(dataset),
            "predictions_generated": len(preds),
            "skipped_samples": skipped_samples,
        }
        return wer, preds, refs, prediction_records, stats

    return wer, preds, refs


def run_baseline(dataset, batch_size=BATCH_SIZE):
    return transcribe_dataset(
        dataset,
        batch_size=batch_size,
        desc="Running Baseline ASR",
    )
