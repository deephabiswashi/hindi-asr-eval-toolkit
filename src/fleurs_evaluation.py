from pathlib import Path

from datasets import Audio, load_dataset

from src.config import BATCH_SIZE, FLEURS_CONFIG, FLEURS_DATASET, FLEURS_REVISION, PROCESSED_DIR
from src.evaluate import transcribe_dataset


def _fleurs_data_files(split):
    directory_pattern = f"hf://datasets/{FLEURS_DATASET}@{FLEURS_REVISION}/{FLEURS_CONFIG}/{split}/*.parquet"
    legacy_file_pattern = f"hf://datasets/{FLEURS_DATASET}@{FLEURS_REVISION}/{FLEURS_CONFIG}/fleurs-{split}.parquet"
    return [directory_pattern, legacy_file_pattern]


def _load_fleurs_split(split):
    try:
        return load_dataset(FLEURS_DATASET, FLEURS_CONFIG, split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

    last_error = None
    for data_file in _fleurs_data_files(split):
        try:
            return load_dataset("parquet", data_files={split: data_file}, split=split)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to load FLEURS split '{split}' from direct parquet files.") from last_error


def _build_fleurs_records(split="test", max_samples=None):
    dataset = _load_fleurs_split(split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    fleurs_audio_dir = PROCESSED_DIR / "fleurs_audio"
    fleurs_audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, sample in enumerate(dataset):
        audio_info = sample["audio"]
        resolved_audio_path = None
        audio_path = audio_info.get("path")

        if audio_path:
            audio_path = Path(audio_path)
            if audio_path.exists():
                resolved_audio_path = str(audio_path)
            else:
                audio_bytes = audio_info.get("bytes")
                if audio_bytes:
                    suffix = audio_path.suffix or ".wav"
                    materialized_path = fleurs_audio_dir / f"{audio_path.stem or f'fleurs_{index:06d}'}{suffix}"
                    if not materialized_path.exists():
                        materialized_path.write_bytes(audio_bytes)
                    resolved_audio_path = str(materialized_path)

        if not resolved_audio_path:
            continue

        records.append(
            {
                "audio_path": resolved_audio_path,
                "text": sample["transcription"],
            }
        )

    if not records:
        raise ValueError("FLEURS evaluation dataset did not yield any audio paths.")

    return records


def eval_fleurs(
    model=None,
    processor=None,
    batch_size=BATCH_SIZE,
    split="test",
    max_samples=None,
    generation_kwargs=None,
    postprocess_fn=None,
    return_records=False,
):
    records = _build_fleurs_records(split=split, max_samples=max_samples)
    return transcribe_dataset(
        records,
        model=model,
        processor=processor,
        batch_size=batch_size,
        desc="Evaluating on FLEURS",
        generation_kwargs=generation_kwargs,
        postprocess_fn=postprocess_fn,
        return_records=return_records,
    )
