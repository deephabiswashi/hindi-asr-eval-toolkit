from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Dataset

from src.config import SAMPLE_RATE, SEGMENTS_DIR


def _resolve_audio_path(sample, segment_dir, index):
    existing_path = sample.get("audio_path")
    if existing_path and Path(existing_path).exists():
        return str(Path(existing_path))

    audio_path = segment_dir / f"segment_{index:06d}.wav"
    audio_array = np.asarray(sample["audio_array"], dtype=np.float32)

    if audio_array.size == 0:
        raise ValueError(f"Cannot create dataset entry for empty audio sample at index {index}.")

    sampling_rate = int(sample.get("sampling_rate", SAMPLE_RATE))
    sf.write(str(audio_path), audio_array, sampling_rate, subtype="PCM_16")
    return str(audio_path)


def create_dataset(samples, output_dir=None):
    segment_dir = Path(output_dir) if output_dir else SEGMENTS_DIR
    segment_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "audio_path": [],
        "text": [],
    }

    for index, sample in enumerate(samples):
        data["audio_path"].append(_resolve_audio_path(sample, segment_dir, index))
        data["text"].append(sample["text"])

    return Dataset.from_dict(data)
