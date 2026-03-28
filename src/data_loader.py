import json

import librosa

from src.config import SAMPLE_RATE, SILENCE_TRIM_DB


def _trim_segment(audio_chunk):
    if len(audio_chunk) == 0:
        return audio_chunk

    trimmed_audio, _ = librosa.effects.trim(audio_chunk, top_db=SILENCE_TRIM_DB)
    return trimmed_audio if len(trimmed_audio) > 0 else audio_chunk


def extract_segments(audio_path, json_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    with open(json_path, "r", encoding="utf-8") as file_obj:
        segments = json.load(file_obj)

    samples = []

    for segment in segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", 0.0) or 0.0)
        text = str(segment.get("text", "") or "")

        if end <= start:
            continue

        start_idx = max(0, min(int(start * sr), len(audio)))
        end_idx = max(start_idx, min(int(end * sr), len(audio)))

        audio_chunk = audio[start_idx:end_idx]
        audio_chunk = _trim_segment(audio_chunk)

        if len(audio_chunk) == 0:
            continue

        samples.append(
            {
                "audio_array": audio_chunk,
                "sampling_rate": sr,
                "text": text,
            }
        )

    return samples
