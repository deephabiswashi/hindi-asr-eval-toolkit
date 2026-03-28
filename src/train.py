import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.config import (
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGE,
    LR,
    MAX_LABEL_LENGTH,
    MODEL_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    TASK,
    TRAIN_BATCH_SIZE,
    WHISPER_MODEL,
)

VALIDATION_RATIO = 0.1


def prepare(batch, processor):
    audio_array, _ = librosa.load(batch["audio_path"], sr=SAMPLE_RATE)
    audio_array = np.asarray(audio_array, dtype=np.float32)

    if audio_array.size == 0:
        raise ValueError(f"Empty audio loaded from: {batch['audio_path']}")

    inputs = processor(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    batch["input_features"] = inputs.input_features[0].cpu().numpy()
    batch["labels"] = processor.tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
    ).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels = torch.tensor(
            [feature["labels"] for feature in features],
            dtype=torch.long,
        )
        labels = labels.masked_fill(labels.eq(self.processor.tokenizer.pad_token_id), -100)
        batch["labels"] = labels
        decoder_input_ids = labels.clone()
        decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids.eq(-100), self.processor.tokenizer.pad_token_id)
        decoder_input_ids[:, 1:] = decoder_input_ids[:, :-1].clone()
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        batch["decoder_input_ids"] = decoder_input_ids
        return batch


def _split_dataset(dataset):
    shuffled = dataset.shuffle(seed=RANDOM_SEED)
    if len(shuffled) < 2:
        return shuffled, shuffled

    eval_size = min(max(1, int(round(len(shuffled) * VALIDATION_RATIO))), len(shuffled) - 1)
    split = shuffled.train_test_split(test_size=eval_size, seed=RANDOM_SEED)
    return split["train"], split["test"]


def _map_features(dataset, processor, desc):
    return dataset.map(
        lambda sample: prepare(sample, processor),
        remove_columns=dataset.column_names,
        desc=desc,
    )


def _additional_training_steps(train_dataset_size, max_steps):
    if max_steps is not None:
        return max(1, int(max_steps))

    effective_batch = max(1, TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    return max(1, math.ceil(train_dataset_size / effective_batch))


def saved_model_available(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return False

    has_config = (model_dir / "config.json").exists()
    has_preprocessor = (model_dir / "preprocessor_config.json").exists()
    has_tokenizer = (model_dir / "tokenizer_config.json").exists()
    has_model_weights = any(model_dir.glob("model*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))
    return has_config and has_preprocessor and has_tokenizer and has_model_weights


def load_saved_model(model_dir=MODEL_DIR):
    if not saved_model_available(model_dir):
        raise FileNotFoundError(f"Saved fine-tuned model is not available at: {model_dir}")

    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE,
        task=TASK,
    )
    model.config.use_cache = False
    return model, processor


def train_model(dataset, max_steps: Optional[int] = None):
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing.")

    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE,
        task=TASK,
    )
    model.config.use_cache = False

    train_split, eval_split = _split_dataset(dataset)
    train_dataset = _map_features(train_split, processor, desc="Preparing training features")
    eval_dataset = _map_features(eval_split, processor, desc="Preparing validation features")

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = processor.tokenizer.bos_token_id
    if decoder_start_token_id is None:
        raise ValueError("Whisper decoder_start_token_id is not configured.")

    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )
    training_steps = _additional_training_steps(len(train_dataset), max_steps)
    eval_steps = max(1, training_steps // 4)
    logging_steps = max(1, min(50, eval_steps))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LR,
        num_train_epochs=1 if max_steps is not None else EPOCHS,
        max_steps=training_steps if max_steps is not None else -1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=RANDOM_SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    print("\nStarting Whisper Fine-Tuning...\n")
    print("Training mode: fresh fine-tune from openai/whisper-small")
    print(f"Model output directory: {MODEL_DIR}")
    print(f"Training samples used: {len(train_dataset)}")
    print(f"Validation samples used: {len(eval_dataset)}")
    if max_steps is not None:
        print(f"Requested max training steps: {training_steps}")
    else:
        print(f"Training epochs: {EPOCHS}")
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    processor.save_pretrained(str(MODEL_DIR))
    print("\nTraining completed.\n")
    return trainer.model, processor
