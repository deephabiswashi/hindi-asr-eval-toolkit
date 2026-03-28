from src.download_dataset import download_dataset
from src.fleurs_evaluation import eval_fleurs
from src.train import train_model

from run_pipeline import build_training_dataset, log_stage


def main():
    log_stage("Quick Test: Build Debug Dataset")
    records = download_dataset("FT Data.xlsx", "data/raw")
    dataset, _ = build_training_dataset(records, debug_mode=True)

    log_stage("Quick Test: Train Smoke Run")
    model, processor = train_model(dataset, max_steps=1)

    log_stage("Quick Test: Fine-Tuned FLEURS Smoke Eval")
    wer, preds, refs = eval_fleurs(model, processor, max_samples=8)
    print("Valid predictions:", len(preds))
    print("Valid references:", len(refs))
    print("Smoke-test WER:", wer)


if __name__ == "__main__":
    main()
