# Hindi ASR Evaluation and Data Cleaning Toolkit

An end-to-end Hindi ASR research codebase built around four assignment-style workflows:

- `Q1`: Whisper-small fine-tuning and FLEURS evaluation
- `Q2`: post-ASR cleanup for Hindi number normalization and English-word tagging
- `Q3`: large-scale Hindi word validation for transcription cleanup
- `Q4`: lattice-based WER evaluation that reduces unfair penalties from noisy references

The repository is organized so each question can be run independently while sharing a consistent project structure. The later Q3 and Q4 pipelines are fully standalone and do not depend on the Whisper training flow.

## Suggested Repository Name

`hindi-asr-eval-toolkit`

## Suggested Repository Description

Production-oriented Hindi ASR toolkit covering Whisper fine-tuning, post-processing, spelling validation, and lattice-based WER evaluation for noisy conversational transcripts.

## Highlights

- Hindi Whisper-small fine-tuning with preprocessing, filtering, evaluation, and report generation
- No-retraining cleanup pipeline for numbers and Devanagari English terms
- Rule-based large-scale spelling validation over `~1.77L` unique Hindi words
- Lattice-based WER that uses model agreement to reduce unfair penalties from noisy references
- Windows-friendly Python code with modular pipelines and JSON/CSV outputs

## Repository Layout

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- outputs/
|-- scripts/
|   |-- generate_q1_report.py
|   `-- run_q4_pipeline.py
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- dataset.py
|   |-- download_dataset.py
|   |-- error_analysis.py
|   |-- evaluate.py
|   |-- fleurs_evaluation.py
|   |-- postprocess.py
|   |-- preprocess.py
|   |-- q1_analysis.py
|   |-- q2_english_detector.py
|   |-- q2_number_normalizer.py
|   |-- q2_pipeline.py
|   |-- train.py
|   |-- q3_pipeline/
|   `-- q4_pipeline/
|-- quick_test_no_baseline.py
|-- run_pipeline.py
`-- run_q4_pipeline.py
```

## Pipeline Overview

### Q1: Whisper Fine-Tuning and Evaluation

`run_pipeline.py` is the main entrypoint for the Q1 workflow.

Stages:

1. Download or verify the dataset manifest
2. Extract aligned speech segments from `data/raw`
3. Preprocess and filter low-quality segments
4. Create the training dataset
5. Run baseline ASR on training segments
6. Fine-tune `openai/whisper-small`
7. Evaluate on Hindi FLEURS
8. Sample error cases and generate report inputs

Q1 also contains:

- stronger segment filtering in `src/preprocess.py`
- safer decoding in `src/evaluate.py` and `src/fleurs_evaluation.py`
- post-processing cleanup in `src/postprocess.py`
- Q1 report generation in `scripts/generate_q1_report.py`

### Q2: Hindi ASR Cleanup

Q2 is integrated into `run_pipeline.py` after the shared dataset is built. It does not require retraining.

It provides:

- Hindi number normalization
- English-word tagging in Hindi transcripts
- JSON outputs with curated examples for submission

Key modules:

- `src/q2_number_normalizer.py`
- `src/q2_english_detector.py`
- `src/q2_pipeline.py`

### Q3: Hindi Word Validation

Q3 is a standalone text pipeline under `src/q3_pipeline/`. It classifies each word as:

- `correct spelling`
- `incorrect spelling`

and also assigns:

- `high`
- `medium`
- `low`

confidence with a short reason.

It combines:

- normalization
- Hindi lexicon checks
- Devanagari English-word detection
- edit-distance and orthographic heuristics
- manual low-confidence review support

### Q4: Lattice-Based WER

Q4 is fully standalone under `src/q4_pipeline/`.

It:

- reads a CSV containing one human reference and multiple ASR outputs
- aligns all sequences at the word level
- builds per-position lattice bins
- uses model agreement to weaken noisy references
- computes baseline WER and lattice-aware WER per model

This is useful when a single human reference is itself noisy or incomplete.

## Requirements

Install dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Core packages include:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `jiwer`
- `librosa`
- `pandas`
- `numpy`

## Expected Local Inputs

This repo assumes some task-specific data files exist locally and may not be committed:

- `FT Data.xlsx` or `FT Data - data.csv`
- `data/raw/` containing paired `.wav` and `.json` files
- `Unique Words Data - Sheet1.csv`
- `Question 4 - Task.csv`

## How To Run

### Q1 + Q2 Full Pipeline

```powershell
python run_pipeline.py
```

Useful flags:

```powershell
python run_pipeline.py --q2-only
python run_pipeline.py --reuse-artifacts
```

### Q1 Report Generation

```powershell
python scripts/generate_q1_report.py
```

### Q3 Standalone Pipeline

```powershell
python -m src.q3_pipeline.q3_main --input-csv "C:\Users\admin\Desktop\JoshTalks\Unique Words Data - Sheet1.csv" --output-dir outputs
```

Interactive low-confidence review:

```powershell
python -m src.q3_pipeline.evaluator --samples outputs/q3_low_confidence_samples.json --annotations-output outputs/q3_low_confidence_annotations.json --evaluation-output outputs/q3_evaluation.json --failure-analysis-output outputs/q3_failure_analysis.json --max-annotations 50 --interactive
```

### Q4 Standalone Pipeline

Either of these works:

```powershell
python run_q4_pipeline.py
```

```powershell
python scripts/run_q4_pipeline.py --input-csv "C:\Users\admin\Desktop\JoshTalks\Question 4 - Task.csv" --output-dir outputs/q4
```

## Outputs

### Q1

- `outputs/results.json`
- `outputs/error_samples.json`
- `outputs/q1_d_to_g_report.md`
- `outputs/q1_d_to_g_analysis.json`

### Q2

- `outputs/q2_results.json`
- `outputs/q2_examples.json`
- `outputs/q2_report.json`
- `outputs/q2_run_summary.json`

### Q3

- `outputs/q3_results.csv`
- `outputs/q3_word_labels.csv`
- `outputs/q3_counts.csv`
- `outputs/q3_summary.json`
- `outputs/q3_low_confidence_samples.json`
- `outputs/q3_evaluation.json`
- `outputs/q3_failure_analysis.json`

### Q4

- `outputs/q4/q4_lattices.json`
- `outputs/q4/q4_aligned_sequences.json`
- `outputs/q4/q4_wer_comparison.csv`
- `outputs/q4/q4_consensus_analysis.json`
- `outputs/q4/q4_error_cases.json`
- `outputs/q4/q4_summary.json`

## Design Notes

- The codebase is optimized for practical experimentation rather than packaging as a pip module.
- Q1/Q2 share dataset-building logic because both operate on speech segments.
- Q3 and Q4 are intentionally independent from the Whisper training path.
- The repository keeps `outputs/` versionable so generated results can be inspected or shared.
- Local raw data and model checkpoints are expected to stay out of Git.

## Notes for Reviewers

- This project mixes model training, text post-processing, and evaluation tooling in one repo because it was developed as a research assignment.
- Some pipelines are intentionally heuristic-heavy because the goal is practical data cleaning and fairer evaluation, not just model training.
- Several outputs in `outputs/` are meant to be submission artifacts rather than intermediate scratch files.
