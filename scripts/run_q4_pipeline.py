import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.q4_pipeline.lattice_builder import build_lattice_for_record
from src.q4_pipeline.utils import load_task_records, save_json, serialize_sequence_map, setup_logging
from src.q4_pipeline.wer import ErrorCounts, compute_baseline_wer, compute_lattice_wer


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone Q4 lattice-based WER evaluation.")
    parser.add_argument(
        "--input-csv",
        default=r"C:\Users\admin\Desktop\JoshTalks\Question 4 - Task.csv",
        help="Path to the Question 4 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/q4",
        help="Directory where Q4 outputs will be written.",
    )
    return parser.parse_args()


def _counts_to_dict(counts: ErrorCounts) -> dict:
    return {
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "reference_tokens": counts.reference_tokens,
        "wer": round(counts.wer, 6),
    }


def _write_wer_comparison(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["model_name", "baseline_wer", "lattice_wer", "delta"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    logger = setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_task_records(args.input_csv)
    if not records:
        raise ValueError("Question 4 CSV produced no records.")

    model_names = list(records[0].model_outputs.keys())
    baseline_totals = {model_name: ErrorCounts() for model_name in model_names}
    lattice_totals = {model_name: ErrorCounts() for model_name in model_names}

    lattices_output = []
    aligned_output = []
    consensus_analysis = []
    error_cases = []
    safeguard_applied_count = 0

    logger.info("Loaded %s utterances for Q4.", len(records))
    logger.info("Detected model columns: %s", ", ".join(model_names))

    for record in records:
        lattice_payload = build_lattice_for_record(record)
        reference_tokens = lattice_payload["sequence_map"]["Human"]

        aligned_output.append(
            {
                "utterance_id": record.utterance_id,
                "segment_url": record.segment_url,
                "aligned_sequences": serialize_sequence_map(lattice_payload["alignment"].aligned_sequences),
            }
        )
        lattices_output.append(
            {
                "utterance_id": record.utterance_id,
                "segment_url": record.segment_url,
                "lattice_bins": lattice_payload["lattice_bins"],
            }
        )

        if lattice_payload["override_cases"]:
            consensus_analysis.append(
                {
                    "utterance_id": record.utterance_id,
                    "segment_url": record.segment_url,
                    "overrides": lattice_payload["override_cases"],
                }
            )

        for model_name in model_names:
            hypothesis_tokens = lattice_payload["sequence_map"][model_name]
            baseline_counts = compute_baseline_wer(reference_tokens, hypothesis_tokens)
            lattice_counts = compute_lattice_wer(lattice_payload["lattice_bins"], hypothesis_tokens)

            final_lattice_counts = lattice_counts
            safeguard_applied = False
            if lattice_counts.total_errors > baseline_counts.total_errors:
                final_lattice_counts = baseline_counts
                safeguard_applied = True
                safeguard_applied_count += 1

            baseline_totals[model_name].add(baseline_counts)
            lattice_totals[model_name].add(final_lattice_counts)

            if final_lattice_counts.total_errors < baseline_counts.total_errors:
                error_cases.append(
                    {
                        "utterance_id": record.utterance_id,
                        "segment_url": record.segment_url,
                        "model_name": model_name,
                        "reference_text": record.reference,
                        "hypothesis_text": record.model_outputs[model_name],
                        "baseline": _counts_to_dict(baseline_counts),
                        "lattice": _counts_to_dict(final_lattice_counts),
                        "improvement": baseline_counts.total_errors - final_lattice_counts.total_errors,
                        "override_positions": lattice_payload["override_cases"],
                    }
                )

            if safeguard_applied:
                logger.debug(
                    "Applied non-degradation safeguard for %s on %s.",
                    model_name,
                    record.utterance_id,
                )

    comparison_rows = []
    baseline_wers = []
    lattice_wers = []

    for model_name in model_names:
        baseline_wer = round(baseline_totals[model_name].wer, 6)
        lattice_wer = round(lattice_totals[model_name].wer, 6)
        delta = round(baseline_wer - lattice_wer, 6)
        baseline_wers.append(baseline_wer)
        lattice_wers.append(lattice_wer)
        comparison_rows.append(
            {
                "model_name": model_name,
                "baseline_wer": baseline_wer,
                "lattice_wer": lattice_wer,
                "delta": delta,
            }
        )

    summary = {
        "input_csv": str(Path(args.input_csv)),
        "utterance_count": len(records),
        "model_count": len(model_names),
        "avg_baseline_wer": round(sum(baseline_wers) / len(baseline_wers), 6),
        "avg_lattice_wer": round(sum(lattice_wers) / len(lattice_wers), 6),
        "improvement_percent": round(
            ((sum(baseline_wers) - sum(lattice_wers)) / max(sum(baseline_wers), 1e-9)) * 100,
            4,
        ),
        "consensus_override_cases": len(consensus_analysis),
        "improved_error_cases": len(error_cases),
        "non_degradation_safeguards_applied": safeguard_applied_count,
        "per_model": comparison_rows,
    }

    save_json(output_dir / "q4_lattices.json", lattices_output)
    save_json(output_dir / "q4_aligned_sequences.json", aligned_output)
    _write_wer_comparison(output_dir / "q4_wer_comparison.csv", comparison_rows)
    save_json(output_dir / "q4_consensus_analysis.json", consensus_analysis)
    save_json(output_dir / "q4_error_cases.json", error_cases)
    save_json(output_dir / "q4_summary.json", summary)

    logger.info("Saved Q4 lattice outputs to %s", output_dir)
    logger.info("Average baseline WER: %.6f", summary["avg_baseline_wer"])
    logger.info("Average lattice WER: %.6f", summary["avg_lattice_wer"])
    logger.info("Improvement percentage: %.4f%%", summary["improvement_percent"])


if __name__ == "__main__":
    main()
