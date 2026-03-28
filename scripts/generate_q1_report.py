import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.q1_analysis import write_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate submission-ready Q1(d)-Q1(g) outputs from existing Whisper ASR artifacts."
    )
    parser.add_argument(
        "--error-samples",
        default="outputs/error_samples.json",
        help="Path to the sampled error JSON created by the pipeline.",
    )
    parser.add_argument(
        "--results",
        default="outputs/results.json",
        help="Path to the results JSON created by the pipeline.",
    )
    parser.add_argument(
        "--markdown-output",
        default="outputs/q1_d_to_g_report.md",
        help="Where to write the submission-ready markdown report.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/q1_d_to_g_analysis.json",
        help="Where to write the structured analysis JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_outputs(
        error_samples_path=args.error_samples,
        results_path=args.results,
        markdown_output_path=args.markdown_output,
        json_output_path=args.json_output,
    )


if __name__ == "__main__":
    main()
