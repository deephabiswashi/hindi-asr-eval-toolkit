import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.report_plots.assignment_plots import (
    generate_q1_plots,
    generate_q2_plots,
    generate_q3_plots,
    generate_q4_plots,
)
from src.report_plots.plot_utils import ensure_dir, setup_logging, setup_plot_style


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report-ready figures for Q1 to Q4.")
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing generated JSON/CSV outputs from the pipelines.",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Directory where PNG figures will be saved.",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        choices=["q1", "q2", "q3", "q4", "all"],
        default=["all"],
        help="Which question figures to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    setup_plot_style()

    outputs_dir = Path(args.outputs_dir)
    figures_dir = ensure_dir(args.figures_dir)

    selected = {"q1", "q2", "q3", "q4"} if "all" in args.questions else set(args.questions)
    generated: dict[str, list[str]] = {}

    plotters = {
        "q1": generate_q1_plots,
        "q2": generate_q2_plots,
        "q3": generate_q3_plots,
        "q4": generate_q4_plots,
    }

    for question in ("q1", "q2", "q3", "q4"):
        if question not in selected:
            continue

        try:
            generated[question] = [str(path) for path in plotters[question](outputs_dir, figures_dir)]
            logger.info("Generated %s figures for %s.", len(generated[question]), question.upper())
        except FileNotFoundError as exc:
            logger.warning("Skipping %s because a required output file is missing: %s", question.upper(), exc)
        except Exception as exc:
            logger.exception("Failed while generating %s figures: %s", question.upper(), exc)

    manifest_path = Path(figures_dir) / "plot_manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")

    total_figures = sum(len(paths) for paths in generated.values())
    logger.info("Generated %s figures across %s question groups.", total_figures, len(generated))
    logger.info("Saved plot manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
