import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.report_plots.plot_utils import ensure_dir, load_json, repair_text, save_figure


EN_TAG_RE = re.compile(r"\[EN\](.*?)\[/EN\]")


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def generate_q1_plots(outputs_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    outputs_dir = Path(outputs_dir)
    figures_dir = ensure_dir(Path(figures_dir) / "q1")

    results = load_json(outputs_dir / "results.json")
    analysis = load_json(outputs_dir / "q1_d_to_g_analysis.json")

    saved_paths: list[Path] = []

    wer_df = pd.DataFrame(
        [
            {"stage": "Train Segment Baseline", "wer": results["baseline_segment_wer"]},
            {"stage": "FLEURS Baseline", "wer": results["fleurs_baseline_wer"]},
            {"stage": "Fine-Tuned\nBefore Fix", "wer": results["fleurs_fine_tuned_wer_before_fix"]},
            {"stage": "Fine-Tuned\nAfter Fix", "wer": results["fleurs_fine_tuned_wer_after_fix"]},
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#4C78A8", "#72B7B2", "#E45756", "#54A24B"]
    sns.barplot(data=wer_df, x="stage", y="wer", palette=palette, ax=ax)
    ax.set_title("Q1: WER Across Baseline, Fine-Tuning, and Decoding Fixes")
    ax.set_xlabel("")
    ax.set_ylabel("Word Error Rate (WER)")
    for patch, value in zip(ax.patches, wer_df["wer"]):
        ax.annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q1_wer_progression.png"))

    taxonomy_rows = []
    severity_rows = []
    for category, payload in analysis["taxonomy"].items():
        taxonomy_rows.append({"category": category, "count": payload["count"]})
        for example in payload.get("examples", []):
            metrics = example.get("metrics", {})
            severity_rows.append(
                {
                    "category": category,
                    "token_wer_proxy": metrics.get("token_wer_proxy", 0.0),
                }
            )

    taxonomy_df = pd.DataFrame(taxonomy_rows).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=taxonomy_df, x="count", y="category", palette="viridis", ax=ax)
    ax.set_title("Q1: Error Taxonomy Frequency")
    ax.set_xlabel("Count in Sampled Error Set")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(
            f"{int(width)}",
            (width, patch.get_y() + patch.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=10,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q1_error_taxonomy.png"))

    severity_df = pd.DataFrame(severity_rows)
    severity_df = (
        severity_df.groupby("category", as_index=False)["token_wer_proxy"]
        .mean()
        .sort_values("token_wer_proxy", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=severity_df, x="token_wer_proxy", y="category", palette="magma", ax=ax)
    ax.set_title("Q1: Mean Severity by Error Category")
    ax.set_xlabel("Mean Token-Level WER Proxy")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(
            f"{width:.2f}",
            (width, patch.get_y() + patch.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=10,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q1_error_severity.png"))

    return saved_paths


def generate_q2_plots(outputs_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    outputs_dir = Path(outputs_dir)
    figures_dir = ensure_dir(Path(figures_dir) / "q2")

    q2_results = load_json(outputs_dir / "q2_results.json")
    saved_paths: list[Path] = []

    stats = {
        "Number normalization": 0,
        "English tagging": 0,
        "Any cleanup": 0,
        "Unchanged": 0,
    }
    overlap_counter = Counter()
    term_counter = Counter()

    for item in q2_results:
        raw_prediction = repair_text(item.get("raw_prediction", ""))
        normalized = repair_text(item.get("normalized", ""))
        tagged = repair_text(item.get("tagged", ""))

        number_changed = raw_prediction != normalized
        english_changed = normalized != tagged
        any_changed = raw_prediction != tagged

        stats["Number normalization"] += int(number_changed)
        stats["English tagging"] += int(english_changed)
        stats["Any cleanup"] += int(any_changed)
        stats["Unchanged"] += int(not any_changed)

        if number_changed and english_changed:
            overlap_counter["Both"] += 1
        elif number_changed:
            overlap_counter["Number only"] += 1
        elif english_changed:
            overlap_counter["English only"] += 1
        else:
            overlap_counter["No change"] += 1

        for term in EN_TAG_RE.findall(tagged):
            normalized_term = repair_text(term).strip()
            if normalized_term:
                term_counter[normalized_term] += 1

    coverage_df = pd.DataFrame(
        [{"metric": metric, "count": count} for metric, count in stats.items()]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=coverage_df, x="metric", y="count", palette=["#4C78A8", "#F58518", "#54A24B", "#B279A2"], ax=ax)
    ax.set_title("Q2: Cleanup Coverage Across Raw ASR Predictions")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Utterances")
    for patch, value in zip(ax.patches, coverage_df["count"]):
        ax.annotate(
            f"{int(value)}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q2_cleanup_coverage.png"))

    overlap_df = pd.DataFrame(
        [{"group": group, "count": count} for group, count in overlap_counter.items()]
    ).sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=overlap_df, x="group", y="count", palette="crest", ax=ax)
    ax.set_title("Q2: Overlap of Number Normalization and English Tagging")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Utterances")
    for patch, value in zip(ax.patches, overlap_df["count"]):
        ax.annotate(
            f"{int(value)}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q2_cleanup_overlap.png"))

    top_terms = pd.DataFrame(term_counter.most_common(12), columns=["term", "count"])
    if not top_terms.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=top_terms, x="count", y="term", palette="flare", ax=ax)
        ax.set_title("Q2: Most Frequently Tagged English Terms")
        ax.set_xlabel("Tag Frequency")
        ax.set_ylabel("")
        for patch in ax.patches:
            width = patch.get_width()
            ax.annotate(
                f"{int(width)}",
                (width, patch.get_y() + patch.get_height() / 2.0),
                ha="left",
                va="center",
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=10,
            )
        saved_paths.append(save_figure(fig, figures_dir / "q2_top_english_terms.png"))

    return saved_paths


def generate_q3_plots(outputs_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    outputs_dir = Path(outputs_dir)
    figures_dir = ensure_dir(Path(figures_dir) / "q3")

    q3_summary = load_json(outputs_dir / "q3_summary.json")
    q3_eval = load_json(outputs_dir / "q3_evaluation.json")
    q3_results = _safe_read_csv(outputs_dir / "q3_results.csv")
    saved_paths: list[Path] = []

    label_df = pd.DataFrame(
        [
            {"label": "Correct", "count": q3_summary["correct_spelling_count"]},
            {"label": "Incorrect", "count": q3_summary["incorrect_spelling_count"]},
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2E8B57", "#C44E52"]
    wedges, texts, autotexts = ax.pie(
        label_df["count"],
        labels=label_df["label"],
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
        textprops={"fontsize": 12},
    )
    ax.set_title("Q3: Correct vs Incorrect Spelling Distribution")
    centre_text = f"Total\n{q3_summary['total_words']:,}"
    ax.text(0, 0, centre_text, ha="center", va="center", fontsize=16, fontweight="bold")
    saved_paths.append(save_figure(fig, figures_dir / "q3_label_distribution.png"))

    confidence_df = (
        q3_results.groupby(["label", "confidence"]).size().reset_index(name="count")
    )
    confidence_order = ["high", "medium", "low"]
    label_order = ["correct_spelling", "incorrect_spelling"]
    confidence_df["confidence"] = pd.Categorical(confidence_df["confidence"], confidence_order, ordered=True)
    confidence_df["label"] = pd.Categorical(confidence_df["label"], label_order, ordered=True)
    confidence_df = confidence_df.sort_values(["label", "confidence"])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=confidence_df,
        x="label",
        y="count",
        hue="confidence",
        palette=["#54A24B", "#ECA82C", "#B279A2"],
        ax=ax,
    )
    ax.set_title("Q3: Confidence Distribution by Spelling Label")
    ax.set_xlabel("")
    ax.set_ylabel("Word Count")
    ax.set_xticklabels(["Correct spelling", "Incorrect spelling"])
    ax.legend(title="Confidence")
    saved_paths.append(save_figure(fig, figures_dir / "q3_confidence_by_label.png"))

    source_df = pd.DataFrame(
        [
            {"source": source.replace("_", " "), "count": count}
            for source, count in q3_summary["classification_source_distribution"].items()
        ]
    ).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=source_df, x="count", y="source", palette="viridis", ax=ax)
    ax.set_title("Q3: Classification Source Distribution")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(
            f"{int(width):,}",
            (width, patch.get_y() + patch.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q3_classification_sources.png"))

    audit_df = pd.DataFrame(
        [
            {"outcome": "System right", "count": q3_eval["system_right"]},
            {"outcome": "System wrong", "count": q3_eval["system_wrong"]},
            {"outcome": "Skipped", "count": q3_eval["skipped_samples"]},
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=audit_df, x="outcome", y="count", palette=["#54A24B", "#E45756", "#9D9D9D"], ax=ax)
    fig.suptitle(
        "Q3: Manual Audit of Low-Confidence Bucket",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Reviewed Samples")
    subtitle = f"Low-confidence accuracy: {q3_eval['low_confidence_accuracy']:.2%}"
    ax.set_title(subtitle, fontsize=11, color="#444444", pad=12, loc="left")
    for patch, value in zip(ax.patches, audit_df["count"]):
        ax.annotate(
            f"{int(value)}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
        )
    fig.subplots_adjust(top=0.84)
    saved_paths.append(save_figure(fig, figures_dir / "q3_low_confidence_audit.png"))

    return saved_paths


def generate_q4_plots(outputs_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    outputs_dir = Path(outputs_dir)
    figures_dir = ensure_dir(Path(figures_dir) / "q4")

    q4_summary = load_json(outputs_dir / "q4" / "q4_summary.json")
    q4_comparison = _safe_read_csv(outputs_dir / "q4" / "q4_wer_comparison.csv")
    q4_error_cases = load_json(outputs_dir / "q4" / "q4_error_cases.json")
    saved_paths: list[Path] = []

    comparison_long = q4_comparison.melt(
        id_vars="model_name",
        value_vars=["baseline_wer", "lattice_wer"],
        var_name="metric",
        value_name="wer",
    )
    comparison_long["metric"] = comparison_long["metric"].map(
        {"baseline_wer": "Baseline WER", "lattice_wer": "Lattice WER"}
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=comparison_long,
        x="model_name",
        y="wer",
        hue="metric",
        palette=["#E45756", "#4C78A8"],
        ax=ax,
    )
    ax.set_title("Q4: Baseline vs Lattice-Aware WER by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("WER")
    ax.legend(title="")
    saved_paths.append(save_figure(fig, figures_dir / "q4_wer_comparison.png"))

    delta_df = q4_comparison.sort_values("delta", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=delta_df, x="model_name", y="delta", palette="rocket", ax=ax)
    ax.set_title("Q4: Absolute WER Improvement from Lattice Evaluation")
    ax.set_xlabel("Model")
    ax.set_ylabel("Baseline WER - Lattice WER")
    for patch, value in zip(ax.patches, delta_df["delta"]):
        ax.annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
        )
    saved_paths.append(save_figure(fig, figures_dir / "q4_delta_improvement.png"))

    improved_case_counts = Counter(case["model_name"] for case in q4_error_cases)
    case_df = pd.DataFrame(
        [{"model_name": model_name, "improved_cases": count} for model_name, count in improved_case_counts.items()]
    ).sort_values("improved_cases", ascending=False)
    if not case_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=case_df, x="model_name", y="improved_cases", palette="crest", ax=ax)
        fig.suptitle(
            "Q4: Number of Utterances Helped by Lattice Scoring",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Improved Utterance Cases")
        info = (
            f"Overrides: {q4_summary['consensus_override_cases']}   "
            f"Safeguards: {q4_summary['non_degradation_safeguards_applied']}   "
            f"Avg improvement: {q4_summary['improvement_percent']:.1f}%"
        )
        ax.set_title(info, fontsize=11, color="#444444", pad=12, loc="left")
        for patch, value in zip(ax.patches, case_df["improved_cases"]):
            ax.annotate(
                f"{int(value)}",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=10,
            )
        fig.subplots_adjust(top=0.84)
        saved_paths.append(save_figure(fig, figures_dir / "q4_improved_cases.png"))

    return saved_paths


def generate_all_plots(outputs_dir: str | Path = "outputs", figures_dir: str | Path = "outputs/figures") -> dict[str, list[str]]:
    outputs_dir = Path(outputs_dir)
    figures_dir = Path(figures_dir)

    generated = {
        "q1": [str(path) for path in generate_q1_plots(outputs_dir, figures_dir)],
        "q2": [str(path) for path in generate_q2_plots(outputs_dir, figures_dir)],
        "q3": [str(path) for path in generate_q3_plots(outputs_dir, figures_dir)],
        "q4": [str(path) for path in generate_q4_plots(outputs_dir, figures_dir)],
    }
    return generated
