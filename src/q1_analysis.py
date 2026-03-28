import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from src.postprocess import collapse_repetition_loops, normalize_spaces, repair_mojibake, repetition_metrics


COMPARISON_PUNCT_RE = re.compile(r"[^0-9A-Za-z\u0900-\u097F\s]")
NUMERIC_TOKEN_RE = re.compile(r"\d[\d,]*")


@dataclass
class ErrorMetrics:
    ref_tokens: int
    pred_tokens: int
    token_edit_distance: int
    token_wer_proxy: float
    char_edit_ratio: float
    length_ratio: float
    overlap_ratio: float
    repeated_token_ratio: float
    max_token_run: int
    max_ngram_repeat: int
    digit_mismatch: bool
    max_unmatched_ref_token_length: int
    max_unmatched_pred_token_length: int


@dataclass
class ErrorSample:
    index: int
    prediction: str
    reference: str
    metrics: ErrorMetrics
    category: str
    error_tag: str
    reason: str


def _comparison_tokens(text):
    text = normalize_spaces(text)
    text = unicodedata.normalize("NFKC", text)
    text = COMPARISON_PUNCT_RE.sub(" ", text)
    return [token for token in text.split() if token]


def _word_edit_distance(left_tokens, right_tokens):
    rows = len(left_tokens) + 1
    cols = len(right_tokens) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if left_tokens[row - 1] == right_tokens[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + substitution_cost,
            )

    return matrix[-1][-1]


def _char_edit_distance(left_text, right_text):
    rows = len(left_text) + 1
    cols = len(right_text) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if left_text[row - 1] == right_text[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + substitution_cost,
            )

    return matrix[-1][-1]


def _overlap_ratio(pred_tokens, ref_tokens):
    if not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    shared = sum((pred_counter & ref_counter).values())
    return shared / len(ref_tokens)


def _has_digit_mismatch(prediction, reference):
    pred_numbers = NUMERIC_TOKEN_RE.findall(normalize_spaces(prediction))
    ref_numbers = NUMERIC_TOKEN_RE.findall(normalize_spaces(reference))
    return bool(pred_numbers or ref_numbers) and pred_numbers != ref_numbers


def _max_unmatched_token_lengths(pred_tokens, ref_tokens):
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    pred_only = list((pred_counter - ref_counter).elements())
    ref_only = list((ref_counter - pred_counter).elements())

    return (
        max((len(token) for token in ref_only), default=0),
        max((len(token) for token in pred_only), default=0),
    )


def _classify_error(metrics):
    if (
        metrics.repeated_token_ratio >= 0.18
        or metrics.max_token_run >= 4
        or metrics.max_ngram_repeat >= 3
        or metrics.length_ratio >= 1.55
    ):
        return (
            "Repetition / hallucination",
            "decoder_loop",
            "The prediction follows the reference for a short prefix, then enters a repetition loop and keeps emitting the same token or phrase.",
        )

    if metrics.digit_mismatch:
        return (
            "Number / numeral mismatch",
            "numeral_surface_form",
            "The sentence scaffold is broadly similar, but numeric tokens are substituted or reformatted incorrectly.",
        )

    if metrics.length_ratio <= 0.78:
        return (
            "Deletion / truncation",
            "content_drop",
            "The hypothesis is substantially shorter than the reference and drops a trailing content span.",
        )

    if (
        metrics.overlap_ratio >= 0.70
        and metrics.char_edit_ratio <= 0.35
        and 0.85 <= metrics.length_ratio <= 1.20
    ):
        return (
            "Surface-form / normalization mismatch",
            "surface_form_variation",
            "The prediction stays close to the reference but differs in spelling, normalized form, or inflectional surface form.",
        )

    if (
        metrics.overlap_ratio >= 0.45
        and max(metrics.max_unmatched_ref_token_length, metrics.max_unmatched_pred_token_length) >= 6
    ):
        return (
            "Rare content-word confusion",
            "rare_word_swap",
            "The sentence frame is preserved, but one or two long content words are replaced with acoustically plausible alternatives.",
        )

    if metrics.overlap_ratio >= 0.35:
        return (
            "Lexical / phonetic substitution",
            "content_word_substitution",
            "Most of the sentence frame is preserved, but content words are swapped with acoustically similar alternatives.",
        )

    return (
        "Semantic drift / insertion",
        "off_target_drift",
        "The decoder diverges from the reference early and inserts off-target content instead of making a local substitution.",
    )


def _build_metrics(prediction, reference):
    normalized_prediction = normalize_spaces(prediction)
    normalized_reference = normalize_spaces(reference)
    pred_tokens = _comparison_tokens(normalized_prediction)
    ref_tokens = _comparison_tokens(normalized_reference)
    edit_distance = _word_edit_distance(pred_tokens, ref_tokens)
    char_edit_distance = _char_edit_distance(
        normalized_prediction.replace(" ", ""),
        normalized_reference.replace(" ", ""),
    )
    repetition = repetition_metrics(prediction)
    max_unmatched_ref_token_length, max_unmatched_pred_token_length = _max_unmatched_token_lengths(
        pred_tokens,
        ref_tokens,
    )

    return ErrorMetrics(
        ref_tokens=len(ref_tokens),
        pred_tokens=len(pred_tokens),
        token_edit_distance=edit_distance,
        token_wer_proxy=edit_distance / max(1, len(ref_tokens)),
        char_edit_ratio=char_edit_distance / max(1, len(normalized_reference.replace(" ", ""))),
        length_ratio=len(pred_tokens) / max(1, len(ref_tokens)),
        overlap_ratio=_overlap_ratio(pred_tokens, ref_tokens),
        repeated_token_ratio=repetition.repeated_token_ratio,
        max_token_run=repetition.max_token_run,
        max_ngram_repeat=repetition.max_ngram_repeat,
        digit_mismatch=_has_digit_mismatch(prediction, reference),
        max_unmatched_ref_token_length=max_unmatched_ref_token_length,
        max_unmatched_pred_token_length=max_unmatched_pred_token_length,
    )


def load_results(results_path):
    with Path(results_path).open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_error_samples(error_samples_path):
    with Path(error_samples_path).open("r", encoding="utf-8") as file_obj:
        raw_samples = json.load(file_obj)

    samples = []
    for index, raw_sample in enumerate(raw_samples, start=1):
        if isinstance(raw_sample, dict):
            prediction = raw_sample.get("prediction", "")
            reference = raw_sample.get("reference", "")
        else:
            prediction, reference = raw_sample

        prediction = repair_mojibake(prediction)
        reference = repair_mojibake(reference)
        metrics = _build_metrics(prediction, reference)
        category, error_tag, reason = _classify_error(metrics)
        samples.append(
            ErrorSample(
                index=index,
                prediction=prediction,
                reference=reference,
                metrics=metrics,
                category=category,
                error_tag=error_tag,
                reason=reason,
            )
        )

    return samples


def sampling_strategy_paragraph(sample_count=25):
    return (
        f"The {sample_count} examples were selected by uniform interval sampling over the full error list, "
        "not by manual cherry-picking. The pipeline first keeps every evaluation pair where prediction and "
        "reference differ, computes a fixed stride of floor(total_errors / 25), and then returns every stride-th "
        "error from that ordered list. This spreads the sample across the evaluation set and makes it less likely "
        "that the report over-focuses on any single utterance, topic, or failure mode."
    )


def taxonomy(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.category].append(sample)

    for category_samples in grouped.values():
        category_samples.sort(key=lambda sample: sample.metrics.token_wer_proxy, reverse=True)

    return dict(sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True))


def category_fix(category):
    fixes = {
        "Repetition / hallucination": (
            "Apply repetition-aware decoding guards and a post-decoding repetition collapse.",
            "This directly targets runaway loops by preventing repeated n-grams during decoding and removing repeated tails after inference.",
        ),
        "Deletion / truncation": (
            "Use beam search with a stronger length penalty and reject hypotheses whose token count is far below the acoustic duration prior.",
            "This makes premature end-of-sequence decisions less attractive and reduces content drops on longer utterances.",
        ),
        "Lexical / phonetic substitution": (
            "Align training/evaluation text normalization and inject a domain hotword list for rare content words, names, and technical terms.",
            "This reduces mismatch between label forms and biases the decoder toward acoustically plausible domain words instead of generic near-sounds.",
        ),
        "Surface-form / normalization mismatch": (
            "Canonicalize both references and predictions with the same Hindi text normalization pipeline before decode-time comparison and final scoring.",
            "When the model is nearly correct but differs in surface form, a shared normalizer removes avoidable penalties without touching the acoustic model.",
        ),
        "Rare content-word confusion": (
            "Add domain hotword biasing or shallow-fusion vocabulary support for rare nouns, names, and technical terms at decode time.",
            "This targets the exact words the model confuses most often, and it is cheaper and faster to deploy than retraining the full ASR model.",
        ),
        "Number / numeral mismatch": (
            "Normalize numeric forms consistently before scoring and add a numeral post-processor for digits, commas, and Hindi number surface forms.",
            "Many number errors are formatting-level mistakes; a deterministic normalizer corrects them without retraining the acoustic model.",
        ),
        "Semantic drift / insertion": (
            "Add hypothesis-level rejection based on extreme length growth and low token overlap, then fall back to a shorter alternative beam.",
            "This suppresses off-target continuations where the decoder leaves the acoustically supported region and starts inventing content.",
        ),
    }
    return fixes[category]


def top_fixes(grouped_taxonomy, limit=5):
    fixes = []
    for category, category_samples in list(grouped_taxonomy.items())[:limit]:
        fix, why = category_fix(category)
        fixes.append(
            {
                "error_type": category,
                "count": len(category_samples),
                "fix": fix,
                "why_it_works": why,
            }
        )
    return fixes


def build_postprocess_demo(samples, demo_size=8):
    repetition_samples = [sample for sample in samples if sample.category == "Repetition / hallucination"]
    repetition_samples.sort(
        key=lambda sample: (
            sample.metrics.repeated_token_ratio,
            sample.metrics.max_ngram_repeat,
            sample.metrics.token_wer_proxy,
        ),
        reverse=True,
    )

    selected = repetition_samples[:demo_size]
    if not selected:
        selected = sorted(samples, key=lambda sample: sample.metrics.token_wer_proxy, reverse=True)[:demo_size]

    demo_rows = []
    improved = 0
    unchanged = 0
    worsened = 0
    before_total = 0.0
    after_total = 0.0

    for sample in selected:
        fixed_prediction = collapse_repetition_loops(sample.prediction)
        before_metrics = _build_metrics(sample.prediction, sample.reference)
        after_metrics = _build_metrics(fixed_prediction, sample.reference)

        if after_metrics.token_edit_distance < before_metrics.token_edit_distance:
            improved += 1
        elif after_metrics.token_edit_distance > before_metrics.token_edit_distance:
            worsened += 1
        else:
            unchanged += 1

        before_total += before_metrics.token_wer_proxy
        after_total += after_metrics.token_wer_proxy

        demo_rows.append(
            {
                "index": sample.index,
                "reference": sample.reference,
                "before": sample.prediction,
                "after": fixed_prediction,
                "before_token_wer_proxy": round(before_metrics.token_wer_proxy, 4),
                "after_token_wer_proxy": round(after_metrics.token_wer_proxy, 4),
            }
        )

    summary = {
        "selected_count": len(demo_rows),
        "selection_rule": "Top repetition-heavy errors ranked by repeated-token ratio, repeated n-gram count, and token-level WER proxy.",
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
        "average_before_token_wer_proxy": round(before_total / max(1, len(demo_rows)), 4),
        "average_after_token_wer_proxy": round(after_total / max(1, len(demo_rows)), 4),
    }

    return summary, demo_rows


def structured_analysis(samples, results):
    grouped = taxonomy(samples)
    fixes = top_fixes(grouped, limit=5)
    postprocess_summary, postprocess_rows = build_postprocess_demo(samples)

    return {
        "results": results,
        "sampling_strategy": sampling_strategy_paragraph(sample_count=len(samples)),
        "wer_diagnosis": (
            "The fine-tuned model performs worse than the pretrained baseline because it appears to overfit to the Josh Talks training distribution. "
            "The training data is conversational, disfluent, and relatively small, while Hindi FLEURS test is cleaner read speech. "
            "The sampled errors show decoder loops, long hallucinated tails, and domain-mismatched substitutions, which means the model learned the training-set style "
            "but lost some of the pretrained model's broader generalization. In short: training loss improved, but cross-domain robustness degraded."
        ),
        "taxonomy": {
            category: {
                "count": len(category_samples),
                "examples": [
                    {
                        "index": sample.index,
                        "reference": sample.reference,
                        "prediction": sample.prediction,
                        "error_tag": sample.error_tag,
                        "reason": sample.reason,
                        "metrics": asdict(sample.metrics),
                    }
                    for sample in category_samples
                ],
            }
            for category, category_samples in grouped.items()
        },
        "top_fixes": fixes,
        "postprocess_demo": {
            "summary": postprocess_summary,
            "rows": postprocess_rows,
        },
    }


def _truncate(text, limit=240):
    text = normalize_spaces(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def render_markdown_report(analysis):
    total_examples = sum(item["count"] for item in analysis["taxonomy"].values())
    lines = []
    lines.append("# Q1(d) Sampling Strategy")
    lines.append("")
    lines.append(analysis["sampling_strategy"])
    lines.append("")

    lines.append("# WER Diagnosis")
    lines.append("")
    lines.append(analysis["wer_diagnosis"])
    lines.append("")

    lines.append("# Q1(e) Error Taxonomy")
    lines.append("")
    for category, payload in analysis["taxonomy"].items():
        lines.append(f"## {category} ({payload['count']}/{total_examples})")
        lines.append("")
        if category == "Repetition / hallucination":
            description = "Predictions start plausibly but then repeat the same token or short phrase, creating long hallucinated tails."
        elif category == "Deletion / truncation":
            description = "Predictions are materially shorter than the reference and drop a content-bearing suffix."
        elif category == "Lexical / phonetic substitution":
            description = "The sentence frame is broadly intact, but content words are replaced by acoustically similar alternatives."
        elif category == "Surface-form / normalization mismatch":
            description = "The prediction is close to the reference, but differs in surface spelling, normalization, or inflectional form."
        elif category == "Rare content-word confusion":
            description = "The model preserves the sentence scaffold but substitutes one or two rare or content-heavy words."
        elif category == "Number / numeral mismatch":
            description = "Numeric content is mistranscribed or reformatted even when the rest of the sentence is relatively aligned."
        else:
            description = "The decoder drifts off the reference and inserts unrelated content rather than making a local substitution."
        lines.append(f"Description: {description}")
        lines.append("")

        for example in payload["examples"][:5]:
            lines.append(f"- Reference: {_truncate(example['reference'])}")
            lines.append(f"- Prediction: {_truncate(example['prediction'])}")
            lines.append(f"- Error Type: {example['error_tag']}")
            lines.append(f"- Reason: {example['reason']}")
            lines.append("")

    lines.append("# Q1(f) Actionable Fixes")
    lines.append("")
    lines.append(
        "The first three fixes below are the highest-priority fixes by category frequency. The next two are supporting fixes "
        "that deepen the engineering diagnosis and can still be deployed without retraining."
    )
    lines.append("")
    for fix in analysis["top_fixes"]:
        lines.append(f"- Error Type: {fix['error_type']}")
        lines.append(f"Fix: {fix['fix']}")
        lines.append(f"Why it works: {fix['why_it_works']}")
        lines.append("")

    lines.append("# Q1(g) Fix Implementation")
    lines.append("")
    lines.append(
        "Implemented fix: a no-retraining repetition-collapsing post-processor that detects repeated tokens or repeated short n-grams "
        "and collapses runaway loops back to a single occurrence."
    )
    lines.append("")
    lines.append(
        "This is a production-grade mitigation because it can be inserted directly into the inference path, validated quickly, and rolled back safely. "
        "By contrast, retraining the ASR model is slower, more expensive, and requires a full re-evaluation cycle."
    )
    lines.append("")

    summary = analysis["postprocess_demo"]["summary"]
    lines.append(
        f"Subset selection: {summary['selection_rule']} "
        f"Selected {summary['selected_count']} samples, improved {summary['improved']}, unchanged {summary['unchanged']}, worsened {summary['worsened']}. "
        f"Average token-WER proxy moved from {summary['average_before_token_wer_proxy']:.4f} to {summary['average_after_token_wer_proxy']:.4f}."
    )
    lines.append("")
    lines.append("| Reference | Before | After |")
    lines.append("| --- | --- | --- |")

    for row in analysis["postprocess_demo"]["rows"]:
        reference = _truncate(row["reference"], limit=180).replace("|", "\\|")
        before = _truncate(row["before"], limit=180).replace("|", "\\|")
        after = _truncate(row["after"], limit=180).replace("|", "\\|")
        lines.append(f"| {reference} | {before} | {after} |")

    lines.append("")
    return "\n".join(lines)


def write_outputs(error_samples_path, results_path, markdown_output_path, json_output_path):
    results = load_results(results_path)
    samples = load_error_samples(error_samples_path)
    analysis = structured_analysis(samples, results)

    markdown_output_path = Path(markdown_output_path)
    json_output_path = Path(json_output_path)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_output_path.write_text(render_markdown_report(analysis), encoding="utf-8")
    json_output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
