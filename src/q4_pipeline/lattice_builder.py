from typing import Dict, List

from src.q4_pipeline.aligner import AlignmentResult, progressive_align
from src.q4_pipeline.consensus import apply_consensus
from src.q4_pipeline.utils import TaskRecord, merge_split_pairs, tokenize_text


def build_lattice_for_record(record: TaskRecord) -> Dict:
    sequence_map: Dict[str, List[str]] = {"Human": tokenize_text(record.reference)}
    for model_name, prediction in record.model_outputs.items():
        sequence_map[model_name] = tokenize_text(prediction)

    sequence_map = merge_split_pairs(sequence_map)
    alignment: AlignmentResult = progressive_align(sequence_map, anchor_name="Human")
    lattice_bins, override_cases = apply_consensus(
        alignment=alignment,
        reference_name="Human",
        model_names=list(record.model_outputs.keys()),
    )

    return {
        "utterance_id": record.utterance_id,
        "segment_url": record.segment_url,
        "reference": record.reference,
        "sequence_map": sequence_map,
        "alignment": alignment,
        "lattice_bins": lattice_bins,
        "override_cases": override_cases,
    }

