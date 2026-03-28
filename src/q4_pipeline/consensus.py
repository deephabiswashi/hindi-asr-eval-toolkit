from collections import Counter
from typing import Dict, List, Optional

from src.q4_pipeline.aligner import AlignmentResult
from src.q4_pipeline.utils import tokens_are_similar


def _cluster_position_tokens(position_tokens: Dict[str, Optional[str]]) -> List[Dict]:
    groups: List[Dict] = []

    for source_name, token in position_tokens.items():
        if not token:
            continue

        assigned = False
        for group in groups:
            if any(tokens_are_similar(token, existing) for existing in group["tokens"]):
                group["tokens"].append(token)
                group["sources"].append(source_name)
                if source_name != "Human":
                    group["model_sources"].append(source_name)
                assigned = True
                break

        if assigned:
            continue

        groups.append(
            {
                "representative": token,
                "tokens": [token],
                "sources": [source_name],
                "model_sources": [] if source_name == "Human" else [source_name],
            }
        )

    for group in groups:
        group["token_counts"] = dict(Counter(group["tokens"]))
        group["model_support"] = len(set(group["model_sources"]))
        group["sources"] = sorted(set(group["sources"]))
        group["model_sources"] = sorted(set(group["model_sources"]))

    return groups


def apply_consensus(
    alignment: AlignmentResult,
    reference_name: str,
    model_names: List[str],
    model_trust_threshold: int = 3,
) -> tuple[list[dict], list[dict]]:
    lattice_bins: List[Dict] = []
    override_cases: List[Dict] = []

    total_positions = len(alignment.bins)
    for position in range(total_positions):
        position_tokens = {
            sequence_name: aligned_tokens[position]
            for sequence_name, aligned_tokens in alignment.aligned_sequences.items()
        }
        reference_token = position_tokens.get(reference_name)
        groups = _cluster_position_tokens(position_tokens)

        trusted_groups = [group for group in groups if group["model_support"] >= model_trust_threshold]
        reference_group = next(
            (group for group in groups if reference_token and reference_token in group["tokens"]),
            None,
        )
        reference_support = reference_group["model_support"] if reference_group else 0
        weak_reference = bool(reference_token) and reference_support < 2

        if trusted_groups:
            accepted_tokens = sorted({token for group in trusted_groups for token in group["tokens"]})
        else:
            accepted_tokens = sorted({token for group in groups for token in group["tokens"]})

        trusted_by_models = bool(trusted_groups)
        reference_overridden = bool(
            trusted_by_models
            and weak_reference
            and reference_token
            and reference_token not in accepted_tokens
        )

        lattice_bin = {
            "position": position,
            "reference_token": reference_token,
            "candidate_tokens": sorted({token for group in groups for token in group["tokens"]}),
            "accepted_tokens": accepted_tokens,
            "weak_reference": weak_reference,
            "trusted_by_models": trusted_by_models,
            "groups": groups,
        }
        lattice_bins.append(lattice_bin)

        if reference_overridden:
            winning_groups = [
                {
                    "representative": group["representative"],
                    "tokens": sorted(set(group["tokens"])),
                    "supporting_models": group["model_sources"],
                }
                for group in trusted_groups
            ]
            override_cases.append(
                {
                    "position": position,
                    "reference_token": reference_token,
                    "reference_support": reference_support,
                    "winning_groups": winning_groups,
                }
            )

    return lattice_bins, override_cases

