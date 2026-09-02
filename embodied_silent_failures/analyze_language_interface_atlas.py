from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.provenance import file_sha256, load_json


VECTOR_FAMILIES = (
    "injection_residual_delta",
    "immediate_residual_delta",
    "immediate_key_delta",
    "immediate_value_delta",
    "final_residual_delta",
    "safe_feature_delta",
    "action_logit_delta",
)

ANALYSIS_ARRAYS = (
    "record_id",
    "injection_residual_delta",
    "immediate_residual_delta",
    "immediate_key_delta",
    "immediate_value_delta",
    "final_residual_delta",
    "safe_feature_delta",
    "action_logit_delta",
    "command_delta",
    "eligible_causal_outcome",
    "command_changed",
    "task_failure",
    "safe_alarm_at_fault",
    "safe_alarm_within_10",
    "safe_alarm_post_fault_any",
    "operational_silent_failure",
    "score_at_fault",
    "control_score_at_fault",
    "score_change_from_control_at_fault",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reduced OpenVLA interface cuts on CPU."
    )
    parser.add_argument(
        "--atlas-dir", action="append", dest="atlas_dirs", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observations-csv", required=True, type=Path)
    parser.add_argument("--head-spectra-csv", required=True, type=Path)
    return parser.parse_args()


def _component_count(energy: Any, threshold: float) -> int | None:
    total = float(energy.sum())
    if total <= 0:
        return None
    return int((energy.cumsum() < total * threshold).sum() + 1)


def _spectrum(np: Any, matrix: Any, *, centered: bool) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"spectrum input must be a matrix, got {values.shape}")
    if centered:
        values = values - values.mean(axis=0, keepdims=True)
    gram = values @ values.T
    energy = np.linalg.eigvalsh(gram.astype(np.float64))[::-1]
    energy = np.maximum(energy, 0.0)
    total = float(energy.sum())
    if total <= 0:
        return {
            "total_energy": 0.0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "components_90": None,
            "components_95": None,
            "components_99": None,
            "energy_fractions": [],
        }
    probability = energy[energy > 0] / total
    effective_rank = math.exp(float(-(probability * np.log(probability)).sum()))
    return {
        "total_energy": total,
        "effective_rank": effective_rank,
        "stable_rank": total / float(energy[0]),
        "components_90": _component_count(energy, 0.90),
        "components_95": _component_count(energy, 0.95),
        "components_99": _component_count(energy, 0.99),
        "energy_fractions": (energy / total).tolist(),
    }


def spectrum_summary(np: Any, matrix: Any) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    return {
        "samples": int(values.shape[0]),
        "dimensions": int(values.shape[1]),
        "nonzero_samples": int((norms > 0).sum()),
        "norm": {
            "minimum": float(norms.min()),
            "median": float(np.median(norms)),
            "maximum": float(norms.max()),
        },
        "uncentered": _spectrum(np, values, centered=False),
        "centered": _spectrum(np, values, centered=True),
    }


def _cosine(np: Any, left: Any, right: Any) -> Any:
    numerator = (left * right).sum(axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float32),
        where=denominator > 0,
    )


def _load_atlases(
    np: Any, atlas_dirs: list[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs = [load_json(path / "run.json") for path in atlas_dirs]
    shards = [int(value["worker_shard"]) for value in runs]
    if len(shards) != len(set(shards)):
        raise ValueError("atlas inputs repeat a worker shard")
    if {value["analysis_split"] for value in runs} != {"development"}:
        raise ValueError("exploratory atlas analysis only accepts development data")
    extractor_hashes = {value["code"]["extractor_sha256"] for value in runs}
    if len(extractor_hashes) != 1:
        raise ValueError("atlas shards were produced by different extractors")

    contexts = []
    errors = []
    for atlas_dir, run in zip(atlas_dirs, runs, strict=True):
        for context_id in run["context_ids"]:
            result_dir = atlas_dir / "contexts" / str(context_id)
            complete_path = result_dir / "context.complete.json"
            if not complete_path.is_file():
                error_path = result_dir / "context.error.json"
                errors.append(
                    load_json(error_path)
                    if error_path.is_file()
                    else {"context_id": str(context_id), "status": "missing"}
                )
                continue
            complete = load_json(complete_path)
            artifact_path = result_dir / str(complete["artifact"]["name"])
            if not artifact_path.is_file():
                errors.append(
                    {
                        "context_id": str(context_id),
                        "status": "missing_artifact",
                    }
                )
                continue
            if file_sha256(artifact_path) != complete["artifact"]["sha256"]:
                errors.append(
                    {"context_id": str(context_id), "status": "artifact_hash_mismatch"}
                )
                continue
            with np.load(artifact_path, allow_pickle=False) as archive:
                missing = sorted(set(ANALYSIS_ARRAYS) - set(archive.files))
                if missing:
                    errors.append(
                        {
                            "context_id": str(context_id),
                            "status": "missing_analysis_arrays",
                            "missing": missing,
                        }
                    )
                    continue
                contexts.append(
                    {
                        "context": complete["context"],
                        "source": complete["source"],
                        "arrays": {
                            name: archive[name].copy() for name in ANALYSIS_ARRAYS
                        },
                    }
                )
    contexts.sort(key=lambda value: str(value["context"]["context_id"]))
    return runs, contexts, errors


def _stack(np: Any, contexts: list[dict[str, Any]], name: str) -> Any:
    return np.stack([value["arrays"][name] for value in contexts])


def _spectra(
    np: Any,
    contexts: list[dict[str, Any]],
    arrays: dict[str, Any],
) -> list[dict[str, Any]]:
    token_positions = np.asarray(
        [int(value["context"]["action_token_position"]) for value in contexts]
    )
    records = []
    for family in VECTOR_FAMILIES:
        values = arrays[family]
        layer_count = int(values.shape[1])
        for layer in range(layer_count):
            flattened = values[:, layer].reshape(len(contexts), -1)
            records.append(
                {
                    "family": family,
                    "source_layer": layer,
                    "action_token_position": "all",
                    **spectrum_summary(np, flattened),
                }
            )
            for token in sorted(set(token_positions.tolist())):
                selected = flattened[token_positions == token]
                if len(selected) < 2:
                    continue
                records.append(
                    {
                        "family": family,
                        "source_layer": layer,
                        "action_token_position": int(token),
                        **spectrum_summary(np, selected),
                    }
                )
    return records


def _head_spectra(
    np: Any, contexts: list[dict[str, Any]], arrays: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for family in ("immediate_key_delta", "immediate_value_delta"):
        values = arrays[family]
        for source_layer in range(values.shape[1]):
            for head in range(values.shape[2]):
                summary = spectrum_summary(np, values[:, source_layer, head])
                rows.append(
                    {
                        "family": family,
                        "source_layer": source_layer,
                        "attention_head": head,
                        "samples": summary["samples"],
                        "dimensions": summary["dimensions"],
                        "nonzero_samples": summary["nonzero_samples"],
                        "norm_median": summary["norm"]["median"],
                        "centered_effective_rank": summary["centered"][
                            "effective_rank"
                        ],
                        "centered_stable_rank": summary["centered"]["stable_rank"],
                        "centered_components_90": summary["centered"][
                            "components_90"
                        ],
                        "centered_components_95": summary["centered"][
                            "components_95"
                        ],
                        "centered_components_99": summary["centered"][
                            "components_99"
                        ],
                    }
                )
    return rows


def _observation_rows(
    np: Any, contexts: list[dict[str, Any]], arrays: dict[str, Any]
) -> list[dict[str, Any]]:
    injection = arrays["injection_residual_delta"]
    immediate = arrays["immediate_residual_delta"]
    keys = arrays["immediate_key_delta"].reshape(len(contexts), 31, -1)
    values = arrays["immediate_value_delta"].reshape(len(contexts), 31, -1)
    safe = arrays["safe_feature_delta"]
    logits = arrays["action_logit_delta"].reshape(len(contexts), 32, -1)
    commands = arrays["command_delta"]

    injection_norm = np.linalg.norm(injection, axis=-1)
    immediate_norm = np.linalg.norm(immediate, axis=-1)
    key_norm = np.linalg.norm(keys, axis=-1)
    value_norm = np.linalg.norm(values, axis=-1)
    safe_norm = np.linalg.norm(safe, axis=-1)
    logit_norm = np.linalg.norm(logits, axis=-1)
    command_norm = np.linalg.norm(commands, axis=-1)
    transition_cosine = _cosine(np, injection[:, :31], immediate)
    key_value_cosine = _cosine(np, keys, values)

    rows = []
    for context_index, item in enumerate(contexts):
        context = item["context"]
        current = item["arrays"]
        for layer in range(32):
            row = {
                "context_id": context["context_id"],
                "worker_shard": context["worker_shard"],
                "task_id": context["task_id"],
                "episode_index": context["episode_index"],
                "phase": context["phase"],
                "policy_step": context["policy_step"],
                "action_token_position": context["action_token_position"],
                "source_layer": layer,
                "record_id": str(current["record_id"][layer]),
                "injection_residual_l2": float(injection_norm[context_index, layer]),
                "safe_final_token_l2": float(safe_norm[context_index, layer]),
                "action_logit_l2": float(logit_norm[context_index, layer]),
                "command_l2": float(command_norm[context_index, layer]),
                "eligible_causal_outcome": int(
                    current["eligible_causal_outcome"][layer]
                ),
                "command_changed": int(current["command_changed"][layer]),
                "task_failure": int(current["task_failure"][layer]),
                "safe_alarm_at_fault": int(current["safe_alarm_at_fault"][layer]),
                "safe_alarm_within_10": int(
                    current["safe_alarm_within_10"][layer]
                ),
                "safe_alarm_post_fault_any": int(
                    current["safe_alarm_post_fault_any"][layer]
                ),
                "operational_silent_failure": int(
                    current["operational_silent_failure"][layer]
                ),
                "score_at_fault": float(current["score_at_fault"][layer]),
                "control_score_at_fault": float(
                    current["control_score_at_fault"][layer]
                ),
                "score_change_from_control_at_fault": float(
                    current["score_change_from_control_at_fault"][layer]
                ),
                "immediate_residual_l2": None,
                "immediate_key_l2": None,
                "immediate_value_l2": None,
                "residual_transition_cosine": None,
                "key_value_cosine": None,
            }
            if layer < 31:
                row.update(
                    {
                        "immediate_residual_l2": float(
                            immediate_norm[context_index, layer]
                        ),
                        "immediate_key_l2": float(key_norm[context_index, layer]),
                        "immediate_value_l2": float(
                            value_norm[context_index, layer]
                        ),
                        "residual_transition_cosine": float(
                            transition_cosine[context_index, layer]
                        ),
                        "key_value_cosine": float(
                            key_value_cosine[context_index, layer]
                        ),
                    }
                )
            rows.append(row)
    return rows


def analyze(
    np: Any, atlas_dirs: list[Path]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    runs, contexts, errors = _load_atlases(np, atlas_dirs)
    if not contexts:
        raise ValueError("no complete atlas contexts were available")
    arrays = {
        "injection_residual_delta": _stack(np, contexts, "injection_residual_delta"),
        "immediate_residual_delta": _stack(np, contexts, "immediate_residual_delta"),
        "immediate_key_delta": _stack(np, contexts, "immediate_key_delta"),
        "immediate_value_delta": _stack(np, contexts, "immediate_value_delta"),
        "final_residual_delta": _stack(np, contexts, "final_residual_delta"),
        "safe_feature_delta": _stack(np, contexts, "safe_feature_delta"),
        "action_logit_delta": _stack(np, contexts, "action_logit_delta"),
        "command_delta": _stack(np, contexts, "command_delta"),
    }
    observations = _observation_rows(np, contexts, arrays)
    head_spectra = _head_spectra(np, contexts, arrays)
    output = {
        "schema_version": 1,
        "analysis": "development interface atlas",
        "status": (
            "exploratory description of mechanically declared signed cuts; no "
            "interface or risk rule is selected here"
        ),
        "population": {
            "planned_contexts": sum(len(value["context_ids"]) for value in runs),
            "complete_contexts": len(contexts),
            "unresolved_contexts": len(errors),
            "context_status_counts": dict(
                sorted(
                    Counter(
                        ["complete"] * len(contexts)
                        + [str(value.get("status", "error")) for value in errors]
                    ).items()
                )
            ),
            "interventions": len(observations),
        },
        "contracts": runs[0]["cut_contract"],
        "spectrum_definition": {
            "uncentered": "energy spectrum of signed interface vectors",
            "centered": "energy spectrum after subtracting the development mean",
            "effective_rank": "exponential entropy of normalized squared singular values",
            "stable_rank": "total squared singular-value energy divided by the largest",
        },
        "source_atlases": [
            {
                "path": str(path.resolve()),
                "run_sha256": file_sha256(path / "run.json"),
            }
            for path in atlas_dirs
        ],
        "spectra": _spectra(np, contexts, arrays),
        "unresolved": errors,
    }
    return output, observations, head_spectra


def main() -> None:
    args = _arguments()
    import numpy as np

    output, observations, head_spectra = analyze(np, args.atlas_dirs)
    write_json_atomic(args.output, output)
    write_csv_atomic(args.observations_csv, observations)
    write_csv_atomic(args.head_spectra_csv, head_spectra)
    print(
        json.dumps(
            {
                "population": output["population"],
                "spectrum_records": len(output["spectra"]),
                "head_spectrum_records": len(head_spectra),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
