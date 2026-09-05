from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    write_json_atomic,
    write_npz_atomic,
)
from embodied_silent_failures.language_scoring import physical_score_index
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
    source_file_record,
)
from embodied_silent_failures.safe_trajectory_geometry import (
    WINDOW_STEPS,
    physical_population,
    trajectory_window_geometry,
)
from embodied_silent_failures.score_safe import SAFE_REVISION, _validate_monitor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how paired physical trajectories move through SAFE's input space."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--site-analysis", action="append", required=True, type=Path)
    parser.add_argument("--physical-analysis", required=True, type=Path)
    parser.add_argument("--physical-scores", required=True, type=Path)
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--array-output", required=True, type=Path)
    parser.add_argument("--window-steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _load_hidden_states(path: Path, torch: Any) -> Any:
    with path.open("rb") as file:
        hidden_states = pickle.load(file)["hidden_states"]
    if isinstance(hidden_states, list):
        hidden_states = torch.stack(hidden_states, dim=0)
    if hidden_states.ndim != 3 or hidden_states.shape[1:] != (7, 4096):
        raise ValueError(f"unexpected physical feature shape {tuple(hidden_states.shape)}")
    return hidden_states


def _branch_feature_path(
    campaign_dir: Path,
    context_id: str,
    branch: dict[str, Any],
) -> Path:
    result = branch["result"]
    if result.get("status") != "complete":
        raise ValueError(f"physical branch {branch['branch']} is not complete")
    path = (
        campaign_dir
        / "attempts"
        / f"{context_id}-{branch['branch']}"
        / str(result["files"]["pickle"])
    )
    if not path.is_file():
        raise FileNotFoundError(f"physical feature archive is missing: {path}")
    return path


def _score_increments(scores: Any, np: Any) -> Any:
    values = np.asarray(scores, dtype=np.float32)
    return np.diff(np.concatenate((np.zeros(1, dtype=np.float32), values)))


def _window_integrity(
    *,
    direct_control: Any,
    direct_faulted: Any,
    stored_control_scores: Any,
    stored_faulted_scores: Any,
    fault_step: int,
    np: Any,
) -> dict[str, float]:
    stop = fault_step + len(direct_control)
    stored_control = _score_increments(stored_control_scores, np)[fault_step:stop]
    stored_faulted = _score_increments(stored_faulted_scores, np)[fault_step:stop]
    return {
        "maximum_control_increment_error": float(
            np.max(np.abs(direct_control - stored_control))
        ),
        "maximum_faulted_increment_error": float(
            np.max(np.abs(direct_faulted - stored_faulted))
        ),
        "signed_response_sum_error": float(
            abs(
                float((direct_faulted - direct_control).sum())
                - float((stored_faulted - stored_control).sum())
            )
        ),
    }


def main() -> None:
    args = _arguments()
    if args.window_steps < 1:
        raise ValueError("window steps must be positive")
    project_root = Path(__file__).resolve().parents[1]
    if git_revision(args.safe_root) != SAFE_REVISION or git_dirty(args.safe_root):
        raise RuntimeError("SAFE source must be the clean pinned revision")

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    site_documents = [load_json(path) for path in args.site_analysis]
    population, site_monitor = physical_population(site_documents)
    if args.limit is not None:
        population = population[: args.limit]

    physical_document = load_json(args.physical_analysis)
    physical_scores, _bands, _alphas = physical_score_index(
        physical_document, args.physical_scores, np
    )
    if physical_document["monitor"] != site_monitor:
        raise ValueError("site and physical analyses used different SAFE monitors")

    monitor_manifest, monitor_paths = _validate_monitor(args.monitor_dir)
    checkpoint_hash = file_sha256(monitor_paths["checkpoint"])
    if checkpoint_hash != site_monitor["checkpoint_sha256"]:
        raise ValueError("geometry extraction selected a different SAFE checkpoint")

    sys.path.insert(0, str(args.safe_root.resolve()))
    from failure_prob.data.utils import process_tensor_idx_rel
    from failure_prob.model import get_model

    cfg = OmegaConf.load(monitor_paths["configuration"])
    expected = {
        "model.name": (str(cfg.model.name), "indep"),
        "model.n_layers": (int(cfg.model.n_layers), 2),
        "model.hidden_dim": (int(cfg.model.hidden_dim), 256),
        "model.final_act_layer": (str(cfg.model.final_act_layer), "sigmoid"),
        "model.n_history_steps": (int(cfg.model.n_history_steps), 1),
        "model.cumsum": (bool(cfg.model.cumsum), True),
        "model.rmean": (bool(cfg.model.rmean), False),
        "dataset.token_idx_rel": (float(cfg.dataset.token_idx_rel), 1.0),
    }
    disagreements = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if disagreements:
        raise ValueError("SAFE geometry does not match monitor: " + "; ".join(disagreements))

    # SAFE@b6036ab, failure_prob/model/indep.py::IndepModel.forward builds the
    # frozen 4096 -> 256 -> 1 sigmoid projector used independently at each step.
    model = get_model(cfg, 4096)
    model.load_state_dict(torch.load(monitor_paths["checkpoint"], map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in population:
        by_context[record["context_id"]].append(record)

    output_records = []
    array_rows: dict[str, list[Any]] = defaultdict(list)
    errors = []
    maxima = Counter()
    for context_id, records in sorted(by_context.items()):
        summary = load_json(
            args.campaign_dir / "contexts" / context_id / "context.complete.json"
        )
        branch_by_name = {
            str(branch["branch"]): branch for branch in summary["branches"]
        }
        control_branch = branch_by_name["control"]
        try:
            control_raw = _load_hidden_states(
                _branch_feature_path(args.campaign_dir, context_id, control_branch),
                torch,
            )
            # SAFE@b6036ab, failure_prob/data/openvla.py::load_rollouts calls
            # process_tensor_idx_rel; token_idx_rel=1.0 selects the final action token.
            control_selected = process_tensor_idx_rel(
                control_raw.float(), float(cfg.dataset.token_idx_rel)
            )
        except Exception as error:
            for record in records:
                errors.append(
                    {
                        **record,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            continue

        for record in records:
            try:
                branch_name = record["physical_run"][len(context_id) + 1 :]
                branch = branch_by_name[branch_name]
                faulted_raw = _load_hidden_states(
                    _branch_feature_path(args.campaign_dir, context_id, branch), torch
                )
                faulted_selected = process_tensor_idx_rel(
                    faulted_raw.float(), float(cfg.dataset.token_idx_rel)
                )
                start = int(record["fault_step"])
                stop = start + args.window_steps
                if stop > len(control_selected) or stop > len(faulted_selected):
                    raise ValueError("paired feature traces do not cover the full window")

                control_window = control_selected[start:stop].to(device)
                faulted_window = faulted_selected[start:stop].to(device)
                geometry, arrays = trajectory_window_geometry(
                    model, control_window, faulted_window, torch
                )
                score_control = physical_scores[f"{context_id}-control"]["scores"]
                score_faulted = physical_scores[record["physical_run"]]["scores"]
                integrity = _window_integrity(
                    direct_control=arrays["clean_monitor_increment"],
                    direct_faulted=arrays["faulted_monitor_increment"],
                    stored_control_scores=score_control,
                    stored_faulted_scores=score_faulted,
                    fault_step=start,
                    np=np,
                )
                for name, value in integrity.items():
                    maxima[name] = max(float(maxima[name]), value)

                pre_fault = None
                if start:
                    pre_fault = {
                        "exact_equal": bool(
                            torch.equal(
                                control_selected[start - 1],
                                faulted_selected[start - 1],
                            )
                        ),
                        "maximum_absolute_difference": float(
                            (control_selected[start - 1] - faulted_selected[start - 1])
                            .abs()
                            .max()
                            .item()
                        ),
                    }
                array_index = len(output_records)
                output_records.append(
                    {
                        **record,
                        "status": "complete",
                        "array_index": array_index,
                        "pre_fault_feature_check": pre_fault,
                        "stored_score_integrity": integrity,
                        **geometry,
                    }
                )
                for name, values in arrays.items():
                    array_rows[name].append(values)
            except Exception as error:
                errors.append(
                    {
                        **record,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    if not output_records:
        raise RuntimeError("no physical branch geometry was extracted")
    archive = {
        "physical_runs": np.asarray(
            [record["physical_run"] for record in output_records]
        ),
        **{name: np.stack(values) for name, values in sorted(array_rows.items())},
    }
    write_npz_atomic(args.array_output, np, archive)
    output = {
        "schema_version": 1,
        "analysis": "paired physical trajectory geometry at the frozen SAFE input",
        "analysis_contract": {
            "unit": "one distinct non-control physical continuation",
            "window_steps": args.window_steps,
            "window_basis": (
                "the same already-declared 25-step window used by the fixed-window "
                "SAFE observability analysis"
            ),
            "decomposition": (
                "paired SAFE-input displacement, projection onto the frozen monitor's "
                "clean-state gradient, ReLU gate changes, and cancellation of signed "
                "per-step score differences"
            ),
            "limits": (
                "the gradient is a local description of SAFE rather than a proposed "
                "failure classifier; outcome comparisons remain post-hoc until repeated"
            ),
        },
        "provenance": {
            "experiment_revision": git_revision(project_root),
            "experiment_dirty": git_dirty(project_root),
            "safe_revision": git_revision(args.safe_root),
            "safe_model_source": source_file_record(model),
            "safe_token_selector": {
                "function": "failure_prob.data.utils.process_tensor_idx_rel",
                "path": str(Path(process_tensor_idx_rel.__code__.co_filename).resolve()),
                "sha256": file_sha256(Path(process_tensor_idx_rel.__code__.co_filename)),
            },
            "monitor": {
                "checkpoint_sha256": checkpoint_hash,
                "configuration_sha256": file_sha256(monitor_paths["configuration"]),
                "manifest_sha256": file_sha256(monitor_paths["monitor"]),
                "frozen_checkpoint": monitor_manifest["checkpoint"],
            },
            "site_analyses": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.site_analysis
            ],
            "physical_analysis": {
                "path": str(args.physical_analysis.resolve()),
                "sha256": file_sha256(args.physical_analysis),
            },
            "physical_scores": {
                "path": str(args.physical_scores.resolve()),
                "sha256": file_sha256(args.physical_scores),
            },
        },
        "device": device,
        "coverage": {
            "declared_physical_continuations": len(population),
            "complete": len(output_records),
            "errors": len(errors),
        },
        "maximum_stored_score_integrity_error": dict(sorted(maxima.items())),
        "array_archive": artifact_record(args.array_output),
        "records": output_records,
        "error_records": errors,
    }
    write_json_atomic(args.output, output)
    print(
        json.dumps(
            {
                "analysis": output["analysis"],
                "device": device,
                "coverage": output["coverage"],
                "maximum_stored_score_integrity_error": output[
                    "maximum_stored_score_integrity_error"
                ],
                "array_archive": output["array_archive"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
