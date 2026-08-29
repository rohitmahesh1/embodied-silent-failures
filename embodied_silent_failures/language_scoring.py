from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.score_safe import alarm_windows


SCORE_ABSOLUTE_TOLERANCE = 1e-6
SCORE_RELATIVE_TOLERANCE = 1e-6


def intervention_sources(
    summary: dict[str, Any], local_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    branches = {str(item["branch"]): item for item in summary["branches"]}
    if "control" not in branches:
        raise ValueError("language context has no control branch")
    command_branches = {
        str(item["command_group"]["command_id"]): item
        for item in summary["branches"]
        if item.get("command_group") is not None
    }
    group_by_layer = {}
    for group in summary["command_groups"]:
        for layer_index in group["member_layer_indices"]:
            if int(layer_index) in group_by_layer:
                raise ValueError("language layer belongs to two command groups")
            group_by_layer[int(layer_index)] = group

    plans = []
    for record in sorted(local_records, key=lambda value: int(value["layer_index"])):
        layer_index = int(record["layer_index"])
        if record.get("status") != "complete":
            plans.append(
                {
                    "layer_index": layer_index,
                    "status": "local_error",
                    "local_record": record,
                }
            )
            continue
        if bool(record["executed_command"]["exact_equal"]):
            plans.append(
                {
                    "layer_index": layer_index,
                    "status": "scoreable",
                    "local_record": record,
                    "physical_branch": branches["control"],
                    "terminal_result": branches["control"]["result"],
                    "terminal_evidence": "inherited_from_exact_command_control",
                    "command_group": None,
                    "monitor_horizon": "complete_physical_trace",
                }
            )
            continue

        group = group_by_layer.get(layer_index)
        if group is None:
            raise ValueError(f"changed layer {layer_index} has no command group")
        branch = command_branches.get(str(group["command_id"]))
        if branch is None:
            plans.append(
                {
                    "layer_index": layer_index,
                    "status": "scoreable",
                    "local_record": record,
                    "physical_branch": branches["control"],
                    "terminal_result": None,
                    "terminal_evidence": "unavailable_without_successful_control",
                    "command_group": group,
                    "monitor_horizon": "through_fault_step_only",
                }
            )
            continue
        plans.append(
            {
                "layer_index": layer_index,
                "status": "scoreable",
                "local_record": record,
                "physical_branch": branch,
                "terminal_result": branch["result"],
                "terminal_evidence": "observed_exact_command_branch",
                "command_group": group,
                "monitor_horizon": "complete_physical_trace",
            }
        )
    return plans


def _load_hidden_states(
    campaign_dir: Path,
    context_id: str,
    branch: dict[str, Any],
    torch: Any,
) -> Any:
    branch_name = str(branch["branch"])
    directory = campaign_dir / "attempts" / f"{context_id}-{branch_name}"
    result = branch["result"]
    if result.get("status") != "complete":
        raise ValueError(f"physical branch is not complete: {directory}")
    pickle_path = directory / str(result["files"]["pickle"])
    if not pickle_path.is_file():
        raise FileNotFoundError(f"physical feature archive is missing: {pickle_path}")
    with pickle_path.open("rb") as file:
        value = pickle.load(file)["hidden_states"]
    if isinstance(value, list):
        value = torch.stack(value, dim=0)
    if value.ndim != 3 or value.shape[1] != 7:
        raise ValueError(
            f"physical SAFE feature trace has unexpected shape {tuple(value.shape)}"
        )
    if len(value) != int(result["policy_steps"]):
        raise ValueError("physical feature trace and completion length disagree")
    return value


def physical_score_index(
    score_json: dict[str, Any], archive_path: Path, np: Any
) -> tuple[dict[str, dict[str, Any]], Any, list[float]]:
    if file_sha256(archive_path) != score_json["score_archive"]["sha256"]:
        raise ValueError("ordinary SAFE score archive hash does not match its record")
    archive = np.load(archive_path)
    runs = [str(value) for value in archive["runs"]]
    lengths = archive["lengths"].astype(int)
    scores = archive["scores"]
    if len(runs) != len(set(runs)):
        raise ValueError("ordinary SAFE score archive contains duplicate run labels")
    if any(
        str(record["run"]) != run
        for run, record in zip(runs, score_json["records"], strict=True)
    ):
        raise ValueError("ordinary SAFE JSON and score archive use different ordering")
    indexed = {
        run: {"scores": scores[index, : lengths[index]], "record": record}
        for index, (run, record) in enumerate(
            zip(runs, score_json["records"], strict=True)
        )
    }
    return indexed, archive["bands"].astype(float), archive["alphas"].astype(float).tolist()


def score_batches(
    model: Any,
    items: list[tuple[str, Any]],
    *,
    batch_size: int,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    result = {}
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        maximum = max(len(features) for _, features in chunk)
        dimension = int(chunk[0][1].shape[-1])
        padded = torch.zeros((len(chunk), maximum, dimension), dtype=torch.float32)
        for index, (_, features) in enumerate(chunk):
            padded[index, : len(features)] = features
        with torch.no_grad():
            values = model({"features": padded.to("cuda")}).squeeze(-1)
        for index, (record_id, features) in enumerate(chunk):
            result[record_id] = (
                values[index, : len(features)].detach().cpu().numpy().astype(np.float32)
            )
    return result


def primary_band(score_json: dict[str, Any], bands: Any, alphas: list[float]) -> Any:
    primary = float(score_json["monitor"]["primary_alpha"])
    matches = [
        index
        for index, alpha in enumerate(alphas)
        if math.isclose(alpha, primary, rel_tol=0, abs_tol=1e-8)
    ]
    if len(matches) != 1:
        raise ValueError("ordinary SAFE scores do not contain one primary alpha")
    return bands[matches[0]]


def composition_check(
    reconstructed: Any, physical: Any, band: Any, np: Any
) -> dict[str, Any]:
    if len(reconstructed) != len(physical):
        return {
            "valid": False,
            "reason": "score_length_mismatch",
            "score_exact_equal": False,
            "maximum_score_difference": None,
            "score_within_diagnostic_tolerance": False,
            "alarm_timeline_exact_equal": False,
        }
    finite = np.isfinite(reconstructed) & np.isfinite(physical)
    difference = np.abs(reconstructed - physical)
    maximum = float(np.max(difference[finite])) if np.any(finite) else None
    exact = bool(np.array_equal(reconstructed, physical, equal_nan=True))
    close = bool(
        np.allclose(
            reconstructed,
            physical,
            rtol=SCORE_RELATIVE_TOLERANCE,
            atol=SCORE_ABSOLUTE_TOLERANCE,
        )
    )
    alarm_equal = bool(
        np.array_equal(
            reconstructed >= band[: len(reconstructed)],
            physical >= band[: len(physical)],
        )
    )
    return {
        "valid": alarm_equal,
        "reason": None if alarm_equal else "ordinary_safe_alarm_timeline_mismatch",
        "score_exact_equal": exact,
        "maximum_score_difference": maximum,
        "score_within_diagnostic_tolerance": close,
        "alarm_timeline_exact_equal": alarm_equal,
        "absolute_tolerance": SCORE_ABSOLUTE_TOLERANCE,
        "relative_tolerance": SCORE_RELATIVE_TOLERANCE,
    }


def composition_verified(
    record: dict[str, Any],
    checks_by_group: dict[tuple[str, str], dict[str, Any]],
    *,
    control_feature_exact: bool,
) -> bool:
    if not control_feature_exact:
        return False
    command_id = record.get("command_id")
    if command_id is None:
        return True
    if record.get("terminal_evidence") != "observed_exact_command_branch":
        return False
    check = checks_by_group.get((str(record["context_id"]), str(command_id)))
    return bool(check and check["valid"])


def score_context(
    *,
    campaign_dir: Path,
    context_id: str,
    cfg: Any,
    model: Any,
    physical_index: dict[str, dict[str, Any]],
    bands: Any,
    alphas: list[float],
    primary: Any,
    batch_size: int,
    torch: Any,
    np: Any,
    process_tensor_idx_rel: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    context_dir = campaign_dir / "contexts" / context_id
    summary = load_json(context_dir / "context.complete.json")
    local = load_json(context_dir / "local.json")
    with (context_dir / "local_features.pkl").open("rb") as file:
        local_features = pickle.load(file)
    context = summary["context"]
    step = int(context["policy_step"])
    plans = intervention_sources(summary, local["interventions"])
    branches = {str(item["branch"]): item for item in summary["branches"]}
    trace_cache = {}

    def branch_trace(branch: dict[str, Any]) -> tuple[Any, Any]:
        name = str(branch["branch"])
        if name not in trace_cache:
            raw = _load_hidden_states(campaign_dir, context_id, branch, torch)
            # SAFE b6036abe, failure_prob/data/openvla.py::load_rollouts,
            # applies this function to stored [step, token, 4096] features. The
            # frozen config selects relative token 1.0, the final action token.
            selected = process_tensor_idx_rel(raw.float(), cfg.dataset.token_idx_rel)
            trace_cache[name] = (raw, selected)
        return trace_cache[name]

    control_raw, _control_selected = branch_trace(branches["control"])
    clean_feature = local_features["clean_hidden_states"]
    control_feature_exact = bool(torch.equal(control_raw[step], clean_feature))
    items = []
    records = []
    record_by_id = {}
    for plan in plans:
        layer_index = int(plan["layer_index"])
        record_id = f"{context_id}:layer{layer_index:02d}"
        base = {
            "record_id": record_id,
            "context": context,
            "context_id": context_id,
            "layer_index": layer_index,
            "status": plan["status"],
            "local_measurements": plan["local_record"],
            "control_success": bool(branches["control"]["result"].get("success")),
            "control_feature_exact": control_feature_exact,
        }
        if plan["status"] != "scoreable":
            records.append(base)
            continue
        branch = plan["physical_branch"]
        raw, selected = branch_trace(branch)
        replacement = local_features["faulted_hidden_states_by_layer"][layer_index]
        if tuple(replacement.shape) != tuple(raw[step].shape):
            raise ValueError(f"saved layer feature has wrong shape for {record_id}")
        features = selected.clone()
        features[step] = process_tensor_idx_rel(
            replacement.float(), cfg.dataset.token_idx_rel
        )
        if plan["monitor_horizon"] == "through_fault_step_only":
            features = features[: step + 1]
        terminal = plan["terminal_result"]
        group = plan["command_group"]
        base.update(
            {
                "physical_run": f"{context_id}-{branch['branch']}",
                "monitor_horizon": plan["monitor_horizon"],
                "terminal_evidence": plan["terminal_evidence"],
                "terminal_success": (
                    bool(terminal["success"])
                    if terminal is not None and terminal.get("status") == "complete"
                    else None
                ),
                "terminal_policy_steps": (
                    int(terminal["policy_steps"])
                    if terminal is not None and terminal.get("status") == "complete"
                    else None
                ),
                "command_id": group["command_id"] if group is not None else None,
                "representative_layer_index": (
                    int(group["representative_layer_index"])
                    if group is not None
                    else None
                ),
                "command_group_size": (
                    len(group["member_layer_indices"]) if group is not None else 1
                ),
            }
        )
        items.append((record_id, features))
        record_by_id[record_id] = base

    scored = score_batches(model, items, batch_size=batch_size, torch=torch, np=np)
    control_scores = physical_index[f"{context_id}-control"]["scores"]
    score_arrays = {}
    checks = []
    for record_id, values in scored.items():
        base = record_by_id[record_id]
        layer_index = int(base["layer_index"])
        finite = np.flatnonzero(~np.isfinite(values))
        base.update(
            {
                "status": "scored",
                "score_length": len(values),
                "score_validity": {
                    "all_finite": not bool(len(finite)),
                    "nonfinite_count": int(len(finite)),
                    "first_nonfinite_step": int(finite[0]) if len(finite) else None,
                },
                "score_at_fault": (
                    float(values[step]) if math.isfinite(float(values[step])) else None
                ),
                "threshold_at_fault": float(primary[step]),
                "alarm_at_fault": bool(values[step] >= primary[step]),
                "alarm_before_fault": bool(np.any(values[:step] >= primary[:step])),
                "alarms": alarm_windows(values.tolist(), alphas, bands, step),
                "control_score_at_fault": float(control_scores[step]),
                "control_alarm_at_fault": bool(control_scores[step] >= primary[step]),
                "score_change_from_control_at_fault": float(
                    values[step] - control_scores[step]
                ),
            }
        )
        score_arrays[record_id] = values
        command_id = base["command_id"]
        if (
            base["terminal_evidence"] == "observed_exact_command_branch"
            and layer_index == base["representative_layer_index"]
        ):
            branch = next(
                item
                for item in summary["branches"]
                if isinstance(item.get("command_group"), dict)
                and item["command_group"].get("command_id") == command_id
            )
            branch_raw, _ = branch_trace(branch)
            feature_exact = bool(
                torch.equal(
                    branch_raw[step],
                    local_features["faulted_hidden_states_by_layer"][layer_index],
                )
            )
            check = composition_check(
                values,
                physical_index[base["physical_run"]]["scores"],
                primary,
                np,
            )
            check.update(
                {
                    "context_id": context_id,
                    "command_id": command_id,
                    "representative_layer_index": layer_index,
                    "feature_exact_equal": feature_exact,
                }
            )
            check["valid"] = bool(check["valid"] and feature_exact)
            if not feature_exact:
                check["reason"] = "representative_feature_mismatch"
            checks.append(check)
        records.append(base)

    checks_by_group = {
        (str(value["context_id"]), str(value["command_id"])): value
        for value in checks
    }
    for record in records:
        verified = composition_verified(
            record,
            checks_by_group,
            control_feature_exact=control_feature_exact,
        )
        record["composition_verified"] = verified
        if not verified and record.get("status") == "scored":
            record["status"] = "composition_unverified"

    context_record = {
        "context_id": context_id,
        "status": "complete",
        "context": context,
        "control_success": bool(branches["control"]["result"].get("success")),
        "control_feature_exact": control_feature_exact,
        "local_interventions": int(summary["local_interventions"]),
        "command_changing_interventions": int(summary["command_changing_interventions"]),
        "unique_faulted_commands": int(summary["unique_faulted_commands"]),
    }
    return context_record, records, score_arrays, checks
