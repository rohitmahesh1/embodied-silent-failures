from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

from embodied_silent_failures.language_scoring import composition_check
from embodied_silent_failures.provenance import load_json
from embodied_silent_failures.score_safe import alarm_windows


def intervention_sources(
    summary: dict[str, Any], local_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    branches = {str(value["branch"]): value for value in summary["branches"]}
    if "control" not in branches:
        raise ValueError("atlas context has no control branch")
    command_branches = {
        str(value["command_group"]["command_id"]): value
        for value in summary["branches"]
        if value.get("command_group") is not None
    }
    group_by_site = {}
    for group in summary["command_groups"]:
        for site_id in group["member_site_ids"]:
            if site_id in group_by_site:
                raise ValueError(f"atlas site belongs to two command groups: {site_id}")
            group_by_site[str(site_id)] = group

    plans = []
    for record in sorted(local_records, key=lambda value: str(value["site_id"])):
        site_id = str(record["site_id"])
        base = {"site_id": site_id, "local_record": record}
        if record.get("status") != "complete":
            plans.append({**base, "status": "local_error"})
            continue
        if bool(record["executed_command"]["exact_equal"]):
            plans.append(
                {
                    **base,
                    "status": "scoreable",
                    "physical_branch": branches["control"],
                    "terminal_result": branches["control"]["result"],
                    "terminal_evidence": "inherited_from_exact_command_control",
                    "command_group": None,
                    "monitor_horizon": "complete_physical_trace",
                }
            )
            continue

        group = group_by_site.get(site_id)
        if group is None:
            raise ValueError(f"changed atlas site has no command group: {site_id}")
        branch = command_branches.get(str(group["command_id"]))
        if branch is None:
            plans.append(
                {
                    **base,
                    "status": "local_only",
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
                **base,
                "status": "scoreable",
                "physical_branch": branch,
                "terminal_result": branch["result"],
                "terminal_evidence": "observed_exact_command_branch",
                "command_group": group,
                "monitor_horizon": "complete_physical_trace",
            }
        )
    return plans


def reconstruct_cumulative_scores(
    physical_scores: Any,
    *,
    fault_step: int,
    replacement_contribution: float,
    physical_contribution: float,
    np: Any,
) -> Any:
    if fault_step < 0 or fault_step >= len(physical_scores):
        raise ValueError("fault step is outside the physical SAFE trace")
    values = np.asarray(physical_scores, dtype=np.float32).copy()
    values[fault_step:] += replacement_contribution - physical_contribution
    return values


def replay_is_exact(result: dict[str, Any]) -> bool:
    replay = result.get("context_replay")
    if not isinstance(replay, dict):
        return False
    observation = replay.get("observation")
    if not isinstance(observation, dict):
        return False
    return bool(
        replay.get("simulator_state_exact_equal") is True
        and float(observation.get("maximum_numeric_error", math.inf)) == 0.0
        and float(observation.get("maximum_image_channel_error", math.inf)) == 0.0
        and int(observation.get("changed_image_channels", -1)) == 0
    )


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
        hidden_states = pickle.load(file)["hidden_states"]
    if isinstance(hidden_states, list):
        hidden_states = torch.stack(hidden_states, dim=0)
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 7:
        raise ValueError(
            f"physical SAFE feature trace has unexpected shape {tuple(hidden_states.shape)}"
        )
    if len(hidden_states) != int(result["policy_steps"]):
        raise ValueError("physical feature trace and completion length disagree")
    return hidden_states


def _score_contributions(
    model: Any,
    items: list[tuple[str, Any]],
    *,
    batch_size: int,
    torch: Any,
) -> dict[str, float]:
    result = {}
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        features = torch.stack([value.float() for _, value in chunk], dim=0).unsqueeze(1)
        with torch.no_grad():
            values = model({"features": features.to("cuda")}).squeeze(-1).squeeze(-1)
        for (record_id, _), value in zip(chunk, values, strict=True):
            result[record_id] = float(value.detach().cpu())
    return result


def _primary_band(score_json: dict[str, Any], bands: Any, alphas: list[float]) -> Any:
    primary = float(score_json["monitor"]["primary_alpha"])
    matches = [
        index
        for index, alpha in enumerate(alphas)
        if math.isclose(alpha, primary, rel_tol=0, abs_tol=1e-8)
    ]
    if len(matches) != 1:
        raise ValueError("physical SAFE scores do not contain one primary alpha")
    return bands[matches[0]]


def _alarm_summary(
    values: Any,
    *,
    step: int,
    bands: Any,
    alphas: list[float],
    primary: Any,
    np: Any,
) -> dict[str, Any]:
    nonfinite = np.flatnonzero(~np.isfinite(values))
    return {
        "score_length": len(values),
        "score_validity": {
            "all_finite": not bool(len(nonfinite)),
            "nonfinite_count": len(nonfinite),
            "first_nonfinite_step": int(nonfinite[0]) if len(nonfinite) else None,
        },
        "score_at_fault": (
            float(values[step]) if math.isfinite(float(values[step])) else None
        ),
        "threshold_at_fault": float(primary[step]),
        "alarm_before_fault": bool(np.any(values[:step] >= primary[:step])),
        "alarm_at_fault": bool(values[step] >= primary[step]),
        "alarms": alarm_windows(values.tolist(), alphas, bands, step),
    }


def score_context(
    *,
    campaign_dir: Path,
    context_id: str,
    site_by_id: dict[str, dict[str, Any]],
    model: Any,
    cfg: Any,
    physical_index: dict[str, dict[str, Any]],
    physical_score_json: dict[str, Any],
    bands: Any,
    alphas: list[float],
    batch_size: int,
    torch: Any,
    np: Any,
    process_tensor_idx_rel: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[Any, Any]], list[dict[str, Any]]]:
    context_dir = campaign_dir / "contexts" / context_id
    summary = load_json(context_dir / "context.complete.json")
    local = load_json(context_dir / "local.json")
    with (context_dir / "local_evidence.pkl").open("rb") as file:
        evidence = pickle.load(file)
    context = summary["context"]
    step = int(context["policy_step"])
    plans = intervention_sources(summary, local["interventions"])
    primary = _primary_band(physical_score_json, bands, alphas)
    branches = {str(value["branch"]): value for value in summary["branches"]}
    trace_cache = {}

    def branch_features(branch: dict[str, Any]) -> tuple[Any, Any]:
        name = str(branch["branch"])
        if name not in trace_cache:
            raw = _load_hidden_states(campaign_dir, context_id, branch, torch)
            # SAFE b6036abe, failure_prob/data/openvla.py::load_rollouts, applies
            # process_tensor_idx_rel to [step, token, 4096]. The frozen config's
            # token_idx_rel=1.0 selects the final action-token representation.
            selected = process_tensor_idx_rel(raw.float(), cfg.dataset.token_idx_rel)
            trace_cache[name] = (raw, selected)
        return trace_cache[name]

    clean_feature = process_tensor_idx_rel(
        evidence["clean"]["hidden_states"].float(), cfg.dataset.token_idx_rel
    )
    contribution_items = [("clean", clean_feature)]
    plan_by_id = {}
    base_feature_by_branch = {}
    expected_feature_by_branch = {}
    for plan in plans:
        site_id = plan["site_id"]
        plan_by_id[site_id] = plan
        faulted = evidence.get("faulted_by_site", {}).get(site_id)
        if plan["status"] not in {"scoreable", "local_only"} or faulted is None:
            continue
        site_feature = process_tensor_idx_rel(
            faulted["hidden_states"].float(), cfg.dataset.token_idx_rel
        )
        contribution_items.append((f"site:{site_id}", site_feature))
        branch = plan["physical_branch"]
        branch_name = str(branch["branch"])
        if branch_name not in base_feature_by_branch:
            _raw, selected = branch_features(branch)
            base_feature_by_branch[branch_name] = selected[step]
            contribution_items.append((f"branch:{branch_name}", selected[step]))
        group = plan["command_group"]
        if group is None:
            expected_feature_by_branch.setdefault(branch_name, clean_feature)
        elif site_id == str(group["representative_site_id"]):
            expected_feature_by_branch[branch_name] = site_feature

    contributions = _score_contributions(
        model, contribution_items, batch_size=batch_size, torch=torch
    )
    clean_contribution = contributions["clean"]
    branch_checks = []
    for branch_name, actual in base_feature_by_branch.items():
        expected = expected_feature_by_branch.get(branch_name)
        feature_exact = bool(expected is not None and torch.equal(actual, expected))
        physical = physical_index[f"{context_id}-{branch_name}"]["scores"]
        observed_contribution = float(
            physical[step] - (physical[step - 1] if step else 0.0)
        )
        direct_contribution = contributions[f"branch:{branch_name}"]
        branch_checks.append(
            {
                "context_id": context_id,
                "physical_branch": branch_name,
                "feature_exact_equal": feature_exact,
                "observed_cumulative_increment": observed_contribution,
                "direct_feature_contribution": direct_contribution,
                "maximum_contribution_difference": abs(
                    observed_contribution - direct_contribution
                ),
            }
        )

    check_by_branch = {
        str(value["physical_branch"]): value for value in branch_checks
    }
    records = []
    score_arrays = {}
    for site_id in sorted(plan_by_id):
        plan = plan_by_id[site_id]
        site = site_by_id[site_id]
        local_record = plan["local_record"]
        base = {
            "record_id": f"{context_id}:{site_id}",
            "context_id": context_id,
            "context": context,
            "site_id": site_id,
            "status": plan["status"],
            "topologies": site["topologies"],
            "sampling": site["sampling"],
            "identity": site["identity"],
            "architecture": site["architecture"],
            "local_measurements": local_record,
        }
        if plan["status"] not in {"scoreable", "local_only"}:
            records.append(base)
            continue
        branch = plan["physical_branch"]
        branch_name = str(branch["branch"])
        physical_run = f"{context_id}-{branch_name}"
        physical_item = physical_index[physical_run]
        physical_scores = physical_item["scores"]
        if plan["monitor_horizon"] == "through_fault_step_only":
            physical_scores = physical_scores[: step + 1]
        branch_contribution = contributions[f"branch:{branch_name}"]
        faulted_scores = reconstruct_cumulative_scores(
            physical_scores,
            fault_step=step,
            replacement_contribution=contributions[f"site:{site_id}"],
            physical_contribution=branch_contribution,
            np=np,
        )
        clean_evidence_scores = reconstruct_cumulative_scores(
            physical_scores,
            fault_step=step,
            replacement_contribution=clean_contribution,
            physical_contribution=branch_contribution,
            np=np,
        )
        terminal = plan["terminal_result"]
        control = branches["control"]["result"]
        terminal_success = (
            bool(terminal["success"])
            if terminal is not None and terminal.get("status") == "complete"
            else None
        )
        control_success = (
            bool(control["success"]) if control.get("status") == "complete" else None
        )
        replay_exact = bool(
            terminal is not None
            and replay_is_exact(control)
            and replay_is_exact(terminal)
        )
        branch_check = check_by_branch[branch_name]
        faulted_summary = _alarm_summary(
            faulted_scores,
            step=step,
            bands=bands,
            alphas=alphas,
            primary=primary,
            np=np,
        )
        clean_summary = _alarm_summary(
            clean_evidence_scores,
            step=step,
            bands=bands,
            alphas=alphas,
            primary=primary,
            np=np,
        )
        group = plan["command_group"]
        base.update(
            {
                "status": "scored",
                "physical_run": physical_run,
                "monitor_horizon": plan["monitor_horizon"],
                "terminal_evidence": plan["terminal_evidence"],
                "control_success": control_success,
                "terminal_success": terminal_success,
                "policy_failure": (
                    bool(control_success and not terminal_success)
                    if terminal_success is not None and control_success is not None
                    else None
                ),
                "context_replay_exact": replay_exact,
                "physical_feature_exact": branch_check["feature_exact_equal"],
                "command_id": group["command_id"] if group is not None else None,
                "representative_site_id": (
                    str(group["representative_site_id"])
                    if group is not None
                    else None
                ),
                "command_group_size": (
                    len(group["member_site_ids"]) if group is not None else 1
                ),
                "safe_faulted_evidence": faulted_summary,
                "safe_clean_evidence_same_suffix": clean_summary,
                "safe_contribution": {
                    "faulted": contributions[f"site:{site_id}"],
                    "clean": clean_contribution,
                    "physical": branch_contribution,
                    "faulted_minus_clean": (
                        contributions[f"site:{site_id}"] - clean_contribution
                    ),
                },
            }
        )
        base["primary_eligible"] = bool(
            plan["status"] == "scoreable"
            and control_success is True
            and terminal_success is not None
            and replay_exact
            and branch_check["feature_exact_equal"]
        )
        record_id = base["record_id"]
        score_arrays[record_id] = (faulted_scores, clean_evidence_scores)
        records.append(base)

    for check in branch_checks:
        branch_name = str(check["physical_branch"])
        group = next(
            (
                value["command_group"]
                for value in summary["branches"]
                if str(value["branch"]) == branch_name
                and value.get("command_group") is not None
            ),
            None,
        )
        representative = (
            str(group["representative_site_id"]) if group is not None else None
        )
        if representative is None:
            check["alarm_timeline_exact_equal"] = True
            continue
        record_id = f"{context_id}:{representative}"
        if record_id not in score_arrays:
            check["alarm_timeline_exact_equal"] = False
            continue
        reconstructed = score_arrays[record_id][0]
        physical = physical_index[f"{context_id}-{branch_name}"]["scores"]
        comparison = composition_check(reconstructed, physical, primary, np)
        check.update(comparison)

    context_record = {
        "context_id": context_id,
        "status": "complete",
        "context": context,
        "control_success": bool(branches["control"]["result"].get("success")),
        "control_replay_exact": replay_is_exact(branches["control"]["result"]),
        "local_interventions": int(summary["local_interventions"]),
        "command_changing_interventions": int(summary["command_changing_interventions"]),
        "unique_faulted_commands": int(summary["unique_faulted_commands"]),
    }
    return context_record, records, score_arrays, branch_checks
