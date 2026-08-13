from typing import Any

from embodied_silent_failures.evidence_graph.record import Recorder
from embodied_silent_failures.evidence_graph.torch_trace import (
    capture_torch_operations,
)


SAFE_REVISION = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
SAFE_PAPER = "paper:safe-v2-arxiv-2506.09937:sec4.2+appendix-b.1+b.2"
REQUIRED_ENDPOINTS = ("safe.monitor_input", "safe.score", "safe.alarm")


# SAFE b6036ab, data.openvla.load_rollouts and process_tensor_idx_rel: load the
# seven final-layer action-token features and, for token_idx_rel=1.0, select the
# seventh token as the 4096-coordinate feature supplied to the frozen monitor.
FEATURE_BASIS = (
    "code:safe@b6036ab:failure_prob.data.openvla.load_rollouts+"
    "failure_prob.data.utils.process_tensor_idx_rel:select-final-action-token-feature"
)

# SAFE b6036ab, IndepModel.forward: project each timestep's feature to one
# scalar and apply the configured cumulative or running-mean aggregation.
SCORE_BASIS = (
    "code:safe@b6036ab:failure_prob.model.indep.IndepModel.forward:"
    "project-feature-and-aggregate-score-over-time"
)


def monitored_features(
    recorder: Recorder, torch: Any, generated: Any
) -> tuple[Any, Any]:
    final_layer_states = [
        token_states[-1] for token_states in generated["hidden_states"]
    ]
    per_token = torch.stack(
        [token_states[0, -1, :] for token_states in final_layer_states],
        dim=0,
    )
    if per_token.ndim != 2 or per_token.shape[0] != 7:
        raise ValueError(
            f"unexpected OpenVLA hidden-state shape: {tuple(per_token.shape)}"
        )
    feature = per_token[-1]
    recorder.mark(
        "safe.final_layer_action_features",
        inputs=final_layer_states,
        outputs=per_token,
        basis=[SAFE_PAPER, FEATURE_BASIS],
        region="safe_feature_extraction",
        fault_interface="final_layer_action_features",
        details={
            "action_token_positions": list(range(int(per_token.shape[0]))),
            "feature_dimension": int(per_token.shape[-1]),
        },
    )
    recorder.mark(
        "safe.monitor_input",
        inputs=per_token,
        outputs=feature,
        basis=[SAFE_PAPER, FEATURE_BASIS],
        region="safe_feature_selection",
        role="monitor_input",
        fault_interface="safe_feature",
    )
    return per_token, feature


def feature_history(
    recorder: Recorder,
    torch: Any,
    prior_features: Any,
    current_feature: Any,
) -> Any:
    prior = torch.as_tensor(prior_features).to(current_feature.device)
    recorder.source(
        "safe.prior_feature_history",
        prior,
        basis="protocol:evidence-graph-v1:replayed-clean-prefix-features",
        region="safe_feature_history",
        lifetime="temporal",
    )
    history = torch.cat((prior, current_feature.unsqueeze(0)), dim=0)
    recorder.mark(
        "safe.feature_history",
        inputs={"prior": prior, "current": current_feature},
        outputs=history,
        basis=[SAFE_PAPER, FEATURE_BASIS],
        region="safe_feature_history",
        lifetime="temporal",
        fault_interface="safe_feature_history",
    )
    return history


def run_monitor(
    recorder: Recorder,
    *,
    torch: Any,
    monitor: Any,
    feature_history: Any,
    policy_step: int,
    threshold: float,
) -> tuple[float, bool]:
    batch = {"features": feature_history.float().unsqueeze(0).to("cuda")}
    recorder.mark(
        "safe.monitor_batch",
        inputs=feature_history,
        outputs=batch,
        basis=SCORE_BASIS,
        region="safe_monitor_input_conversion",
        fault_interface="safe_monitor_batch",
    )
    with recorder.scope(phase="monitor"), capture_torch_operations(
        recorder, {"safe_monitor": monitor}
    ), torch.no_grad():
        scores = monitor(batch).squeeze(0).squeeze(-1)
    score = float(scores[-1].item())
    alarm = bool(score >= threshold)
    recorder.mark(
        "safe.score",
        inputs=scores,
        outputs=scores,
        basis=[SAFE_PAPER, SCORE_BASIS],
        region="safe_monitor",
        role="sink",
        fault_interface="safe_score",
        details={"policy_step": policy_step, "current_score": score},
    )
    recorder.mark(
        "safe.alarm",
        inputs={"scores": scores, "threshold": threshold},
        outputs=alarm,
        basis="protocol:frozen-safe-band:score-greater-than-or-equal-to-threshold",
        region="safe_alarm",
        role="sink",
        details={"policy_step": policy_step, "threshold": threshold},
    )
    return score, alarm


def operator_annotations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations = []
    for event in events:
        if event["kind"] not in {"module", "operator", "state"}:
            continue
        phase = event.get("context", {}).get("phase")
        details = event.get("details", {})
        registrations = details.get("registrations", [])
        is_monitor_state = details.get("root") == "safe_monitor" and bool(registrations)
        if phase != "monitor" and not is_monitor_state:
            continue
        calls = details.get("module_calls", [])
        scope = details.get("module_path") or (
            calls[-1].get("path") if calls else ""
        )
        call_index = details.get("module_call_index")
        if call_index is None and calls:
            call_index = calls[-1].get("call_index")
        annotations.append(
            {
                "event_id": event["event_id"],
                "region": (
                    "safe_monitor_parameters" if is_monitor_state else "safe_monitor"
                ),
                "basis": [SAFE_PAPER, SCORE_BASIS, "observed:torch-dispatch"],
                "lifetime": "step",
                "fault_interface": (
                    "safe_monitor_model_state"
                    if is_monitor_state
                    else "safe_private_compute"
                ),
                "semantic_key": (
                    f"safe_monitor_parameters/state/{registrations[0]['module_path']}"
                    if is_monitor_state
                    else (
                        f"safe_monitor/{scope}/call_{int(call_index)}"
                        if scope and call_index is not None
                        else f"safe_monitor/{scope or event['name']}"
                    )
                ),
            }
        )
    return annotations


def contract_issues(events: list[dict[str, Any]]) -> list[str]:
    feature_events = [
        event for event in events if event["name"] == "safe.final_layer_action_features"
    ]
    if not feature_events:
        return ["SAFE feature boundary is missing"]
    issues = []
    for event in feature_events:
        details = event.get("details", {})
        if details.get("action_token_positions") != list(range(7)):
            issues.append("SAFE feature extraction did not observe all seven action tokens")
        if details.get("feature_dimension") != 4096:
            issues.append(
                "SAFE feature dimension is "
                f"{details.get('feature_dimension')}, expected 4096 for pinned OpenVLA"
            )
    return issues
