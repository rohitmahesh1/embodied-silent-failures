from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


FRESHNESS_GATES = (
    "none",
    "source_metadata",
    "exact_duplicate",
    "either",
)
FRESHNESS_RESPONSES = ("observe", "hold")


@dataclass(frozen=True)
class FreshnessSignals:
    policy_step: int
    image_source_policy_step: int
    source_metadata_age_steps: int
    source_metadata_alarm: bool
    relabelled_metadata_alarm: bool
    exact_duplicate_alarm: bool
    input_sha256: str
    previous_input_sha256: str | None

    def selected_alarm(self, gate: str) -> bool:
        if gate == "none":
            return False
        if gate == "source_metadata":
            return self.source_metadata_alarm
        if gate == "exact_duplicate":
            return self.exact_duplicate_alarm
        if gate == "either":
            return self.source_metadata_alarm or self.exact_duplicate_alarm
        raise ValueError(f"unknown freshness gate: {gate}")

    def to_row(self, gate: str, response_applied: bool) -> dict[str, Any]:
        return {
            "freshness/image_source_policy_step": self.image_source_policy_step,
            "freshness/source_metadata_age_steps": self.source_metadata_age_steps,
            "freshness/source_metadata_alarm": self.source_metadata_alarm,
            "freshness/relabelled_metadata_alarm": self.relabelled_metadata_alarm,
            "freshness/exact_duplicate_alarm": self.exact_duplicate_alarm,
            "freshness/selected_gate_alarm": self.selected_alarm(gate),
            "freshness/response_applied": response_applied,
        }

    def intervention_record(
        self, gate: str, response_applied: bool
    ) -> dict[str, Any]:
        return {
            "image_source_policy_step": self.image_source_policy_step,
            "source_metadata_age_steps": self.source_metadata_age_steps,
            "source_metadata_alarm": self.source_metadata_alarm,
            "relabelled_metadata_alarm": self.relabelled_metadata_alarm,
            "exact_duplicate_alarm": self.exact_duplicate_alarm,
            "selected_gate_alarm": self.selected_alarm(gate),
            "response_applied": response_applied,
            "input_sha256": self.input_sha256,
            "previous_input_sha256": self.previous_input_sha256,
        }


def _image_sha256(np: Any, image: Any) -> str:
    array = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def observe_freshness(
    np: Any,
    *,
    policy_step: int,
    policy_image: Any,
    previous_policy_image: Any | None,
    image_source_policy_step: int,
) -> FreshnessSignals:
    if policy_step < 0:
        raise ValueError("policy step must be non-negative")
    if not 0 <= image_source_policy_step <= policy_step:
        raise ValueError("image source step must be between zero and the policy step")

    input_sha256 = _image_sha256(np, policy_image)
    previous_sha256 = (
        _image_sha256(np, previous_policy_image)
        if previous_policy_image is not None
        else None
    )
    exact_duplicate = bool(
        previous_policy_image is not None
        and np.array_equal(policy_image, previous_policy_image)
    )
    age = policy_step - image_source_policy_step

    # OpenVLA 300dce2, run_libero_eval.py:245 and :365, consumes synchronous
    # observations without camera timestamps. The stale-image injector's
    # source step is therefore a simulator proxy for metadata that remained
    # bound to its pixels, not evidence supplied by LIBERO or OpenVLA.
    return FreshnessSignals(
        policy_step=policy_step,
        image_source_policy_step=image_source_policy_step,
        source_metadata_age_steps=age,
        source_metadata_alarm=age > 0,
        # If a downstream consumer assigns current metadata to old pixels, a
        # timestamp or frame-ID check sees a current packet and cannot alarm.
        relabelled_metadata_alarm=False,
        exact_duplicate_alarm=exact_duplicate,
        input_sha256=input_sha256,
        previous_input_sha256=previous_sha256,
    )


def hold_action(np: Any, previous_action: Any) -> Any:
    previous = np.asarray(previous_action)
    if previous.shape != (7,):
        raise ValueError(f"previous action has shape {previous.shape}, expected (7,)")

    # ActFovea arXiv:2607.29169, Risk-Adaptive Execution and Safe Failure:
    # holding suppresses motion while carrying forward the gripper command.
    action = np.zeros(7, dtype=previous.dtype)
    action[-1] = previous[-1]
    return action


def summarize_freshness(
    rows: list[dict[str, Any]], *, gate: str, response: str
) -> dict[str, Any]:
    if gate not in FRESHNESS_GATES:
        raise ValueError(f"unknown freshness gate: {gate}")
    if response not in FRESHNESS_RESPONSES:
        raise ValueError(f"unknown freshness response: {response}")

    key = "freshness/selected_gate_alarm"
    evaluated = [row for row in rows if key in row]
    return {
        "schema_version": 1,
        "gate": gate,
        "response": response,
        "response_scope": "declared_intervention_step_only",
        "evaluated_policy_steps": len(evaluated),
        "source_metadata_alarms": sum(
            bool(row["freshness/source_metadata_alarm"]) for row in evaluated
        ),
        "relabelled_metadata_alarms": sum(
            bool(row["freshness/relabelled_metadata_alarm"]) for row in evaluated
        ),
        "exact_duplicate_alarms": sum(
            bool(row["freshness/exact_duplicate_alarm"]) for row in evaluated
        ),
        "selected_gate_alarms": sum(bool(row[key]) for row in evaluated),
        "responses_applied": sum(
            bool(row["freshness/response_applied"]) for row in evaluated
        ),
    }
