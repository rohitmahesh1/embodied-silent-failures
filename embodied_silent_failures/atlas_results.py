from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, load_json


def _find_context_dir(
    campaign_dirs: list[Path], context_id: str
) -> tuple[Path | None, list[Path]]:
    matches = [
        root / "contexts" / context_id
        for root in campaign_dirs
        if (root / "contexts" / context_id).is_dir()
    ]
    return (matches[0] if len(matches) == 1 else None), matches


def _physical_plan(
    summary: dict[str, Any], local_record: dict[str, Any]
) -> dict[str, Any]:
    control = next(
        (value for value in summary.get("branches", []) if value.get("branch") == "control"),
        None,
    )
    if control is None:
        return {
            "physical_status": "unavailable",
            "physical_reason": "control_branch_missing",
        }
    command_groups = {}
    for group in summary.get("command_groups", []):
        for site_id in group["member_site_ids"]:
            if site_id in command_groups:
                raise ValueError(f"site {site_id} belongs to two command groups")
            command_groups[site_id] = group
    command_branches = {
        str(value["command_group"]["command_id"]): value
        for value in summary.get("branches", [])
        if value.get("command_group") is not None
    }
    if local_record["executed_command"]["exact_equal"]:
        selected = control
        evidence = "matched clean branch because the executed command is exact"
        command_group = None
    else:
        command_group = command_groups.get(str(local_record["site_id"]))
        if command_group is None:
            return {
                "physical_status": "unavailable",
                "physical_reason": "changed_command_has_no_group",
            }
        selected = command_branches.get(str(command_group["command_id"]))
        if selected is None:
            return {
                "physical_status": "unavailable",
                "physical_reason": (
                    summary.get("faulted_terminal_skip_reason")
                    or "terminal_branch_deferred_or_missing"
                ),
                "command_group": command_group,
            }
        evidence = "observed terminal branch for this exact executed command"
    result = selected["result"]
    if result.get("status") != "complete":
        return {
            "physical_status": "unresolved",
            "physical_reason": result.get("reason", "terminal_branch_unresolved"),
            "physical_branch": selected["branch"],
            "command_group": command_group,
        }
    control_result = control["result"]
    control_success = (
        bool(control_result["success"])
        if control_result.get("status") == "complete"
        else None
    )
    terminal_success = bool(result["success"])
    return {
        "physical_status": "complete",
        "physical_branch": selected["branch"],
        "physical_evidence": evidence,
        "control_success": control_success,
        "terminal_success": terminal_success,
        "policy_failure": (
            bool(control_success and not terminal_success)
            if control_success is not None
            else None
        ),
        "command_group": command_group,
    }


def consolidate_intervention_atlas(
    manifest_path: Path, campaign_dirs: list[Path]
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    expected_manifest_hash = file_sha256(manifest_path)
    worker_runs = []
    for directory in campaign_dirs:
        run_path = directory / "run.json"
        run = load_json(run_path)
        if run.get("campaign") != manifest.get("campaign"):
            raise ValueError(f"campaign identity mismatch: {directory}")
        if run.get("execution", {}).get("manifest_file_sha256") != expected_manifest_hash:
            raise ValueError(f"manifest hash mismatch: {directory}")
        worker_runs.append(
            {
                "directory": str(directory.resolve()),
                "run_sha256": file_sha256(run_path),
                "worker_shard": int(run["worker_shard"]),
            }
        )
    if len({value["worker_shard"] for value in worker_runs}) != len(worker_runs):
        raise ValueError("two campaign directories claim the same worker shard")

    contexts = []
    interventions = []
    counts = Counter()
    for context in manifest["contexts"]:
        context_id = str(context["context_id"])
        context_dir, matches = _find_context_dir(campaign_dirs, context_id)
        if len(matches) > 1:
            raise ValueError(f"context appears in multiple campaign directories: {context_id}")
        if context_dir is None:
            contexts.append(
                {"context_id": context_id, "status": "missing", "context": context}
            )
            counts["contexts_missing"] += 1
            continue
        complete_path = context_dir / "context.complete.json"
        local_path = context_dir / "local.json"
        if not complete_path.is_file() or not local_path.is_file():
            unresolved_path = context_dir / "context.unresolved.json"
            unresolved = load_json(unresolved_path) if unresolved_path.is_file() else {}
            contexts.append(
                {
                    "context_id": context_id,
                    "status": str(unresolved.get("status", "incomplete")),
                    "context": context,
                    "errors": unresolved.get("errors", []),
                }
            )
            counts["contexts_unresolved"] += 1
            continue
        summary = load_json(complete_path)
        local = load_json(local_path)
        contexts.append(
            {
                "context_id": context_id,
                "status": "complete",
                "context": context,
                "local_complete": int(summary["local_complete"]),
                "local_unresolved": int(summary["local_unresolved"]),
                "unique_faulted_commands": int(summary["unique_faulted_commands"]),
                "terminal_unresolved": int(summary["terminal_unresolved"]),
                "source_collection": local["source_collection"],
                "current_collection": local["current_collection"],
            }
        )
        counts["contexts_complete"] += 1
        for record in local["interventions"]:
            base = {
                "record_id": f"{context_id}:{record['site_id']}",
                "context_id": context_id,
                "context": context,
                "site_id": record["site_id"],
                "status": record["status"],
            }
            if record.get("status") != "complete":
                interventions.append({**base, "local_record": record})
                counts["interventions_unresolved"] += 1
                continue
            physical = _physical_plan(summary, record)
            interventions.append(
                {
                    **base,
                    "topologies": record["topologies"],
                    "sampling": record["sampling"],
                    "local_measurements": {
                        key: record[key]
                        for key in (
                            "fault",
                            "raw_action",
                            "executed_command",
                            "action_tokens",
                            "action_logits",
                            "safe_input",
                            "inference_seconds",
                        )
                    },
                    **physical,
                    "safe_scoring_status": "pending",
                }
            )
            counts["interventions_complete"] += 1
            counts[f"physical_{physical['physical_status']}"] += 1
            if physical.get("policy_failure") is True:
                counts["policy_failures"] += 1
    return {
        "schema_version": 1,
        "analysis": "lossless consolidation of the graph-derived intervention atlas",
        "source": {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": expected_manifest_hash,
            "worker_runs": worker_runs,
        },
        "analysis_contract": manifest["analysis_contract"],
        "coverage": dict(sorted(counts.items())),
        "contexts": contexts,
        "interventions": interventions,
    }
