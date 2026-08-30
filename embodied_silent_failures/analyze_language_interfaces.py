from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_interface import boundary_replay_targets
from embodied_silent_failures.provenance import file_sha256, load_json


REPLAY_KINDS = ["immediate", "final"]
EXACT_COMPONENTS = (
    "cache_cut_keys_exact",
    "cache_cut_values_exact",
    "downstream_residuals_exact",
    "downstream_cache_keys_exact",
    "downstream_cache_values_exact",
    "action_logits_exact",
    "action_tokens_exact",
    "raw_action_exact",
    "executed_command_exact",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cache-aware OpenVLA boundary replays."
    )
    parser.add_argument(
        "--campaign-dir",
        action="append",
        dest="campaign_dirs",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--split", choices=("development", "holdout", "all"), default="development"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    return parser.parse_args()


def _exact(record: dict[str, Any], *path: str) -> bool:
    value: Any = record
    for name in path:
        if not isinstance(value, dict) or name not in value:
            return False
        value = value[name]
    return value is True


def replay_row(
    *,
    context: dict[str, Any],
    intervention: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    boundary_layer = int(replay["boundary_layer"])
    propagation = {
        int(value["layer_index"]): value
        for value in intervention.get("propagation", [])
    }
    boundary = propagation.get(boundary_layer)
    if boundary is None:
        raise ValueError(
            f"layer {intervention['layer_index']} has no propagation record at "
            f"boundary {boundary_layer}"
        )
    status = str(replay.get("status"))
    row = {
        "context_id": str(context["context_id"]),
        "analysis_split": str(context["analysis_split"]),
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "phase": str(context["phase"]),
        "action_token_position": int(context["action_token_position"]),
        "injection_layer": int(replay["injection_layer"]),
        "boundary_kind": str(replay["boundary_kind"]),
        "boundary_layer": boundary_layer,
        "status": status,
        "injection_nontrivial": not bool(intervention["injection"]["exact_equal"]),
        "boundary_residual_nontrivial": not bool(boundary["exact_equal"]),
        "fault_action_tokens_changed": not bool(
            intervention["action_tokens_exact_equal"]
        ),
        "fault_command_changed": not bool(
            intervention["executed_command"]["exact_equal"]
        ),
        "cache_precondition_keys_exact": _exact(
            intervention, "cache_precondition", "key", "all_coordinates_exact"
        ),
        "cache_precondition_values_exact": _exact(
            intervention, "cache_precondition", "value", "all_coordinates_exact"
        ),
        "cache_cut_keys_exact": _exact(
            replay, "cache_cut", "keys", "all_coordinates_exact"
        ),
        "cache_cut_values_exact": _exact(
            replay, "cache_cut", "values", "all_coordinates_exact"
        ),
        "downstream_residuals_exact": _exact(
            replay, "residual_path", "all_coordinates_exact"
        ),
        "downstream_cache_keys_exact": _exact(
            replay, "attention_cache_keys", "all_coordinates_exact"
        ),
        "downstream_cache_values_exact": _exact(
            replay, "attention_cache_values", "all_coordinates_exact"
        ),
        "action_logits_exact": replay.get("action_logits_exact_equal") is True,
        "action_tokens_exact": replay.get("action_tokens_exact_equal") is True,
        "raw_action_exact": _exact(replay, "raw_action", "exact_equal"),
        "executed_command_exact": _exact(
            replay, "executed_command", "exact_equal"
        ),
    }
    row["closure_exact"] = bool(
        status == "complete" and all(row[name] is True for name in EXACT_COMPONENTS)
    )
    return row


def replay_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    nontrivial_injection = [row for row in complete if row["injection_nontrivial"]]
    nontrivial_boundary = [
        row for row in complete if row["boundary_residual_nontrivial"]
    ]

    def exact_count(values: list[dict[str, Any]]) -> int:
        return sum(row["closure_exact"] for row in values)

    return {
        "records": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "complete_records": len(complete),
        "nontrivial_injection_records": len(nontrivial_injection),
        "nontrivial_boundary_records": len(nontrivial_boundary),
        "exact_closure_records": exact_count(complete),
        "exact_closure_given_nontrivial_injection": {
            "numerator": exact_count(nontrivial_injection),
            "denominator": len(nontrivial_injection),
        },
        "exact_closure_given_nontrivial_boundary": {
            "numerator": exact_count(nontrivial_boundary),
            "denominator": len(nontrivial_boundary),
        },
        "component_exact_counts": {
            name: sum(row[name] is True for row in complete)
            for name in EXACT_COMPONENTS
        },
        "cache_precondition_exact_records": sum(
            row["cache_precondition_keys_exact"]
            and row["cache_precondition_values_exact"]
            for row in complete
        ),
    }


def analyze(campaign_dirs: list[Path], split: str) -> dict[str, Any]:
    runs = [load_json(path / "run.json") for path in campaign_dirs]
    shards = [int(value["worker_shard"]) for value in runs]
    if len(shards) != len(set(shards)):
        raise ValueError("interface campaign inputs repeat a worker shard")
    manifest_hashes = {
        str(value["execution"]["manifest_content_sha256"]) for value in runs
    }
    if len(manifest_hashes) != 1:
        raise ValueError("interface campaign shards use different manifests")

    planned_ids = [
        str(context_id) for value in runs for context_id in value["context_ids"]
    ]
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError("interface campaign shards repeat a context")

    rows: list[dict[str, Any]] = []
    contexts = []
    for campaign_dir, run in zip(campaign_dirs, runs, strict=True):
        for context_id in run["context_ids"]:
            context_dir = campaign_dir / "contexts" / str(context_id)
            complete_path = context_dir / "context.complete.json"
            if not complete_path.is_file():
                unresolved_path = context_dir / "context.unresolved.json"
                unresolved = (
                    load_json(unresolved_path) if unresolved_path.is_file() else {}
                )
                context = unresolved.get("context")
                if not isinstance(context, dict):
                    raise ValueError(f"unresolved context lacks metadata: {context_id}")
                if split == "all" or context.get("analysis_split") == split:
                    contexts.append(
                        {
                            "context_id": str(context_id),
                            "status": "unresolved",
                            "context": context,
                            "interface_archive": None,
                        }
                    )
                continue

            complete = load_json(complete_path)
            local = load_json(context_dir / "local.json")
            context = local["context"]
            if str(context["context_id"]) != str(context_id):
                raise ValueError(f"context identity mismatch: {context_id}")
            if split != "all" and context.get("analysis_split") != split:
                continue
            interventions = {
                int(value["layer_index"]): value
                for value in local["interventions"]
                if value.get("status") == "complete"
            }
            expected = {
                (layer, kind, boundary)
                for layer in interventions
                for kind, boundary in boundary_replay_targets(layer, REPLAY_KINDS)
            }
            actual_records = local.get("boundary_replays", [])
            actual = [
                (
                    int(value["injection_layer"]),
                    str(value["boundary_kind"]),
                    int(value["boundary_layer"]),
                )
                for value in actual_records
            ]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError(
                    f"boundary replay target coverage mismatch: {context_id}"
                )
            for replay in actual_records:
                layer = int(replay["injection_layer"])
                rows.append(
                    replay_row(
                        context=context,
                        intervention=interventions[layer],
                        replay=replay,
                    )
                )
            archive = local.get("interface_archive")
            archive_path = None
            archive_hash_verified = None
            if isinstance(archive, dict):
                archive_path = context_dir / str(archive["artifact"]["name"])
                if archive_path.is_file():
                    archive_hash_verified = (
                        file_sha256(archive_path) == archive["artifact"]["sha256"]
                    )
            contexts.append(
                {
                    "context_id": str(context_id),
                    "status": "complete",
                    "context": context,
                    "complete_record_sha256": file_sha256(complete_path),
                    "local_record_sha256": file_sha256(context_dir / "local.json"),
                    "complete_interventions": len(interventions),
                    "local_errors": int(complete.get("local_errors", 0)),
                    "interface_archive": archive,
                    "interface_archive_recorded_complete": bool(
                        complete.get("interface_archive_complete")
                    ),
                    "interface_archive_locally_present": bool(
                        archive_path is not None and archive_path.is_file()
                    ),
                    "interface_archive_hash_verified": archive_hash_verified,
                    "boundary_replays_complete": int(
                        complete.get("boundary_replays_complete", 0)
                    ),
                    "boundary_replays_error": int(
                        complete.get("boundary_replays_error", 0)
                    ),
                }
            )

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["boundary_kind"])].append(row)
        by_layer[str(row["injection_layer"])].append(row)
        by_token[str(row["action_token_position"])].append(row)
    complete_contexts = [value for value in contexts if value["status"] == "complete"]
    return {
        "schema_version": 1,
        "analysis": "cache-aware OpenVLA boundary replay audit",
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "analysis_split": split,
        "interface_claim": (
            "Replay the measured post-block residual and exact differential "
            "current-token key/value cache entries through the declared boundary."
        ),
        "source_campaigns": [
            {
                "directory": str(path.resolve()),
                "run_sha256": file_sha256(path / "run.json"),
                "worker_shard": int(run["worker_shard"]),
            }
            for path, run in zip(campaign_dirs, runs, strict=True)
        ],
        "manifest_content_sha256": next(iter(manifest_hashes)),
        "coverage": {
            "declared_contexts_all_splits": len(planned_ids),
            "selected_contexts": len(contexts),
            "complete_contexts": len(complete_contexts),
            "unresolved_contexts": len(contexts) - len(complete_contexts),
            "interface_archives_recorded_complete": sum(
                value.get("interface_archive_recorded_complete") is True
                for value in complete_contexts
            ),
            "interface_archives_locally_present": sum(
                value.get("interface_archive_locally_present") is True
                for value in complete_contexts
            ),
            "interface_archives_hash_verified": sum(
                value.get("interface_archive_hash_verified") is True
                for value in complete_contexts
            ),
            "interface_archive_bytes": sum(
                int(value["interface_archive"]["artifact"]["bytes"])
                for value in complete_contexts
                if isinstance(value.get("interface_archive"), dict)
            ),
        },
        "overall": replay_summary(rows),
        "groups": {
            "boundary_kind": {
                key: replay_summary(value) for key, value in sorted(by_kind.items())
            },
            "injection_layer": {
                key: replay_summary(value)
                for key, value in sorted(
                    by_layer.items(), key=lambda item: int(item[0])
                )
            },
            "action_token_position": {
                key: replay_summary(value)
                for key, value in sorted(
                    by_token.items(), key=lambda item: int(item[0])
                )
            },
        },
        "contexts": contexts,
        "records": rows,
    }


def main() -> None:
    args = _arguments()
    output = analyze(args.campaign_dirs, args.split)
    write_json_atomic(args.output, output)
    columns = sorted({name for row in output["records"] for name in row})
    write_csv_atomic(
        args.records_csv,
        [{name: row.get(name, "") for name in columns} for row in output["records"]],
    )
    print(
        json.dumps(
            {
                "analysis_split": output["analysis_split"],
                "coverage": output["coverage"],
                "overall": output["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
