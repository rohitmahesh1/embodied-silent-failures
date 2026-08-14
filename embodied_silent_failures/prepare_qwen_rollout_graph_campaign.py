import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.qwen_artifacts import file_sha256


SELECTION_BASIS = (
    "protocol:qwen-rollout-graph-v1:reuse-existing-safe-viewer-"
    "representatives-before-qwen-results"
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze video-backed reruns of the existing SAFE viewer rollouts."
    )
    parser.add_argument("--reference-campaign-dir", required=True, type=Path)
    parser.add_argument("--reference-output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--safe-viewer-data",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "viewer/public/data/openvla-safe-graph.json"
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_path(reference: Path, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = reference / "manifests" / path.name
    if not candidate.is_file():
        raise FileNotFoundError(f"reference manifest is unavailable: {value}")
    return candidate


def _representatives(
    reference: Path, campaign: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    stages = campaign.get("stages", [])
    ordinary_stage = next(
        (stage for stage in stages if stage.get("name") == "census-wave-0"), None
    )
    stale_stage = next(
        (
            stage
            for stage in stages
            if stage.get("kind") == "stale_image"
            and stage.get("image_input_mode") == "stale"
        ),
        None,
    )
    control_stage = next(
        (
            stage
            for stage in stages
            if stage.get("kind") == "stale_image"
            and stage.get("image_input_mode") == "current_control"
            and stale_stage is not None
            and Path(str(stage.get("manifest"))).name
            == Path(str(stale_stage.get("manifest"))).name
        ),
        None,
    )
    if ordinary_stage is None or stale_stage is None or control_stage is None:
        raise ValueError("reference campaign has no ordinary and stale representatives")
    ordinary_manifest = _load(
        _manifest_path(reference, str(ordinary_stage["manifest"]))
    )
    pair_manifest = _load(_manifest_path(reference, str(stale_stage["manifest"])))
    ordinary = ordinary_manifest.get("trials", [])[0]
    pair = pair_manifest.get("trials", [])[0]
    if not isinstance(ordinary, dict) or not isinstance(pair, dict):
        raise ValueError("reference representative manifests are empty")
    return {
        "ordinary": {"stage": ordinary_stage["name"], "trial": ordinary},
        "control": {
            "stage": control_stage["name"],
            "trial": pair,
        },
        "stale": {"stage": stale_stage["name"], "trial": pair},
    }


def _verify_published_graphs(
    reference_output: Path,
    viewer: dict[str, Any],
    representatives: dict[str, dict[str, Any]],
) -> dict[str, str]:
    published = {
        source["mode"]: source["graphSha256"]
        for source in viewer.get("sources", [])
        if source.get("mode") in representatives
    }
    if set(published) != set(representatives):
        raise ValueError("SAFE viewer does not identify all three source modes")
    hashes = {}
    for mode, record in representatives.items():
        trial = record["trial"]
        graph = (
            reference_output
            / "evidence"
            / record["stage"]
            / f"task{trial['task_id']}--ep{trial['episode_index']}"
            / "graph.json"
        )
        digest = file_sha256(graph)
        if digest != published[mode]:
            raise ValueError(f"{mode} reference graph differs from the SAFE viewer")
        hashes[mode] = digest
    return hashes


def prepare_campaign(
    reference_campaign_dir: Path,
    reference_output_root: Path,
    output_dir: Path,
    safe_viewer_data: Path,
) -> dict[str, Any]:
    reference_campaign_dir = reference_campaign_dir.resolve()
    reference_output_root = reference_output_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"campaign output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign = _load(reference_campaign_dir / "campaign.json")
    viewer = _load(safe_viewer_data.resolve())
    representatives = _representatives(reference_campaign_dir, campaign)
    graph_hashes = _verify_published_graphs(
        reference_output_root, viewer, representatives
    )

    ordinary_manifest = output_dir / "ordinary.json"
    intervention_manifest = output_dir / "intervention.json"
    write_json_atomic(
        ordinary_manifest,
        {
            "schema_version": 1,
            "selection_basis": SELECTION_BASIS,
            "trials": [representatives["ordinary"]["trial"]],
        },
    )
    write_json_atomic(
        intervention_manifest,
        {
            "schema_version": 1,
            "selection_basis": SELECTION_BASIS,
            "trials": [representatives["stale"]["trial"]],
        },
    )
    stages = [
        {
            "name": "ordinary",
            "kind": "clean",
            "manifest": str(ordinary_manifest),
            "trace_steps": [100],
            "expected_trials": 1,
            "allow_exclusions": False,
            "save_video": True,
        },
        {
            "name": "control",
            "kind": "stale_image",
            "image_input_mode": "current_control",
            "manifest": str(intervention_manifest),
            "trace_steps": [],
            "expected_trials": 1,
            "allow_exclusions": False,
            "save_video": True,
        },
        {
            "name": "stale",
            "kind": "stale_image",
            "image_input_mode": "stale",
            "manifest": str(intervention_manifest),
            "trace_steps": [],
            "expected_trials": 1,
            "allow_exclusions": False,
            "save_video": True,
        },
    ]
    result = {
        "schema_version": 1,
        "selection_basis": SELECTION_BASIS,
        "reference_campaign_sha256": file_sha256(
            reference_campaign_dir / "campaign.json"
        ),
        "reference_output_root": str(reference_output_root),
        "safe_viewer_data_sha256": file_sha256(safe_viewer_data.resolve()),
        "reference_graph_sha256": graph_hashes,
        "reference_trials": {
            mode: {
                "task_id": int(record["trial"]["task_id"]),
                "episode_index": int(record["trial"]["episode_index"]),
            }
            for mode, record in representatives.items()
        },
        "stages": stages,
    }
    write_json_atomic(output_dir / "campaign.json", result)
    return result


def main() -> None:
    args = _parse_arguments()
    result = prepare_campaign(
        args.reference_campaign_dir,
        args.reference_output_root,
        args.output_dir,
        args.safe_viewer_data,
    )
    print(json.dumps({"selection_basis": result["selection_basis"], "stages": 3}))


if __name__ == "__main__":
    main()
