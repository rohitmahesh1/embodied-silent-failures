from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.intervention_atlas_followups import (
    POSTHOC_MODEL_SPECS,
    STABILITY_MODEL_NAMES,
    classification_metrics,
    fit_posthoc_model,
    fit_stability_model,
    paired_metric_bootstrap,
    phase_rank_stability,
    physical_equivalence_audit,
    probability_summary,
    rank_stability,
    validate_frozen_models,
)
from embodied_silent_failures.intervention_atlas_risk import (
    MODEL_SPECS,
    load_analysis_rows,
    model_features,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_state,
    load_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc follow-up analyses on the OpenVLA intervention atlas."
    )
    parser.add_argument("--development", action="append", required=True, type=Path)
    parser.add_argument("--holdout", action="append", required=True, type=Path)
    parser.add_argument("--frozen-fit", required=True, type=Path)
    parser.add_argument("--frozen-models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def _predict_frozen(models, rows, name):
    spec = MODEL_SPECS[name]
    features = [model_features(row, spec["features"]) for row in rows]
    return models["models"][name].predict_proba(features)[:, 1]


def _predict_posthoc(model, rows, feature_names):
    features = [model_features(row, feature_names) for row in rows]
    return model.predict_proba(features)[:, 1]


def _stability_analysis(
    development_rows,
    holdout_rows,
    *,
    folds,
    bootstrap_samples,
    seed,
):
    import numpy as np

    targets = {
        "policy_failure": {
            "outcome": "policy_failure",
            "select": lambda row: True,
        },
        "silent_failure": {
            "outcome": "silent_failure",
            "select": lambda row: True,
        },
        "safe_miss_given_policy_failure": {
            "outcome": "safe_miss_given_failure",
            "select": lambda row: bool(row["policy_failure"]),
        },
    }
    output = {}
    for target_offset, (target_name, target) in enumerate(targets.items()):
        selected_holdout = [
            row for row in holdout_rows if target["select"](row)
        ]
        labels = np.asarray(
            [int(row[target["outcome"]]) for row in selected_holdout], dtype=int
        )
        models = {}
        predictions = {}
        for name in STABILITY_MODEL_NAMES:
            result, prediction = fit_stability_model(
                development_rows,
                holdout_rows,
                name=name,
                outcome=target["outcome"],
                select=target["select"],
                folds=folds,
            )
            models[name] = result
            predictions[name] = prediction
        output[target_name] = {
            "models": models,
            "paired_holdout_comparisons": {
                "site_plus_context_over_context_only": paired_metric_bootstrap(
                    selected_holdout,
                    labels,
                    predictions["site_plus_context"],
                    predictions["context_only"],
                    samples=bootstrap_samples,
                    seed=seed + target_offset * 10,
                ),
                "interactions_over_additive": paired_metric_bootstrap(
                    selected_holdout,
                    labels,
                    predictions["site_context_interactions"],
                    predictions["site_plus_context"],
                    samples=bootstrap_samples,
                    seed=seed + target_offset * 10 + 1,
                ),
            },
            "site_rate_stability": rank_stability(
                development_rows,
                holdout_rows,
                target["outcome"],
                select=target["select"],
                samples=bootstrap_samples,
                seed=seed + target_offset * 10 + 2,
            ),
            "site_rank_stability_between_phases": {
                "development": phase_rank_stability(
                    development_rows,
                    target["outcome"],
                    select=target["select"],
                ),
                "holdout": phase_rank_stability(
                    holdout_rows,
                    target["outcome"],
                    select=target["select"],
                ),
            },
        }
    return output


def main() -> None:
    args = _arguments()
    if args.folds < 2:
        raise ValueError("at least two trajectory folds are required")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")

    import numpy as np

    development_rows, development_source = load_analysis_rows(
        args.development, analysis_split="development"
    )
    holdout_rows, holdout_source = load_analysis_rows(
        args.holdout, analysis_split="holdout"
    )
    frozen_fit = load_json(args.frozen_fit)
    if not development_rows or not holdout_rows:
        raise ValueError("development and holdout analyses must both be nonempty")
    if development_source["manifest_sha256"] != holdout_source["manifest_sha256"]:
        raise ValueError("development and holdout use different intervention manifests")
    if development_source["monitor"] != holdout_source["monitor"]:
        raise ValueError("development and holdout use different SAFE monitors")
    if frozen_fit.get("development_only") is not True:
        raise ValueError("frozen fit is not marked development-only")
    if (
        frozen_fit["source"]["manifest_sha256"]
        != development_source["manifest_sha256"]
    ):
        raise ValueError(
            "frozen fit and follow-up use different intervention manifests"
        )
    if frozen_fit["source"]["monitor"] != development_source["monitor"]:
        raise ValueError("frozen fit and follow-up use different SAFE monitors")
    if file_sha256(args.frozen_models) != frozen_fit["models"]["artifact"]["sha256"]:
        raise ValueError("frozen model hash differs from the development fit record")
    with args.frozen_models.open("rb") as file:
        frozen_models = pickle.load(file)
    validate_frozen_models(frozen_models)

    holdout_silent = np.asarray(
        [int(row["silent_failure"]) for row in holdout_rows], dtype=int
    )
    holdout_failures = np.asarray(
        [int(row["policy_failure"]) for row in holdout_rows], dtype=int
    )
    conventional = _predict_frozen(
        frozen_models, holdout_rows, "conventional_vulnerability"
    )
    direct_local = _predict_frozen(frozen_models, holdout_rows, "local_residual")
    direct_monitor = _predict_frozen(
        frozen_models, holdout_rows, "monitor_aware_residual"
    )
    development_failures = [
        row for row in development_rows if row["policy_failure"]
    ]
    constant_miss_probability = sum(
        row["safe_miss_given_failure"] for row in development_failures
    ) / len(development_failures)
    constant_composition = conventional * constant_miss_probability

    miss_models = {}
    miss_predictions = {}
    compositions = {}
    for offset, name in enumerate(
        ("miss_phase", "miss_graph_context", "miss_action_context", "miss_full")
    ):
        result, model, conditional_predictions = fit_posthoc_model(
            development_rows,
            holdout_rows,
            features=POSTHOC_MODEL_SPECS[name],
            outcome="safe_miss_given_failure",
            select=lambda row: bool(row["policy_failure"]),
            folds=args.folds,
        )
        all_predictions = _predict_posthoc(
            model, holdout_rows, POSTHOC_MODEL_SPECS[name]
        )
        composed = conventional * all_predictions
        result["holdout_probability_distribution"] = probability_summary(
            conditional_predictions
        )
        miss_models[name] = result
        miss_predictions[name] = all_predictions
        compositions[name] = {
            "holdout": classification_metrics(holdout_silent, composed),
            "paired_against_conventional_vulnerability": paired_metric_bootstrap(
                holdout_rows,
                holdout_silent,
                composed,
                conventional,
                samples=args.bootstrap_samples,
                seed=args.seed + offset,
            ),
            "paired_against_constant_miss_composition": paired_metric_bootstrap(
                holdout_rows,
                holdout_silent,
                composed,
                constant_composition,
                samples=args.bootstrap_samples,
                seed=args.seed + offset + 10,
            ),
        }

    no_graph_models = {}
    no_graph_predictions = {}
    for name in ("silent_action_context", "silent_action_monitor_context"):
        result, _, predictions = fit_posthoc_model(
            development_rows,
            holdout_rows,
            features=POSTHOC_MODEL_SPECS[name],
            outcome="silent_failure",
            folds=args.folds,
        )
        no_graph_models[name] = result
        no_graph_predictions[name] = predictions

    local_only_ablation = {
        "models_without_graph_features": no_graph_models,
        "paired_holdout_comparisons": {
            "graph_action_context_over_action_context": paired_metric_bootstrap(
                holdout_rows,
                holdout_silent,
                direct_local,
                no_graph_predictions["silent_action_context"],
                samples=args.bootstrap_samples,
                seed=args.seed + 100,
            ),
            "graph_action_monitor_context_over_action_monitor_context": (
                paired_metric_bootstrap(
                    holdout_rows,
                    holdout_silent,
                    direct_monitor,
                    no_graph_predictions["silent_action_monitor_context"],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 101,
                )
            ),
        },
    }

    full_composition = conventional * miss_predictions["miss_full"]
    decomposition = {
        "frozen_failure_model": {
            "target": "policy_failure",
            "holdout": classification_metrics(holdout_failures, conventional),
            "same_scores_against_silent_failure": classification_metrics(
                holdout_silent, conventional
            ),
        },
        "constant_miss_composition": {
            "development_miss_probability": constant_miss_probability,
            "holdout": classification_metrics(
                holdout_silent, constant_composition
            ),
            "purpose": (
                "controls for calibration gained merely by scaling failure risk "
                "by the overall development-set SAFE miss rate"
            ),
        },
        "conditional_miss_models": miss_models,
        "composed_silent_risk": compositions,
        "full_composition_against_frozen_direct_models": {
            "against_local_residual": paired_metric_bootstrap(
                holdout_rows,
                holdout_silent,
                full_composition,
                direct_local,
                samples=args.bootstrap_samples,
                seed=args.seed + 20,
            ),
            "against_monitor_aware_residual": paired_metric_bootstrap(
                holdout_rows,
                holdout_silent,
                full_composition,
                direct_monitor,
                samples=args.bootstrap_samples,
                seed=args.seed + 21,
            ),
        },
    }

    output = {
        "schema_version": 1,
        "analysis": "post-hoc intervention-atlas probability and stability analyses",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("intervention_atlas_followups.py")
            ),
        },
        "analysis_contract": {
            "status": "exploratory post-hoc analysis",
            "reason": (
                "these model specifications were chosen after the original holdout "
                "results had been inspected"
            ),
            "split_usage": (
                "models fit only development trajectories and report performance on "
                "the original holdout trajectories; confirmation requires new data"
            ),
            "uncertainty": (
                "paired percentile bootstraps resample whole task/episode trajectories"
            ),
            "site_scope": (
                "site stability concerns the same 55 sampled sites in both splits, "
                "not unseen sites or the full graph"
            ),
            "interaction_scope": (
                "context means task and early/middle/late phase; it does not represent "
                "the full robot and object state"
            ),
            "physical_dependence": (
                "site units that produced the same command can share one physical "
                "continuation; trajectory resampling does not remove this dependence"
            ),
        },
        "sources": {
            "development": development_source,
            "holdout": holdout_source,
            "frozen_fit": {
                "path": str(args.frozen_fit.resolve()),
                "sha256": file_sha256(args.frozen_fit),
            },
            "frozen_models": {
                "path": str(args.frozen_models.resolve()),
                "sha256": file_sha256(args.frozen_models),
            },
        },
        "population": {
            "development_rows": len(development_rows),
            "holdout_rows": len(holdout_rows),
            "development_trajectories": len(
                {(row["task_id"], row["episode_index"]) for row in development_rows}
            ),
            "holdout_trajectories": len(
                {(row["task_id"], row["episode_index"]) for row in holdout_rows}
            ),
            "sites": len({row["site_id"] for row in development_rows}),
        },
        "probability_decomposition": decomposition,
        "graph_increment_beyond_local_change": local_only_ablation,
        "site_context_stability": _stability_analysis(
            development_rows,
            holdout_rows,
            folds=args.folds,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1_000,
        ),
        "physical_equivalence_audit": physical_equivalence_audit(
            development_rows + holdout_rows
        ),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
