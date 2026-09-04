from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, load_json

PRIMARY_ALPHA = "0.1"
PRIMARY_WINDOW = "post_fault_any"


def _first_alarm(record: dict[str, Any], alpha: str) -> int | None:
    value = record["alarms"][alpha][PRIMARY_WINDOW]["first_step"]
    return None if value is None else int(value)


def _alarm_pair(
    faulted_first: int | None,
    control_first: int | None,
    *,
    common_stop: int | None,
) -> tuple[bool, bool]:
    faulted = faulted_first is not None and (
        common_stop is None or faulted_first < common_stop
    )
    control = control_first is not None and (
        common_stop is None or control_first < common_stop
    )
    return faulted, control


def _pair_table(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    counts = Counter(pairs)
    return {
        "both": counts[(True, True)],
        "faulted_only": counts[(True, False)],
        "control_only": counts[(False, True)],
        "neither": counts[(False, False)],
    }


def _timing_summary(delays: list[int]) -> dict[str, float | int | None]:
    if not delays:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(delays),
        "minimum": min(delays),
        "median": statistics.median(delays),
        "maximum": max(delays),
    }


def _paired_summary(
    records: list[dict[str, Any]], *, alpha: str
) -> dict[str, Any]:
    full_pairs = []
    common_pairs = []
    comparable_delays = []
    later_faulted_alarms = 0
    later_control_alarms = 0
    unequal_horizons = 0
    for record in records:
        faulted_first = record["faulted_first"]
        control_first = record["control_first"]
        common_stop = min(record["faulted_length"], record["control_length"])
        full_pairs.append(
            _alarm_pair(faulted_first, control_first, common_stop=None)
        )
        common_pair = _alarm_pair(
            faulted_first, control_first, common_stop=common_stop
        )
        common_pairs.append(common_pair)
        if common_pair[0]:
            comparable_delays.append(faulted_first - record["fault_step"])
        later_faulted_alarms += int(
            faulted_first is not None and faulted_first >= common_stop
        )
        later_control_alarms += int(
            control_first is not None and control_first >= common_stop
        )
        unequal_horizons += int(
            record["faulted_length"] != record["control_length"]
        )

    common_table = _pair_table(common_pairs)
    discordant = common_table["faulted_only"] + common_table["control_only"]
    if discordant:
        from scipy.stats import binomtest

        p_value = float(
            binomtest(
                common_table["faulted_only"], discordant, 0.5
            ).pvalue
        )
    else:
        p_value = 1.0
    return {
        "pairs": len(records),
        "own_terminal_horizon": _pair_table(full_pairs),
        "equal_observation_horizon": {
            **common_table,
            "faulted_detection_probability": (
                sum(pair[0] for pair in common_pairs) / len(common_pairs)
                if common_pairs
                else None
            ),
            "control_alarm_probability": (
                sum(pair[1] for pair in common_pairs) / len(common_pairs)
                if common_pairs
                else None
            ),
            "exact_paired_binomial_p_value": p_value,
            "discordant_pairs": discordant,
        },
        "horizon_audit": {
            "unequal_terminal_lengths": unequal_horizons,
            "faulted_alarms_only_after_common_horizon": later_faulted_alarms,
            "control_alarms_only_after_common_horizon": later_control_alarms,
        },
        "comparable_faulted_alarm_delay_steps": _timing_summary(
            comparable_delays
        ),
        "alpha": alpha,
    }


def analyze_alarm_specificity(
    pairs: list[tuple[Path, Path]], *, alpha: str = PRIMARY_ALPHA
) -> dict[str, Any]:
    site_records = []
    physical_records: dict[tuple[str, str], dict[str, Any]] = {}
    sources = []
    seen_site_records = set()
    for site_path, physical_path in pairs:
        sites = load_json(site_path)
        physical = load_json(physical_path)
        physical_index = {record["run"]: record for record in physical["records"]}
        physical_source = file_sha256(physical_path)
        sources.append(
            {
                "site_analysis": {
                    "path": str(site_path.resolve()),
                    "sha256": file_sha256(site_path),
                    "split": sites["analysis_split"],
                },
                "physical_scores": {
                    "path": str(physical_path.resolve()),
                    "sha256": physical_source,
                },
            }
        )
        for site in sites["records"]:
            if not site.get("primary_eligible") or not site["policy_failure"]:
                continue
            identity = (physical_source, site["record_id"])
            if identity in seen_site_records:
                raise ValueError(f"duplicate failed site record: {site['record_id']}")
            seen_site_records.add(identity)
            faulted = physical_index[site["physical_run"]]
            control = physical_index[f"{site['context_id']}-control"]
            if faulted["success"] is not False or control["success"] is not True:
                raise ValueError(
                    "alarm specificity requires a failed continuation and a "
                    "successful matched control"
                )
            site_records.append(
                {
                    "faulted_first": _first_alarm(
                        site["safe_faulted_evidence"], alpha
                    ),
                    "control_first": _first_alarm(control, alpha),
                    "faulted_length": int(
                        site["safe_faulted_evidence"]["score_length"]
                    ),
                    "control_length": int(control["length"]),
                    "fault_step": int(site["context"]["policy_step"]),
                }
            )
            physical_identity = (physical_source, site["physical_run"])
            physical_record = {
                "faulted_first": _first_alarm(faulted, alpha),
                "control_first": _first_alarm(control, alpha),
                "faulted_length": int(faulted["length"]),
                "control_length": int(control["length"]),
                "fault_step": int(site["context"]["policy_step"]),
            }
            previous = physical_records.setdefault(
                physical_identity, physical_record
            )
            if previous != physical_record:
                raise ValueError(
                    f"inconsistent metadata for physical run {site['physical_run']}"
                )

    return {
        "schema_version": 1,
        "analysis": "matched specificity of eventual SAFE alarms",
        "analysis_contract": {
            "primary_unit": "distinct faulted physical continuation",
            "primary_comparison": (
                "faulted continuation and its successful control are observed only "
                "through the shorter recorded horizon"
            ),
            "own_terminal_comparison": (
                "reported as a horizon audit, not as evidence of fault-specific "
                "detection"
            ),
            "inference": (
                "the exact paired binomial test uses only discordant physical pairs"
            ),
        },
        "sources": sources,
        "physical_continuations": _paired_summary(
            list(physical_records.values()), alpha=alpha
        ),
        "site_intervention_units": _paired_summary(site_records, alpha=alpha),
    }
