"""
Deterministic benchmark scenarios and headless runners for 3v6 and 3v10 experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import Environment
import Evader
import Pursuer


DEFAULT_STRATEGIES = (
    "weighted_sequential",
    "unweighted_sequential",
    "random_sequential",
    "nearest_single",
)

SUITE_ALL = "all"
SUITE_3V6 = "3v6"
SUITE_3V10 = "3v10"


BENCHMARK_FAMILY_BASES = {
    "spread_mild_hetero": {
        "t": 0.1,
        "pursuer_positions": [(0, 0, 12), (24, 0, 12), (12, 18, 12)],
        "pursuer_speeds": [1.6, 1.5, 1.45],
        "evader_positions": [(3, -4, 12), (7, 4, 12), (11, -4, 12), (15, 4, 12), (19, -4, 12), (23, 4, 12)],
        "evader_speeds": [1.0, 0.95, 1.05, 1.0, 1.1, 0.9],
    },
    "clustered_center": {
        "t": 0.1,
        "pursuer_positions": [(0, 0, 12), (24, 0, 12), (12, 18, 12)],
        "pursuer_speeds": [1.55, 1.5, 1.45],
        "evader_positions": [(9, 7, 12), (10, 9, 12), (12, 8, 12), (13, 10, 12), (15, 7, 12), (16, 9, 12)],
        "evader_speeds": [1.0, 1.05, 1.1, 0.95, 1.0, 1.08],
    },
    "split_lanes": {
        "t": 0.1,
        "pursuer_positions": [(0, 2, 12), (24, 2, 12), (12, 20, 12)],
        "pursuer_speeds": [1.7, 1.45, 1.5],
        "evader_positions": [(4, 5, 12), (8, 5, 12), (12, 5, 12), (12, 15, 12), (16, 15, 12), (20, 15, 12)],
        "evader_speeds": [1.0, 1.0, 1.1, 0.95, 1.05, 1.0],
    },
    "asymmetric_fast_intruders": {
        "t": 0.1,
        "pursuer_positions": [(0, -2, 12), (16, 1, 12), (28, 10, 12)],
        "pursuer_speeds": [1.55, 1.5, 1.4],
        "evader_positions": [(6, 12, 12), (10, 9, 12), (14, 11, 12), (18, 8, 12), (22, 10, 12), (26, 7, 12)],
        "evader_speeds": [1.15, 1.1, 1.05, 1.0, 1.12, 1.08],
    },
}


BENCHMARK_FAMILY_BASES_3V10 = {
    "three_vs_ten_spread_mild_hetero": {
        "t": 0.1,
        "pursuer_positions": [(0, 0, 12), (24, 0, 12), (12, 18, 12)],
        "pursuer_speeds": [1.6, 1.5, 1.45],
        "evader_positions": [
            (1, -4, 12),
            (4, 4, 12),
            (7, -4, 12),
            (10, 4, 12),
            (13, -4, 12),
            (16, 4, 12),
            (19, -4, 12),
            (22, 4, 12),
            (25, -4, 12),
            (28, 4, 12),
        ],
        "evader_speeds": [1.0, 0.95, 1.05, 1.0, 1.1, 0.9, 1.02, 0.98, 1.08, 0.92],
    },
    "three_vs_ten_clustered_center": {
        "t": 0.1,
        "pursuer_positions": [(0, 0, 12), (24, 0, 12), (12, 18, 12)],
        "pursuer_speeds": [1.55, 1.5, 1.45],
        "evader_positions": [
            (8.5, 7.0, 12),
            (9.5, 9.0, 12),
            (10.5, 8.0, 12),
            (11.5, 10.0, 12),
            (12.5, 7.5, 12),
            (13.5, 9.5, 12),
            (14.5, 8.5, 12),
            (15.5, 10.5, 12),
            (11.0, 6.5, 12),
            (14.0, 11.0, 12),
        ],
        "evader_speeds": [1.0, 1.05, 1.1, 0.95, 1.0, 1.08, 1.03, 0.97, 1.06, 0.99],
    },
    "three_vs_ten_split_lanes": {
        "t": 0.1,
        "pursuer_positions": [(0, 2, 12), (24, 2, 12), (12, 20, 12)],
        "pursuer_speeds": [1.7, 1.45, 1.5],
        "evader_positions": [
            (2, 5, 12),
            (6, 5, 12),
            (10, 5, 12),
            (14, 5, 12),
            (18, 5, 12),
            (6, 15, 12),
            (10, 15, 12),
            (14, 15, 12),
            (18, 15, 12),
            (22, 15, 12),
        ],
        "evader_speeds": [1.0, 1.0, 1.1, 0.95, 1.05, 1.0, 1.08, 0.98, 1.06, 0.97],
    },
    "three_vs_ten_asymmetric_fast_intruders": {
        "t": 0.1,
        "pursuer_positions": [(0, -2, 12), (16, 1, 12), (28, 10, 12)],
        "pursuer_speeds": [1.55, 1.5, 1.4],
        "evader_positions": [
            (4.0, 13.0, 12),
            (7.5, 11.0, 12),
            (11.0, 12.5, 12),
            (14.5, 10.5, 12),
            (18.0, 11.5, 12),
            (21.5, 9.5, 12),
            (25.0, 10.5, 12),
            (28.5, 8.5, 12),
            (16.0, 14.0, 12),
            (23.0, 13.0, 12),
        ],
        "evader_speeds": [1.16, 1.12, 1.1, 1.06, 1.14, 1.08, 1.13, 1.09, 1.11, 1.07],
    },
    "three_vs_ten_x_crossing": {
        "t": 0.1,
        "pursuer_positions": [(0, 2, 12), (24, 2, 12), (12, 20, 12)],
        "pursuer_speeds": [1.82, 1.52, 1.56],
        "evader_positions": [
            (20, 5, 12),
            (4, 5, 12),
            (18, 14, 12),
            (6, 14, 12),
            (2, 5, 12),
            (22, 5, 12),
            (10, 5, 12),
            (14, 14, 12),
            (8, 14, 12),
            (16, 5, 12),
        ],
        "evader_speeds": [0.99, 0.99, 1.01, 0.94, 0.97, 0.97, 0.99, 0.94, 0.94, 0.98],
    },
    "three_vs_ten_braided_corridors": {
        "t": 0.1,
        "pursuer_positions": [(0, 2, 12), (24, 2, 12), (12, 20, 12)],
        "pursuer_speeds": [1.74, 1.48, 1.51],
        "evader_positions": [
            (18, 5, 12),
            (6, 15, 12),
            (4, 5, 12),
            (20, 15, 12),
            (2, 5, 12),
            (22, 5, 12),
            (8, 15, 12),
            (16, 15, 12),
            (10, 5, 12),
            (14, 5, 12),
        ],
        "evader_speeds": [1.01, 0.97, 1.0, 0.96, 0.99, 0.99, 0.98, 0.98, 1.03, 1.02],
    },
    "three_vs_ten_switchback_gate": {
        "t": 0.1,
        "pursuer_positions": [(1, 1.5, 12), (23, 1.5, 12), (12, 21, 12)],
        "pursuer_speeds": [1.76, 1.47, 1.53],
        "evader_positions": [
            (19, 5, 12),
            (5, 5, 12),
            (17, 14, 12),
            (7, 14, 12),
            (3, 6, 12),
            (21, 6, 12),
            (11, 5, 12),
            (13, 14, 12),
            (9, 14, 12),
            (15, 5, 12),
        ],
        "evader_speeds": [1.01, 1.0, 1.04, 0.97, 0.99, 0.99, 1.02, 0.97, 0.97, 1.01],
    },
}


ALL_BENCHMARK_FAMILY_BASES = {
    **BENCHMARK_FAMILY_BASES,
    **BENCHMARK_FAMILY_BASES_3V10,
}


VARIANT_TEMPLATES = (
    {
        "variant": "v1",
        "global_shift": (0.0, 0.0, 0.0),
        "pursuer_offsets": [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        "evader_offsets": [(0.0, 0.0, 0.0)] * 10,
        "pursuer_speed_delta": [0.0, 0.0, 0.0],
        "evader_speed_delta": [0.0] * 10,
    },
    {
        "variant": "v2",
        "global_shift": (0.75, 0.4, 0.0),
        "pursuer_offsets": [(-0.2, 0.2, 0.0), (0.15, -0.15, 0.0), (0.05, 0.2, 0.0)],
        "evader_offsets": [
            (0.25, -0.35, 0.0),
            (0.15, 0.2, 0.0),
            (-0.2, -0.15, 0.0),
            (0.2, 0.25, 0.0),
            (-0.15, -0.2, 0.0),
            (0.25, 0.15, 0.0),
            (-0.2, 0.2, 0.0),
            (0.15, -0.25, 0.0),
            (-0.1, 0.15, 0.0),
            (0.2, -0.1, 0.0),
        ],
        "pursuer_speed_delta": [0.02, 0.0, -0.01],
        "evader_speed_delta": [0.0, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0],
    },
    {
        "variant": "v3",
        "global_shift": (-0.8, -0.45, 0.0),
        "pursuer_offsets": [(0.1, -0.2, 0.0), (-0.1, 0.1, 0.0), (-0.15, -0.1, 0.0)],
        "evader_offsets": [
            (-0.25, 0.25, 0.0),
            (-0.15, -0.2, 0.0),
            (0.2, 0.15, 0.0),
            (-0.2, -0.25, 0.0),
            (0.15, 0.2, 0.0),
            (-0.25, -0.1, 0.0),
            (0.2, 0.1, 0.0),
            (-0.15, -0.15, 0.0),
            (0.1, 0.25, 0.0),
            (-0.2, 0.05, 0.0),
        ],
        "pursuer_speed_delta": [-0.01, 0.02, 0.0],
        "evader_speed_delta": [0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01],
    },
    {
        "variant": "v4",
        "global_shift": (0.35, -0.7, 0.0),
        "pursuer_offsets": [(0.0, 0.3, 0.0), (0.0, -0.25, 0.0), (0.15, 0.1, 0.0)],
        "evader_offsets": [
            (0.1, 0.15, 0.0),
            (0.25, -0.1, 0.0),
            (-0.15, 0.2, 0.0),
            (0.15, -0.2, 0.0),
            (-0.1, 0.15, 0.0),
            (0.2, -0.15, 0.0),
            (-0.15, 0.25, 0.0),
            (0.1, -0.2, 0.0),
            (-0.2, 0.1, 0.0),
            (0.15, -0.1, 0.0),
        ],
        "pursuer_speed_delta": [0.01, -0.01, 0.02],
        "evader_speed_delta": [-0.01, 0.0, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01, -0.01],
    },
    {
        "variant": "v5",
        "global_shift": (-0.4, 0.75, 0.0),
        "pursuer_offsets": [(-0.15, 0.0, 0.0), (0.2, 0.15, 0.0), (-0.05, -0.2, 0.0)],
        "evader_offsets": [
            (-0.1, -0.2, 0.0),
            (-0.25, 0.15, 0.0),
            (0.15, -0.1, 0.0),
            (-0.15, 0.25, 0.0),
            (0.1, -0.15, 0.0),
            (-0.2, 0.1, 0.0),
            (0.15, -0.2, 0.0),
            (-0.1, 0.2, 0.0),
            (0.2, -0.05, 0.0),
            (-0.15, 0.1, 0.0),
        ],
        "pursuer_speed_delta": [0.0, 0.01, -0.02],
        "evader_speed_delta": [0.0, -0.01, 0.01, 0.0, -0.01, 0.01, 0.0, -0.01, 0.01, 0.0],
    },
)


def _apply_position_variant(
    base_positions: Sequence[Sequence[float]],
    global_shift: Sequence[float],
    local_offsets: Sequence[Sequence[float]],
):
    positions = []
    global_shift = np.array(global_shift, dtype=float)
    for base_position, local_offset in zip(base_positions, local_offsets):
        position = np.array(base_position, dtype=float) + global_shift + np.array(local_offset, dtype=float)
        positions.append(tuple(position.tolist()))
    return positions


def _apply_speed_variant(base_speeds: Sequence[float], speed_delta: Sequence[float]):
    return [round(base_speed + delta, 4) for base_speed, delta in zip(base_speeds, speed_delta)]


def _build_variant_config(family_name: str, base_config: Dict, variant_template: Dict):
    return {
        "scenario_type": family_name,
        "scenario_variant": variant_template["variant"],
        "t": base_config["t"],
        "pursuer_positions": _apply_position_variant(
            base_positions=base_config["pursuer_positions"],
            global_shift=variant_template["global_shift"],
            local_offsets=variant_template["pursuer_offsets"],
        ),
        "pursuer_speeds": _apply_speed_variant(
            base_speeds=base_config["pursuer_speeds"],
            speed_delta=variant_template["pursuer_speed_delta"],
        ),
        "evader_positions": _apply_position_variant(
            base_positions=base_config["evader_positions"],
            global_shift=variant_template["global_shift"],
            local_offsets=variant_template["evader_offsets"],
        ),
        "evader_speeds": _apply_speed_variant(
            base_speeds=base_config["evader_speeds"],
            speed_delta=variant_template["evader_speed_delta"],
        ),
    }


def _build_all_scenarios():
    scenarios = {}
    scenario_families = {}
    for family_name, base_config in ALL_BENCHMARK_FAMILY_BASES.items():
        family_scenarios = []
        for variant_template in VARIANT_TEMPLATES:
            scenario_name = f"{family_name}_{variant_template['variant']}"
            scenarios[scenario_name] = _build_variant_config(family_name, base_config, variant_template)
            family_scenarios.append(scenario_name)
        scenario_families[family_name] = tuple(family_scenarios)
    return scenarios, scenario_families


BENCHMARK_SCENARIOS, BENCHMARK_SCENARIO_FAMILIES = _build_all_scenarios()


def _scenario_in_suite(scenario_name: str, suite: str) -> bool:
    is_three_vs_ten = scenario_name.startswith("three_vs_ten_")
    if suite == SUITE_ALL:
        return True
    if suite == SUITE_3V10:
        return is_three_vs_ten
    if suite == SUITE_3V6:
        return not is_three_vs_ten
    raise ValueError(f"Unknown suite '{suite}'.")


def _family_in_suite(family_name: str, suite: str) -> bool:
    return _scenario_in_suite(BENCHMARK_SCENARIO_FAMILIES[family_name][0], suite)


def scenario_family_names(suite: str = SUITE_ALL) -> List[str]:
    return [
        family_name
        for family_name in BENCHMARK_SCENARIO_FAMILIES.keys()
        if _family_in_suite(family_name, suite)
    ]


def scenario_names(family: Optional[str] = None, suite: str = SUITE_ALL) -> List[str]:
    if family is None:
        return [
            scenario_name
            for scenario_name in BENCHMARK_SCENARIOS.keys()
            if _scenario_in_suite(scenario_name, suite)
        ]
    if family not in BENCHMARK_SCENARIO_FAMILIES:
        raise ValueError(f"Unknown scenario family '{family}'.")
    if not _family_in_suite(family, suite):
        raise ValueError(f"Scenario family '{family}' is not part of suite '{suite}'.")
    return list(BENCHMARK_SCENARIO_FAMILIES[family])


def resolve_scenarios(
    scenarios: Optional[Iterable[str]] = None,
    scenario_families: Optional[Iterable[str]] = None,
    suite: str = SUITE_ALL,
):
    resolved = []
    seen = set()

    for scenario_name in scenarios or []:
        if scenario_name not in BENCHMARK_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_name}'.")
        if not _scenario_in_suite(scenario_name, suite):
            raise ValueError(f"Scenario '{scenario_name}' is not part of suite '{suite}'.")
        if scenario_name not in seen:
            resolved.append(scenario_name)
            seen.add(scenario_name)

    for family_name in scenario_families or []:
        for scenario_name in scenario_names(family=family_name, suite=suite):
            if scenario_name not in seen:
                resolved.append(scenario_name)
                seen.add(scenario_name)

    if not resolved:
        return scenario_names(suite=suite)
    return resolved


def build_environment_for_scenario(
    scenario_name: str,
    strategy: str,
    replan_interval_steps: int = 100,
    rng_seed: Optional[int] = None,
    unmatched_evader_policy: str = "stationary",
):
    config = BENCHMARK_SCENARIOS[scenario_name]
    pursuers = [
        Pursuer.Pursuer(np.array(position, dtype=float), speed, index)
        for index, (position, speed) in enumerate(zip(config["pursuer_positions"], config["pursuer_speeds"]))
    ]
    evaders = [
        Evader.Evader(np.array(position, dtype=float), speed, index)
        for index, (position, speed) in enumerate(zip(config["evader_positions"], config["evader_speeds"]))
    ]

    return Environment.Environment(
        N=len(pursuers),
        M=len(evaders),
        t=config["t"],
        pursuers=pursuers,
        evaders=evaders,
        strategy=strategy,
        replan_interval_steps=replan_interval_steps,
        rng_seed=rng_seed,
        unmatched_evader_policy=unmatched_evader_policy,
    )


def _benchmark_summary(summary: Dict, scenario_name: str) -> Dict:
    config = BENCHMARK_SCENARIOS[scenario_name]
    return {
        "scenario": scenario_name,
        "scenario_type": config["scenario_type"],
        "scenario_variant": config["scenario_variant"],
        "strategy": summary["strategy"],
        "unmatched_evader_policy": summary["unmatched_evader_policy"],
        "replan_interval_steps": summary["replan_interval_steps"],
        "total_steps": summary["total_steps"],
        "average_path_tortuosity": summary["average_path_tortuosity"],
        "angular_control_effort": summary["angular_control_effort"],
        "captured_count": summary["captured_count"],
        "escaped_count": summary["escaped_count"],
        "average_capture_height": summary["average_capture_height"],
        "captured_evaders": summary["captured_evaders"],
        "escaped_evaders": summary["escaped_evaders"],
        "capture_step_by_evader": summary["capture_step_by_evader"],
        "capture_height_by_evader": summary["capture_height_by_evader"],
        "escape_step_by_evader": summary["escape_step_by_evader"],
        "termination_reason": summary["termination_reason"],
    }


def run_single_benchmark(
    scenario_name: str,
    strategy: str,
    max_steps: int = 2000,
    replan_interval_steps: int = 100,
    rng_seed: Optional[int] = None,
    unmatched_evader_policy: str = "stationary",
):
    env = build_environment_for_scenario(
        scenario_name=scenario_name,
        strategy=strategy,
        replan_interval_steps=replan_interval_steps,
        rng_seed=rng_seed,
        unmatched_evader_policy=unmatched_evader_policy,
    )
    summary = env.run_episode(max_steps=max_steps, render=False, record_trajectories=False)
    return _benchmark_summary(summary, scenario_name=scenario_name)


def run_benchmark_suite(
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    scenarios: Optional[Iterable[str]] = None,
    scenario_families: Optional[Iterable[str]] = None,
    suite: str = SUITE_ALL,
    max_steps: int = 2000,
    replan_interval_steps: int = 100,
    rng_seed: Optional[int] = None,
    unmatched_evader_policy: str = "stationary",
):
    resolved_scenarios = resolve_scenarios(
        scenarios=scenarios,
        scenario_families=scenario_families,
        suite=suite,
    )
    results = []
    for scenario_name in resolved_scenarios:
        for strategy in strategies:
            seed = rng_seed if strategy == "random_sequential" else None
            results.append(
                run_single_benchmark(
                    scenario_name=scenario_name,
                    strategy=strategy,
                    max_steps=max_steps,
                    replan_interval_steps=replan_interval_steps,
                    rng_seed=seed,
                    unmatched_evader_policy=unmatched_evader_policy,
                )
            )
    return results


def print_benchmark_table(results: Iterable[Dict]):
    results = list(results)
    if not results:
        print("No benchmark results.")
        return

    header = (
        f"{'Scenario':<34} {'Strategy':<20} {'EvPolicy':<10} {'Steps':>7} "
        f"{'Captured':>8} {'Escaped':>7} {'AvgCatchZ':>10} {'AvgTau':>8} {'AngEffort':>11} {'Reason':<14}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        average_capture_height = result["average_capture_height"]
        average_height_text = f"{average_capture_height:.3f}" if average_capture_height is not None else "n/a"
        print(
            f"{result['scenario']:<34} {result['strategy']:<20} {result['unmatched_evader_policy']:<10} "
            f"{result['total_steps']:>7} "
            f"{result['captured_count']:>8} {result['escaped_count']:>7} "
            f"{average_height_text:>10} {result['average_path_tortuosity']:>8.3f} "
            f"{result['angular_control_effort']:>11.3f} {result['termination_reason']:<14}"
        )


def save_results_json(results: Iterable[Dict], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, indent=2)


def save_results_csv(results: Iterable[Dict], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    fieldnames = [
        "scenario",
        "scenario_type",
        "scenario_variant",
        "strategy",
        "unmatched_evader_policy",
        "replan_interval_steps",
        "total_steps",
        "average_path_tortuosity",
        "angular_control_effort",
        "captured_count",
        "escaped_count",
        "average_capture_height",
        "captured_evaders",
        "escaped_evaders",
        "capture_step_by_evader",
        "capture_height_by_evader",
        "escape_step_by_evader",
        "termination_reason",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], (dict, list)) else row[key]
                    for key in fieldnames
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Run deterministic 3v6 and 3v10 benchmark scenarios.")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario name to run. Repeatable.")
    parser.add_argument(
        "--scenario-family",
        action="append",
        dest="scenario_families",
        help="Scenario family to run all five initializations for. Repeatable.",
    )
    parser.add_argument(
        "--suite",
        choices=(SUITE_3V6, SUITE_3V10, SUITE_ALL),
        default=SUITE_ALL,
        help="Limit the run to the 3v6 suite, the 3v10 suite, or all scenarios.",
    )
    parser.add_argument("--strategy", action="append", dest="strategies", help="Strategy to run. Repeatable.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum number of simulation steps.")
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=100,
        help="Periodic receding-horizon replanning interval in steps.",
    )
    parser.add_argument(
        "--unmatched-evader-policy",
        choices=Environment.UNMATCHED_EVADER_POLICIES,
        default="stationary",
        help="Behavior for evaders that are active but not currently assigned to a coalition.",
    )
    parser.add_argument("--rng-seed", type=int, default=0, help="Seed used for the random sequential strategy.")
    parser.add_argument("--json-output", type=str, help="Optional path for JSON export.")
    parser.add_argument("--csv-output", type=str, help="Optional path for CSV export.")
    args = parser.parse_args()

    strategies = args.strategies if args.strategies else DEFAULT_STRATEGIES
    results = run_benchmark_suite(
        strategies=strategies,
        scenarios=args.scenarios,
        scenario_families=args.scenario_families,
        suite=args.suite,
        max_steps=args.max_steps,
        replan_interval_steps=args.replan_interval,
        rng_seed=args.rng_seed,
        unmatched_evader_policy=args.unmatched_evader_policy,
    )
    print_benchmark_table(results)

    if args.json_output:
        save_results_json(results, args.json_output)
    if args.csv_output:
        save_results_csv(results, args.csv_output)


if __name__ == "__main__":
    main()
