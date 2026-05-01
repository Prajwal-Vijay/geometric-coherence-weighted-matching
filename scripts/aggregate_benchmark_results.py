#!/usr/bin/env python3
"""Aggregate raw benchmark JSON exports into per-family paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _normalize_family_name(raw_family: str) -> str:
    for prefix in ("three_vs_ten_", "three_vs_six_"):
        if raw_family.startswith(prefix):
            return raw_family[len(prefix) :]
    return raw_family


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def load_rows(paths: Iterable[str]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        rows.extend(payload)
    return rows


def aggregate_rows(rows: Iterable[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    overall: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

    for row in rows:
        policy = row["unmatched_evader_policy"]
        family = _normalize_family_name(row["scenario_type"])
        strategy = row["strategy"]
        grouped[(policy, family, strategy)].append(row)
        overall[(policy, strategy)].append(row)

    aggregates: List[Dict] = []

    def build_record(policy: str, family: str, strategy: str, items: List[Dict]) -> Dict:
        return {
            "policy": policy,
            "family": family,
            "strategy": strategy,
            "scenario_count": len(items),
            "avg_steps": _mean(item["total_steps"] for item in items),
            "avg_captured": _mean(item["captured_count"] for item in items),
            "avg_escaped": _mean(item["escaped_count"] for item in items),
            "avg_capture_height": _mean(item.get("average_capture_height") for item in items),
            "avg_tortuosity": _mean(item["average_path_tortuosity"] for item in items),
            "avg_angular_effort": _mean(item["angular_control_effort"] for item in items),
            "max_steps_count": sum(1 for item in items if item["termination_reason"] == "max_steps"),
        }

    family_rows = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    for (policy, family, strategy), items in family_rows:
        aggregates.append(build_record(policy, family, strategy, items))

    overall_rows = sorted(overall.items(), key=lambda item: (item[0][0], item[0][1]))
    for (policy, strategy), items in overall_rows:
        aggregates.append(build_record(policy, "OVERALL", strategy, items))

    return aggregates


def save_csv(rows: List[Dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "family",
        "strategy",
        "scenario_count",
        "avg_steps",
        "avg_captured",
        "avg_escaped",
        "avg_capture_height",
        "avg_tortuosity",
        "avg_angular_effort",
        "max_steps_count",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows: List[Dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw benchmark JSON exports into paper tables.")
    parser.add_argument("inputs", nargs="+", help="Input JSON files from benchmark_suite.py.")
    parser.add_argument("--csv-output", required=True, help="Path for aggregated CSV output.")
    parser.add_argument("--json-output", help="Optional path for aggregated JSON output.")
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    aggregates = aggregate_rows(rows)
    save_csv(aggregates, args.csv_output)
    if args.json_output:
        save_json(aggregates, args.json_output)


if __name__ == "__main__":
    main()
