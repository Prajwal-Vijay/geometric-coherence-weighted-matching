#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

OUTPUT_DIR="results/paper_review"
mkdir -p "$OUTPUT_DIR"

printf 'Running unit tests...\n'
python -m unittest discover -s Code/tests -v

printf '\nRunning stationary benchmark suite...\n'
python Code/benchmark_suite.py \
  --suite 3v10 \
  --unmatched-evader-policy stationary \
  --max-steps 2000 \
  --replan-interval 100 \
  --csv-output "$OUTPUT_DIR/stationary_raw.csv" \
  --json-output "$OUTPUT_DIR/stationary_raw.json"

printf '\nRunning downward benchmark suite...\n'
python Code/benchmark_suite.py \
  --suite 3v10 \
  --unmatched-evader-policy downward \
  --max-steps 2000 \
  --replan-interval 100 \
  --csv-output "$OUTPUT_DIR/downward_raw.csv" \
  --json-output "$OUTPUT_DIR/downward_raw.json"

printf '\nAggregating paper tables...\n'
python scripts/aggregate_benchmark_results.py \
  "$OUTPUT_DIR/stationary_raw.json" \
  "$OUTPUT_DIR/downward_raw.json" \
  --csv-output "$OUTPUT_DIR/aggregates.csv" \
  --json-output "$OUTPUT_DIR/aggregates.json"

printf '\nReproduction complete.\n'
printf 'Wrote raw and aggregated outputs to %s\n' "$OUTPUT_DIR"
