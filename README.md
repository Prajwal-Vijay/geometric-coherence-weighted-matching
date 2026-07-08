# Geometric Coherence via Weighted Matching for 3D Heterogeneous Multi-Agent Reach-Avoid Games

Code and benchmark artifact accompanying the paper:

**[Geometric Coherence via Weighted Matching for 3D Heterogeneous Multi-Agent Reach-Avoid Games](https://openreview.net/forum?id=0WHJGj07mJ)**  
Prajwal Vijay  
Accepted at _MARS Workshop at ICRA 2026_


## Paper

- Paper PDF: [LINK](https://github.com/Prajwal-Vijay/geometric-coherence-weighted-matching)
- Project page: `TODO`
- Video / demo link: `TODO`

## Teaser Figure
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/a4c23d49-27f0-4254-bcf4-8fae96825ccf" />

<!-- Place repository-local figures in `media/` and uncomment the placeholder below when you are ready.

<!--
![Teaser figure](media/teaser.png)
-->

## Overview

This repository contains the simulator, assignment solvers, and deterministic benchmark suite used in the paper above. The project studies assignment quality in 3D heterogeneous multi-pursuer reach-avoid games and introduces a cardinality-first weighted sequential matching method that uses interception height as a secondary assignment objective.

The released artifact is intended to support reproducible evaluation of assignment strategies in geometrically structured reach-avoid scenarios. In particular, it includes the weighted and unweighted sequential baselines, nearest-single and random baselines, and the scenario families used in the workshop submission.

## Repository Layout

- `Code/`
  Python implementation of the simulator, assignment strategies, benchmark runner, and tests.
- `Code/benchmark_suite.py`
  Entry point for deterministic 3v6 and 3v10 benchmark runs.
- `Code/tests/`
  Unit tests for the solver, benchmark suite, and environment behavior.
- `requirements.txt`
  Core Python dependencies for the public artifact.
- `requirements-optional.txt`
  Optional extras for experimental scripts.
- `CITATION.cff`
  Citation metadata for GitHub and other tooling.
- `media/`
  Placeholder directory for README figures, GIFs, and demo videos.
- `results/`
  Suggested output directory for benchmark CSV/JSON files.

## Requirements

The benchmark and simulator use standard scientific Python packages.

- Python 3.10+ recommended
- `numpy`
- `scipy`
- `matplotlib`
- `sympy`

Optional:

- `cvxpy` only for the standalone experimental script `Code/optim_test.py`

For the paper results and the benchmark suite, `cvxpy` is **not** required.
The released min-cost max-flow solver in `Code/minCostMaxFlow_implemented.py`
is a combinatorial SSP/MCMF implementation, and the interception-point
optimization in the benchmark path is performed with SciPy SLSQP in
`Code/Environment.py`. The helper name `_solve_value_function_cvxpy` is a
legacy name retained for compatibility.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you plan to run the experimental optimization script as well:

```bash
python -m pip install -r requirements-optional.txt
```

If you prefer Conda, an `environment.yml` file is also provided.

## Reviewer Reproduction

For reviewers, the simplest end-to-end reproduction path is:

```bash
bash scripts/reproduce_paper_results.sh
```

This script:

- runs the unit tests
- runs the full deterministic 3-vs-10 benchmark for both paper regimes
- exports raw CSV/JSON benchmark outputs
- aggregates the raw outputs into per-family and overall averages

All generated files are written under `results/paper_review/`.

## Quick Start

Run the deterministic 3-vs-10 benchmark with the two sequential methods:

```bash
mkdir -p results
python Code/benchmark_suite.py \
  --suite 3v10 \
  --strategy weighted_sequential \
  --strategy unweighted_sequential \
  --unmatched-evader-policy stationary \
  --max-steps 2000 \
  --replan-interval 100 \
  --csv-output results/stationary.csv \
  --json-output results/stationary.json
```

Run the hybrid benchmark used in the paper:

```bash
python Code/benchmark_suite.py \
  --suite 3v10 \
  --strategy weighted_sequential \
  --strategy unweighted_sequential \
  --unmatched-evader-policy downward \
  --max-steps 2000 \
  --replan-interval 100 \
  --csv-output results/downward.csv \
  --json-output results/downward.json
```

To include the random and nearest-single baselines as well, either omit the `--strategy` flags or add:

```bash
--strategy random_sequential --strategy nearest_single
```

## Reproducing the Main Paper Results

The paper reports results on a deterministic 35-scenario benchmark with:

- 7 scenario families
- 5 initialization variants per family
- two unmatched-evader policies:
  - `stationary` for the diagnostic benchmark
  - `downward` for the hybrid adversarial benchmark
- `max_steps = 2000`
- `replan_interval = 100`

The main comparison in the paper is between:

- `weighted_sequential`
- `unweighted_sequential`

The additional baselines available in the code are:

- `random_sequential`
- `nearest_single`

To reproduce the paper tables manually, run:

```bash
python Code/benchmark_suite.py \
  --suite 3v10 \
  --unmatched-evader-policy stationary \
  --max-steps 2000 \
  --replan-interval 100 \
  --csv-output results/paper_review/stationary_raw.csv \
  --json-output results/paper_review/stationary_raw.json

python Code/benchmark_suite.py \
  --suite 3v10 \
  --unmatched-evader-policy downward \
  --max-steps 2000 \
  --replan-interval 100 \
  --csv-output results/paper_review/downward_raw.csv \
  --json-output results/paper_review/downward_raw.json

python scripts/aggregate_benchmark_results.py \
  results/paper_review/stationary_raw.json \
  results/paper_review/downward_raw.json \
  --csv-output results/paper_review/aggregates.csv \
  --json-output results/paper_review/aggregates.json
```

## Running Tests

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
python -m unittest discover -s Code/tests -v
```

## Output Files

Benchmark runs can export:

- CSV summaries via `--csv-output`
- JSON summaries via `--json-output`
- aggregated per-family and overall summaries via `scripts/aggregate_benchmark_results.py`

The JSON output includes per-scenario metrics such as:

- total steps
- termination reason
- captured and escaped counts
- average capture height
- path tortuosity
- angular-effort proxy

The `scripts/reproduce_paper_results.sh` script creates:

- `results/paper_review/stationary_raw.csv`
- `results/paper_review/stationary_raw.json`
- `results/paper_review/downward_raw.csv`
- `results/paper_review/downward_raw.json`
- `results/paper_review/aggregates.csv`
- `results/paper_review/aggregates.json`

## Figures
`TODO`
<!-- Use this section for static figures from the paper, qualitative trajectory screenshots, or benchmark summary plots.

<!--
### Example figure slot

![Braided corridors qualitative comparison](media/braided_corridors.png)
-->

## Video / Demo
`TODO`
<!-- Use this section for a GitHub-hosted video, a GIF preview, or an external demo link.

<!--
### Example video slot

[Watch the demo video](media/demo.mp4)
-->

## Citation

If you use this repository, please cite the paper once the bibliographic details are finalized.

```bibtex
@inproceedings{
vijay2026geometric,
title={Geometric Coherence via Weighted Matching for 3D Heterogeneous Multi-Agent Reach-Avoid Games},
author={Prajwal Vijay},
booktitle={ICRA 2026 Workshop on Multi-Agent Robotic Systems: Real-World Collaboration and Interaction},
year={2026},
url={https://openreview.net/forum?id=0WHJGj07mJ}
}
```

## Contact

Prajwal Vijay  
Department of Electrical Engineering, IIT Madras  
`ee23b057@smail.iitm.ac.in`

## License

This repository currently includes a conservative placeholder [LICENSE](LICENSE)
file so you can publish it safely. Replace it with your preferred open-source
license before advertising the code as reusable.
