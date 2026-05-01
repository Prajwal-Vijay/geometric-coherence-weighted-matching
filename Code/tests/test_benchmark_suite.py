import os
import sys
import unittest
from unittest import mock

import numpy as np


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import Environment
import Evader
import benchmark_suite
import Pursuer


def fake_solver(self, coalition, evader):
    coalition_positions = np.array([self.pursuers[idx].get_pos().reshape(-1) for idx in coalition])
    coalition_center = np.mean(coalition_positions, axis=0)
    evader_position = evader.get_pos().reshape(-1)
    intercept_point = 0.5 * (coalition_center + evader_position)
    return float(intercept_point[2]), intercept_point


class BenchmarkSuiteTests(unittest.TestCase):
    def test_each_family_has_five_initializations(self):
        self.assertEqual(len(benchmark_suite.scenario_family_names()), 11)
        self.assertIn("spread_mild_hetero", benchmark_suite.scenario_family_names())
        self.assertIn("three_vs_ten_spread_mild_hetero", benchmark_suite.scenario_family_names())
        self.assertIn("three_vs_ten_x_crossing", benchmark_suite.scenario_family_names())
        self.assertIn("three_vs_ten_braided_corridors", benchmark_suite.scenario_family_names())
        self.assertIn("three_vs_ten_switchback_gate", benchmark_suite.scenario_family_names())
        for family_name in benchmark_suite.scenario_family_names():
            self.assertEqual(len(benchmark_suite.scenario_names(family=family_name)), 5)

    def test_suite_filters_split_3v6_and_3v10(self):
        suite_3v6_families = benchmark_suite.scenario_family_names(suite=benchmark_suite.SUITE_3V6)
        suite_3v10_families = benchmark_suite.scenario_family_names(suite=benchmark_suite.SUITE_3V10)

        self.assertTrue(all(not family.startswith("three_vs_ten_") for family in suite_3v6_families))
        self.assertTrue(all(family.startswith("three_vs_ten_") for family in suite_3v10_families))
        self.assertEqual(len(suite_3v6_families), 4)
        self.assertEqual(len(suite_3v10_families), 7)

    def test_benchmark_suite_runs_all_strategies_and_scenarios(self):
        with mock.patch.object(Environment.Environment, "_solve_value_function_cvxpy", new=fake_solver):
            results = benchmark_suite.run_benchmark_suite(
                strategies=benchmark_suite.DEFAULT_STRATEGIES,
                scenarios=benchmark_suite.scenario_names(),
                max_steps=1,
                replan_interval_steps=100,
                rng_seed=7,
            )

        self.assertEqual(
            len(results),
            len(benchmark_suite.DEFAULT_STRATEGIES) * len(benchmark_suite.scenario_names()),
        )

        expected_keys = {
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
        }
        for result in results:
            self.assertTrue(expected_keys.issubset(result.keys()))
            self.assertEqual(result["unmatched_evader_policy"], "stationary")
            self.assertGreaterEqual(result["average_path_tortuosity"], 1.0)
            self.assertGreaterEqual(result["angular_control_effort"], 0.0)

    def test_one_step_run_has_unit_tortuosity_and_zero_angular_effort(self):
        with mock.patch.object(Environment.Environment, "_solve_value_function_cvxpy", new=fake_solver):
            result = benchmark_suite.run_single_benchmark(
                scenario_name=benchmark_suite.scenario_names()[0],
                strategy="weighted_sequential",
                max_steps=1,
                replan_interval_steps=100,
            )

        self.assertAlmostEqual(result["average_path_tortuosity"], 1.0)
        self.assertAlmostEqual(result["angular_control_effort"], 0.0)

    def test_downward_policy_moves_unmatched_evaders_toward_goal_plane(self):
        with mock.patch.object(Environment.Environment, "_solve_value_function_cvxpy", new=fake_solver):
            stationary_env = Environment.Environment(
                N=1,
                M=2,
                t=0.1,
                pursuers=[Pursuer.Pursuer(np.array([0.0, 0.0, 12.0]), 1.5, 0)],
                evaders=[
                    Evader.Evader(np.array([1.0, 0.0, 12.0]), 1.0, 0),
                    Evader.Evader(np.array([3.0, 0.0, 12.0]), 1.0, 1),
                ],
                strategy="weighted_sequential",
                unmatched_evader_policy="stationary",
            )
            downward_env = Environment.Environment(
                N=1,
                M=2,
                t=0.1,
                pursuers=[Pursuer.Pursuer(np.array([0.0, 0.0, 12.0]), 1.5, 0)],
                evaders=[
                    Evader.Evader(np.array([1.0, 0.0, 12.0]), 1.0, 0),
                    Evader.Evader(np.array([3.0, 0.0, 12.0]), 1.0, 1),
                ],
                strategy="weighted_sequential",
                unmatched_evader_policy="downward",
            )

            stationary_env.run_episode(max_steps=1, render=False, record_trajectories=False)
            downward_env.run_episode(max_steps=1, render=False, record_trajectories=False)

        self.assertEqual(stationary_env.evaders[1].get_pos()[2, 0], 12.0)
        self.assertLess(downward_env.evaders[1].get_pos()[2, 0], 12.0)

    def test_run_single_benchmark_accepts_unmatched_evader_policy(self):
        with mock.patch.object(Environment.Environment, "_solve_value_function_cvxpy", new=fake_solver):
            result = benchmark_suite.run_single_benchmark(
                scenario_name=benchmark_suite.scenario_names()[0],
                strategy="weighted_sequential",
                max_steps=1,
                replan_interval_steps=100,
                unmatched_evader_policy="downward",
            )

        self.assertEqual(result["unmatched_evader_policy"], "downward")


if __name__ == "__main__":
    unittest.main()
