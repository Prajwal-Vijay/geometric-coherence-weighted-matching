import os
import sys
import unittest

import numpy as np


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import assignment_strategies


def candidate(coalition, evader_index, score=1.0, feasible=True, distance=None):
    coalition = tuple(coalition)
    pursuer_mask = 0
    for pursuer_idx in coalition:
        pursuer_mask |= 1 << pursuer_idx
    return assignment_strategies.AssignmentCandidate(
        coalition=coalition,
        evader_index=evader_index,
        coalition_size=len(coalition),
        score=score,
        intercept_point=np.array([0.0, 0.0, 1.0]),
        feasible=feasible,
        distance=distance,
        pursuer_mask=pursuer_mask,
    )


def snapshot(*, singles=(), doubles=(), triples=(), pursuer_count=3):
    return assignment_strategies.AssignmentSnapshot(
        pursuer_count=pursuer_count,
        candidates_by_size={
            1: tuple(singles),
            2: tuple(doubles),
            3: tuple(triples),
        },
    )


class AssignmentStrategyTests(unittest.TestCase):
    def test_weighted_prefers_higher_total_value_on_tie(self):
        test_snapshot = snapshot(
            singles=(
                candidate((0,), 0, score=1.0),
                candidate((1,), 1, score=1.0),
                candidate((0,), 1, score=10.0),
                candidate((1,), 0, score=10.0),
            ),
            pursuer_count=2,
        )

        strategy = assignment_strategies.build_strategy("weighted_sequential")
        result = strategy.select(test_snapshot)

        self.assertEqual(
            [assignment.identifier for assignment in result.assignments],
            [((0,), 1), ((1,), 0)],
        )

    def test_unweighted_ignores_value_differences(self):
        test_snapshot = snapshot(
            singles=(
                candidate((0,), 0, score=1.0),
                candidate((1,), 1, score=1.0),
                candidate((0,), 1, score=10.0),
                candidate((1,), 0, score=10.0),
            ),
            pursuer_count=2,
        )

        strategy = assignment_strategies.build_strategy("unweighted_sequential")
        result = strategy.select(test_snapshot)

        self.assertEqual(
            [assignment.identifier for assignment in result.assignments],
            [((0,), 0), ((1,), 1)],
        )

    def test_no_assignment_reuses_pursuer_or_evader(self):
        test_snapshot = snapshot(
            doubles=(
                candidate((0, 1), 0, score=5.0),
                candidate((0, 2), 1, score=6.0),
                candidate((1, 2), 1, score=7.0),
            ),
            pursuer_count=3,
        )

        strategy = assignment_strategies.build_strategy("weighted_sequential")
        result = strategy.select(test_snapshot)

        used_pursuers = set()
        used_evaders = set()
        for assignment in result.assignments:
            self.assertNotIn(assignment.evader_index, used_evaders)
            used_evaders.add(assignment.evader_index)
            for pursuer_idx in assignment.coalition:
                self.assertNotIn(pursuer_idx, used_pursuers)
                used_pursuers.add(pursuer_idx)

    def test_stage_filtering_removes_committed_pursuers_and_evaders(self):
        test_snapshot = snapshot(
            singles=(candidate((0,), 0, score=5.0),),
            doubles=(
                candidate((0, 1), 1, score=100.0),
                candidate((1, 2), 2, score=10.0),
            ),
            pursuer_count=3,
        )

        strategy = assignment_strategies.build_strategy("weighted_sequential")
        result = strategy.select(test_snapshot)

        self.assertEqual(
            [assignment.identifier for assignment in result.assignments],
            [((0,), 0), ((1, 2), 2)],
        )

    def test_weighted_solver_handles_multi_coalition_pair_stage(self):
        test_snapshot = snapshot(
            doubles=(
                candidate((0, 1), 0, score=5.0),
                candidate((2, 3), 1, score=5.0),
                candidate((0, 2), 0, score=10.0),
                candidate((1, 3), 1, score=10.0),
            ),
            pursuer_count=4,
        )

        strategy = assignment_strategies.build_strategy("weighted_sequential")
        result = strategy.select(test_snapshot)

        self.assertEqual(
            [assignment.identifier for assignment in result.assignments],
            [((0, 2), 0), ((1, 3), 1)],
        )

    def test_random_sequential_is_reproducible(self):
        test_snapshot = snapshot(
            singles=(
                candidate((0,), 0, score=1.0),
                candidate((0,), 1, score=1.0),
                candidate((1,), 0, score=1.0),
                candidate((1,), 1, score=1.0),
            ),
            pursuer_count=2,
        )

        strategy_a = assignment_strategies.build_strategy("random_sequential", rng_seed=7)
        strategy_b = assignment_strategies.build_strategy("random_sequential", rng_seed=7)

        result_a = strategy_a.select(test_snapshot)
        result_b = strategy_b.select(test_snapshot)

        self.assertEqual(
            [assignment.identifier for assignment in result_a.assignments],
            [assignment.identifier for assignment in result_b.assignments],
        )

    def test_nearest_single_never_emits_multi_pursuer_coalitions(self):
        test_snapshot = snapshot(
            singles=(
                candidate((0,), 0, distance=5.0),
                candidate((1,), 1, distance=2.0),
            ),
            doubles=(candidate((0, 1), 0, score=100.0),),
            pursuer_count=2,
        )

        strategy = assignment_strategies.build_strategy("nearest_single")
        result = strategy.select(test_snapshot)

        self.assertTrue(result.assignments)
        self.assertTrue(all(assignment.coalition_size == 1 for assignment in result.assignments))


if __name__ == "__main__":
    unittest.main()
