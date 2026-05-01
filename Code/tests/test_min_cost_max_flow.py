import os
import sys
import unittest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import minCostMaxFlow_implemented


class MinCostMaxFlowTests(unittest.TestCase):
    def test_solver_prefers_negative_cost_augmenting_path(self):
        solver = minCostMaxFlow_implemented.MinCostMaxFlow()
        solver.add_edge("source", "middle", capacity=1, cost=-5, label="pick-middle")
        solver.add_edge("middle", "sink", capacity=1, cost=0)
        solver.add_edge("source", "sink", capacity=1, cost=0)

        result = solver.min_cost_flow("source", "sink", max_flow=1)

        self.assertEqual(result["flow"], 1)
        self.assertEqual(solver.used_labeled_edges(), ["pick-middle"])


if __name__ == "__main__":
    unittest.main()
