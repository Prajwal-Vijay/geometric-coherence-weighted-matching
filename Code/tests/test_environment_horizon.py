import os
import sys
import unittest

import numpy as np


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import Environment
import Evader
import Pursuer
import assignment_strategies


class ScriptedStrategy(assignment_strategies.AssignmentStrategy):
    def __init__(self, scripts):
        super().__init__(name="scripted")
        self.scripts = list(scripts)
        self.calls = 0

    def select(self, snapshot):
        script_index = min(self.calls, len(self.scripts) - 1)
        planned_assignments = []
        for coalition, evader_index in self.scripts[script_index]:
            planned_assignments.append(
                assignment_strategies.AssignmentRecord(
                    coalition=tuple(coalition),
                    evader_index=evader_index,
                    coalition_size=len(coalition),
                    score=1.0,
                    intercept_point=np.array([float(self.calls), float(evader_index), 1.0]),
                )
            )
        self.calls += 1
        planned_assignments.sort(
            key=lambda record: (record.coalition_size, record.coalition, record.evader_index)
        )
        return assignment_strategies.AssignmentResult(assignments=tuple(planned_assignments))


class ScriptedEnvironment(Environment.Environment):
    def __init__(self, scripts, replan_interval_steps=100, event_schedule=None):
        pursuers = [
            Pursuer.Pursuer(np.array([0.0, 0.0, 10.0]), 1.0, 0),
            Pursuer.Pursuer(np.array([5.0, 0.0, 10.0]), 1.0, 1),
        ]
        evaders = [
            Evader.Evader(np.array([0.0, 0.0, 10.0]), 1.0, 0),
            Evader.Evader(np.array([5.0, 0.0, 10.0]), 1.0, 1),
        ]
        super().__init__(
            N=2,
            M=2,
            t=0.1,
            pursuers=pursuers,
            evaders=evaders,
            strategy="weighted_sequential",
            replan_interval_steps=replan_interval_steps,
        )
        self.assignment_strategy = ScriptedStrategy(scripts)
        self.event_schedule = dict(event_schedule or {})

    def _build_assignment_snapshot(self):
        return assignment_strategies.AssignmentSnapshot(
            pursuer_count=len(self.pursuers),
            candidates_by_size={1: tuple(), 2: tuple(), 3: tuple()},
        )

    def _compute_candidate_for_pair(self, coalition, evader):
        mask = 0
        for pursuer_idx in coalition:
            mask |= 1 << pursuer_idx
        return assignment_strategies.AssignmentCandidate(
            coalition=tuple(coalition),
            evader_index=evader.index,
            coalition_size=len(coalition),
            score=1.0,
            intercept_point=np.array([float(self.step_count), float(evader.index), float(len(coalition))]),
            feasible=True,
            distance=1.0,
            pursuer_mask=mask,
        )

    def _apply_current_assignment_motion(self):
        return None

    def _update_resolution_status(self, event_step):
        event_kind = self.event_schedule.get(event_step)
        if event_kind:
            unresolved = [
                evader
                for evader in self.evaders
                if not evader.captured and not evader.escaped
            ]
            if unresolved:
                evader = unresolved[0]
                if event_kind == "capture":
                    evader.captured = True
                    self.capture_step_by_evader.setdefault(evader.index, event_step)
                if event_kind == "escape":
                    evader.escaped = True
                    self.escape_step_by_evader.setdefault(evader.index, event_step)
                self.replan_due_to_event = True
        self._refresh_active_evaders()
        self._refresh_current_assignments()
        return event_kind == "capture", event_kind == "escape"


class EnvironmentHorizonTests(unittest.TestCase):
    def test_planning_occurs_at_step_zero_and_at_interval(self):
        env = ScriptedEnvironment(scripts=[[((0,), 0)]], replan_interval_steps=2)

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 1)

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 1)

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 2)

    def test_capture_event_triggers_immediate_replan(self):
        env = ScriptedEnvironment(
            scripts=[[((0,), 0)], [((1,), 1)]],
            replan_interval_steps=10,
            event_schedule={1: "capture"},
        )

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 1)

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 2)

    def test_escape_event_triggers_immediate_replan(self):
        env = ScriptedEnvironment(
            scripts=[[((0,), 0)], [((1,), 1)]],
            replan_interval_steps=10,
            event_schedule={1: "escape"},
        )

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 1)

        env.step()
        self.assertEqual(env.assignment_strategy.calls, 2)

    def test_new_assignment_replaces_old_assignment_on_replan(self):
        env = ScriptedEnvironment(
            scripts=[[((0,), 0)], [((1,), 1)]],
            replan_interval_steps=2,
        )

        env.step()
        self.assertEqual([assignment.identifier for assignment in env.current_assignment_result.assignments], [((0,), 0)])

        env.step()
        env.step()
        self.assertEqual([assignment.identifier for assignment in env.current_assignment_result.assignments], [((1,), 1)])

    def test_pairing_stays_fixed_between_planning_events(self):
        env = ScriptedEnvironment(
            scripts=[[((0,), 0)], [((1,), 1)]],
            replan_interval_steps=5,
        )

        env.step()
        first_ids = [assignment.identifier for assignment in env.current_assignment_result.assignments]

        env.step()
        second_ids = [assignment.identifier for assignment in env.current_assignment_result.assignments]

        self.assertEqual(first_ids, second_ids)

    def test_intercepts_refresh_between_planning_events(self):
        env = ScriptedEnvironment(
            scripts=[[((0,), 0)]],
            replan_interval_steps=5,
        )

        env.step()
        intercept_before = env.current_assignment_result.assignments[0].intercept_point.copy()

        env.step()
        intercept_after = env.current_assignment_result.assignments[0].intercept_point.copy()

        self.assertFalse(np.array_equal(intercept_before, intercept_after))


if __name__ == "__main__":
    unittest.main()
