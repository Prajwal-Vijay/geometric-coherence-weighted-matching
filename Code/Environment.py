"""
Simulation environment for the 3D multiplayer pursuit-evasion game.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, minimize

import assignment_strategies

UNMATCHED_EVADER_POLICIES = ("stationary", "downward")


class Environment:
    def __init__(
        self,
        N,
        M,
        t,
        pursuers,
        evaders,
        strategy="weighted_sequential",
        replan_interval_steps=100,
        rng_seed=None,
        unmatched_evader_policy="stationary",
    ):
        if unmatched_evader_policy not in UNMATCHED_EVADER_POLICIES:
            raise ValueError(
                f"Unknown unmatched evader policy '{unmatched_evader_policy}'. "
                f"Expected one of {UNMATCHED_EVADER_POLICIES}."
            )
        self.N = N
        self.M = M
        self.timestep = t
        self.pursuers = list(pursuers)
        self.evaders = list(evaders)
        self.strategy_name = strategy
        self.replan_interval_steps = replan_interval_steps
        self.rng_seed = rng_seed
        self.unmatched_evader_policy = unmatched_evader_policy
        self.assignment_strategy = assignment_strategies.build_strategy(strategy, rng_seed=rng_seed)
        self.max_coalition_size = min(3, len(self.pursuers))
        self._reset_runtime_state()

    def _reset_runtime_state(self):
        self.step_count = 0
        self.last_plan_step = None
        self.replan_due_to_event = True
        self.current_assignment_result = assignment_strategies.empty_assignment_result()
        self.initial_pursuer_positions = [
            np.array(pursuer.get_pos(), dtype=float).reshape(-1)
            for pursuer in self.pursuers
        ]
        self.pursuer_path_lengths = [0.0 for _ in self.pursuers]
        self.previous_heading_vectors = [None for _ in self.pursuers]
        self.angular_control_effort = 0.0
        self.capture_step_by_evader: Dict[int, int] = {}
        self.capture_height_by_evader: Dict[int, float] = {}
        self.escape_step_by_evader: Dict[int, int] = {}
        self.termination_reason = None
        self._refresh_active_evaders()

    def _refresh_active_evaders(self):
        self.active_evaders = [
            evader
            for evader in self.evaders
            if not evader.captured and not getattr(evader, "escaped", False)
        ]

    def _iter_coalitions(self, max_coalition_size=None) -> Iterable[Tuple[int, ...]]:
        if max_coalition_size is None:
            max_coalition_size = self.max_coalition_size
        for size in range(1, min(max_coalition_size, len(self.pursuers)) + 1):
            for coalition in combinations(range(len(self.pursuers)), size):
                yield coalition

    def _pursuer_mask(self, coalition: Tuple[int, ...]) -> int:
        mask = 0
        for pursuer_idx in coalition:
            mask |= 1 << pursuer_idx
        return mask

    def check_initialization(self, verbose=False):
        """Check whether the initial positions satisfy the deployment constraints."""
        self._refresh_active_evaders()
        for pursuer in self.pursuers:
            for evader in self.active_evaders:
                if np.linalg.norm(pursuer.get_pos() - evader.get_pos()) < pursuer.capture_radius:
                    if verbose:
                        print(f"Evader {evader.index} is too close to the pursuer at initialization.")
                    return False
        for evader in self.active_evaders:
            if evader.get_pos()[2, 0] < 0:
                if verbose:
                    print(f"Evader {evader.index} is not in the play region at initialization.")
                return False

        evasion_matrix, coalition_list = self.create_evasion_matrix()
        for evader in self.active_evaders:
            for coalition in coalition_list:
                if evasion_matrix[(evader, coalition)] == 1:
                    break
            else:
                if verbose:
                    print(f"Evader {evader.index} has no winning-counter coalition at initialization.")
                return False
        return True

    def create_evasion_matrix(self, max_coalition_size=3):
        coalition_list = list(self._iter_coalitions(max_coalition_size=max_coalition_size))
        evasion_matrix = {}
        for evader in self.active_evaders:
            for coalition in coalition_list:
                min_z = self._compute_min_z_in_bes(evader, coalition)
                evasion_matrix[(evader, coalition)] = 1 if min_z > 0 else -1
        return evasion_matrix, coalition_list

    def _compute_min_z_in_bes(self, evader, coalition):
        x0 = np.array([0.0, 0.0, 0.0])

        def make_constraint(pursuer_pos, evader_pos, alpha_ij, capture_radius):
            return {
                "type": "ineq",
                "fun": lambda x, p=pursuer_pos, e=evader_pos, a=alpha_ij, r=capture_radius: (
                    np.linalg.norm(x - p) - a * np.linalg.norm(x - e) - r
                ),
            }

        bounds = Bounds([-100, -100, -np.inf], [+100, +100, +np.inf])
        constraints = []
        for pursuer_idx in coalition:
            pursuer = self.pursuers[pursuer_idx]
            evader_pos = np.array(evader.get_pos()).reshape(-1)
            pursuer_pos = np.array(pursuer.get_pos()).reshape(-1)
            alpha_ij = pursuer.speed / evader.speed
            capture_radius = pursuer.capture_radius
            constraints.append(make_constraint(pursuer_pos, evader_pos, alpha_ij, capture_radius))
        objective = lambda x: x[2]
        res = minimize(objective, x0, constraints=constraints, bounds=bounds, method="SLSQP")
        return res.x[2]

    def _solve_value_function_cvxpy(self, coalition, evader):
        # Legacy helper name retained for compatibility. The released benchmark
        # path uses SciPy SLSQP here, not cvxpy.
        x0 = np.array([0.0, 0.0, 0.0])

        def make_constraint(pursuer_pos, evader_pos, alpha_ij, capture_radius):
            return {
                "type": "ineq",
                "fun": lambda x, p=pursuer_pos, e=evader_pos, a=alpha_ij, r=capture_radius: (
                    np.linalg.norm(x - p) - a * np.linalg.norm(x - e) - r
                ),
            }

        bounds = Bounds([-100, -100, -np.inf], [+100, +100, +np.inf])
        constraints = []
        for pursuer_idx in coalition:
            pursuer = self.pursuers[pursuer_idx]
            evader_pos = np.array(evader.get_pos()).reshape(-1)
            pursuer_pos = np.array(pursuer.get_pos()).reshape(-1)
            alpha_ij = pursuer.speed / evader.speed
            capture_radius = pursuer.capture_radius
            constraints.append(make_constraint(pursuer_pos, evader_pos, alpha_ij, capture_radius))
        objective = lambda x: x[2]
        res = minimize(objective, x0, constraints=constraints, bounds=bounds, method="SLSQP")
        return float(res.x[2]), np.array(res.x, dtype=float).reshape(-1)

    def valueFunctionMatrix(self, max_coalition_size=3):
        value_matrix = {}
        optimal_points_matrix = {}
        coalition_list = list(self._iter_coalitions(max_coalition_size=max_coalition_size))

        for evader in self.active_evaders:
            for coalition in coalition_list:
                try:
                    value, optimal_point = self._solve_value_function_cvxpy(coalition, evader)
                except Exception:
                    value = -1.0
                    optimal_point = np.array(evader.get_pos(), dtype=float).reshape(-1)
                value_matrix[(evader, coalition)] = value
                optimal_points_matrix[(evader, coalition)] = optimal_point

        return value_matrix, optimal_points_matrix, coalition_list

    def plot_current_positions(self):
        """Plot the current positions of pursuers and evaders."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, pursuer in enumerate(self.pursuers):
            pos = pursuer.get_pos().flatten()
            ax.scatter(pos[0], pos[1], pos[2], c="#FF0000", label=f"Pursuer {i}")
        for i, evader in enumerate(self.evaders):
            pos = evader.get_pos().flatten()
            ax.scatter(pos[0], pos[1], pos[2], c="#0000FF", label=f"Evader {i}")
        ax.set_title("Pursuit Evasion Game")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        plt.legend()
        plt.grid()
        plt.show()

    def _compute_candidate_for_pair(self, coalition, evader):
        try:
            score, intercept_point = self._solve_value_function_cvxpy(coalition, evader)
        except Exception:
            score = -1.0
            intercept_point = np.array(evader.get_pos(), dtype=float).reshape(-1)

        distance = None
        if len(coalition) == 1:
            distance = float(np.linalg.norm(self.pursuers[coalition[0]].get_pos() - evader.get_pos()))

        return assignment_strategies.AssignmentCandidate(
            coalition=tuple(coalition),
            evader_index=evader.index,
            coalition_size=len(coalition),
            score=float(score),
            intercept_point=np.array(intercept_point, dtype=float).reshape(-1),
            feasible=bool(np.isfinite(score) and score > 0),
            distance=distance,
            pursuer_mask=self._pursuer_mask(coalition),
        )

    def _build_assignment_snapshot(self):
        candidates_by_size = {1: [], 2: [], 3: []}
        for coalition in self._iter_coalitions():
            for evader in self.active_evaders:
                candidates_by_size[len(coalition)].append(self._compute_candidate_for_pair(coalition, evader))
        return assignment_strategies.AssignmentSnapshot(
            pursuer_count=len(self.pursuers),
            candidates_by_size={size: tuple(candidates) for size, candidates in candidates_by_size.items()},
        )

    def _should_replan(self):
        if self.last_plan_step is None:
            return True
        if self.replan_due_to_event:
            return True
        return (self.step_count - self.last_plan_step) >= self.replan_interval_steps

    def _adopt_new_plan(self):
        snapshot = self._build_assignment_snapshot()
        self.current_assignment_result = self.assignment_strategy.select(snapshot)
        self.last_plan_step = self.step_count
        self.replan_due_to_event = False

    def _refresh_current_assignments(self):
        refreshed_assignments = []
        for assignment in self.current_assignment_result.assignments:
            evader = self.evaders[assignment.evader_index]
            if evader.captured or evader.escaped:
                continue
            candidate = self._compute_candidate_for_pair(assignment.coalition, evader)
            refreshed_assignments.append(assignment_strategies.candidate_to_record(candidate))
        refreshed_assignments.sort(
            key=lambda record: (record.coalition_size, record.coalition, record.evader_index)
        )
        self.current_assignment_result = assignment_strategies.AssignmentResult(
            assignments=tuple(refreshed_assignments)
        )

    def _apply_current_assignment_motion(self):
        pursuer_velocities = [None for _ in self.pursuers]
        evader_velocities = {}

        for assignment in self.current_assignment_result.assignments:
            evader = self.evaders[assignment.evader_index]
            if evader.captured or evader.escaped:
                continue
            target = np.array(assignment.intercept_point, dtype=float).reshape(-1, 1)
            for pursuer_idx in assignment.coalition:
                pursuer = self.pursuers[pursuer_idx]
                pursuer_velocities[pursuer_idx] = pursuer.heading_velocity(target)
            evader_velocities[assignment.evader_index] = evader.heading_velocity(target)

        if self.unmatched_evader_policy == "downward":
            for evader in self.active_evaders:
                if evader.index not in evader_velocities:
                    evader_velocities[evader.index] = evader.downward_velocity()

        for pursuer_idx, pursuer_velocity in enumerate(pursuer_velocities):
            if pursuer_velocity is None:
                self.previous_heading_vectors[pursuer_idx] = None
                continue

            pursuer_velocity = np.array(pursuer_velocity, dtype=float).reshape(-1, 1)
            speed_norm = float(np.linalg.norm(pursuer_velocity))
            if speed_norm > 0:
                heading = pursuer_velocity.reshape(-1) / speed_norm
                previous_heading = self.previous_heading_vectors[pursuer_idx]
                if previous_heading is not None:
                    heading_delta = heading - previous_heading
                    self.angular_control_effort += float(np.dot(heading_delta, heading_delta)) / self.timestep
                self.previous_heading_vectors[pursuer_idx] = heading
            else:
                self.previous_heading_vectors[pursuer_idx] = None

            pursuer_step = self.timestep * pursuer_velocity
            self.pursuer_path_lengths[pursuer_idx] += float(np.linalg.norm(pursuer_step))
            pursuer = self.pursuers[pursuer_idx]
            pursuer.update_pos(pursuer.position + pursuer_step)

        for evader_index, evader_velocity in evader_velocities.items():
            evader_velocity = np.array(evader_velocity, dtype=float).reshape(-1, 1)
            evader = self.evaders[evader_index]
            evader.update_pos(evader.position + self.timestep * evader_velocity)

    def _update_resolution_status(self, event_step):
        capture_event = False
        escape_event = False

        for evader in self.evaders:
            if evader.captured or evader.escaped:
                continue

            for pursuer in self.pursuers:
                if np.linalg.norm(pursuer.get_pos() - evader.get_pos()) < pursuer.capture_radius:
                    evader.captured = True
                    self.capture_step_by_evader.setdefault(evader.index, event_step)
                    self.capture_height_by_evader.setdefault(evader.index, float(evader.get_pos()[2, 0]))
                    capture_event = True
                    break

            if evader.captured:
                continue

            if evader.get_pos()[2, 0] < 0:
                evader.escaped = True
                self.escape_step_by_evader.setdefault(evader.index, event_step)
                escape_event = True

        if capture_event or escape_event:
            self.replan_due_to_event = True

        self._refresh_active_evaders()
        self._refresh_current_assignments()
        return capture_event, escape_event

    def step(self):
        self._refresh_active_evaders()
        if not self.active_evaders:
            self.termination_reason = "all_resolved"
            return True

        if self._should_replan():
            self._adopt_new_plan()
        else:
            self._refresh_current_assignments()

        self._apply_current_assignment_motion()
        event_step = self.step_count + 1
        self._update_resolution_status(event_step=event_step)
        self.step_count = event_step

        if not self.active_evaders:
            self.termination_reason = "all_resolved"
            return True

        return False

    def _initialize_trajectories(self):
        pursuer_trajectories = [[] for _ in self.pursuers]
        evader_trajectories = [[] for _ in self.evaders]
        self._record_positions(pursuer_trajectories, evader_trajectories)
        return pursuer_trajectories, evader_trajectories

    def _record_positions(self, pursuer_trajectories, evader_trajectories):
        for idx, pursuer in enumerate(self.pursuers):
            pursuer_trajectories[idx].append(pursuer.get_pos().flatten())
        for idx, evader in enumerate(self.evaders):
            evader_trajectories[idx].append(evader.get_pos().flatten())

    def _render_frame(self, ax, pursuer_trajectories, evader_trajectories, title, animation_speed):
        ax.clear()
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title(title)

        all_positions = [pursuer.get_pos().flatten() for pursuer in self.pursuers]
        all_positions.extend(evader.get_pos().flatten() for evader in self.evaders)
        all_positions = np.array(all_positions)
        margin = 5
        ax.set_xlim(np.min(all_positions[:, 0]) - margin, np.max(all_positions[:, 0]) + margin)
        ax.set_ylim(np.min(all_positions[:, 1]) - margin, np.max(all_positions[:, 1]) + margin)
        ax.set_zlim(np.min(all_positions[:, 2]) - margin, np.max(all_positions[:, 2]) + margin)

        xx, yy = np.meshgrid(
            np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10),
            np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10),
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color="green")

        pursuer_colors = plt.cm.Reds(np.linspace(0.4, 1, len(self.pursuers)))
        evader_colors = plt.cm.Blues(np.linspace(0.4, 1, len(self.evaders)))

        for idx, trajectory in enumerate(pursuer_trajectories):
            trajectory = np.array(trajectory)
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=pursuer_colors[idx], linewidth=2)
            pos = self.pursuers[idx].get_pos().flatten()
            ax.scatter(pos[0], pos[1], pos[2], color=pursuer_colors[idx], s=100, marker="o", edgecolors="black")

        for idx, trajectory in enumerate(evader_trajectories):
            trajectory = np.array(trajectory)
            evader = self.evaders[idx]
            if len(trajectory) > 1:
                color = "gray" if evader.captured else ("green" if evader.escaped else evader_colors[idx])
                linestyle = "--" if (evader.captured or evader.escaped) else "-"
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=2, linestyle=linestyle)
            pos = evader.get_pos().flatten()
            if evader.captured:
                ax.scatter(pos[0], pos[1], pos[2], color="gray", s=70, marker="x")
            elif evader.escaped:
                ax.scatter(pos[0], pos[1], pos[2], color="green", s=70, marker="^", edgecolors="black")
            else:
                ax.scatter(pos[0], pos[1], pos[2], color=evader_colors[idx], s=100, marker="^", edgecolors="black")

        status_text = (
            f"Step: {self.step_count}\n"
            f"Strategy: {self.strategy_name}\n"
            f"Unmatched Evaders: {self.unmatched_evader_policy}\n"
            f"Replan Interval: {self.replan_interval_steps}\n"
            f"Active Evaders: {len(self.active_evaders)}\n"
            f"Captured: {len(self.capture_step_by_evader)}\n"
            f"Escaped: {len(self.escape_step_by_evader)}"
        )
        ax.text2D(
            0.02,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.draw()
        plt.pause(animation_speed)

    def _build_summary(self, pursuer_trajectories=None, evader_trajectories=None):
        captured_evaders = sorted(self.capture_step_by_evader)
        escaped_evaders = sorted(self.escape_step_by_evader)
        average_capture_height = None
        if self.capture_height_by_evader:
            average_capture_height = float(
                np.mean([self.capture_height_by_evader[evader_idx] for evader_idx in sorted(self.capture_height_by_evader)])
            )

        tortuosity_values = []
        for pursuer_idx, pursuer in enumerate(self.pursuers):
            path_length = float(self.pursuer_path_lengths[pursuer_idx])
            chord_length = float(
                np.linalg.norm(pursuer.get_pos().reshape(-1) - self.initial_pursuer_positions[pursuer_idx])
            )
            if path_length <= 1e-12 and chord_length <= 1e-12:
                tortuosity_values.append(1.0)
            else:
                tortuosity_values.append(max(1.0, path_length / max(chord_length, 1e-12)))

        average_path_tortuosity = float(np.mean(tortuosity_values)) if tortuosity_values else 1.0
        summary = {
            "strategy": self.strategy_name,
            "unmatched_evader_policy": self.unmatched_evader_policy,
            "replan_interval_steps": self.replan_interval_steps,
            "steps": self.step_count,
            "total_steps": self.step_count,
            "average_path_tortuosity": average_path_tortuosity,
            "angular_control_effort": float(self.angular_control_effort),
            "captured_count": len(captured_evaders),
            "escaped_count": len(escaped_evaders),
            "captured_evaders": captured_evaders,
            "escaped_evaders": escaped_evaders,
            "capture_step_by_evader": dict(sorted(self.capture_step_by_evader.items())),
            "capture_height_by_evader": dict(sorted(self.capture_height_by_evader.items())),
            "average_capture_height": average_capture_height,
            "escape_step_by_evader": dict(sorted(self.escape_step_by_evader.items())),
            "termination_reason": self.termination_reason,
            "final_positions": {
                "pursuers": [pursuer.get_pos().flatten() for pursuer in self.pursuers],
                "evaders": [evader.get_pos().flatten() for evader in self.evaders],
            },
        }
        if pursuer_trajectories is not None and evader_trajectories is not None:
            summary["pursuer_trajectories"] = pursuer_trajectories
            summary["evader_trajectories"] = evader_trajectories
        return summary

    def run_episode(self, max_steps=10000, render=False, record_trajectories=False, animation_speed=0.1):
        if render:
            record_trajectories = True

        trajectories = None
        if record_trajectories:
            trajectories = self._initialize_trajectories()

        fig = None
        ax = None
        if render:
            plt.ion()
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        game_over = False
        while self.step_count < max_steps and not game_over:
            if render:
                self._render_frame(
                    ax,
                    trajectories[0],
                    trajectories[1],
                    title=f"Pursuit-Evasion Game - Step {self.step_count}",
                    animation_speed=animation_speed,
                )

            game_over = self.step()

            if trajectories is not None:
                self._record_positions(trajectories[0], trajectories[1])

        if self.termination_reason is None:
            self.termination_reason = "max_steps" if self.active_evaders else "all_resolved"

        if render:
            self._render_frame(
                ax,
                trajectories[0],
                trajectories[1],
                title="Final Trajectories - Pursuit-Evasion Game",
                animation_speed=animation_speed,
            )
            plt.ioff()
            plt.show()

        if trajectories is None:
            return self._build_summary()
        return self._build_summary(
            pursuer_trajectories=trajectories[0],
            evader_trajectories=trajectories[1],
        )

    def obtain_trajectories(self, max_steps=10000, animation_speed=0.1):
        """
        Run the simulation and display real-time 3D animation with trajectory curves.
        """

        return self.run_episode(
            max_steps=max_steps,
            render=True,
            record_trajectories=True,
            animation_speed=animation_speed,
        )
