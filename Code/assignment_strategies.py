"""
Assignment strategies for modular coalition-to-evader planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import random

import numpy as np
import minCostMaxFlow_implemented


Coalition = Tuple[int, ...]
CandidateId = Tuple[Coalition, int]


@dataclass(frozen=True)
class AssignmentCandidate:
    coalition: Coalition
    evader_index: int
    coalition_size: int
    score: float
    intercept_point: np.ndarray = field(compare=False, repr=False)
    feasible: bool = True
    distance: Optional[float] = None
    pursuer_mask: int = 0

    @property
    def identifier(self) -> CandidateId:
        return (self.coalition, self.evader_index)


@dataclass(frozen=True)
class AssignmentRecord:
    coalition: Coalition
    evader_index: int
    coalition_size: int
    score: float
    intercept_point: np.ndarray = field(compare=False, repr=False)

    @property
    def identifier(self) -> CandidateId:
        return (self.coalition, self.evader_index)


@dataclass(frozen=True)
class AssignmentResult:
    assignments: Tuple[AssignmentRecord, ...]


@dataclass(frozen=True)
class AssignmentSnapshot:
    pursuer_count: int
    candidates_by_size: Dict[int, Tuple[AssignmentCandidate, ...]]


def empty_assignment_result() -> AssignmentResult:
    return AssignmentResult(assignments=tuple())


def candidate_to_record(candidate: AssignmentCandidate) -> AssignmentRecord:
    return AssignmentRecord(
        coalition=candidate.coalition,
        evader_index=candidate.evader_index,
        coalition_size=candidate.coalition_size,
        score=float(candidate.score),
        intercept_point=np.array(candidate.intercept_point, dtype=float).reshape(-1),
    )


def build_strategy(name: str, rng_seed: Optional[int] = None) -> "AssignmentStrategy":
    if name == "weighted_sequential":
        return WeightedSequentialStrategy(name=name)
    if name == "unweighted_sequential":
        return UnweightedSequentialStrategy(name=name)
    if name == "random_sequential":
        return RandomSequentialStrategy(name=name, rng_seed=rng_seed)
    if name == "nearest_single":
        return NearestSingleStrategy(name=name)
    raise ValueError(f"Unknown strategy '{name}'.")


class AssignmentStrategy:
    def __init__(self, name: str):
        self.name = name

    def select(self, snapshot: AssignmentSnapshot) -> AssignmentResult:
        raise NotImplementedError


class SequentialStrategyBase(AssignmentStrategy):
    def __init__(self, name: str, use_weights: bool):
        super().__init__(name=name)
        self.use_weights = use_weights

    def select(self, snapshot: AssignmentSnapshot) -> AssignmentResult:
        selected: List[AssignmentRecord] = []
        used_pursuer_mask = 0
        used_evaders = set()

        for coalition_size in (1, 2, 3):
            stage_candidates = [
                candidate
                for candidate in snapshot.candidates_by_size.get(coalition_size, tuple())
                if candidate.feasible
                and candidate.evader_index not in used_evaders
                and candidate.pursuer_mask & used_pursuer_mask == 0
            ]
            stage_selection = self._select_stage(stage_candidates)
            for candidate in stage_selection:
                selected.append(candidate_to_record(candidate))
                used_evaders.add(candidate.evader_index)
                used_pursuer_mask |= candidate.pursuer_mask

        selected.sort(key=lambda record: (record.coalition_size, record.coalition, record.evader_index))
        return AssignmentResult(assignments=tuple(selected))

    def _select_stage(self, candidates: Iterable[AssignmentCandidate]) -> List[AssignmentCandidate]:
        candidates = sorted(candidates, key=lambda candidate: (candidate.evader_index, candidate.coalition))
        if not candidates:
            return []

        candidates_by_id = {candidate.identifier: candidate for candidate in candidates}
        grouped: Dict[int, List[AssignmentCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.evader_index, []).append(candidate)

        evader_indices = tuple(sorted(grouped))
        source = ("source",)
        sink = ("sink",)
        solver = minCostMaxFlow_implemented.MinCostMaxFlow()

        score_scale = 1_000_000
        if self.use_weights:
            scaled_scores = {
                candidate.identifier: max(0, int(round(float(candidate.score) * score_scale)))
                for candidate in candidates
            }
        else:
            scaled_scores = {candidate.identifier: 0 for candidate in candidates}

        ordered_candidate_ids = tuple(sorted(candidates_by_id))
        candidate_rank_rewards = {
            candidate_id: len(ordered_candidate_ids) - rank
            for rank, candidate_id in enumerate(ordered_candidate_ids)
        }
        selected_slots = len(evader_indices)
        rank_base = len(ordered_candidate_ids) + 1
        position_weights = [
            rank_base ** (selected_slots - 1 - selected_position)
            for selected_position in range(selected_slots)
        ]
        max_tertiary_sum = sum(
            position_weight * len(ordered_candidate_ids)
            for position_weight in position_weights
        )
        tie_unit = max_tertiary_sum + 1
        max_secondary_sum = sum(scaled_scores.values())
        primary_unit = max_secondary_sum * tie_unit + max_tertiary_sum + 1

        solver.add_edge(source, ("layer", 0, 0, 0), capacity=1, cost=0)
        reachable_states = {(0, 0)}
        for evader_position, evader_index in enumerate(evader_indices):
            next_reachable_states = set()
            for used_pursuer_mask, selected_count in sorted(reachable_states):
                current_node = ("layer", evader_position, used_pursuer_mask, selected_count)
                skip_node = ("layer", evader_position + 1, used_pursuer_mask, selected_count)
                solver.add_edge(current_node, skip_node, capacity=1, cost=0)
                next_reachable_states.add((used_pursuer_mask, selected_count))

                for candidate in grouped[evader_index]:
                    if candidate.pursuer_mask & used_pursuer_mask:
                        continue
                    next_mask = used_pursuer_mask | candidate.pursuer_mask
                    next_count = selected_count + 1
                    next_node = ("layer", evader_position + 1, next_mask, next_count)
                    tertiary_reward = (
                        position_weights[selected_count]
                        * candidate_rank_rewards[candidate.identifier]
                    )
                    reward = (
                        primary_unit
                        + scaled_scores[candidate.identifier] * tie_unit
                        + tertiary_reward
                    )
                    solver.add_edge(
                        current_node,
                        next_node,
                        capacity=1,
                        cost=-reward,
                        label=candidate.identifier,
                    )
                    next_reachable_states.add((next_mask, next_count))

            reachable_states = next_reachable_states

        final_layer = len(evader_indices)
        for used_pursuer_mask, selected_count in sorted(reachable_states):
            solver.add_edge(("layer", final_layer, used_pursuer_mask, selected_count), sink, capacity=1, cost=0)

        result = solver.min_cost_flow(source, sink, max_flow=1)
        if result["flow"] != 1:
            return []

        candidate_ids = solver.used_labeled_edges()
        candidate_ids = sorted(set(candidate_ids), key=lambda candidate_id: (candidate_id[1], candidate_id[0]))
        return [candidates_by_id[candidate_id] for candidate_id in candidate_ids]


class WeightedSequentialStrategy(SequentialStrategyBase):
    def __init__(self, name: str):
        super().__init__(name=name, use_weights=True)


class UnweightedSequentialStrategy(SequentialStrategyBase):
    def __init__(self, name: str):
        super().__init__(name=name, use_weights=False)


class RandomSequentialStrategy(AssignmentStrategy):
    def __init__(self, name: str, rng_seed: Optional[int]):
        super().__init__(name=name)
        self.rng = random.Random(rng_seed)

    def select(self, snapshot: AssignmentSnapshot) -> AssignmentResult:
        selected: List[AssignmentRecord] = []
        used_pursuer_mask = 0
        used_evaders = set()

        for coalition_size in (1, 2, 3):
            stage_candidates = [
                candidate
                for candidate in snapshot.candidates_by_size.get(coalition_size, tuple())
                if candidate.feasible
                and candidate.evader_index not in used_evaders
                and candidate.pursuer_mask & used_pursuer_mask == 0
            ]
            self.rng.shuffle(stage_candidates)
            for candidate in stage_candidates:
                if candidate.evader_index in used_evaders:
                    continue
                if candidate.pursuer_mask & used_pursuer_mask:
                    continue
                selected.append(candidate_to_record(candidate))
                used_evaders.add(candidate.evader_index)
                used_pursuer_mask |= candidate.pursuer_mask

        selected.sort(key=lambda record: (record.coalition_size, record.coalition, record.evader_index))
        return AssignmentResult(assignments=tuple(selected))


class NearestSingleStrategy(AssignmentStrategy):
    def __init__(self, name: str):
        super().__init__(name=name)

    def select(self, snapshot: AssignmentSnapshot) -> AssignmentResult:
        selected: List[AssignmentRecord] = []
        used_pursuer_mask = 0
        used_evaders = set()
        single_candidates = list(snapshot.candidates_by_size.get(1, tuple()))
        single_candidates.sort(
            key=lambda candidate: (
                float("inf") if candidate.distance is None else candidate.distance,
                candidate.coalition,
                candidate.evader_index,
            )
        )

        for candidate in single_candidates:
            if candidate.evader_index in used_evaders:
                continue
            if candidate.pursuer_mask & used_pursuer_mask:
                continue
            selected.append(candidate_to_record(candidate))
            used_evaders.add(candidate.evader_index)
            used_pursuer_mask |= candidate.pursuer_mask

        selected.sort(key=lambda record: (record.coalition_size, record.coalition, record.evader_index))
        return AssignmentResult(assignments=tuple(selected))
