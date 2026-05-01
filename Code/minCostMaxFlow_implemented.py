"""
Reusable min-cost max-flow solver based on successive shortest paths.

The implementation supports negative edge costs through an initial
Bellman-Ford pass and then uses Johnson reweighting so the augmenting
path search can run with Dijkstra's algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Dict, List, Optional


@dataclass
class Edge:
    to: int
    rev: int
    capacity: int
    cost: int
    label: Optional[Any] = None
    initial_capacity: int = 0


class MinCostMaxFlow:
    def __init__(self):
        self._node_ids: Dict[Any, int] = {}
        self._node_keys: List[Any] = []
        self._graph: List[List[Edge]] = []

    def add_node(self, key: Any) -> int:
        if key in self._node_ids:
            return self._node_ids[key]
        node_id = len(self._node_keys)
        self._node_ids[key] = node_id
        self._node_keys.append(key)
        self._graph.append([])
        return node_id

    def add_edge(
        self,
        from_key: Any,
        to_key: Any,
        capacity: int,
        cost: int,
        label: Optional[Any] = None,
    ) -> None:
        if capacity < 0:
            raise ValueError("Edge capacity must be non-negative.")

        from_id = self.add_node(from_key)
        to_id = self.add_node(to_key)

        forward = Edge(
            to=to_id,
            rev=len(self._graph[to_id]),
            capacity=capacity,
            cost=cost,
            label=label,
            initial_capacity=capacity,
        )
        backward = Edge(
            to=from_id,
            rev=len(self._graph[from_id]),
            capacity=0,
            cost=-cost,
            label=None,
            initial_capacity=0,
        )

        self._graph[from_id].append(forward)
        self._graph[to_id].append(backward)

    def _require_node(self, key: Any) -> int:
        if key not in self._node_ids:
            raise KeyError(f"Unknown node '{key}'.")
        return self._node_ids[key]

    def _initial_potentials(self, source_id: int) -> List[int]:
        node_count = len(self._graph)
        inf = 10**18
        distance = [inf] * node_count
        distance[source_id] = 0

        for _ in range(node_count - 1):
            updated = False
            for node_id, edges in enumerate(self._graph):
                if distance[node_id] == inf:
                    continue
                for edge in edges:
                    if edge.capacity <= 0:
                        continue
                    candidate_distance = distance[node_id] + edge.cost
                    if candidate_distance < distance[edge.to]:
                        distance[edge.to] = candidate_distance
                        updated = True
            if not updated:
                break

        return [0 if value == inf else value for value in distance]

    def min_cost_flow(self, source_key: Any, sink_key: Any, max_flow: Optional[int] = None) -> Dict[str, int]:
        source_id = self._require_node(source_key)
        sink_id = self._require_node(sink_key)

        if max_flow is None:
            max_flow = sum(edge.capacity for edge in self._graph[source_id])

        if max_flow < 0:
            raise ValueError("max_flow must be non-negative.")

        node_count = len(self._graph)
        inf = 10**18
        potentials = self._initial_potentials(source_id)
        flow = 0
        total_cost = 0

        while flow < max_flow:
            distance = [inf] * node_count
            parent_node = [-1] * node_count
            parent_edge_index = [-1] * node_count
            distance[source_id] = 0
            queue = [(0, source_id)]

            while queue:
                current_distance, node_id = heapq.heappop(queue)
                if current_distance != distance[node_id]:
                    continue

                for edge_index, edge in enumerate(self._graph[node_id]):
                    if edge.capacity <= 0:
                        continue
                    reduced_cost = edge.cost + potentials[node_id] - potentials[edge.to]
                    next_distance = current_distance + reduced_cost
                    if next_distance < distance[edge.to]:
                        distance[edge.to] = next_distance
                        parent_node[edge.to] = node_id
                        parent_edge_index[edge.to] = edge_index
                        heapq.heappush(queue, (next_distance, edge.to))

            if distance[sink_id] == inf:
                break

            for node_id, node_distance in enumerate(distance):
                if node_distance != inf:
                    potentials[node_id] += node_distance

            augment = max_flow - flow
            node_id = sink_id
            while node_id != source_id:
                prev_node_id = parent_node[node_id]
                edge_index = parent_edge_index[node_id]
                if prev_node_id < 0 or edge_index < 0:
                    raise RuntimeError("Broken predecessor chain while augmenting flow.")
                edge = self._graph[prev_node_id][edge_index]
                augment = min(augment, edge.capacity)
                node_id = prev_node_id

            node_id = sink_id
            while node_id != source_id:
                prev_node_id = parent_node[node_id]
                edge_index = parent_edge_index[node_id]
                edge = self._graph[prev_node_id][edge_index]
                reverse_edge = self._graph[edge.to][edge.rev]
                edge.capacity -= augment
                reverse_edge.capacity += augment
                total_cost += edge.cost * augment
                node_id = prev_node_id

            flow += augment

        return {"flow": flow, "cost": total_cost}

    def used_labeled_edges(self) -> List[Any]:
        labels: List[Any] = []
        for edges in self._graph:
            for edge in edges:
                if edge.label is None:
                    continue
                used_capacity = edge.initial_capacity - edge.capacity
                if used_capacity > 0:
                    labels.extend([edge.label] * used_capacity)
        return labels
