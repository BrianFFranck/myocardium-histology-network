import math
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from .models import Cell


def estimate_threshold(cells: List[Cell], sample_size: int = 500) -> float:
    if len(cells) < 2:
        return 20.0
    rng = random.Random(42)
    sample = rng.sample(cells, min(sample_size, len(cells)))
    nearest_distances = []
    for cell in sample:
        best = float("inf")
        for other in cells:
            if cell.id == other.id:
                continue
            d = math.hypot(cell.x - other.x, cell.y - other.y)
            if d < best:
                best = d
        nearest_distances.append(best)
    import numpy as np
    mean_nn = np.mean(nearest_distances)
    threshold = mean_nn * 1.8
    threshold = max(12, threshold)
    threshold = min(35, threshold)
    return float(threshold)


class SpatialNetwork:
    def __init__(self, cells: List[Cell]) -> None:
        self.cells = cells
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {c.id: [] for c in cells}

    def build_naive(self, threshold: float) -> float:
        for c in self.cells:
            self.adjacency[c.id] = []
        t0 = time.perf_counter()
        n = len(self.cells)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.cells[i], self.cells[j]
                dist = math.hypot(a.x - b.x, a.y - b.y)
                if dist <= threshold:
                    self.adjacency[a.id].append((b.id, dist))
                    self.adjacency[b.id].append((a.id, dist))
        return time.perf_counter() - t0

    def build_optimized(self, threshold: float) -> float:
        for c in self.cells:
            self.adjacency[c.id] = []
        t0 = time.perf_counter()
        grid: Dict[Tuple[int, int], List[Cell]] = defaultdict(list)
        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)
            grid[(gx, gy)].append(cell)
        temp_neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby = grid.get((gx + dx, gy + dy), [])
                    for other in nearby:
                        if cell.id == other.id:
                            continue
                        dist = math.hypot(cell.x - other.x, cell.y - other.y)
                        local_threshold = threshold
                        if cell.cell_type != other.cell_type:
                            local_threshold *= 0.75
                        if dist <= local_threshold:
                            temp_neighbors[cell.id].append((other.id, dist))
        MAX_NEIGHBORS = 12
        for cid, neighbors in temp_neighbors.items():
            neighbors.sort(key=lambda x: x[1])
            for nid, dist in neighbors[:MAX_NEIGHBORS]:
                self.adjacency[cid].append((nid, dist))
        return time.perf_counter() - t0

    def degree(self, cell_id: str) -> int:
        return len(self.adjacency[cell_id])

    def find_clusters(self) -> List[List[str]]:
        visited: set = set()
        clusters: List[List[str]] = []
        for start_id in self.adjacency:
            if start_id in visited:
                continue
            cluster: List[str] = []
            queue = deque([start_id])
            visited.add(start_id)
            while queue:
                cid = queue.popleft()
                cluster.append(cid)
                for neighbor_id, _ in self.adjacency[cid]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            clusters.append(cluster)
        return sorted(clusters, key=len, reverse=True)

    def find_hubs(self, top_n: int = 5) -> List[Tuple[str, int]]:
        degrees = [(cid, len(neighbors)) for cid, neighbors in self.adjacency.items()]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    def edge_count(self) -> int:
        return sum(len(v) for v in self.adjacency.values()) // 2

    def all_edge_distances(self) -> List[float]:
        """Devolve lista de distâncias de todas as arestas (sem duplicados)."""
        seen: set = set()
        dists: List[float] = []
        for cid, neighbors in self.adjacency.items():
            for nid, dist in neighbors:
                key = (min(cid, nid), max(cid, nid))
                if key not in seen:
                    seen.add(key)
                    dists.append(dist)
        return dists
