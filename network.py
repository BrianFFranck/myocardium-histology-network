import math
import random
import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from models import Cell

def estimate_threshold(cells: List[Cell], sample_size: int = 500) -> float:
    if len(cells) < 2:
        return 20.0
    rng = random.Random(42)
    sample = rng.sample(cells, min(sample_size, len(cells)))

    all_xy = np.array([[c.x, c.y] for c in cells], dtype=np.float32)
    nearest_distances = []

    for cell in sample:
        diff = all_xy - np.array([cell.x, cell.y], dtype=np.float32)
        dists = np.hypot(diff[:, 0], diff[:, 1])
        dists[dists == 0] = np.inf
        nearest_distances.append(float(np.min(dists)))

    mean_nn = np.mean(nearest_distances)
    return float(np.clip(mean_nn * 1.8, 12, 35))

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
         
        seen_edges = set()
        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby = grid.get((gx + dx, gy + dy), [])
                    for other in nearby:
                        if cell.id == other.id:
                            continue
                        edge_key = tuple(sorted((cell.id, other.id)))

                        if edge_key not in seen_edges:
                            dist = math.hypot(cell.x - other.x, cell.y - other.y)
                            local_threshold = threshold
                            if cell.cell_type != other.cell_type:
                                local_threshold *= 0.75

                            if dist < local_threshold:
                                self.adjacency[cell.id].append((other.id, dist))
                                self.adjacency[other.id].append((cell.id, dist))
                                seen_edges.add(edge_key)

        for cid in self.adjacency:
            self.adjacency[cid] = tuple(self.adjacency[cid])
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
        seen: set = set()
        dists: List[float] = []
        for cid, neighbors in self.adjacency.items():
            for nid, dist in neighbors:
                key = (min(cid, nid), max(cid, nid))
                if key not in seen:
                    seen.add(key)
                    dists.append(dist)
        return dists
