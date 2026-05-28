from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .models import Cell
from .network import SpatialNetwork


def compute_density(cells: List[Cell], image_size: Tuple[int, int]) -> Dict[str, float]:
    """Densidade = número de células / área total da imagem (em Mpx²)."""
    W, H = image_size
    area_px2 = W * H
    density = len(cells) / (area_px2 / 1_000_000)
    return {
        "n_cells":            len(cells),
        "image_width":        W,
        "image_height":       H,
        "area_Mpx2":          area_px2 / 1_000_000,
        "density_per_Mpx2":   density,
    }


def compute_degree_stats(cells: List[Cell], network: SpatialNetwork) -> Dict[str, float]:
    degrees = [network.degree(c.id) for c in cells]
    arr = np.array(degrees, dtype=np.float64)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


def find_hubs_statistical(
    cells: List[Cell],
    network: SpatialNetwork,
    deg_stats: Dict[str, float],
) -> List[Tuple[str, int]]:
    """Retorna células com grau > média + 2×std."""
    threshold = deg_stats["mean"] + 2 * deg_stats["std"]
    hubs = [
        (c.id, network.degree(c.id))
        for c in cells
        if network.degree(c.id) > threshold
    ]
    return sorted(hubs, key=lambda x: x[1], reverse=True)


def compute_cluster_stats(clusters: List[List[str]]) -> Dict[str, float]:
    sizes = [len(cl) for cl in clusters]
    arr = np.array(sizes, dtype=np.float64)
    size_dist: Dict[int, int] = defaultdict(int)
    for s in sizes:
        size_dist[s] += 1
    return {
        "n_clusters":   len(clusters),
        "mean_size":    float(np.mean(arr)),
        "median_size":  float(np.median(arr)),
        "std_size":     float(np.std(arr)),
        "max_size":     int(np.max(arr)),
        "min_size":     int(np.min(arr)),
        "n_isolated":   int(size_dist.get(1, 0)),
        "size_dist":    dict(sorted(size_dist.items())),
    }


def compute_edge_distance_stats(network: SpatialNetwork) -> Dict[str, float]:
    dists = network.all_edge_distances()
    if not dists:
        return {"n_edges": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0}
    arr = np.array(dists, dtype=np.float64)
    return {
        "n_edges": len(dists),
        "mean":    float(np.mean(arr)),
        "median":  float(np.median(arr)),
        "std":     float(np.std(arr)),
        "min":     float(np.min(arr)),
        "max":     float(np.max(arr)),
        "p25":     float(np.percentile(arr, 25)),
        "p75":     float(np.percentile(arr, 75)),
    }


def compute_type_comparison(
    cells: List[Cell],
    network: SpatialNetwork,
) -> Dict[str, Dict[str, float]]:
    """Para cada tipo celular calcula: mean/std de intensity, area, circularity e grau da rede."""
    buckets: Dict[str, List[Cell]] = defaultdict(list)
    for c in cells:
        buckets[c.cell_type].append(c)

    def _stats(lst: list) -> Tuple[float, float]:
        a = np.array(lst, dtype=np.float64)
        return (float(np.mean(a)) if len(a) else 0.0,
                float(np.std(a))  if len(a) else 0.0)

    result: Dict[str, Dict[str, float]] = {}
    for ctype, group in sorted(buckets.items()):
        intensities   = [c.mean_intensity for c in group if c.mean_intensity  is not None]
        areas         = [c.approx_area    for c in group if c.approx_area     is not None]
        circularities = [c.circularity    for c in group if c.circularity     is not None]
        degrees       = [network.degree(c.id) for c in group]

        i_m, i_s = _stats(intensities)
        a_m, a_s = _stats(areas)
        c_m, c_s = _stats(circularities)
        d_m, d_s = _stats(degrees)

        result[ctype] = {
            "n":                len(group),
            "intensity_mean":   i_m, "intensity_std":  i_s,
            "area_mean":        a_m, "area_std":        a_s,
            "circularity_mean": c_m, "circularity_std": c_s,
            "degree_mean":      d_m, "degree_std":      d_s,
        }
    return result
