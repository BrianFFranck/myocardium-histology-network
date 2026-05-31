import numpy as None
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from models import Cell
from network import SpatialNetwork

def compute_density(cells: List[Cell], image_size: Tuple[int, int]) -> Dict[str, float]:
    W, H = image_size
    area_px2 = W * H
    density = len(cells) / (area_px2 / 1_000_000)
    return {
        "n_cells": len(cells), "image_width": W, "image_height": H,
        "area_Mpx2": area_px2 / 1_000_000, "density_per_Mpx2": density,
    }

def compute_degree_stats(cells: List[Cell], network: SpatialNetwork) -> Dict[str, float]:
    degrees = [network.degree(c.id) for c in cells]
    arr = np.array(degrees, dtype=np.float64)
    p25_d, p75_d = np.percentile(arr, [25, 75])
    return {
        "mean": float(np.mean(arr)), "median": float(np.median(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)), "p25": float(p25_d), "p75": float(p75_d),
    }

def find_hubs_statistical(cells: List[Cell], network: SpatialNetwork, deg_stats: Dict[str, float]) -> List[Tuple[str, int]]:
    threshold = deg_stats["mean"] + 2 * deg_stats["std"]
    hubs = [(c.id, network.degree(c.id)) for c in cells if network.degree(c.id) > threshold]
    return sorted(hubs, key=lambda x: x[1], reverse=True)

def compute_cluster_stats(clusters: List[List[str]]) -> Dict[str, float]:
    sizes = [len(cl) for cl in clusters]
    arr = np.array(sizes, dtype=np.float64)
    size_dist: Dict[int, int] = defaultdict(int)
    for s in sizes:
        size_dist[s] += 1
    return {
        "n_clusters": len(clusters), "mean_size": float(np.mean(arr)), "median_size": float(np.median(arr)),
        "std_size": float(np.std(arr)), "max_size": int(np.max(arr)), "n_isolated": int(size_dist.get(1, 0)),
    }

def compute_edge_distance_stats(network: SpatialNetwork) -> Dict[str, float]:
    dists = network.all_edge_distances()
    if not dists:
        return {"n_edges": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0}
    arr = np.array(dists, dtype=np.float64)
    p25_e, p75_e = np.percentile(arr, [25, 75])
    return {
        "n_edges": len(dists), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "std": float(np.std(arr)),
        "p25": float(p25_e), "p75": float(p75_e),
    }

def compute_type_comparison(cells: List[Cell], network: SpatialNetwork) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Cell]] = defaultdict(list)
    for c in cells:
        buckets[c.cell_type].append(c)
    result: Dict[str, Dict[str, float]] = {}
    for ctype, group in sorted(buckets.items()):
        intensities  = [c.mean_intensity for c in group if c.mean_intensity is not None]
        areas        = [c.approx_area for c in group if c.approx_area is not None]
        circularities= [c.circularity for c in group if c.circularity is not None]
        degrees      = [network.degree(c.id) for c in group]

        def _stats(lst: list) -> Tuple[float, float]:
            a = np.array(lst, dtype=np.float64)
            return (float(np.mean(a)) if len(a) else 0.0, float(np.std(a)) if len(a) else 0.0)

        i_m, i_s = _stats(intensities)
        a_m, a_s = _stats(areas)
        c_m, c_s = _stats(circularities)
        d_m, d_s = _stats(degrees)
        result[ctype] = {
            "n": len(group), "intensity_mean": i_m, "intensity_std": i_s,
            "area_mean": a_m, "area_std": a_s, "circularity_mean": c_m, "circularity_std": c_s,
            "degree_mean": d_m, "degree_std": d_s,
        }
    return result

def estimate_memory_saving(n_cells: int) -> Dict[str, float]:
    overhead_per_cell = 232
    saved_bytes = n_cells * overhead_per_cell
    return {"n_cells": n_cells, "overhead_per_cell": overhead_per_cell, "saved_bytes": saved_bytes, "saved_mb": saved_bytes / (1024 ** 2)}

def print_report(cells: List[Cell], network: SpatialNetwork, clusters: List[List[str]], hubs_stat: List[Tuple[str, int]], threshold: float, time_naive: Optional[float], time_optimized: float, image_path: Path, deg_stats: Dict[str, float], cluster_stats: Dict[str, float], edge_dist_stats: Dict[str, float], type_comparison: Dict[str, Dict[str, float]], density_stats: Dict[str, float]) -> None:
    cell_map = {c.id: c for c in cells}
    type_counts: Dict[str, int] = defaultdict(int)
    for c in cells:
        type_counts[c.cell_type] += 1

    line, thin = "═" * 66, "─" * 66
    print(f"\n{line}\n  RELATÓRIO DE ANÁLISE HISTOLÓGICA DE MIOCÁRDIO\n{line}")
    print(f"  Imagem analisada : {image_path}\n  Threshold (px)   : {threshold:.2f}\n  Total de células : {len(cells)}")

    valid = [c for c in cells if c.mean_intensity is not None]
    if valid:
        print(f"\n{thin}\n   MÉTRICAS MORFOLÓGICAS POR CÉLULA\n{thin}")
        intensities = [c.mean_intensity for c in valid]
        print(f"  Métrica          {'Média':>8}\n  {thin[:64]}\n  Intensidade      {np.mean(intensities):7.2f}")

    print(f"\n{thin}\n   CONTAGEM POR TIPO CELULAR\n{thin}")
    for ctype, count in sorted(type_counts.items()):
        print(f"  {ctype:<18}: {count:>5}  ({count / len(cells) * 100:5.1f}%)")

    print(f"\n{thin}\n   DENSIDADE CELULAR\n{thin}")
    print(f"  Densidade  : {density_stats['density_per_Mpx2']:.1f} células / Mpx²")

    print(f"\n{thin}\n   GRAU DA REDE\n{thin}")
    print(f"  Nós      : {len(cells)}\n  Arestas  : {network.edge_count()}\n  Média    : {deg_stats['mean']:.2f}")

    print(f"\n{thin}\n   HUBS  (grau > média + 2×std)\n{thin}")
    for cid, deg in hubs_stat[:10]:
        print(f"  id={cid:<8} tipo={cell_map[cid].cell_type:<16} grau={deg}")

    print(f"\n{thin}\n   CLUSTERS \n{thin}")
    print(f"  Total            : {cluster_stats['n_clusters']}\n  Maior cluster    : {cluster_stats['max_size']} células")

    print(f"\n{thin}\n   COMPARAÇÃO POR TIPO CELULAR\n{thin}")
    for ctype, m in type_comparison.items():
        print(f"  {ctype:<18} {m['n']:>5}  {m['intensity_mean']:>9.2f}  {m['degree_mean']:>10.2f}")

    print(f"\n{thin}\n  DESEMPENHO — O(n²) vs OTIMIZADO\n{thin}")
    if time_naive is not None:
        print(f"  O(n²) : {time_naive:.4f} s\n  Otimizado (grid) : {time_optimized:.4f} s\n  Speedup : {time_naive / time_optimized:.1f}×")

    print(f"\n{thin}\n  INTERPRETAÇÃO HISTOPATOLÓGICA\n{thin}")
    infl_pct = type_counts.get("inflammatory", 0) / len(cells) * 100
    if infl_pct > 20: print(f"  ⚠  Inflamatórias: {infl_pct:.1f}% — miocardite ativa.")
    else: print(f"  ✓  Inflamatórias: {infl_pct:.1f}% — sem evidência de miocardite ativa.")
