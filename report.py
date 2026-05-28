from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Cell
from .network import SpatialNetwork


def print_report(
    cells: List[Cell],
    network: SpatialNetwork,
    clusters: List[List[str]],
    hubs_topn: List[Tuple[str, int]],
    hubs_stat: List[Tuple[str, int]],
    threshold: float,
    time_naive: Optional[float],
    time_optimized: float,
    image_path: Path,
    image_size: Tuple[int, int],
    deg_stats: Dict[str, float],
    cluster_stats: Dict[str, float],
    edge_dist_stats: Dict[str, float],
    type_comparison: Dict[str, Dict[str, float]],
    density_stats: Dict[str, float],
) -> None:
    cell_map = {c.id: c for c in cells}
    type_counts: Dict[str, int] = defaultdict(int)
    for c in cells:
        type_counts[c.cell_type] += 1

    line = "═" * 66
    thin = "─" * 66

    print(f"\n{line}")
    print("  RELATÓRIO DE ANÁLISE HISTOLÓGICA DE MIOCÁRDIO")
    print(f"{line}")
    print(f"  Imagem analisada : {image_path}")
    print(f"  Threshold (px)   : {threshold:.2f}")
    print(f"  Total de células : {len(cells)}")

    # ── 1. Métricas morfológicas por célula ───────────────────────
    print(f"\n{thin}")
    print("  1. MÉTRICAS MORFOLÓGICAS POR CÉLULA")
    print(thin)
    valid = [c for c in cells if c.mean_intensity is not None]
    if valid:
        intensities = [c.mean_intensity for c in valid]
        medians     = [c.median_intensity for c in valid]
        stds        = [c.std_intensity for c in valid]
        mins        = [c.min_intensity for c in valid]
        maxs        = [c.max_intensity for c in valid]
        p25s        = [c.p25_intensity for c in valid]
        p75s        = [c.p75_intensity for c in valid]
        print(f"  Métrica          {'Média':>8} {'Mediana':>8} {'Std':>8} {'Mín':>8} {'Máx':>8} {'P25':>8} {'P75':>8}")
        print(f"  {thin[:64]}")
        print(f"  Intensidade    "
              f"  {np.mean(intensities):7.2f}"
              f"  {np.mean(medians):7.2f}"
              f"  {np.mean(stds):7.2f}"
              f"  {np.mean(mins):7.2f}"
              f"  {np.mean(maxs):7.2f}"
              f"  {np.mean(p25s):7.2f}"
              f"  {np.mean(p75s):7.2f}")

    # ── 2. Área aproximada ────────────────────────────────────────
    print(f"\n{thin}")
    print("  2. ÁREA APROXIMADA")
    print(thin)
    areas = [c.approx_area for c in valid if c.approx_area is not None]
    if areas:
        arr = np.array(areas, dtype=np.float64)
        print(f"  Média    : {np.mean(arr):.1f} px²")
        print(f"  Mediana  : {np.median(arr):.1f} px²")
        print(f"  Std      : {np.std(arr):.1f} px²")
        print(f"  Mín/Máx  : {np.min(arr):.0f} / {np.max(arr):.0f} px²")
        print(f"  P25/P75  : {np.percentile(arr,25):.1f} / {np.percentile(arr,75):.1f} px²")

    # ── 3. Circularidade ──────────────────────────────────────────
    print(f"\n{thin}")
    print("  3. CIRCULARIDADE")
    print(thin)
    circs = [c.circularity for c in valid if c.circularity is not None]
    if circs:
        arr = np.array(circs, dtype=np.float64)
        print(f"  Média    : {np.mean(arr):.4f}  (1.0 = círculo perfeito)")
        print(f"  Mediana  : {np.median(arr):.4f}")
        print(f"  Std      : {np.std(arr):.4f}")
        print(f"  Mín/Máx  : {np.min(arr):.4f} / {np.max(arr):.4f}")

    # ── 4. Contagem por tipo celular ──────────────────────────────
    print(f"\n{thin}")
    print("  4. CONTAGEM POR TIPO CELULAR")
    print(thin)
    for ctype, count in sorted(type_counts.items()):
        pct = count / len(cells) * 100
        bar = "█" * int(pct / 2)
        print(f"  {ctype:<18} {count:>5}  ({pct:5.1f}%)  {bar}")

    # ── 5. Densidade celular ──────────────────────────────────────
    print(f"\n{thin}")
    print("  5. DENSIDADE CELULAR")
    print(thin)
    print(f"  Imagem             : {density_stats['image_width']:.0f} × {density_stats['image_height']:.0f} px")
    print(f"  Área total         : {density_stats['area_Mpx2']:.2f} Mpx²")
    print(f"  Densidade          : {density_stats['density_per_Mpx2']:.1f} células / Mpx²")

    # ── 6. Grau da rede ────────────────────────────────────────────
    print(f"\n{thin}")
    print("  6. GRAU DA REDE")
    print(thin)
    print(f"  Nós              : {len(cells)}")
    print(f"  Arestas          : {network.edge_count()}")
    print(f"  Média            : {deg_stats['mean']:.2f}")
    print(f"  Mediana          : {deg_stats['median']:.2f}")
    print(f"  Std              : {deg_stats['std']:.2f}")
    print(f"  Mín / Máx        : {deg_stats['min']:.0f} / {deg_stats['max']:.0f}")
    print(f"  P25 / P75        : {deg_stats['p25']:.1f} / {deg_stats['p75']:.1f}")

    # ── 7. Hubs ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("  7. HUBS  (grau > média + 2×std)")
    print(thin)
    hub_thresh = deg_stats["mean"] + 2 * deg_stats["std"]
    print(f"  Threshold de hub : {hub_thresh:.2f}")
    print(f"  Hubs encontrados : {len(hubs_stat)}")
    for cid, deg in hubs_stat[:10]:
        c = cell_map[cid]
        print(f"  id={cid:<8} tipo={c.cell_type:<16} grau={deg}")
    if not hubs_stat:
        print("  (nenhuma célula excede o critério)")

    # ── 8. Clusters ────────────────────────────────────────────────
    print(f"\n{thin}")
    print("  8. CLUSTERS (componentes conexas)")
    print(thin)
    cs = cluster_stats
    print(f"  Total            : {cs['n_clusters']}")
    print(f"  Tamanho médio    : {cs['mean_size']:.1f}  (mediana: {cs['median_size']:.1f})")
    print(f"  Std tamanho      : {cs['std_size']:.1f}")
    print(f"  Maior cluster    : {cs['max_size']} células")
    print(f"  Menor cluster    : {cs['min_size']} células")
    print(f"  Isoladas (size=1): {cs['n_isolated']}")
    dist_items = sorted(cs["size_dist"].items())
    buckets_disp = [(s, n) for s, n in dist_items if s <= 10]
    large = sum(n for s, n in dist_items if s > 10)
    disp = "  Distribuição     : " + "  ".join(f"size={s}:{n}" for s, n in buckets_disp)
    if large:
        disp += f"  size>10:{large}"
    print(disp)
    for i, cl in enumerate(clusters[:3], 1):
        comp: Dict[str, int] = defaultdict(int)
        for cid in cl:
            comp[cell_map[cid].cell_type] += 1
        comp_str = ", ".join(f"{k}:{v}" for k, v in sorted(comp.items()))
        print(f"  Cluster #{i}: {len(cl)} células  [{comp_str}]")

    # ── 9. Distância entre células ligadas ────────────────────────
    print(f"\n{thin}")
    print("  9. DISTÂNCIA ENTRE CÉLULAS LIGADAS")
    print(thin)
    ed = edge_dist_stats
    print(f"  Arestas          : {ed['n_edges']}")
    print(f"  Média            : {ed['mean']:.2f} px")
    print(f"  Mediana          : {ed['median']:.2f} px")
    print(f"  Std              : {ed['std']:.2f} px")
    print(f"  Mín / Máx        : {ed['min']:.2f} / {ed['max']:.2f} px")
    print(f"  P25 / P75        : {ed['p25']:.2f} / {ed['p75']:.2f} px")

    # ── 10. Comparação por tipo celular ───────────────────────────
    print(f"\n{thin}")
    print("  10. COMPARAÇÃO POR TIPO CELULAR")
    print(thin)
    hdr = f"  {'Tipo':<18} {'N':>5}  {'Int.média':>9}  {'Área média':>10}  {'Circ. média':>11}  {'Grau médio':>10}"
    print(hdr)
    print(f"  {thin[:64]}")
    for ctype, m in type_comparison.items():
        print(
            f"  {ctype:<18} {m['n']:>5}"
            f"  {m['intensity_mean']:>9.2f}"
            f"  {m['area_mean']:>10.1f}"
            f"  {m['circularity_mean']:>11.4f}"
            f"  {m['degree_mean']:>10.2f}"
        )

    # ── Performance ───────────────────────────────────────────────
    print(f"\n{thin}")
    print("  DESEMPENHO — NAÏVE vs OTIMIZADO")
    print(thin)
    if time_naive is not None:
        speedup = time_naive / time_optimized if time_optimized > 0 else float("inf")
        print(f"  Naïve O(n²)      : {time_naive:.4f} s")
        print(f"  Otimizado (grid) : {time_optimized:.4f} s")
        print(f"  Speedup          : {speedup:.1f}×")
    else:
        print(f"  Otimizado (grid) : {time_optimized:.4f} s")
        print("  (use --compare para incluir versão naïve)")

    # ── Interpretação histopatológica ─────────────────────────────
    print(f"\n{thin}")
    print("  INTERPRETAÇÃO HISTOPATOLÓGICA")
    print(thin)
    infl_pct   = type_counts.get("inflammatory", 0) / len(cells) * 100
    cardio_pct = type_counts.get("cardiomyocyte", 0) / len(cells) * 100
    fibro_pct  = type_counts.get("fibroblast", 0) / len(cells) * 100
    if infl_pct > 20:
        print(f"  ⚠  Inflamatórias: {infl_pct:.1f}% — possível miocardite ativa.")
    elif infl_pct > 10:
        print(f"  ℹ  Inflamatórias: {infl_pct:.1f}% — infiltrado moderado. Monitorizar.")
    else:
        print(f"  ✓  Inflamatórias: {infl_pct:.1f}% — sem evidência de miocardite ativa.")
    if fibro_pct > cardio_pct:
        print("  ⚠  Fibroblastos superam cardiomiócitos — padrão sugestivo de fibrose.")
    print(f"{line}\n")
