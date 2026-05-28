import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from myocardium_pipeline import (
    WINDOW_SIZE,
    load_cells,
    generate_image,
    extract_metrics,
    SpatialNetwork,
    estimate_threshold,
    compute_density,
    compute_degree_stats,
    find_hubs_statistical,
    compute_cluster_stats,
    compute_edge_distance_stats,
    compute_type_comparison,
    print_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de análise histológica de miocárdio."
    )
    parser.add_argument("--csv",        required=True,  type=Path)
    parser.add_argument("--image",      required=True,  type=Path)
    parser.add_argument("--threshold",  type=float,     default=50.0)
    parser.add_argument("--compare",    action="store_true")
    parser.add_argument("--top-hubs",   type=int,       default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/5] A carregar células de: {args.csv}")
    cells = load_cells(args.csv)
    if not cells:
        print("ERRO: Nenhuma célula carregada.", file=sys.stderr)
        sys.exit(1)
    print(f"      {len(cells)} células carregadas.")

    print(f"[2/5] A gerar imagem sintética → {args.image}")
    img = generate_image(cells, args.image)
    print(f"      Imagem guardada ({img.size[0]}×{img.size[1]} px).")

    print(f"[3/5] A extrair métricas morfológicas (janela={WINDOW_SIZE}px)...")
    t_metrics = time.perf_counter()
    extract_metrics(cells, img, window=WINDOW_SIZE)
    print(f"      Concluído em {time.perf_counter() - t_metrics:.3f}s.")

    network = SpatialNetwork(cells)
    time_naive: Optional[float] = None
    if args.compare:
        net_naive = SpatialNetwork(cells)
        time_naive = net_naive.build_naive(args.threshold)
        print(f"      Naïve: {time_naive:.4f}s")

    auto_threshold = estimate_threshold(cells)
    threshold = min(args.threshold, auto_threshold)
    print(f"[4/5] A construir rede (otimizada, threshold={threshold:.2f}px)...")
    print(f"      Threshold automático estimado: {auto_threshold:.2f}px")
    time_opt = network.build_optimized(threshold)
    print(f"      Otimizado: {time_opt:.4f}s")

    print("[5/5] A calcular análises...")
    clusters        = network.find_clusters()
    hubs_topn       = network.find_hubs(top_n=args.top_hubs)
    deg_stats       = compute_degree_stats(cells, network)
    hubs_stat       = find_hubs_statistical(cells, network, deg_stats)
    cluster_stats   = compute_cluster_stats(clusters)
    edge_dist_stats = compute_edge_distance_stats(network)
    type_comparison = compute_type_comparison(cells, network)
    density_stats   = compute_density(cells, img.size)

    print_report(
        cells=cells,
        network=network,
        clusters=clusters,
        hubs_topn=hubs_topn,
        hubs_stat=hubs_stat,
        threshold=threshold,
        time_naive=time_naive,
        time_optimized=time_opt,
        image_path=args.image,
        image_size=img.size,
        deg_stats=deg_stats,
        cluster_stats=cluster_stats,
        edge_dist_stats=edge_dist_stats,
        type_comparison=type_comparison,
        density_stats=density_stats,
    )


if __name__ == "__main__":
    main()
