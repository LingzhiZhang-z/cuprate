"""Embedding workflow implementation."""

import os
import sys
import time
from datetime import datetime

import numpy as np

from cuprate.io import (
    EMBED_CLI_SPEC,
    build_run_suffixes,
    embed_root_dir,
    lce_root_dir,
    load_spin_coupling_catalog,
    parse_embed_cli_args,
    print_cli_help,
    write_cluster_points,
)
from .core import EmbeddedCouplings, populate_embedded_couplings
from .reporting import print_grid, write_couplings_embed


def main(params):
    base_dir, run_dir, _ = build_run_suffixes(params)
    lce_root = lce_root_dir(base_dir)
    prefix_output = f"{embed_root_dir(base_dir)}/Ncell{params.Ncell}_Ncut{params.Ncut}{run_dir}"
    print(f"Working directory: {prefix_output}")

    clusters = load_spin_coupling_catalog(lce_root, run_dir, params.Ncut)
    square_cell = [(x, y) for x in range(params.Ncell) for y in range(params.Ncell)]
    nsites = params.Ncell * params.Ncell
    couplings_pbc = EmbeddedCouplings(
        constant=0.0,
        two_site=np.zeros([nsites, nsites], dtype=complex),
    )
    populate_embedded_couplings(clusters, lce_root, run_dir, params.Ncell, couplings_pbc)

    os.makedirs(prefix_output, exist_ok=True)
    with open(f"{prefix_output}/couplings_pbc.txt", "w") as f:
        write_cluster_points(f, square_cell)
        f.write("\n=== Grid ===\n")
        f.write("\n" + print_grid(params.Ncell) + "\n")
        write_couplings_embed(f, couplings_pbc, square_cell)


def run_cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_cli_help(EMBED_CLI_SPEC)
        sys.exit(0)

    params = parse_embed_cli_args()
    t0 = time.time()
    print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main(params)
    t1 = time.time()
    print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task is finished in {t1 - t0} seconds")


if __name__ == "__main__":
    run_cli()
