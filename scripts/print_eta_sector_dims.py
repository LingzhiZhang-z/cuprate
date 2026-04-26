from __future__ import annotations

from cuprate.clusters import ClusterSets
from cuprate.hubbard import HubbardModel


def _sector_dims(cluster, mode: str) -> str:
    model = HubbardModel(cluster, U=3.0, t=1.0)
    model.set_symmetry(mode)
    entries = []
    for block in model.blocks:
        dim = block.basis_transform.shape[1] if block.basis_transform is not None else len(block.basis_states)
        eta = "" if block.eta is None else f",eta={block.eta}"
        entries.append(f"({block.twoSz},{block.twoS}{eta}):{dim}/{block.spin_dim}")
    return ", ".join(entries)


def main() -> None:
    for N in range(2, 7):
        print(f"N={N}")
        for cluster in ClusterSets(N).generate().clusters:
            szs2 = _sector_dims(cluster, "SzS2")
            eta0 = _sector_dims(cluster, "SzS2eta2")
            print(f"  {cluster.label()}")
            print(f"    SzS2     : {szs2}")
            print(f"    SzS2eta0 : {eta0}")


if __name__ == "__main__":
    main()
