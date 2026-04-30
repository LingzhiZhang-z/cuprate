"""Shared test fixtures for the cuprate test suite."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.clusters import Cluster
from cuprate.hubbard import HubbardModel


def make_chain_model(N, U, t, mode="full", twoSz=None, twoS=None, match_spin_sectors=False):
    """Build a HubbardModel for a simple chain of N sites."""
    if match_spin_sectors:
        raise NotImplementedError("match_spin_sectors is not part of the current HubbardModel API")

    sites = tuple((i, 0) for i in range(N))
    bonds = tuple((i, i + 1) for i in range(N - 1))
    cluster = Cluster(sites=sites, bonds=bonds)
    model = HubbardModel(cluster, U, t)
    model.set_symmetry(mode, twoSz=twoSz, twoS=twoS)
    model.build_hamiltonians()
    model.solve()
    return model, getattr(model, "_mode_spec", None)
