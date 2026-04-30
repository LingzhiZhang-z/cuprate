"""Integration: current HubbardModel modes must yield consistent spectra."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.clusters import Cluster
from cuprate.hubbard import HubbardModel


def _make_cluster(N: int, bonds: list[tuple[int, int]]) -> Cluster:
    """Fabricate a Cluster with N sites laid out on a line and explicit bonds.

    The site coordinates are synthetic — only `bonds` drives the Hubbard hopping
    graph in these tests, so the chain placement is a placeholder.
    """
    sites = tuple((i, 0) for i in range(N))
    return Cluster(sites=sites, bonds=tuple(bonds))


def _run_mode(N, U, t, bonds, mode_name, *, twoSz=None, twoS=None, scope="nonnegative"):
    """Solve a HubbardModel mode and return its sorted real spectrum."""
    cluster = _make_cluster(N, bonds)
    result = _solve_model(cluster, U, t, mode_name, twoSz=twoSz, twoS=twoS, scope=scope)
    eigvals = np.concatenate([np.real(block.eigvals) for block in result.blocks])
    return np.sort(eigvals)


def _solve_model(cluster, U, t, mode_name, *, twoSz=None, twoS=None, scope="nonnegative"):
    model = HubbardModel(cluster, U, t)
    model.set_symmetry(mode_name, twoSz=twoSz, twoS=twoS, scope=scope)
    model.build_hamiltonians()
    model.solve()
    return model


@pytest.mark.parametrize("N,bonds", [(2, [(0, 1)]), (3, [(0, 1), (1, 2), (0, 2)])])
def test_sz_pm_matches_full(N, bonds):
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    spec_sz = _run_mode(N, 4.0, 1.0, bonds, "Sz", scope="pm")
    assert np.allclose(spec_full, spec_sz, atol=1e-10)


@pytest.mark.parametrize("N,bonds", [(2, [(0, 1)]), (3, [(0, 1), (1, 2), (0, 2)])])
def test_szs2_pm_matches_full(N, bonds):
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    spec_s2 = _run_mode(N, 4.0, 1.0, bonds, "SzS2", scope="pm")
    assert np.allclose(spec_full, spec_s2, atol=1e-10)


def test_sz_fixed_twoSz_is_subset_of_full():
    N = 2
    bonds = [(0, 1)]
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    spec_sz0 = _run_mode(N, 4.0, 1.0, bonds, "Sz", twoSz=0)
    for ev in spec_sz0:
        assert np.any(np.isclose(spec_full, ev, atol=1e-10)), f"{ev} not in full spectrum"


def test_szs2_fixed_twoS_is_subset_of_full():
    N = 2
    bonds = [(0, 1)]
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    # (twoSz=0, twoS=0) = singlet subspace
    spec_sing = _run_mode(N, 4.0, 1.0, bonds, "SzS2", twoSz=0, twoS=0)
    for ev in spec_sing:
        assert np.any(np.isclose(spec_full, ev, atol=1e-10)), f"{ev} not in full spectrum"


def test_szs2_all_twoS_matches_sz_fixed_twoSz():
    N = 2
    bonds = [(0, 1)]
    spec_sz0 = _run_mode(N, 4.0, 1.0, bonds, "Sz", twoSz=0)
    spec_sz0_s2 = _run_mode(N, 4.0, 1.0, bonds, "SzS2", twoSz=0)
    assert np.allclose(spec_sz0, spec_sz0_s2, atol=1e-10)


def test_block_quantum_numbers_sz_fixed_twoSz():
    cluster = _make_cluster(2, [(0, 1)])
    result = _solve_model(cluster, 4.0, 1.0, "Sz", twoSz=0)
    block = result.blocks[0]

    assert len(block.eigenstate_twoSz()) == len(block.eigvals)
    assert set(block.eigenstate_twoSz()) == {0}
    assert np.all((0.0 <= block.eigenstate_D()) & (block.eigenstate_D() <= 2.0))


def test_block_quantum_numbers_szs2_all_twoS():
    cluster = _make_cluster(2, [(0, 1)])
    result = _solve_model(cluster, 4.0, 1.0, "SzS2", twoSz=0)

    assert {(block.twoSz, block.twoS) for block in result.blocks} == {(0, 0), (0, 2)}

    for block in result.blocks:
        assert len(block.eigenstate_twoSz()) == len(block.eigvals)
        assert set(block.eigenstate_twoSz()) == {block.twoSz}
        assert set(block.eigenstate_twoS()) == {block.twoS}
        assert np.all((0.0 <= block.eigenstate_D()) & (block.eigenstate_D() <= 2.0))
