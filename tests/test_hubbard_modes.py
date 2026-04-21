"""Integration: all HubbardModel modes must yield the same half-filled spectrum.

Modes that block-diagonalize (and optionally reconstruct) must produce the
exact same eigenvalue set as the single-matrix full diagonalization.
"""

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


def _run_mode(N, U, t, bonds, mode_name, *, twoSz=None, twoS=None):
    """Solve a HubbardModel mode and return its sorted real spectrum."""
    cluster = _make_cluster(N, bonds)
    result = HubbardModel(cluster, U, t).solve(mode_name, twoSz=twoSz, twoS=twoS)
    eigvals = np.concatenate([np.real(block.eigvals) for block in result.blocks])
    return np.sort(eigvals)


@pytest.mark.parametrize("N,bonds", [(2, [(0, 1)]), (3, [(0, 1), (1, 2), (0, 2)])])
def test_block_sz_full_matches_full(N, bonds):
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    spec_sz = _run_mode(N, 4.0, 1.0, bonds, "block_sz_full")
    assert np.allclose(spec_full, spec_sz, atol=1e-10)


@pytest.mark.parametrize("N,bonds", [(2, [(0, 1)]), (3, [(0, 1), (1, 2), (0, 2)])])
def test_block_sz_s2_full_matches_full(N, bonds):
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    spec_s2 = _run_mode(N, 4.0, 1.0, bonds, "block_sz_s2_full")
    assert np.allclose(spec_full, spec_s2, atol=1e-10)


def test_fixed_sz_is_subset_of_full():
    N = 2
    bonds = [(0, 1)]
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    spec_sz0 = _run_mode(N, 4.0, 1.0, bonds, "fixed_sz", twoSz=0)
    for ev in spec_sz0:
        assert np.any(np.isclose(spec_full, ev, atol=1e-10)), f"{ev} not in full spectrum"


def test_fixed_sz_s2_is_subset_of_full():
    N = 2
    bonds = [(0, 1)]
    spec_full = _run_mode(N, 4.0, 1.0, bonds, "full")
    # (twoSz=0, twoS=0) = singlet subspace
    spec_sing = _run_mode(N, 4.0, 1.0, bonds, "fixed_sz_s2", twoSz=0, twoS=0)
    for ev in spec_sing:
        assert np.any(np.isclose(spec_full, ev, atol=1e-10)), f"{ev} not in full spectrum"


def test_fixed_sz_s2_all_matches_fixed_sz():
    N = 2
    bonds = [(0, 1)]
    spec_sz0 = _run_mode(N, 4.0, 1.0, bonds, "fixed_sz", twoSz=0)
    spec_sz0_s2 = _run_mode(N, 4.0, 1.0, bonds, "fixed_sz_s2_all", twoSz=0)
    assert np.allclose(spec_sz0, spec_sz0_s2, atol=1e-10)


def test_block_quantum_numbers_fixed_sz():
    cluster = _make_cluster(2, [(0, 1)])
    result = HubbardModel(cluster, 4.0, 1.0).solve("fixed_sz", twoSz=0)
    block = result.blocks[0]
    qnums = block.quantum_numbers()

    assert len(qnums) == len(block.eigvals)
    assert {entry["twoSz"] for entry in qnums} == {0}
    assert all(0.0 <= entry["D"] <= 2.0 for entry in qnums)


def test_block_quantum_numbers_fixed_sz_s2_all():
    cluster = _make_cluster(2, [(0, 1)])
    result = HubbardModel(cluster, 4.0, 1.0).solve("fixed_sz_s2_all", twoSz=0)

    assert {(block.twoSz, block.twoS) for block in result.blocks} == {(0, 0), (0, 2)}

    for block in result.blocks:
        qnums = block.quantum_numbers()
        assert len(qnums) == len(block.eigvals)
        assert {entry["twoSz"] for entry in qnums} == {block.twoSz}
        assert {entry["twoS"] for entry in qnums} == {block.twoS}
        assert all(0.0 <= entry["D"] <= 2.0 for entry in qnums)
