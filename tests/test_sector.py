"""Tests for build_S2_sectors and build_S2_transforms.

For each output column, verify that the labeled (twoSz, twoS, D) quantum
numbers match what the Sz / 4S² / D operators report. Also verify column
orthonormality, (twoSz, D) completeness, key consistency between the two
structures, and multiplet counts against the closed-form formula.
"""

from __future__ import annotations

import pathlib
import sys
from math import comb

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate import ATOL
from cuprate.sectors import (
    build_S2_multiplets,
    build_S2_sectors,
    build_S2_transforms,
)
from cuprate.states import (
    calc_S_plus_matrix,
    calc_Sz,
    calc_double_occupation_matrix,
    calc_fourS2_matrix,
    generate_all_states,
    group_states,
    sort_states,
)

TOL = ATOL["loose"]


def _pipeline(N):
    grouped = group_states(generate_all_states(N), N)
    _hw, multiplets = build_S2_multiplets(grouped, N)
    sector_blocks = build_S2_sectors(grouped, multiplets)
    transformers = build_S2_transforms(grouped, N, sector_blocks)
    return grouped, sector_blocks, transformers


def _safe_comb(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def _expected_cols(N, twoSz, twoS, D):
    """Closed-form column count for sector block (twoSz, twoS, D) at N sites."""
    if abs(twoSz) > twoS or (twoSz - twoS) % 2 or not 0 <= D <= N:
        return 0
    return _safe_comb(N, D) * sum(
        _safe_comb(N - D, M)
        * (_safe_comb(M, (M - twoS) // 2) - _safe_comb(M, (M - twoS) // 2 - 1))
        for M in range(twoS, N - D + 1, 2)
    )


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_build_S2_sectors_labels(N):
    """Each sector block column carries the labeled (twoSz, twoS, D); columns orthonormal; (twoSz, D) complete."""
    grouped, sector_blocks, _ = _pipeline(N)
    cols_per_tsz_d = {}

    for block in sector_blocks:
        twoSz, twoS, D = block.twoSz, block.twoS, block.D
        states, coeff = block.basis_states, block.transform
        coeff = np.asarray(coeff)
        Sz = np.diag([calc_Sz(s, N) for s in states])
        fourS2 = calc_fourS2_matrix(states, N)
        Docc = calc_double_occupation_matrix(states, N)

        assert np.allclose(coeff.conj().T @ coeff, np.eye(coeff.shape[1]), atol=TOL)

        for j in range(coeff.shape[1]):
            v = coeff[:, j]
            assert abs(v.conj() @ Sz @ v - 0.5 * twoSz) < TOL
            assert abs(v.conj() @ fourS2 @ v - twoS * (twoS + 2)) < TOL
            assert abs(v.conj() @ Docc @ v - D) < TOL

        cols_per_tsz_d[(twoSz, D)] = cols_per_tsz_d.get((twoSz, D), 0) + coeff.shape[1]

    for (twoSz, D), states in grouped.items():
        assert cols_per_tsz_d.get((twoSz, D), 0) == len(states)


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_build_S2_transforms_labels(N):
    """Each transformer column carries the labeled (twoSz, twoS); columns orthonormal; keys match sector_blocks."""
    grouped, sector_blocks, transformers = _pipeline(N)

    basis = {}
    for (twoSz, _D), st in grouped.items():
        basis.setdefault(twoSz, []).extend(st)
    for twoSz in basis:
        basis[twoSz] = sort_states(basis[twoSz], N)

    sector_keys = {(block.twoSz, block.twoS) for block in sector_blocks}
    assert sector_keys == set(transformers)

    for (twoSz, twoS), U in transformers.items():
        U = np.asarray(U)
        states = basis[twoSz]
        Sz = np.diag([calc_Sz(s, N) for s in states])
        fourS2 = calc_fourS2_matrix(states, N)

        assert np.allclose(U.conj().T @ U, np.eye(U.shape[1]), atol=TOL)

        for j in range(U.shape[1]):
            v = U[:, j]
            assert abs(v.conj() @ Sz @ v - 0.5 * twoSz) < TOL
            assert abs(v.conj() @ fourS2 @ v - twoS * (twoS + 2)) < TOL


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_sector_block_counts_analytic(N):
    """Column counts per (twoSz, twoS, D) match the closed-form multiplet formula."""
    _grouped, sector_blocks, _ = _pipeline(N)
    got = {
        (block.twoSz, block.twoS, block.D): block.transform.shape[1]
        for block in sector_blocks
    }

    for key, cnt in got.items():
        assert cnt == _expected_cols(N, *key), f"{key}: got {cnt}, expected {_expected_cols(N, *key)}"

    for twoSz in range(-N, N + 1):
        for twoS in range(0, N + 1):
            for D in range(0, N + 1):
                if _expected_cols(N, twoSz, twoS, D) > 0:
                    assert (twoSz, twoS, D) in got, f"missing block {(twoSz, twoS, D)}"


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_build_S2_multiplets_hw_is_kernel_of_S_plus(N):
    """hw[(twoSz, D)] columns must be annihilated by S_+ within the (twoSz, D) block."""
    grouped = group_states(generate_all_states(N), N)
    hw, _multiplets = build_S2_multiplets(grouped, N)

    for (twoSz, D), hw_block in hw.items():
        if hw_block.shape[1] == 0:
            continue
        src_states = grouped[(twoSz, D)]
        dst_states = grouped.get((twoSz + 2, D), [])
        S_plus = calc_S_plus_matrix(src_states, dst_states, N)
        residual = S_plus @ hw_block
        assert np.allclose(residual, 0, atol=TOL), (
            f"S_+ did not annihilate hw at (twoSz={twoSz}, D={D}); "
            f"max |residual| = {np.max(np.abs(residual))}"
        )
