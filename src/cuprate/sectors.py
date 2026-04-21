"""Symmetry-sector construction from S+/S- in fixed-(twoSz, D) blocks.

Core data flow:
  1. build states
  2. group by (twoSz, D)
  3. build highest-weight blocks from ker(S+)
  4. lower each highest-weight block with S-
"""

from __future__ import annotations

import numpy as np

from cuprate import ATOL
from cuprate.states import (
    calc_S_minus_matrix,
    calc_S_plus_matrix,
    sort_states,
)


# ── Small helpers ───────────────────────────────────────────


def _null_space(matrix):
    """Orthonormal basis for ker(matrix) via SVD."""
    n = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(n, dtype=complex)
    _u, sv, vh = np.linalg.svd(matrix, full_matrices=True)
    if len(sv) == 0:
        return np.eye(n, dtype=matrix.dtype)
    null_idx = list(np.where(sv <= ATOL["loose"])[0])
    null_idx.extend(range(len(sv), n))
    if not null_idx:
        return np.zeros((n, 0), dtype=matrix.dtype)
    return vh[null_idx, :].conj().T


def _lowering_coeff(twoS, twoSz):
    return 0.5 * np.sqrt((twoS + twoSz) * (twoS - twoSz + 2))


# ── Core: group -> highest weight -> lower with S- ──────────
def _build_highest_weight(grouped_states, N):
    """For each non-negative twoS and each D, compute ker(S+) = highest-weight space.

    Returns dict[(twoS, D)] -> ndarray of shape (block_dim, n_hw).
    """
    hw = {}
    for (twoSz, D), src_states in grouped_states.items():
        dst_states = grouped_states.get((twoSz + 2, D), [])
        hw[(twoSz, D)] = _null_space(calc_S_plus_matrix(src_states, dst_states, N))
    return hw


def _generate_spin_multiplet_blocks(grouped_states, hw, twoS, D, N):
    """Generate all accessible twoSz blocks in one fixed-(twoS, D) multiplet.

    Here hw is the highest-weight block in the fixed-(twoSz=twoS, D) basis, i.e.
    the columns representing |S, Sz=S>. Repeated S_- then generates the remaining
    |S, Sz> blocks with the same fixed D, stopping at twoSz = -twoS.
    """
    multiplet_blocks = {twoS: hw}
    current = hw
    current_twoSz = twoS
    while current_twoSz > -twoS and (current_twoSz - 2, D) in grouped_states:
        # Apply S_- : H_(twoSz=current_twoSz,D) -> H_(twoSz=current_twoSz-2,D).
        # The raw lowered vector carries the standard SU(2) factor
        # sqrt(S(S+1) - Sz(Sz-1)), so divide it out to recover the normalized
        # |S, Sz-1> block. In integer labels this coefficient is
        # 0.5 * sqrt((twoS + current_twoSz) * (twoS - current_twoSz + 2)).
        current = (
            calc_S_minus_matrix(
                grouped_states[(current_twoSz, D)],
                grouped_states[(current_twoSz - 2, D)],
                N,
            )
            @ current
            / _lowering_coeff(twoS, current_twoSz)
        )
        current_twoSz -= 2
        multiplet_blocks[current_twoSz] = current
    return multiplet_blocks


def build_S2_multiplets(grouped_states, N):
    """Build the simple sector data: highest-weight blocks and their S_- descendants.

    Input:
      grouped_states[(twoSz, D)] = states in one fixed-(twoSz, D) block.

    Returns:
      hw[(twoS, D)] = orthonormal highest-weight columns in the (twoSz=twoS, D) basis
      multiplets[(twoS, D)][twoSz] = the same multiplet expressed in the
      fixed-(twoSz, D) basis after repeated S_- lowering
    """
    hw = _build_highest_weight(grouped_states, N)
    multiplets = {}
    for (twoSz, D), hw_block in hw.items():
        if hw_block.shape[1] == 0:
            continue
        multiplets[(twoSz, D)] = _generate_spin_multiplet_blocks(
            grouped_states,
            hw_block,
            twoSz,
            D,
            N,
        )
    return hw, multiplets


def build_S2_sectors(grouped_states, multiplets):
    """Collect all lowered blocks and sort them by (twoSz, twoS, D).

    Each item contains:
      twoSz, twoS, D, states_block, coeff_block
    """
    sector_blocks = []
    for (twoS, D), multiplet_blocks in multiplets.items():
        for twoSz, coeff_block in multiplet_blocks.items():
            sector_blocks.append(
                (twoSz, twoS, D, grouped_states[(twoSz, D)], coeff_block)
            )
    sector_blocks.sort(key=lambda item: (item[0], item[1], item[2]))
    return sector_blocks


def build_S2_transforms(grouped_states, N, sector_blocks):
    """Build one transform matrix U for each (twoSz, twoS) sector.

    Input:
      grouped_states[(twoSz, D)] = fixed-(twoSz, D) basis states
      sector_blocks sorted by (twoSz, twoS, D), from build_S2_sectors

    Returns dict[(twoSz, twoS)] -> U in the fixed-twoSz Fock basis.
    """
    basis = {}
    basis_map = {}
    for (twoSz, D), states in grouped_states.items():
        basis.setdefault(twoSz, []).extend(states)

    for twoSz in basis:
        basis[twoSz] = sort_states(basis[twoSz], N)
        basis_map[twoSz] = {state: idx for idx, state in enumerate(basis[twoSz])}

    blocks_by_sector = {}
    for twoSz, twoS, D, states_block, coeff_block in sector_blocks:
        blocks_by_sector.setdefault((twoSz, twoS), []).append(
            (states_block, np.asarray(coeff_block))
        )

    transformers = {}
    for (twoSz, twoS), blocks in blocks_by_sector.items():
        n_rows = len(basis[twoSz])
        n_cols = sum(coeff_block.shape[1] for _states_block, coeff_block in blocks)
        U = np.zeros((n_rows, n_cols), dtype=complex)
        col_start = 0
        for states_block, coeff_block in blocks:
            indices = [basis_map[twoSz][state] for state in states_block]
            col_stop = col_start + coeff_block.shape[1]
            U[np.ix_(indices, range(col_start, col_stop))] = coeff_block
            col_start = col_stop
        transformers[(twoSz, twoS)] = U

    return transformers


def _format_complex(value):
    real = float(np.real(value))
    imag = float(np.imag(value))
    sign = "+" if imag >= 0 else "-"
    return f"{real:.16g}{sign}{abs(imag):.16g}j"


def write_S2_blocks(filename, sector_blocks, N):
    """Write all sorted (twoSz, twoS, D) blocks to one text file.

    For each block, write
      line 1: twoSz twoS D nstate nvec
      line 2: state_1 ... state_nstate as fixed-width 2N-bit strings
      next nvec lines: coeff_block[:, j] for j = 0 .. nvec-1
    """
    width = 2 * N
    with open(filename, "w", encoding="ascii") as f:
        for twoSz, twoS, D, states_block, coeff_block in sector_blocks:
            coeff_block = np.asarray(coeff_block)
            nstate = len(states_block)
            nvec = coeff_block.shape[1]
            f.write(f"{twoSz} {twoS} {D} {nstate} {nvec}\n")
            f.write(" ".join(format(int(state), f"0{width}b") for state in states_block) + "\n")
            for j in range(nvec):
                f.write(
                    " ".join(_format_complex(value) for value in coeff_block[:, j])
                    + "\n"
                )


def load_S2_blocks(filename):
    """Read sorted (twoSz, twoS, D) blocks from the text format written above."""
    sector_blocks = []
    with open(filename, "r", encoding="ascii") as f:
        while True:
            header = f.readline()
            if header == "":
                break
            twoSz, twoS, D, nstate, nvec = map(int, header.split())
            states_block = [int(bits, 2) for bits in f.readline().split()]
            coeff_block = np.empty((nstate, nvec), dtype=complex)
            for j in range(nvec):
                coeff_block[:, j] = np.asarray(
                    [complex(value) for value in f.readline().split()],
                    dtype=complex,
                )
            sector_blocks.append((twoSz, twoS, D, states_block, coeff_block))
    return sector_blocks


