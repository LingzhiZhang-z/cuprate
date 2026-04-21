"""Hamiltonian construction and diagonalisation (02-HAMILTONIAN §8-11).

Pure functions — no dependency on HubbardModel.
"""

from __future__ import annotations

import numpy as np

from cuprate.states import count_double_occ, sign_below


# ============================================================
# Matrix element helpers
# ============================================================


def apply_hop(state, src, dst, spin):
    """Apply c†_{dst,σ} c_{src,σ} to *state*.

    Returns (new_state, sign) or (None, 0) if the operator annihilates it.
    """
    offset = 0 if spin == "up" else 1
    src_bit = 2 * src + offset
    dst_bit = 2 * dst + offset

    if not (state >> src_bit) & 1:
        return None, 0
    if (state >> dst_bit) & 1:
        return None, 0

    sign = sign_below(state, src_bit)
    new_state = state ^ (1 << src_bit)
    sign *= sign_below(new_state, dst_bit)
    new_state ^= 1 << dst_bit
    return new_state, sign


# ============================================================
# Hamiltonian construction
# ============================================================

def build_hamiltonian_t(states, bonds, hoppings):
    """Build the hopping matrix H_t on a given basis."""
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    dim = len(states)
    H_t = np.zeros([dim, dim], dtype=complex)
    for col, state in enumerate(states):
        for (site_i, site_j), hopping in zip(bonds, hoppings):
            for spin in ("up", "down"):
                moved_state, sign = apply_hop(state, src=site_j, dst=site_i, spin=spin)
                if moved_state is not None:
                    row = state_to_idx.get(moved_state)
                    if row is not None:
                        H_t[row, col] += -hopping * sign

                moved_state, sign = apply_hop(state, src=site_i, dst=site_j, spin=spin)
                if moved_state is not None:
                    row = state_to_idx.get(moved_state)
                    if row is not None:
                        H_t[row, col] += -hopping.conjugate() * sign
    return H_t


def build_hamiltonian_U(states, N, U):
    """Build the diagonal interaction matrix H_U on a given basis."""
    diag = [U * count_double_occ(state, N) for state in states]
    return np.diag(np.asarray(diag, dtype=complex))


def build_hamiltonian(states, N, U, bonds, hoppings):
    """Build the full Hamiltonian matrix for a set of states."""
    H_t = build_hamiltonian_t(states, bonds, hoppings)
    H_U = build_hamiltonian_U(states, N, U)
    return H_t + H_U


# ============================================================
# Diagonalisation
# ============================================================

def diagonalise(H):
    """Diagonalise a single Hermitian matrix."""
    return np.linalg.eigh(H)


def diagonalise_blocked(hams):
    """Diagonalise a list of Hermitian matrices."""
    eigvals_list = []
    eigvecs_list = []
    for H in hams:
        ev, evec = np.linalg.eigh(H)
        eigvals_list.append(ev)
        eigvecs_list.append(evec)
    return eigvals_list, eigvecs_list
