"""Hamiltonian construction and diagonalisation (02-HAMILTONIAN §8-11).

Pure functions — no dependency on HubbardModel.
"""

from __future__ import annotations

import numpy as np

from cuprate.sectors import canonicalize_eigenpairs
from cuprate.states import sign_state


# ============================================================
# Matrix element helpers
# ============================================================

def hopping_element(state, site, spin_type, N):
    """Annihilate a spin from *site* and return (sign, modified_occupation).

    spin_type: 'up' or 'down'
    Returns (0.0, original_occ) if no such spin is present.
    """
    occ = state[site]
    if spin_type == 'up':
        if occ == 2:
            return sign_state(state, site), -1
        elif occ == 1:
            return sign_state(state, site), 0
        else:
            return 0.0, occ
    else:  # down
        if occ == 2:
            return sign_state(state, site) * (-1), 1
        elif occ == -1:
            return sign_state(state, site), 0
        else:
            return 0.0, occ


def calc_ham_t_ij(state1_ref, state2_ref, N, bond_map, hoppings):
    """Hopping matrix element <state1|H_t|state2>."""
    state1 = list(state1_ref)
    state2 = list(state2_ref)

    res = 0.0 + 0.0j
    for m in range(N):
        for n in range(N):
            index = bond_map.get((m, n) if m <= n else (n, m))
            if index is None:
                continue
            t = hoppings[index] if m < n else hoppings[index].conjugate()

            for spin_type in ('up', 'down'):
                occ_m = state1[m]
                occ_n = state2[n]
                sign_m, new_m = hopping_element(state1, m, spin_type, N)
                sign_n, new_n = hopping_element(state2, n, spin_type, N)
                state1[m] = new_m
                state2[n] = new_n

                if sign_m != 0.0 and sign_n != 0.0 and state1 == state2:
                    res -= sign_m * sign_n * t

                state1[m] = occ_m
                state2[n] = occ_n

    return res


def calc_ham_U_ij(state1_ref, state2_ref, N, U):
    """On-site repulsion matrix element <state1|H_U|state2>."""
    if state1_ref != state2_ref:
        return 0j
    return complex(sum(U for m in range(N) if state2_ref[m] == 2))


def calc_ham_ij(state1, state2, N, U, bond_map, hoppings):
    return calc_ham_t_ij(state1, state2, N, bond_map, hoppings) + \
           calc_ham_U_ij(state1, state2, N, U)


# ============================================================
# Hamiltonian construction
# ============================================================

def build_hamiltonian(states, N, U, bonds, hoppings):
    """Build the full Hamiltonian matrix for a set of states."""
    bond_map = {tuple(bond): i for i, bond in enumerate(bonds)}
    dim = len(states)
    H = np.zeros([dim, dim], dtype=complex)
    for i in range(dim):
        for j in range(i, dim):
            val = calc_ham_ij(states[i], states[j], N, U, bond_map, hoppings)
            H[i, j] = val
            H[j, i] = val.conjugate()
    return H


def build_hamiltonian_blocked(sz_sectors_states, N, U, bonds, hoppings):
    """Build per-Sz-sector Hamiltonian matrices."""
    hams = []
    for states in sz_sectors_states:
        hams.append(build_hamiltonian(states, N, U, bonds, hoppings))
    return hams


def build_hamiltonian_S2(sz_hams, s2_sector_list, s2_transforms):
    """Transform Sz-blocked Hamiltonians to the (Sz, S²) basis."""
    hams = [None] * len(s2_sector_list)
    for idx, (twoSz, twoS, idx_sz) in enumerate(s2_sector_list):
        U = s2_transforms[idx]
        hams[idx] = U.conj().T @ sz_hams[idx_sz] @ U
    return hams


# ============================================================
# Diagonalisation
# ============================================================

def diagonalise(H):
    """Diagonalise a single Hermitian matrix."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return canonicalize_eigenpairs(eigvals, eigvecs)


def diagonalise_blocked(hams):
    """Diagonalise a list of Hermitian matrices."""
    eigvals_list = []
    eigvecs_list = []
    for H in hams:
        ev, evec = np.linalg.eigh(H)
        ev, evec = canonicalize_eigenpairs(ev, evec)
        eigvals_list.append(ev)
        eigvecs_list.append(evec)
    return eigvals_list, eigvecs_list
