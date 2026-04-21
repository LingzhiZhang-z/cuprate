"""Spin-basis operators and least-squares fit of H_eff to spin couplings."""

from __future__ import annotations

from functools import reduce
from operator import matmul

import numpy as np

from cuprate.states import calc_double_occupation_matrix, set_site, site_code

# site_code 1 = ↑ (Sz=+1/2), 2 = ↓ (Sz=-1/2) in the D=0 subspace
_SZ = {1: 0.5, 2: -0.5}


def _spin_pair(states: list[int], i: int, j: int) -> np.ndarray:
    """S_i · S_j on the singly-occupied (D=0) Fock basis `states`."""
    dim = len(states)
    index = {s: k for k, s in enumerate(states)}
    M = np.zeros((dim, dim), dtype=complex)

    for k, s in enumerate(states):
        ci, cj = site_code(s, i), site_code(s, j)
        M[k, k] = _SZ[ci] * _SZ[cj]
        if ci == 2 and cj == 1:  # S+_i S-_j
            M[index[set_site(set_site(s, i, 1), j, 2)], k] = 0.5
        elif ci == 1 and cj == 2:  # S-_i S+_j
            M[index[set_site(set_site(s, i, 2), j, 1)], k] = 0.5
    return M


def spin_matrix(states: list[int], bond) -> np.ndarray:
    """Product of S·S pair operators over consecutive site pairs in `bond`."""
    pairs = zip(bond[::2], bond[1::2])
    return reduce(matmul, (_spin_pair(states, i, j) for i, j in pairs))


def spin_fit(A: np.ndarray, b: np.ndarray):
    """Least-squares fit A x = b. Returns (x, (rel_err, residual, r2))."""
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    r = A @ x - b
    ss_res = np.real(np.dot(r.conj(), r))
    centered = b - np.mean(b)
    ss_tot = np.real(np.dot(centered.conj(), centered))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual = np.linalg.norm(r)
    rel_err = residual / np.linalg.norm(b)
    return x, (rel_err, residual, r2)


def _block_double_occ_expectation(block) -> np.ndarray:
    dom = calc_double_occupation_matrix(block.basis_states, block.N)
    return np.real(np.diag(block.eigvecs.conj().T @ dom @ block.eigvecs))


def _joint_t11_norm(blocks, selected_list) -> float:
    norms_sq = 0.0
    for block, selected in zip(blocks, selected_list):
        spin_rows = block.spin_rows()
        spin_basis = block.spin_basis()
        s_bd = spin_basis @ block.eigvecs[np.ix_(spin_rows, selected)]
        U, Sigma, _ = np.linalg.svd(s_bd, full_matrices=False)
        t11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(block.spin_dim())
        norms_sq += float(np.linalg.norm(t11m1.flatten()) ** 2)
    return float(np.sqrt(norms_sq))


def select_occ(spectrum) -> list[list[int]]:
    selected = []
    for block in spectrum.blocks:
        dimspin = block.spin_dim()
        double_occ = _block_double_occ_expectation(block)
        n = len(block.eigvals)
        order = np.lexsort((np.arange(n), np.real(block.eigvals), np.real(double_occ)))
        selected.append(order[:dimspin].tolist())
    return selected


def select_energy(spectrum) -> list[list[int]]:
    selected = []
    for block in spectrum.blocks:
        dimspin = block.spin_dim()
        double_occ = _block_double_occ_expectation(block)
        energy_order = np.argsort(np.real(block.eigvals), kind="stable")[:dimspin]
        reorder = np.lexsort((
            np.arange(len(energy_order)),
            np.real(block.eigvals[energy_order]),
            np.real(double_occ[energy_order]),
        ))
        selected.append(energy_order[reorder].tolist())
    return selected


def select_greedy(spectrum, ratio: int = 5) -> list[list[int]]:
    raise NotImplementedError("select_greedy: reserved for later implementation")


def select_multi(spectrum, **kwargs) -> list[list[int]]:
    raise NotImplementedError("select_multi: reserved for later implementation")


def select_adiabatic(spectrum, seed) -> list[list[int]]:
    raise NotImplementedError("select_adiabatic: reserved for later implementation")


def select(spectrum, method: str = "occ", **kwargs) -> list[list[int]]:
    methods = {
        "occ": select_occ,
        "energy": select_energy,
        "greedy": select_greedy,
        "multi": select_multi,
        "adiabatic": select_adiabatic,
    }
    if method not in methods:
        raise ValueError(f"Unknown selection method: {method}")
    return methods[method](spectrum, **kwargs)
