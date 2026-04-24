"""Downfolding to effective spin Hamiltonian (04-DOWNFOLDING).

Eigenstate selection, SVD projection, H_eff construction,
spin-operator basis, and least-squares fitting.
Incorporates the former selection.py and spin-coupling parts of spin.py.
"""

from __future__ import annotations

import os
import time

import numpy as np

from cuprate import ATOL
from cuprate.back.io import Params, block_root_dir, build_run_dirnames
from cuprate.mpi import rank
from cuprate.sectors import INVALID_OCCUPATION
from cuprate.states import calc_double_occupation_matrix


# ============================================================
# Spin-operator primitives (formerly in spin.py)
# ============================================================


def Spin_z(occupation):
    if occupation == 2 or occupation == 0:
        return 0
    elif occupation == 1:
        return 1 / 2
    elif occupation == -1:
        return -1 / 2


def Spin_Flip_Plus(occupation):
    if occupation == -1:
        return 1
    elif occupation in (1, 2, 0):
        return INVALID_OCCUPATION


def Spin_Flip_Minus(occupation):
    if occupation == 1:
        return -1
    elif occupation in (-1, 2, 0):
        return INVALID_OCCUPATION


def spin_matrix_J_ij(states, bond):
    dimspin = len(states)
    site1, site2 = bond[0], bond[1]

    SzSz = np.zeros((dimspin, dimspin), dtype=complex)
    SpSm = np.zeros((dimspin, dimspin), dtype=complex)
    SmSp = np.zeros((dimspin, dimspin), dtype=complex)

    state_index = {tuple(s): i for i, s in enumerate(states)}

    for i, state in enumerate(states):
        SzSz[i, i] = Spin_z(state[site1]) * Spin_z(state[site2])

        occ1, occ2 = state[site1], state[site2]

        # S+S-
        fp1, fm2 = Spin_Flip_Plus(occ1), Spin_Flip_Minus(occ2)
        if fp1 is not INVALID_OCCUPATION and fm2 is not INVALID_OCCUPATION:
            flipped = list(state)
            flipped[site1], flipped[site2] = fp1, fm2
            j = state_index.get(tuple(flipped))
            if j is not None:
                SpSm[j, i] = 1.0

        # S-S+
        fm1, fp2 = Spin_Flip_Minus(occ1), Spin_Flip_Plus(occ2)
        if fm1 is not INVALID_OCCUPATION and fp2 is not INVALID_OCCUPATION:
            flipped = list(state)
            flipped[site1], flipped[site2] = fm1, fp2
            j = state_index.get(tuple(flipped))
            if j is not None:
                SmSp[j, i] = 1.0

    return SzSz + 0.5 * (SpSm + SmSp)


def spin_matrix_JJ_ij(states, bond):
    site1, site2, site3, site4 = bond[0], bond[1], bond[2], bond[3]
    spin_matrix1 = spin_matrix_J_ij(states, [site1, site2])
    spin_matrix2 = spin_matrix_J_ij(states, [site3, site4])
    return spin_matrix1 @ spin_matrix2


def spin_matrix_JJJ_ij(states, bond):
    site1, site2, site3, site4, site5, site6 = bond[0], bond[1], bond[2], bond[3], bond[4], bond[5]
    spin_matrix1 = spin_matrix_J_ij(states, [site1, site2])
    spin_matrix2 = spin_matrix_J_ij(states, [site3, site4])
    spin_matrix3 = spin_matrix_J_ij(states, [site5, site6])
    return spin_matrix1 @ spin_matrix2 @ spin_matrix3

# ============================================================
# T11 norm
# ============================================================

def compute_t11_norm(eigvecs_selected, dimspin):
    try:
        S_BD = eigvecs_selected[:dimspin, :]
        U, Sigma, VH = np.linalg.svd(S_BD)
        T11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(dimspin)
        return np.linalg.norm(T11m1.flatten())
    except np.linalg.LinAlgError:
        return np.inf


def _block_t11_norm(eigvecs_blocks, selected_blocks, dimspin_blocks):
    dimspin = sum(dimspin_blocks)
    selected = np.concatenate(
        [ev[:, idx] for ev, idx in zip(eigvecs_blocks, selected_blocks)], axis=1
    )
    return compute_t11_norm(selected, dimspin)


def _selection_log_stem(tmp_dir, *, twoSz=None, block_idx=None):
    os.makedirs(tmp_dir, exist_ok=True)
    labels = [f"rank{rank}"]
    if twoSz is not None:
        labels.append(f"twoSz_{twoSz}")
    if block_idx is not None:
        labels.append(f"block_{block_idx}")
    return os.path.join(tmp_dir, "_".join(labels))


def _open_log_file(path):
    return open(path, "w") if path is not None else open(os.devnull, "w")


# ============================================================
# Selection strategies
# ============================================================

def _argsort_real(values, secondary=None):
    values = np.asarray(values)
    if secondary is None:
        secondary = np.arange(len(values), dtype=float)
    secondary = np.asarray(secondary)
    return np.lexsort((np.arange(len(values), dtype=int), np.real(secondary), np.real(values)))


def _argsort_double_occ(double_occ, eigvals=None):
    secondary = np.real(eigvals) if eigvals is not None else np.arange(len(double_occ), dtype=float)
    return np.lexsort((np.arange(len(double_occ), dtype=int), secondary, np.real(double_occ)))


def select_by_occupation(double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks=None):
    if eigvals_blocks is None:
        eigvals_blocks = [None for _ in double_occ_blocks]
    sorted_blocks = [_argsort_double_occ(do, ev) for do, ev in zip(double_occ_blocks, eigvals_blocks)]
    selected = [si[:ds] for si, ds in zip(sorted_blocks, dimspin_blocks)]
    return selected, _block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)


def select_by_energy(double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks):
    sorted_blocks = [_argsort_real(ev) for ev in eigvals_blocks]
    selected = [si[:ds] for si, ds in zip(sorted_blocks, dimspin_blocks)]
    sorted_by_occ = [
        _argsort_double_occ(do[sel], ev[sel])
        for do, ev, sel in zip(double_occ_blocks, eigvals_blocks, selected)
    ]
    selected = [sel[sbo] for sel, sbo in zip(selected, sorted_by_occ)]
    return selected, _block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)


def greedy_swap(init_blocks, others_blocks, eigvecs_blocks, dimspin_blocks, log=None):
    selected = [idx.copy() for idx in init_blocks]
    swap = [oth.copy() for oth in others_blocks]

    best_norm = _block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)
    if np.isinf(best_norm) and log is not None:
        log.write("Failure in SVD, norm is set to inf\n")

    for blk in range(len(dimspin_blocks)):
        if len(swap[blk]) == 0:
            continue
        for i in range(len(selected[blk])):
            norm_list = []
            for j in range(len(swap[blk])):
                selected[blk][i], swap[blk][j] = swap[blk][j], selected[blk][i]

                t0 = time.time()
                norm = _block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)
                if log is not None:
                    dt = (time.time() - t0) * 1000
                    if not np.isinf(norm):
                        log.write(f"Time cost in SVD: {dt:.4f}ms, norm: {norm:.12f}\n")
                    else:
                        log.write(f"Time cost in SVD: {dt:.4f}ms, failure in SVD\n")
                norm_list.append(norm)

                selected[blk][i], swap[blk][j] = swap[blk][j], selected[blk][i]

            index_min = np.argmin(norm_list)
            if norm_list[index_min] < best_norm:
                best_norm = norm_list[index_min]
                selected[blk][i], swap[blk][index_min] = swap[blk][index_min], selected[blk][i]

    if np.isinf(best_norm):
        if log is not None:
            log.write("All svd failed, norm is set to inf, return the initial indices\n")
        return [idx.copy() for idx in init_blocks], [oth.copy() for oth in others_blocks], best_norm

    return selected, swap, best_norm


def select_greedy(double_occ_blocks, eigvecs_blocks, dimspin_blocks, ratio=5,
                  tmp_dir=None, twoSz=None, block_idx=None, eigvals_blocks=None):
    t0 = time.time()
    if eigvals_blocks is None:
        eigvals_blocks = [None for _ in double_occ_blocks]
    sorted_blocks = [_argsort_double_occ(do, ev) for do, ev in zip(double_occ_blocks, eigvals_blocks)]
    sizes = [min(ratio * ds, len(do)) for ds, do in zip(dimspin_blocks, double_occ_blocks)]
    init_blocks = [si[:ds].copy() for si, ds in zip(sorted_blocks, dimspin_blocks)]
    others_blocks = [si[ds:ss].copy() for si, ss, ds in zip(sorted_blocks, sizes, dimspin_blocks)]

    if tmp_dir is not None:
        log_stem = _selection_log_stem(tmp_dir, twoSz=twoSz, block_idx=block_idx)
        with open(f"{log_stem}_single.txt", "w") as log:
            log.write("Start the greedy algorithm to find the best t11 indices\n")
            selected, _, best_norm = greedy_swap(init_blocks, others_blocks, eigvecs_blocks, dimspin_blocks, log)
            log.write(f"Best |T11-1| = {best_norm:.12f}\n")
            log.write(f"Total time cost: {time.time() - t0:.1f}s\n")
    else:
        selected, _, best_norm = greedy_swap(init_blocks, others_blocks, eigvecs_blocks, dimspin_blocks, None)

    return selected, best_norm


def select_multi_restart(double_occ, eigvecs, dimspin, N, U, t_value,
                         eigvals=None,
                         ratio=4, rand_frac=0.10, ratio_rand_swap=2,
                         n_restarts=10, max_iters_rand=None,
                         tmp_dir=None, twoSz=None, block_idx=None):
    t0 = time.time()
    iter_rand = 0
    flag_rand = False
    sorted_indices = _argsort_double_occ(double_occ, eigvals)
    size_space = min(ratio * dimspin, len(double_occ))

    if max_iters_rand is None:
        max_iters_rand = n_restarts

    best_selected = sorted_indices[:dimspin].copy()
    best_swap = sorted_indices[dimspin:size_space].copy()
    best_norm = compute_t11_norm(eigvecs[:, best_selected], dimspin)

    log_stem = None
    if tmp_dir is not None:
        log_stem = _selection_log_stem(tmp_dir, twoSz=twoSz, block_idx=block_idx)

    with _open_log_file(None if log_stem is None else f"{log_stem}_multi.txt") as log:
        log.write(f"N={N}, U={U:.4f}, t={t_value:.4f}, rank={rank}\n")
        log.write(f"n_restarts={n_restarts}, ratio={ratio}, rand_frac={rand_frac}, max_iters_rand={max_iters_rand}\n\n")
        log.write("Start to find the best t11 indices\n")
        log.flush()

        if np.isinf(best_norm):
            log.write("Failure in SVD, norm is set to inf\n")

        for it in range(n_restarts):
            t1 = time.time()
            iter_path = None if log_stem is None else f"{log_stem}_multi_iter{it}.txt"
            with _open_log_file(iter_path) as iter_log:
                iter_log.write("Start the greedy algorithm to find the best t11 indices\n")

                if not flag_rand:
                    sel_blocks, swap_blocks, cur_norm = greedy_swap(
                        [best_selected], [best_swap], [eigvecs], [dimspin], iter_log
                    )
                    cur_selected, cur_swap = sel_blocks[0], swap_blocks[0]
                else:
                    sorted_selected = best_selected[
                        _argsort_double_occ(
                            double_occ[best_selected],
                            None if eigvals is None else eigvals[best_selected],
                        )
                    ].copy()
                    sorted_swap = best_swap[
                        _argsort_double_occ(
                            double_occ[best_swap],
                            None if eigvals is None else eigvals[best_swap],
                        )
                    ].copy()

                    n_rand = max(8, int(dimspin * rand_frac))
                    if n_rand > 0 and len(sorted_swap) >= n_rand and dimspin >= n_rand:
                        select_space = np.concatenate([sorted_selected[-n_rand:], sorted_swap[:n_rand * ratio_rand_swap]])
                        chosen = np.random.choice(len(select_space), size=n_rand, replace=False)
                        remaining = np.setdiff1d(np.arange(len(select_space)), chosen)
                        cur_selected = sorted_selected.copy()
                        cur_swap = sorted_swap.copy()
                        cur_selected[-n_rand:] = select_space[chosen]
                        cur_swap[:n_rand * ratio_rand_swap] = select_space[remaining]
                    else:
                        cur_selected = sorted_selected.copy()
                        cur_swap = sorted_swap.copy()

                    sel_blocks, swap_blocks, cur_norm = greedy_swap(
                        [cur_selected], [cur_swap], [eigvecs], [dimspin], iter_log
                    )
                    cur_selected, cur_swap = sel_blocks[0], swap_blocks[0]

                iter_log.write(f"Best |T11-1| = {cur_norm:.12f}\n")
                iter_log.write(f"Total time cost: {time.time() - t1:.1f}s\n")

            if cur_norm < best_norm and abs(cur_norm - best_norm) > ATOL["loose"]:
                msg = (f"Succeed to find better solution at iteration {it + 1} / {n_restarts}, "
                       f"time cost: {time.time() - t1:.1f}s, current norm: {cur_norm:.12f}, best norm: {best_norm:.12f}")
                if not flag_rand:
                    log.write(msg + "\n")
                else:
                    log.write(msg + f", random iteration {iter_rand + 1} / {max_iters_rand}\n")
                flag_rand = False
                iter_rand = 0
                best_norm = cur_norm
                best_selected = cur_selected.copy()
                best_swap = cur_swap.copy()
            else:
                msg = (f" Failed to find better solution at iteration {it + 1} / {n_restarts}, "
                       f"time cost: {time.time() - t1:.1f}s, current norm: {cur_norm:.12f}, best norm: {best_norm:.12f}")
                if not flag_rand:
                    log.write(msg + ", random swap starts!\n")
                    flag_rand = True
                    iter_rand = 0
                else:
                    log.write(msg + f", random iteration {iter_rand + 1} / {max_iters_rand}\n")
                    iter_rand += 1
                    if iter_rand >= max_iters_rand:
                        log.write("Break the loop!\n")
                        break
            log.flush()

        log.write(f"Best |T11-1| = {best_norm:.12f}\n")
        log.write(f"Total time cost: {time.time() - t0:.1f}s\n")

    selected_double_occ = double_occ[best_selected]
    sorted_by_occ = _argsort_double_occ(
        selected_double_occ,
        None if eigvals is None else eigvals[best_selected],
    )
    return best_selected[sorted_by_occ], best_norm


def select_adiabatic(eigvecs_blocks, dimspin_blocks, filename_eigvecs, filename_indices):
    eigvecs_all = np.concatenate(eigvecs_blocks, axis=1)
    eigvecs_previous = np.load(f"{filename_eigvecs}_eigvecs.npy", allow_pickle=False)
    selected_previous = np.atleast_1d(
        np.load(f"{filename_indices}_t11_selected_indices.npy", allow_pickle=False)
    ).astype(int)
    eigvecs_previous = eigvecs_previous[:, selected_previous]

    norms_matrix = np.abs(eigvecs_previous.conj().T @ eigvecs_all) ** 2
    norms_vector = np.sum(norms_matrix, axis=0)

    block_sizes = [ev.shape[1] for ev in eigvecs_blocks]
    offsets = np.cumsum([0] + block_sizes)
    norms_blocks = [norms_vector[offsets[i]:offsets[i + 1]] for i in range(len(eigvecs_blocks))]

    selected = [np.argsort(nb)[-ds:] for nb, ds in zip(norms_blocks, dimspin_blocks)]
    overlap = sum(np.sum(nb[sel]) for nb, sel in zip(norms_blocks, selected))
    best_norm = _block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)
    return selected, best_norm, overlap


def _adiabatic_seed_paths(params, params_cluster):
    previous_params = Params(
        N=params.N,
        U=params.U,
        t=params.t - params.delta,
        mode=params.mode,
        twoSz=params.twoSz,
        twoS=params.twoS,
        match_spin_sectors=params.match_spin_sectors,
        workflow=params.workflow,
        restart=False,
    )
    base_dir, run_dir, data_dir = build_run_dirnames(previous_params)
    if params.restart:
        run_dir = f"{run_dir}_restart"

    block_root = block_root_dir(base_dir)
    previous_indices = (
        f"{block_root}/{run_dir}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
    )
    previous_eigvecs = (
        f"{block_root}/{data_dir}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
    )

    if (
        os.path.exists(f"{previous_indices}_t11_selected_indices.npy")
        and os.path.exists(f"{previous_eigvecs}_eigvecs.npy")
    ):
        return previous_eigvecs, previous_indices

    seed_params = Params(
        N=params.N,
        U=params.U,
        t=params.t,
        mode=params.mode,
        twoSz=params.twoSz,
        twoS=params.twoS,
        match_spin_sectors=params.match_spin_sectors,
        workflow=None,
        restart=False,
    )
    base_dir, run_dir, data_dir = build_run_dirnames(seed_params)
    block_root = block_root_dir(base_dir)
    seed_indices = (
        f"{block_root}/{run_dir}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
    )
    seed_eigvecs = (
        f"{block_root}/{data_dir}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
    )

    if not os.path.exists(f"{seed_indices}_t11_selected_indices.npy"):
        raise FileNotFoundError(
            f"Missing adiabatic seed selected indices: {seed_indices}_t11_selected_indices.npy"
        )
    if not os.path.exists(f"{seed_eigvecs}_eigvecs.npy"):
        raise FileNotFoundError(
            f"Missing adiabatic seed eigvecs: {seed_eigvecs}_eigvecs.npy"
        )

    return seed_eigvecs, seed_indices


def select_eigenstates(method, global_indices, eigvals_blocks, eigvecs_blocks,
                       double_occ_blocks, dimspin_blocks,
                       params=None, params_cluster=None,
                       tmp_dir=None, twoSz=None,
                       N=None, U=None, t=None):
    """Unified dispatcher for eigenstate selection."""
    overlap = None
    best_norm = np.inf

    if method is None:
        selected_blocks, best_norm = select_by_occupation(
            double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks
        )
        if np.isinf(best_norm):
            selected_blocks, best_norm = select_greedy(
                double_occ_blocks, eigvecs_blocks, dimspin_blocks, ratio=8,
                tmp_dir=tmp_dir, twoSz=twoSz, eigvals_blocks=eigvals_blocks,
            )
    elif method.lower() == "occ":
        selected_blocks, best_norm = select_by_occupation(
            double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks
        )
    elif method.lower() == "energy":
        selected_blocks, best_norm = select_by_energy(double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks)
    elif method.lower() == "single":
        selected_blocks, best_norm = select_greedy(
            double_occ_blocks, eigvecs_blocks, dimspin_blocks, ratio=8,
            tmp_dir=tmp_dir, twoSz=twoSz, eigvals_blocks=eigvals_blocks,
        )
    elif method.lower() == "multi":
        selected_blocks = []
        for block_idx, (double_occ_block, eigvecs_block, dimspin_block) in enumerate(
            zip(double_occ_blocks, eigvecs_blocks, dimspin_blocks)
        ):
            selected_block, _ = select_multi_restart(
                double_occ_block,
                eigvecs_block,
                dimspin_block,
                N=N,
                U=U,
                t_value=t,
                eigvals=eigvals_blocks[block_idx],
                ratio=8,
                n_restarts=40,
                max_iters_rand=4,
                tmp_dir=tmp_dir,
                twoSz=twoSz,
                block_idx=block_idx,
            )
            selected_blocks.append(selected_block)
        best_norm = _block_t11_norm(eigvecs_blocks, selected_blocks, dimspin_blocks)
    elif method.lower() == "adiabatic":
        filename_eigvecs, filename_indices = _adiabatic_seed_paths(params, params_cluster)
        selected_blocks, best_norm, overlap = select_adiabatic(
            eigvecs_blocks, dimspin_blocks, filename_eigvecs, filename_indices,
        )
    else:
        raise ValueError(f"Invalid selection method: {method}")

    indices_selected = []
    for blk_idx, sel in enumerate(selected_blocks):
        indices_selected.extend(global_indices[blk_idx][i] for i in sel)

    return indices_selected, best_norm, overlap


def fit_error_metrics(x, A, b):
    """Return (relative_error, residual_norm, R²) for the fit Ax ~ b."""
    residual_vec = A @ x - b
    relative_error = np.linalg.norm(residual_vec) / np.linalg.norm(b)
    residual = np.linalg.norm(residual_vec)
    ss_res = np.real(np.dot(residual_vec.conj(), residual_vec))
    centered_b = b - np.mean(b)
    ss_tot = np.real(np.dot(centered_b.conj(), centered_b))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (relative_error, residual, r2)


# ============================================================
# H_eff extraction
# ============================================================

def calc_double_occ_expectation(states, eigvecs):
    dom = calc_double_occupation_matrix(states)
    return np.diag(eigvecs.conj().T @ dom @ eigvecs)


def extract_heff(eigvals, eigvecs, selected_indices, dimspin):
    """SVD-extract effective spin Hamiltonian.

    Returns (heff, t11m1, t11m1_norm).
    """
    S_BD = eigvecs[:dimspin, selected_indices]
    U, Sigma, VH = np.linalg.svd(S_BD)
    t11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(dimspin)
    t11m1_norm = np.linalg.norm(t11m1.flatten())

    Lambda = np.diag(eigvals[selected_indices])
    heff = U @ VH @ Lambda @ VH.conj().T @ U.conj().T

    return heff, t11m1, t11m1_norm


# ============================================================
# Spin-coupling fitting
# ============================================================

def solve_spin_fit(A, b):
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs


def calc_spin_coeff(heff, states_spin, dimspin, spin_operator_catalog):
    """Fit Heff as a linear combination of spin operators.

    Returns (coeffs, error_metrics).
    """
    Ms = [np.eye(dimspin, dtype=complex)]
    grouped_terms = [bond_group for _, bond_group in spin_operator_catalog.two_site_classes]
    if spin_operator_catalog.four_site_terms:
        grouped_terms.append(spin_operator_catalog.four_site_terms)
    if spin_operator_catalog.six_site_terms:
        grouped_terms.append(spin_operator_catalog.six_site_terms)

    for bond_group in grouped_terms:
        for bond in bond_group:
            if len(bond) == 2:
                Ms.append(spin_matrix_J_ij(states_spin, bond))
            elif len(bond) == 4:
                Ms.append(spin_matrix_JJ_ij(states_spin, bond))
            elif len(bond) == 6:
                Ms.append(spin_matrix_JJJ_ij(states_spin, bond))

    b = heff.flatten()
    A = np.array([m.flatten() for m in Ms]).T
    coeffs_flat = solve_spin_fit(A, b)

    coeffs = [coeffs_flat[0]]
    idx = 1
    for bond_group in grouped_terms:
        if bond_group:
            coeffs.append(list(coeffs_flat[idx:idx + len(bond_group)]))
            idx += len(bond_group)
    error = fit_error_metrics(coeffs_flat, A, b)
    return (coeffs, error)
