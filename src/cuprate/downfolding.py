"""Spin-basis operators, selection, and least-squares spin fitting."""

from __future__ import annotations

import os
import time
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


def _selection_log_stem(tmp_dir, *, twoSz=None, block_idx=None):
    os.makedirs(tmp_dir, exist_ok=True)
    labels = ["rank0"]
    if twoSz is not None:
        labels.append(f"twoSz_{twoSz}")
    if block_idx is not None:
        labels.append(f"block_{block_idx}")
    return os.path.join(tmp_dir, "_".join(labels))


def _open_log_file(path):
    return open(path, "w") if path is not None else open(os.devnull, "w")


def _argsort_real(values, secondary=None):
    values = np.asarray(values)
    if secondary is None:
        secondary = np.arange(len(values), dtype=float)
    secondary = np.asarray(secondary)
    return np.lexsort((np.arange(len(values), dtype=int), np.real(secondary), np.real(values)))


def _argsort_double_occ(double_occ, eigvals=None):
    secondary = np.real(eigvals) if eigvals is not None else np.arange(len(double_occ), dtype=float)
    return np.lexsort((np.arange(len(double_occ), dtype=int), secondary, np.real(double_occ)))


def _select_occ_info(spectrum) -> tuple[list[list[int]], dict]:
    selected = []
    for block in spectrum.blocks:
        dimspin = block.spin_dim()
        double_occ = _block_double_occ_expectation(block)
        n = len(block.eigvals)
        order = np.lexsort((np.arange(n), np.real(block.eigvals), np.real(double_occ)))
        selected.append(order[:dimspin].tolist())
    return selected, {"t11m1_norm": _joint_t11_norm(spectrum.blocks, selected), "overlap": None}


def select_occ(spectrum) -> list[list[int]]:
    return _select_occ_info(spectrum)[0]


def _select_energy_info(spectrum) -> tuple[list[list[int]], dict]:
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
    return selected, {"t11m1_norm": _joint_t11_norm(spectrum.blocks, selected), "overlap": None}


def select_energy(spectrum) -> list[list[int]]:
    return _select_energy_info(spectrum)[0]


def greedy_swap(init_blocks, others_blocks, blocks, log=None):
    selected = [idx.copy() for idx in init_blocks]
    swap = [oth.copy() for oth in others_blocks]

    best_norm = _joint_t11_norm(blocks, selected)
    if np.isinf(best_norm) and log is not None:
        log.write("Failure in SVD, norm is set to inf\n")

    for blk in range(len(selected)):
        if len(swap[blk]) == 0:
            continue
        for i in range(len(selected[blk])):
            norm_list = []
            for j in range(len(swap[blk])):
                selected[blk][i], swap[blk][j] = swap[blk][j], selected[blk][i]

                t0 = time.time()
                norm = _joint_t11_norm(blocks, selected)
                if log is not None:
                    dt = (time.time() - t0) * 1000
                    if not np.isinf(norm):
                        log.write(f"Time cost in SVD: {dt:.4f}ms, norm: {norm:.12f}\n")
                    else:
                        log.write(f"Time cost in SVD: {dt:.4f}ms, failure in SVD\n")
                norm_list.append(norm)

                selected[blk][i], swap[blk][j] = swap[blk][j], selected[blk][i]

            index_min = int(np.argmin(norm_list))
            if norm_list[index_min] < best_norm:
                best_norm = norm_list[index_min]
                selected[blk][i], swap[blk][index_min] = swap[blk][index_min], selected[blk][i]

    if np.isinf(best_norm):
        if log is not None:
            log.write("All svd failed, norm is set to inf, return the initial indices\n")
        return [idx.copy() for idx in init_blocks], [oth.copy() for oth in others_blocks], best_norm

    return selected, swap, best_norm


def _select_greedy_info(
    spectrum,
    ratio: int = 5,
    tmp_dir: str | None = None,
    twoSz: int | None = None,
) -> tuple[list[list[int]], dict]:
    t0 = time.time()
    sorted_blocks = []
    sizes = []
    init_blocks = []
    others_blocks = []

    for block in spectrum.blocks:
        double_occ = _block_double_occ_expectation(block)
        sorted_indices = _argsort_double_occ(double_occ, block.eigvals)
        dimspin = block.spin_dim()
        size_space = min(ratio * dimspin, len(double_occ))
        sorted_blocks.append(sorted_indices)
        sizes.append(size_space)
        init_blocks.append(sorted_indices[:dimspin].copy())
        others_blocks.append(sorted_indices[dimspin:size_space].copy())

    if tmp_dir is not None:
        log_stem = _selection_log_stem(tmp_dir, twoSz=twoSz)
        with open(f"{log_stem}_single.txt", "w") as log:
            log.write("Start the greedy algorithm to find the best t11 indices\n")
            selected, _, best_norm = greedy_swap(init_blocks, others_blocks, spectrum.blocks, log)
            log.write(f"Best |T11-1| = {best_norm:.12f}\n")
            log.write(f"Total time cost: {time.time() - t0:.1f}s\n")
    else:
        selected, _, best_norm = greedy_swap(init_blocks, others_blocks, spectrum.blocks, None)

    return selected, {"t11m1_norm": best_norm, "overlap": None}


def select_greedy(spectrum, ratio: int = 5, **kwargs) -> list[list[int]]:
    return _select_greedy_info(spectrum, ratio=ratio, **kwargs)[0]


def select_single(spectrum, ratio: int = 5, **kwargs) -> list[list[int]]:
    return select_greedy(spectrum, ratio=ratio, **kwargs)


def select_multi_restart(
    block,
    *,
    ratio: int = 4,
    rand_frac: float = 0.10,
    ratio_rand_swap: int = 2,
    n_restarts: int = 10,
    max_iters_rand: int | None = None,
    tmp_dir: str | None = None,
    twoSz: int | None = None,
    block_idx: int | None = None,
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    iter_rand = 0
    flag_rand = False

    double_occ = _block_double_occ_expectation(block)
    eigvals = block.eigvals
    dimspin = block.spin_dim()
    spin_basis = block.spin_basis()
    eigvecs = block.eigvecs

    sorted_indices = _argsort_double_occ(double_occ, eigvals)
    size_space = min(ratio * dimspin, len(double_occ))

    if max_iters_rand is None:
        max_iters_rand = n_restarts

    best_selected = sorted_indices[:dimspin].copy()
    best_swap = sorted_indices[dimspin:size_space].copy()
    best_norm = _joint_t11_norm([block], [best_selected])

    log_stem = None
    if tmp_dir is not None:
        log_stem = _selection_log_stem(tmp_dir, twoSz=twoSz, block_idx=block_idx)

    with _open_log_file(None if log_stem is None else f"{log_stem}_multi.txt") as log:
        log.write(
            f"dimspin={dimspin}, n_restarts={n_restarts}, ratio={ratio}, "
            f"rand_frac={rand_frac}, max_iters_rand={max_iters_rand}\n\n"
        )
        log.write("Start to find the best t11 indices\n")
        log.flush()

        if np.isinf(best_norm):
            log.write("Failure in initial SVD, norm is set to inf\n")

        for it in range(n_restarts):
            t1 = time.time()
            iter_path = None if log_stem is None else f"{log_stem}_multi_iter{it}.txt"
            with _open_log_file(iter_path) as iter_log:
                iter_log.write("Start the greedy algorithm to find the best t11 indices\n")

                if not flag_rand:
                    sel_blocks, swap_blocks, cur_norm = greedy_swap(
                        [best_selected],
                        [best_swap],
                        [block],
                        iter_log,
                    )
                    cur_selected, cur_swap = sel_blocks[0], swap_blocks[0]
                else:
                    sorted_selected = best_selected[
                        _argsort_double_occ(double_occ[best_selected], eigvals[best_selected])
                    ].copy()
                    sorted_swap = best_swap[
                        _argsort_double_occ(double_occ[best_swap], eigvals[best_swap])
                    ].copy()

                    n_rand = max(8, int(dimspin * rand_frac))
                    if n_rand > 0 and len(sorted_swap) >= n_rand and dimspin >= n_rand:
                        select_space = np.concatenate(
                            [sorted_selected[-n_rand:], sorted_swap[: n_rand * ratio_rand_swap]]
                        )
                        chosen = np.random.choice(len(select_space), size=n_rand, replace=False)
                        remaining = np.setdiff1d(np.arange(len(select_space)), chosen)
                        cur_selected = sorted_selected.copy()
                        cur_swap = sorted_swap.copy()
                        cur_selected[-n_rand:] = select_space[chosen]
                        cur_swap[: n_rand * ratio_rand_swap] = select_space[remaining]
                    else:
                        cur_selected = sorted_selected.copy()
                        cur_swap = sorted_swap.copy()

                    sel_blocks, swap_blocks, cur_norm = greedy_swap(
                        [cur_selected],
                        [cur_swap],
                        [block],
                        iter_log,
                    )
                    cur_selected, cur_swap = sel_blocks[0], swap_blocks[0]

                iter_log.write(f"Best |T11-1| = {cur_norm:.12f}\n")
                iter_log.write(f"Total time cost: {time.time() - t1:.1f}s\n")

            if cur_norm < best_norm:
                msg = (
                    f"Succeed to find better solution at iteration {it + 1} / {n_restarts}, "
                    f"time cost: {time.time() - t1:.1f}s, current norm: {cur_norm:.12f}, "
                    f"best norm: {best_norm:.12f}"
                )
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
                msg = (
                    f"Failed to find better solution at iteration {it + 1} / {n_restarts}, "
                    f"time cost: {time.time() - t1:.1f}s, current norm: {cur_norm:.12f}, "
                    f"best norm: {best_norm:.12f}"
                )
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
    sorted_by_occ = _argsort_double_occ(selected_double_occ, eigvals[best_selected])
    return best_selected[sorted_by_occ], float(best_norm)


def _select_multi_info(
    spectrum,
    *,
    ratio: int = 8,
    rand_frac: float = 0.10,
    ratio_rand_swap: int = 2,
    n_restarts: int = 40,
    max_iters_rand: int = 4,
    tmp_dir: str | None = None,
    twoSz: int | None = None,
) -> tuple[list[list[int]], dict]:
    selected = []
    for block_idx, block in enumerate(spectrum.blocks):
        selected_block, _ = select_multi_restart(
            block,
            ratio=ratio,
            rand_frac=rand_frac,
            ratio_rand_swap=ratio_rand_swap,
            n_restarts=n_restarts,
            max_iters_rand=max_iters_rand,
            tmp_dir=tmp_dir,
            twoSz=twoSz,
            block_idx=block_idx,
        )
        selected.append(selected_block.tolist())
    return selected, {"t11m1_norm": _joint_t11_norm(spectrum.blocks, selected), "overlap": None}


def select_multi(spectrum, **kwargs) -> list[list[int]]:
    return _select_multi_info(spectrum, **kwargs)[0]


def _load_seed(seed):
    if isinstance(seed, dict):
        eigvecs_path = seed["eigvecs"]
        indices_path = seed["indices"]
        reuse_indices = bool(seed.get("reuse_indices", False))
    else:
        eigvecs_path, indices_path = seed
        reuse_indices = False
    eigvecs_previous = np.load(eigvecs_path, allow_pickle=False)
    selected_previous = np.atleast_1d(
        np.load(indices_path, allow_pickle=False)
    ).astype(int)
    return eigvecs_previous, selected_previous, reuse_indices


def _select_adiabatic_info(spectrum, seed) -> tuple[list[list[int]], dict]:
    eigvecs_all = np.concatenate([block.eigvecs for block in spectrum.blocks], axis=1)
    eigvecs_previous, selected_previous, reuse_indices = _load_seed(seed)
    block_sizes = [block.eigvecs.shape[1] for block in spectrum.blocks]

    if reuse_indices:
        offsets = np.cumsum([0] + block_sizes)
        selected = []
        for i in range(len(spectrum.blocks)):
            start, stop = offsets[i], offsets[i + 1]
            mask = (selected_previous >= start) & (selected_previous < stop)
            selected.append((selected_previous[mask] - start).tolist())
        best_norm = _joint_t11_norm(spectrum.blocks, selected)
        return selected, {"t11m1_norm": best_norm, "overlap": None}

    eigvecs_previous = eigvecs_previous[:, selected_previous]

    norms_matrix = np.abs(eigvecs_previous.conj().T @ eigvecs_all) ** 2
    norms_vector = np.sum(norms_matrix, axis=0)

    dimspin_blocks = [block.spin_dim() for block in spectrum.blocks]
    offsets = np.cumsum([0] + block_sizes)
    norms_blocks = [norms_vector[offsets[i] : offsets[i + 1]] for i in range(len(spectrum.blocks))]

    selected = [np.argsort(nb)[-ds:].tolist() for nb, ds in zip(norms_blocks, dimspin_blocks)]
    overlap = float(sum(np.sum(nb[sel]) for nb, sel in zip(norms_blocks, selected)))
    best_norm = _joint_t11_norm(spectrum.blocks, selected)
    return selected, {"t11m1_norm": best_norm, "overlap": overlap}


def select_adiabatic(spectrum, seed) -> list[list[int]]:
    return _select_adiabatic_info(spectrum, seed)[0]


def select(spectrum, method: str = "occ", return_info: bool = False, **kwargs):
    methods = {
        "occ": _select_occ_info,
        "energy": _select_energy_info,
        "single": _select_greedy_info,
        "greedy": _select_greedy_info,
        "multi": _select_multi_info,
        "adiabatic": _select_adiabatic_info,
    }
    if method not in methods:
        raise ValueError(f"Unknown selection method: {method}")
    selected, info = methods[method](spectrum, **kwargs)
    if return_info:
        return selected, info
    return selected
