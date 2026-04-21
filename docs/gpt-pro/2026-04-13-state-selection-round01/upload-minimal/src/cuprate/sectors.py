"""Symmetry sector decomposition (03-SYMMETRY_SECTORS).

S² matrix construction, Sz/S² blocking, and spectrum reconstruction.
Also contains eigenpair canonicalization utilities used across modules.
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.linalg import block_diag

from cuprate import ATOL
from cuprate.states import (
    calc_double_occupation,
    calc_double_occupation_matrix,
    calc_twoSz,
    sort_by_twoSz as sort_states,
)


# ============================================================
# Combinatorics
# ============================================================

def comb_safe(n, k):
    if k < 0 or k > n:
        return 0
    return comb(n, k)


# ============================================================
# Eigenpair canonicalization (shared utility)
# ============================================================

def canonicalize_vector_phases(eigvecs, atol=ATOL["tight"]):
    eigvecs = np.array(eigvecs, dtype=complex, copy=True)
    for col in range(eigvecs.shape[1]):
        vec = eigvecs[:, col]
        pivot = int(np.argmax(np.abs(vec)))
        value = vec[pivot]
        if abs(value) <= atol:
            continue
        phase = np.exp(-1j * np.angle(value))
        vec = vec * phase
        if vec[pivot].real < 0:
            vec = -vec
        if abs(vec[pivot].imag) <= atol:
            vec[pivot] = vec[pivot].real + 0.0j
        eigvecs[:, col] = vec
    return eigvecs


def canonicalize_eigenpairs(eigvals, eigvecs, atol=ATOL["tight"], tie_breaker=None):
    eigvals = np.array(eigvals, copy=True)
    eigvecs = np.array(eigvecs, dtype=complex, copy=True)
    if eigvecs.size == 0:
        return eigvals, eigvecs

    if tie_breaker is None:
        weights = np.arange(1, eigvecs.shape[0] + 1, dtype=float)
        tie_breaker = np.diag(weights)

    start = 0
    while start < len(eigvals):
        end = start + 1
        while end < len(eigvals) and abs(eigvals[end] - eigvals[start]) <= atol:
            end += 1

        if end - start > 1:
            subspace = eigvecs[:, start:end]
            projected = subspace.conj().T @ tie_breaker @ subspace
            _, rotation = np.linalg.eigh(projected)
            eigvecs[:, start:end] = subspace @ rotation
        start = end

    eigvecs = canonicalize_vector_phases(eigvecs, atol=atol)
    return eigvals, eigvecs


def canonicalize_eigenpairs_by_operator(
    eigvals,
    eigvecs,
    operator,
    atol=ATOL["tight"],
    secondary_tie_breaker=None,
):
    eigvals = np.array(eigvals, copy=True)
    eigvecs = np.array(eigvecs, dtype=complex, copy=True)
    if eigvecs.size == 0:
        return eigvals, eigvecs

    if secondary_tie_breaker is None:
        weights = np.arange(1, eigvecs.shape[0] + 1, dtype=float)
        secondary_tie_breaker = np.diag(weights)

    start = 0
    while start < len(eigvals):
        end = start + 1
        while end < len(eigvals) and abs(eigvals[end] - eigvals[start]) <= atol:
            end += 1

        if end - start > 1:
            subspace = eigvecs[:, start:end]
            projected = subspace.conj().T @ operator @ subspace
            op_vals, rotation = np.linalg.eigh(projected)
            subspace = subspace @ rotation

            sub_start = 0
            while sub_start < len(op_vals):
                sub_end = sub_start + 1
                while sub_end < len(op_vals) and abs(op_vals[sub_end] - op_vals[sub_start]) <= atol:
                    sub_end += 1
                if sub_end - sub_start > 1:
                    block = subspace[:, sub_start:sub_end]
                    projected_secondary = block.conj().T @ secondary_tie_breaker @ block
                    _, secondary_rotation = np.linalg.eigh(projected_secondary)
                    subspace[:, sub_start:sub_end] = block @ secondary_rotation
                sub_start = sub_end

            eigvecs[:, start:end] = subspace
        start = end

    eigvecs = canonicalize_vector_phases(eigvecs, atol=atol)
    return eigvals, eigvecs


def canonicalize_eigenpairs_with_S2(basis, eigvals, eigvecs, atol=ATOL["tight"]):
    s2_matrix = compute_S2_matrix(basis)
    double_occ_matrix = calc_double_occupation_matrix(basis)
    return canonicalize_eigenpairs_by_operator(
        eigvals,
        eigvecs,
        s2_matrix,
        atol=atol,
        secondary_tie_breaker=double_occ_matrix,
    )


# ============================================================
# S² matrix construction
# ============================================================

# None means the spin-flip operator annihilates the state at this site.
INVALID_OCCUPATION = None


def compute_S2_element(state1, state2):
    N = len(state1)
    if state1 == state2:
        s_vals = np.where(np.abs(state1) == 1, 0.5, 0.0)
        m_vals = [0.0 for _ in range(N)]
        for i in range(N):
            if state1[i] == 1 or state1[i] == -1:
                m_vals[i] = 0.5 * state1[i]
            else:
                m_vals[i] = 0.0

        S2_site = np.sum(s_vals * (s_vals + 1))
        S2_pair = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                S2_pair += m_vals[i] * m_vals[j]

        S2_total = S2_site + 2 * S2_pair
    else:
        S2_site = 0.0
        S2_pair = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if (state1[i], state2[i]) == (1, -1) \
                   and (state1[j], state2[j]) == (-1, 1) \
                   and all(state1[k] == state2[k] for k in range(N) if k not in (i, j)):
                    S2_pair += 0.5
                if (state1[i], state2[i]) == (-1, 1) \
                   and (state1[j], state2[j]) == (1, -1) \
                   and all(state1[k] == state2[k] for k in range(N) if k not in (i, j)):
                    S2_pair += 0.5

        S2_total = S2_site + 2 * S2_pair

    return S2_total


def compute_S2_vec(basis, eigvec1, eigvec2):
    N = len(basis)
    S2_val = 0.0j
    for i in range(N):
        for j in range(N):
            S2_val += np.conjugate(eigvec1[i]) * eigvec2[j] * compute_S2_element(basis[i], basis[j])
    return S2_val


def compute_S2_matrix(basis):
    N = len(basis)
    basis_matrix = np.zeros([N, N], dtype=complex)
    for i in range(N):
        basis_matrix[i][i] = compute_S2_element(basis[i], basis[i])
        for j in range(i + 1, N):
            basis_matrix[i][j] = compute_S2_element(basis[i], basis[j])
            basis_matrix[j][i] = np.conjugate(basis_matrix[i][j])
    return basis_matrix


# ============================================================
# S² eigenvalue classification
# ============================================================

def sort_S2(values):
    S2_groups = {}
    bad_indices = []
    for idx, val in enumerate(values):
        val_real = float(np.real(val))
        s_raw = (-1.0 + np.sqrt(1.0 + 4.0 * val_real)) / 2.0
        s = round(s_raw * 2) / 2
        if abs(s * (s + 1) - val_real) > ATOL["loose"]:
            bad_indices.append(idx)
            s = s_raw
        S2_groups.setdefault(s, []).append(idx)
    s_list = sorted(S2_groups.keys())
    index_groups = [S2_groups[s] for s in s_list]
    return s_list, index_groups, (bad_indices if bad_indices else None)


# ============================================================
# S² block diagonalisation
# ============================================================

def solve_S2_blocks(basis):
    states_by_double = {}
    for idx, state in enumerate(basis):
        double_occ = calc_double_occupation(state)
        states_by_double.setdefault(double_occ, []).append(state)

    eigvals_list = []
    eigvecs_list = []
    for double_occ in sorted(states_by_double.keys()):
        grouped_states, _ = sort_states(states_by_double[double_occ])
        basis_matrix = compute_S2_matrix(grouped_states)
        eigvals, eigvecs = np.linalg.eigh(basis_matrix)
        eigvals, eigvecs = canonicalize_eigenpairs(eigvals, eigvecs)
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)

    eigvals = np.concatenate(eigvals_list)
    eigvecs = block_diag(*eigvecs_list)

    s_list, index_groups, bad = sort_S2(eigvals)
    if bad is not None:
        raise RuntimeError(f"Failed to sort eigvals for S^2 matrix")

    indices = np.concatenate(index_groups)
    return eigvals[indices], eigvecs[:, indices]


# ============================================================
# Sector counting
# ============================================================

def compute_all_sector_counts(N, print_out=False):
    """Compute counts for all (S, Sz) sectors for N spin-1/2 sites."""
    smax = N * 0.5
    n_half = N * 0.5

    results = []
    for nup in range(N + 1):
        ndo = N - nup
        twoSz = nup - ndo
        dim_sz = comb_safe(N, nup)
        sz_entry = {"twoSz": twoSz, "dim_sz": dim_sz, "S2_counts": []}
        for idx_s in range(int(smax - abs(twoSz * 0.5)) + 1):
            twoS = abs(twoSz) + 2 * idx_s
            s = twoS * 0.5
            S2_val = s * (s + 1)
            dim_sz_s = comb_safe(N, int(n_half + s)) * comb_safe(N, int(n_half - s))
            dim_sz_s -= comb_safe(N, int(n_half + s + 1)) * comb_safe(N, int(n_half - s - 1))
            sz_entry["S2_counts"].append((twoS, S2_val, dim_sz_s))
        results.append(sz_entry)

    if print_out:
        print("Sz sectors:")
        for entry in results:
            print(f"  twoSz={entry['twoSz']}: dim={entry['dim_sz']}")
        print("Sz, S^2 sectors:")
        for entry in results:
            for twoS, S2_val, dim_sz_s in entry["S2_counts"]:
                print(f"  twoSz={entry['twoSz']}, twoS={twoS}, S2={S2_val:.1f}: dim={dim_sz_s}")

    return results


# ============================================================
# Spectrum reconstruction helpers
# ============================================================

def spin_flip_state(state):
    return tuple(-s if abs(s) == 1 else s for s in state)


def build_sz_perm(all_states, sz_sector_states):
    state_to_global = {tuple(s): i for i, s in enumerate(all_states)}
    return [
        [state_to_global[tuple(s)] for s in sector_states]
        for sector_states in sz_sector_states
    ]


def build_mirrored_rows(all_states, states_local):
    state_to_global = {tuple(s): i for i, s in enumerate(all_states)}
    return [state_to_global[spin_flip_state(s)] for s in states_local]


def build_mirrored_eigvecs(states_local, local_eigvecs):
    signs = np.array([(-1) ** calc_double_occupation(s) for s in states_local], dtype=float)
    mirrored = signs[:, np.newaxis] * local_eigvecs
    return canonicalize_vector_phases(mirrored)


def assemble_reconstructed_eigensystem(sector_entries, n_total):
    """Reconstruct the global eigensystem from sector entries.

    Returns (eigvals, eigvecs, block_global_indices).
    """
    eigvals_cat = np.concatenate([e[0] for e in sector_entries])
    sort_idx = np.argsort(eigvals_cat)
    eigvals = eigvals_cat[sort_idx]

    eigvecs = np.zeros((n_total, len(eigvals_cat)), dtype=complex)
    block_ranges = []
    col_offset = 0
    for eigvals_block, rows, local_eigvecs in sector_entries:
        n_cols = len(eigvals_block)
        for local_i, global_i in enumerate(rows):
            eigvecs[global_i, col_offset:col_offset + n_cols] = local_eigvecs[local_i, :]
        block_ranges.append(np.arange(col_offset, col_offset + n_cols))
        col_offset += n_cols

    eigvecs = eigvecs[:, sort_idx]
    inv_sort = np.argsort(sort_idx)
    global_indices = [inv_sort[br] for br in block_ranges]

    return eigvals, eigvecs, global_indices


def reconstruct_from_sz(all_states, sz_sectors):
    """Reconstruct full spectrum from Sz-blocked eigensystems."""
    perm_per_sz = build_sz_perm(all_states, sz_sectors.states)
    entries = []
    for twoSz, sector_states, ev, evec, perm in zip(
        sz_sectors.twoSz_list, sz_sectors.states,
        sz_sectors.eigvals, sz_sectors.eigvecs, perm_per_sz
    ):
        entries.append((ev, perm, evec))
        if twoSz > 0:
            entries.append((
                ev,
                build_mirrored_rows(all_states, sector_states),
                build_mirrored_eigvecs(sector_states, evec),
            ))
    return assemble_reconstructed_eigensystem(entries, len(all_states))


def reconstruct_from_S2(all_states, sz_sectors, s2_sectors):
    """Reconstruct full spectrum from (Sz, S²)-blocked eigensystems."""
    perm_per_sz = build_sz_perm(all_states, sz_sectors.states)
    entries = []
    for idx, (twoSz, twoS, idx_sz) in enumerate(s2_sectors.sector_list):
        U = s2_sectors.transforms[idx]
        evec_sz = canonicalize_vector_phases(U @ s2_sectors.eigvecs[idx])
        entries.append((s2_sectors.eigvals[idx], perm_per_sz[idx_sz], evec_sz))
        if twoSz > 0:
            entries.append((
                s2_sectors.eigvals[idx],
                build_mirrored_rows(all_states, sz_sectors.states[idx_sz]),
                build_mirrored_eigvecs(sz_sectors.states[idx_sz], evec_sz),
            ))
    return assemble_reconstructed_eigensystem(entries, len(all_states))


# ============================================================
# S² transform construction
# ============================================================

def construct_transform_matrix(N, sz_sectors, s2_sectors):
    """Build unitary Sz -> (Sz, S²) transform.

    Mutates s2_sectors in place (fills sector_list, transforms, dimspin).
    """
    s2_sectors.sector_list = []
    s2_sectors.transforms = []
    s2_sectors.dimspin = []
    for idx, (twoSz, states, dimspin) in enumerate(
        zip(sz_sectors.twoSz_list, sz_sectors.states, sz_sectors.dimspin)
    ):
        eigvals = s2_sectors.basis_eigvals[idx]
        eigvecs = s2_sectors.basis_eigvecs[idx]

        s_list, index_groups, bad = sort_S2(eigvals)
        if bad is not None:
            raise RuntimeError(f"Failed to sort eigvals for N={N}, twoSz={twoSz}")

        for s in s_list:
            twoS = int(round(s * 2))
            s2_sectors.sector_list.append((twoSz, twoS, idx))
        s2_sectors.transforms.extend([eigvecs[:, indices] for indices in index_groups])
        s2_sectors.dimspin.extend([
            comb_safe(N, int(N * 0.5 - s)) - comb_safe(N, int(N * 0.5 - s - 1))
            for s in s_list
        ])


def find_fixed_S2_block_index(s2_sectors, target_twoSz, target_twoS):
    """Find the block index for a fixed (Sz, S) sector."""
    if s2_sectors is None or s2_sectors.sector_list is None:
        raise RuntimeError("S^2 blocks unavailable; call construct_transform_matrix first.")
    if target_twoS is None:
        raise RuntimeError("Target S sector not configured for fixed_sz_s2 mode.")

    matching = [
        (idx, twoS)
        for idx, (twoSz, twoS, _) in enumerate(s2_sectors.sector_list)
        if target_twoSz is None or twoSz == target_twoSz
    ]

    if not matching:
        raise RuntimeError(f"No S^2 blocks found for target twoSz={target_twoSz}")

    for idx, twoS in matching:
        if twoS == target_twoS:
            return idx
    raise RuntimeError(
        f"No S^2 block found for target twoS={target_twoS} "
        f"within twoSz={target_twoSz}"
    )


# ============================================================
# Diagnostics
# ============================================================

def analyze_S2_transform_columns(sz_sectors, s2_sectors, atol=ATOL["loose"],
                                  check=True, print_out=True):
    S2_cache = {}
    diag_cache = {}
    results = []

    if print_out:
        print("blk  col  twoSz_blk  sz_cal  twoS_blk  s_cal  double_occ  S2_cal")

    for blk_idx, (twoSz, twoS, idx_sz) in enumerate(s2_sectors.sector_list):
        if idx_sz not in S2_cache:
            basis = sz_sectors.states[idx_sz]
            S2_cache[idx_sz] = compute_S2_matrix(basis)
            sz_vals = np.array([calc_twoSz(state) * 0.5 for state in basis], dtype=float)
            do_vals = np.array([calc_double_occupation(state) for state in basis], dtype=float)
            diag_cache[idx_sz] = (sz_vals, do_vals)

        S2_matrix = S2_cache[idx_sz]
        sz_vals, do_vals = diag_cache[idx_sz]
        U = s2_sectors.transforms[blk_idx]
        s = twoS * 0.5
        target_S2 = s * (s + 1)

        for col in range(U.shape[1]):
            vec = U[:, col]
            S2_exp = float(np.real(np.vdot(vec, S2_matrix @ vec)))
            sz_exp = float(np.real(np.vdot(vec, sz_vals * vec)))
            do_exp = float(np.real(np.vdot(vec, do_vals * vec)))
            s_cal = round((-1.0 + np.sqrt(1.0 + 4.0 * S2_exp)) / 2.0 * 2) / 2

            if print_out:
                print(
                    f"{blk_idx:3d}  {col:3d}  {twoSz:10d}  {sz_exp:6.2f}  "
                    f"{twoS:9d}  {s_cal:5.1f}  {do_exp:10.6f}  {S2_exp:8.6f}"
                )

            if check:
                if abs(sz_exp - twoSz * 0.5) > atol:
                    raise AssertionError(
                        f"Sz mismatch in block {blk_idx}, col {col}: "
                        f"expected {twoSz * 0.5}, got {sz_exp}"
                    )
                if abs(S2_exp - target_S2) > atol:
                    raise AssertionError(
                        f"S^2 mismatch in block {blk_idx}, col {col}: "
                        f"expected {target_S2}, got {S2_exp}"
                    )

            results.append({
                "block": blk_idx, "col": col, "idx_sz": idx_sz,
                "twoS": twoS, "sz_exp": sz_exp, "S2_exp": S2_exp,
                "double_occ": do_exp,
            })

    return results


def summarize_S2_blocks(sz_sectors, s2_sectors, atol=ATOL["loose"],
                         print_out=True, print_do_values=True, print_do_sequence=False):
    S2_cache = {}
    diag_cache = {}
    summary = []

    if print_out:
        print("blk  twoSz  twoS  ncols  sz_ok  s_ok  do_unique  do_zero  dimspin  do_match  do_sorted  bad_sz  bad_s")

    for blk_idx, (twoSz, twoS, idx_sz) in enumerate(s2_sectors.sector_list):
        if idx_sz not in S2_cache:
            basis = sz_sectors.states[idx_sz]
            S2_cache[idx_sz] = compute_S2_matrix(basis)
            sz_vals = np.array([calc_twoSz(state) * 0.5 for state in basis], dtype=float)
            do_vals = np.array([calc_double_occupation(state) for state in basis], dtype=float)
            diag_cache[idx_sz] = (sz_vals, do_vals)

        S2_matrix = S2_cache[idx_sz]
        sz_vals, do_vals = diag_cache[idx_sz]
        U_mat = s2_sectors.transforms[blk_idx]
        s = twoS * 0.5

        bad_sz = bad_s = do_zero = 0
        do_set = set()
        do_seq = []
        basis = sz_sectors.states[idx_sz]
        nsites = len(basis[0]) if basis else 0

        for col in range(U_mat.shape[1]):
            vec = U_mat[:, col]
            S2_exp = float(np.real(np.vdot(vec, S2_matrix @ vec)))
            sz_exp = float(np.real(np.vdot(vec, sz_vals * vec)))
            do_exp = float(np.real(np.vdot(vec, do_vals * vec)))
            s_cal = round((-1.0 + np.sqrt(1.0 + 4.0 * S2_exp)) / 2.0 * 2) / 2

            if abs(sz_exp - twoSz * 0.5) > atol:
                bad_sz += 1
            if abs(s_cal - s) > atol or abs(S2_exp - s * (s + 1)) > atol:
                bad_s += 1

            do_round = round(do_exp)
            if abs(do_exp - do_round) <= atol:
                do_int = int(do_round)
                do_set.add(do_int)
                if do_int == 0:
                    do_zero += 1
                do_seq.append(float(do_int))
            else:
                do_set.add(round(do_exp, 6))
                do_seq.append(do_exp)

        sz_ok = (bad_sz == 0)
        s_ok = (bad_s == 0)
        do_values = sorted(do_set)
        do_sorted = all(do_seq[i] <= do_seq[i + 1] + atol for i in range(len(do_seq) - 1))
        dimspin_ref = None
        if s2_sectors.dimspin is not None and len(s2_sectors.dimspin) == len(s2_sectors.sector_list):
            dimspin_ref = s2_sectors.dimspin[blk_idx]
        if dimspin_ref is None and nsites > 0:
            dimspin_ref = comb_safe(nsites, int(nsites * 0.5 - s)) - comb_safe(nsites, int(nsites * 0.5 - s - 1))
        do_match = (dimspin_ref is None) or (do_zero == dimspin_ref)

        entry = {
            "block": blk_idx, "twoSz": twoSz, "twoS": twoS,
            "ncols": U_mat.shape[1], "sz_ok": sz_ok, "s_ok": s_ok,
            "do_unique": len(do_set), "do_values": do_values,
            "do_zero": do_zero, "dimspin": dimspin_ref,
            "do_match": do_match, "do_sorted": do_sorted,
            "do_sequence": do_seq, "bad_sz": bad_sz, "bad_s": bad_s,
        }
        summary.append(entry)

        if print_out:
            print(
                f"{blk_idx:3d}  {twoSz:6d}  {twoS:5d}  {U_mat.shape[1]:5d}  "
                f"{str(sz_ok):5s}  {str(s_ok):4s}  {len(do_set):9d}  "
                f"{do_zero:7d}  {str(dimspin_ref):7s}  {str(do_match):8s}  "
                f"{str(do_sorted):9s}  {bad_sz:6d}  {bad_s:5d}"
            )
            if print_do_values:
                print(f"     do_values: [{', '.join(str(v) for v in do_values)}]")
            if print_do_sequence:
                do_seq_text = ", ".join(
                    f"{v:.0f}" if abs(v - round(v)) <= atol else f"{v:.6f}" for v in do_seq
                )
                print(f"     do_sequence: [{do_seq_text}]")

    return summary
