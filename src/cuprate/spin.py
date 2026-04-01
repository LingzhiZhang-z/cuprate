import itertools
import numpy as np
from scipy.linalg import block_diag
from math import comb

# ============================================================
# States Utilities (copied from core/states.py to avoid circular imports)
# ============================================================
def sort_s2(values):
    s2_groups = {}
    bad_indices = []
    for idx, val in enumerate(values):
        val_real = float(np.real(val))
        s_raw = (-1.0 + np.sqrt(1.0 + 4.0 * val_real)) / 2.0
        s = round(s_raw * 2) / 2
        if abs(s * (s + 1) - val_real) > 1e-6:
            bad_indices.append(idx)
            s = s_raw
        s2_groups.setdefault(s, []).append(idx)
    s_list = sorted(s2_groups.keys())
    index_groups = [s2_groups[s] for s in s_list]
    return s_list, index_groups, (bad_indices if bad_indices else None)

def calc_double_occupation(state):
    return state.count(2)

def sort_states(state_list):
    sorted_list = sorted(state_list, key=lambda s: (abs(total_mag(s)), -total_mag(s)))
    sorted_mags = [total_mag(s) for s in sorted_list]
    counts = {mag: len(list(group)) for mag, group in itertools.groupby(sorted_mags)}
    return sorted_list, counts

def total_mag(state):
    return sum([spin for spin in state if abs(spin)==1])

def comb_safe(n, k):
    if k < 0 or k > n:
        return 0
    return comb(n, k)

def compute_S2_basis(state1, state2):
    N = len(state1)
    if state1==state2:
        s_vals = np.where(np.abs(state1) == 1, 0.5, 0.0)
        m_vals = [ 0.0 for _ in range(N)]
        for i in range(N):
            if state1[i] == 1 or state1[i] == -1:
                m_vals[i] = 0.5 * state1[i]
            else:
                m_vals[i] = 0.0

        S2_site = np.sum(s_vals * (s_vals + 1))
        S2_pair = 0.0
        for i in range(N):
            for j in range(i+1, N):
                S2_pair += m_vals[i] * m_vals[j]

        S2_total = S2_site + 2 * S2_pair
    else:
        S2_site=0
        S2_pair = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if (state1[i],state2[i]) == ( 1,-1) \
                and (state1[j],state2[j]) == (-1, 1) \
                and all(state1[k]==state2[k] for k in range(N) if k not in (i,j)):
                    S2_pair += 0.5
                if (state1[i],state2[i]) == (-1, 1) \
                and (state1[j],state2[j]) == ( 1,-1) \
                and all(state1[k]==state2[k] for k in range(N) if k not in (i,j)):
                    S2_pair += 0.5

        S2_total = S2_site + 2 * S2_pair

    return S2_total

def compute_S2_vec(basis, eigvec1, eigvec2):
    N=len(basis)
    S2=0.0j
    for i in range(N):
        for j in range(N):
            S2 += np.conjugate(eigvec1[i])*eigvec2[j]*compute_S2_basis(basis[i], basis[j])
    return S2

def compute_S2_matrix(basis):
    N=len(basis)
    basis_matrix=np.zeros([N,N], dtype=complex)
    for i in range(N):
        basis_matrix[i][i] = compute_S2_basis(basis[i], basis[i])
        for j in range(i+1, N):
            basis_matrix[i][j] = compute_S2_basis(basis[i], basis[j])
            basis_matrix[j][i] = np.conjugate(basis_matrix[i][j])
    return basis_matrix


def canonicalize_vector_phases(eigvecs, atol=1e-12):
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


def canonicalize_eigenpairs(eigvals, eigvecs, atol=1e-10, tie_breaker=None):
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

    eigvals=np.concatenate(eigvals_list)
    eigvecs = block_diag(*eigvecs_list)

    s_list, index_groups, bad = sort_s2(eigvals)
    if bad is not None:
        raise RuntimeError(f"Failed to sort eigvals for S^2 matrix")

    indices=np.concatenate(index_groups)
    return eigvals[indices], eigvecs[:, indices]


def compute_all_s2_sz_counts(N, print_out=False):
    """Compute counts for all (S, Sz) sectors for N spin-1/2 sites."""
    smax = N * 0.5
    n_half = N * 0.5

    results = []
    for nup in range(N + 1):
        ndo = N - nup
        sz = (nup - ndo) * 0.5
        dim_sz = comb_safe(N, nup)
        sz_entry = {"sz": sz, "dim_sz": dim_sz, "s2_counts": []}
        for idx_s in range(int(smax - abs(sz)) + 1):
            s = abs(sz) + idx_s
            s2 = s * (s + 1)
            dim_sz_s = comb_safe(N, int(n_half + s)) * comb_safe(N, int(n_half - s))
            dim_sz_s -= comb_safe(N, int(n_half + s + 1)) * comb_safe(N, int(n_half - s - 1))
            sz_entry["s2_counts"].append((s, s2, dim_sz_s))
        results.append(sz_entry)

    if print_out:
        print("Sz sectors:")
        for entry in results:
            print(f"  sz={entry['sz']:.1f}: dim={entry['dim_sz']}")
        print("Sz, S^2 sectors:")
        for entry in results:
            for s, s2, dim_sz_s in entry["s2_counts"]:
                print(f"  sz={entry['sz']:.1f}, s={s:.1f}, s2={s2:.1f}: dim={dim_sz_s}")

    return results

def spin_moments(states, bond):
    dimspin = len(states)
    SzSz = np.zeros((dimspin, dimspin), dtype=complex)
    for i, state1 in enumerate(states):
        SzSz[i, i] = Spin_z(state1[bond[0]]) * Spin_z(state1[bond[1]])
    return SzSz

def Spin_z(occupation):
    if occupation == 2 or occupation == 0:
        return 0
    elif occupation == 1:
        return 1/2
    elif occupation == -1:
        return -1/2
    else:
        raise ValueError(f"Invalid occupation: {occupation}")

def Spin_Flip_Plus(occupation):
    if occupation == -1:
        return 1
    elif occupation in (1, 2, 0):
        return -100
    else:
        raise ValueError(f"Invalid occupation: {occupation}")

def Spin_Flip_Minus(occupation):
    if occupation == 1:
        return -1
    elif occupation in (-1, 2, 0):
        return -100
    else:
        raise ValueError(f"Invalid occupation: {occupation}")

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
        if fp1 != -100 and fm2 != -100:
            flipped = list(state)
            flipped[site1], flipped[site2] = fp1, fm2
            j = state_index.get(tuple(flipped))
            if j is not None:
                SpSm[j, i] = 1.0

        # S-S+
        fm1, fp2 = Spin_Flip_Minus(occ1), Spin_Flip_Plus(occ2)
        if fm1 != -100 and fp2 != -100:
            flipped = list(state)
            flipped[site1], flipped[site2] = fm1, fp2
            j = state_index.get(tuple(flipped))
            if j is not None:
                SmSp[j, i] = 1.0

    return SzSz + 0.5 * (SpSm + SmSp)

def spin_matrix_JJ_ij(states, bond):
    site1, site2, site3, site4 = bond[0], bond[1], bond[2], bond[3]

    spin_matrix1=spin_matrix_J_ij(states, [site1, site2])
    spin_matrix2=spin_matrix_J_ij(states, [site3, site4])

    return spin_matrix1 @ spin_matrix2

def spin_matrix_JJJ_ij(states, bond):
    site1, site2, site3, site4, site5, site6 = bond[0], bond[1], bond[2], bond[3], bond[4], bond[5]

    spin_matrix1=spin_matrix_J_ij(states, [site1, site2])
    spin_matrix2=spin_matrix_J_ij(states, [site3, site4])
    spin_matrix3=spin_matrix_J_ij(states, [site5, site6])

    return spin_matrix1 @ spin_matrix2 @ spin_matrix3

def spin_matrix_square(states, plaquette):
    bonds_list = [
        [plaquette[0], plaquette[1], plaquette[2], plaquette[3]],
        [plaquette[0], plaquette[3], plaquette[1], plaquette[2]],
        [plaquette[0], plaquette[2], plaquette[1], plaquette[3]]
    ]
    factors = [1, 1, -1]

    dimspin = len(states)
    spin_matrix = np.zeros((dimspin, dimspin), dtype=complex)
    for idx, bonds in enumerate(bonds_list):
        spin_matrix += factors[idx] * spin_matrix_JJ_ij(states, bonds)

    return spin_matrix


def spin_matrix_square_bonds(states, plaquette):
    bonds_list = [
        [plaquette[0], plaquette[1], plaquette[2], plaquette[3]],
        [plaquette[0], plaquette[3], plaquette[1], plaquette[2]],
        [plaquette[0], plaquette[2], plaquette[1], plaquette[3]]
    ]

    spin_matrix=[]
    for bonds in bonds_list:
        spin_matrix.append(spin_matrix_JJ_ij(states, bonds))

    return spin_matrix, bonds_list
