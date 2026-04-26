"""Fock-space state utilities for the single-band Hubbard model.

Bit encoding used throughout this file:
- one many-body state is one Python int
- site i occupies two bits
- bit 2*i     = n_{i,up}
- bit 2*i + 1 = n_{i,down}

So the local site code is:
- 0 = 00 = empty
- 1 = 01 = up
- 2 = 10 = down
- 3 = 11 = double

Code-level consequences:
- site_code(...) reads one local two-bit code
- set_site(...) overwrites one local two-bit code
- calc_twoSz(...) counts up bits minus down bits
- count_double_occ(...) counts sites whose two local bits are both 1
- calc_S_plus_matrix(...) / calc_S_minus_matrix(...) flip 2 <-> 1 inside fixed-many-body bases
"""

from __future__ import annotations

import itertools
from functools import reduce
from operator import matmul

import numpy as np


# site_code 1 = up (Sz=+1/2), 2 = down (Sz=-1/2) on a D=0 site
_SZ = {1: 0.5, 2: -0.5}


def up_mask(N: int) -> int:
    return ((1 << (2 * N)) - 1) // 3


def down_mask(N: int) -> int:
    return up_mask(N) << 1


def count_electrons(state: int) -> int:
    return state.bit_count()


def count_double_occ(state: int, N: int) -> int:
    return (state & ((state >> 1) & up_mask(N))).bit_count()


def calc_twoSz(state: int, N: int) -> int:
    return (state & up_mask(N)).bit_count() - (state & down_mask(N)).bit_count()


def calc_Sz(state: int, N: int) -> float:
    return 0.5 * calc_twoSz(state, N)


def spin_flip_state(state: int, N: int) -> int:
    return ((state & up_mask(N)) << 1) | ((state & down_mask(N)) >> 1)


def is_pure_spin_state(state: int, N: int) -> bool:
    return count_electrons(state) == N and count_double_occ(state, N) == 0


def site_code(state: int, i: int) -> int:
    return (state >> (2 * i)) & 3


def set_site(state: int, i: int, code: int) -> int:
    mask = 3 << (2 * i)
    return (state & ~mask) | (code << (2 * i))


def sign_below(state: int, bit_idx: int) -> int:
    return 1 if (state & ((1 << bit_idx) - 1)).bit_count() % 2 == 0 else -1


def apply_hop(state: int, src: int, dst: int, spin: str) -> tuple[int | None, int]:
    """Apply c†_{dst,σ} c_{src,σ} to `state`; return (new_state, sign) or (None, 0)."""
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


def _calc_site_S_plus_state(state: int, i: int) -> int | None:
    return set_site(state, i, 1) if site_code(state, i) == 2 else None


def _calc_site_S_minus_state(state: int, i: int) -> int | None:
    return set_site(state, i, 2) if site_code(state, i) == 1 else None


def _calc_spin_ladder_matrix(states_src: list[int], states_dst: list[int], N: int, lowering: bool) -> np.ndarray:
    state_to_row = {state: idx for idx, state in enumerate(states_dst)}
    matrix = np.zeros((len(states_dst), len(states_src)), dtype=complex)
    for col, state in enumerate(states_src):
        for i in range(N):
            moved_state = (
                _calc_site_S_minus_state(state, i) if lowering else _calc_site_S_plus_state(state, i)
            )
            if moved_state is None:
                continue
            row = state_to_row.get(moved_state)
            if row is not None:
                matrix[row, col] += 1.0
    return matrix


def calc_S_plus_matrix(states_src: list[int], states_dst: list[int], N: int) -> np.ndarray:
    return _calc_spin_ladder_matrix(states_src, states_dst, N, lowering=False)


def calc_S_minus_matrix(states_src: list[int], states_dst: list[int], N: int) -> np.ndarray:
    return _calc_spin_ladder_matrix(states_src, states_dst, N, lowering=True)


def pure_spin_state_indices(states: list[int], N: int) -> list[int]:
    return [idx for idx, state in enumerate(states) if is_pure_spin_state(state, N)]


def calc_double_occupation_matrix(states: list[int], N: int) -> np.ndarray:
    diag = [count_double_occ(state, N) for state in states]
    return np.diag(diag)


def calc_eta2_matrix_direct(states: list[int], N: int, signs) -> np.ndarray:
    """Build eta2 directly in one fixed half-filled Fock basis."""
    if len(signs) != N:
        raise ValueError(f"eta signs length {len(signs)} does not match N={N}")

    state_to_row = {state: row for row, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=complex)

    for col, state in enumerate(states):
        empty_sites = []
        doublon_sites = []
        for site in range(N):
            code = site_code(state, site)
            if code == 0:
                empty_sites.append(site)
            elif code == 3:
                doublon_sites.append(site)

        matrix[col, col] += len(empty_sites)
        for empty_site in empty_sites:
            for doublon_site in doublon_sites:
                moved_state = set_site(
                    set_site(state, empty_site, 3),
                    doublon_site,
                    0,
                )
                matrix[state_to_row[moved_state], col] += (
                    int(signs[empty_site]) * int(signs[doublon_site])
                )
    return matrix


def calc_eta_plus_matrix(states_src: list[int], states_dst: list[int], N: int, signs) -> np.ndarray:
    """Build eta+ between two Fock bases with the same twoSz."""
    if len(signs) != N:
        raise ValueError(f"eta signs length {len(signs)} does not match N={N}")

    state_to_row = {state: row for row, state in enumerate(states_dst)}
    matrix = np.zeros((len(states_dst), len(states_src)), dtype=complex)
    for col, state in enumerate(states_src):
        for site in range(N):
            if site_code(state, site) != 0:
                continue
            row = state_to_row.get(set_site(state, site, 3))
            if row is not None:
                matrix[row, col] += int(signs[site])
    return matrix


def _site_twoSz_projection(code: int) -> int:
    if code == 1:
        return 1
    if code == 2:
        return -1
    return 0


def _calc_diagonal_fourS2_element(state: int, N: int) -> int:
    projections = [_site_twoSz_projection(site_code(state, i)) for i in range(N)]
    local_term = 3 * sum(1 for projection in projections if projection != 0)
    pair_term = 0
    for i in range(N):
        for j in range(i + 1, N):
            pair_term += projections[i] * projections[j]
    return local_term + 2 * pair_term


def _is_spin_exchange(code1_i: int, code2_i: int, code1_j: int, code2_j: int) -> bool:
    return (code1_i, code2_i, code1_j, code2_j) in (
        (1, 2, 2, 1),
        (2, 1, 1, 2),
    )


def _calc_offdiagonal_fourS2_element(state1: int, state2: int, N: int) -> int:
    differing_sites = [i for i in range(N) if site_code(state1, i) != site_code(state2, i)]
    if len(differing_sites) != 2:
        return 0

    i, j = differing_sites
    return 4 if _is_spin_exchange(
        site_code(state1, i),
        site_code(state2, i),
        site_code(state1, j),
        site_code(state2, j),
    ) else 0


def calc_fourS2_element(state1: int, state2: int, N: int) -> int:
    if state1 == state2:
        return _calc_diagonal_fourS2_element(state1, N)
    return _calc_offdiagonal_fourS2_element(state1, state2, N)


def calc_fourS2_matrix(states: list[int], N: int) -> np.ndarray:
    dim = len(states)
    fourS2_matrix = np.zeros([dim, dim], dtype=complex)
    for i in range(dim):
        fourS2_matrix[i][i] = calc_fourS2_element(states[i], states[i], N)
        for j in range(i + 1, dim):
            fourS2_matrix[i][j] = calc_fourS2_element(states[i], states[j], N)
            fourS2_matrix[j][i] = np.conjugate(fourS2_matrix[i][j])
    return fourS2_matrix


def calc_fourS2(state: int, N: int) -> int:
    return _calc_diagonal_fourS2_element(state, N)


def _spin_pair(states: list[int], i: int, j: int) -> np.ndarray:
    """S_i · S_j on the singly occupied (D=0) Fock basis `states`."""
    dim = len(states)
    index = {state: idx for idx, state in enumerate(states)}
    matrix = np.zeros((dim, dim), dtype=complex)

    for idx, state in enumerate(states):
        code_i = site_code(state, i)
        code_j = site_code(state, j)
        matrix[idx, idx] = _SZ[code_i] * _SZ[code_j]
        if code_i == 2 and code_j == 1:
            matrix[index[set_site(set_site(state, i, 1), j, 2)], idx] = 0.5
        elif code_i == 1 and code_j == 2:
            matrix[index[set_site(set_site(state, i, 2), j, 1)], idx] = 0.5
    return matrix


def spin_matrix(states: list[int], bond) -> np.ndarray:
    """Product of S·S pair operators over consecutive site pairs in `bond`."""
    pairs = zip(bond[::2], bond[1::2])
    return reduce(matmul, (_spin_pair(states, i, j) for i, j in pairs))


def sort_states(state_list: list[int], N: int) -> list[int]:
    return sorted(
        state_list,
        key=lambda state: (calc_twoSz(state, N), count_double_occ(state, N), state),
    )


def generate_all_states(N: int) -> list[int]:
    return list(range(1 << (2 * N)))


def _generate_spin_occupation_masks(N: int, nocc: int, offset: int) -> list[int]:
    masks = []
    for occupied_sites in itertools.combinations(range(N), nocc):
        mask = 0
        for site in occupied_sites:
            mask |= 1 << (2 * site + offset)
        masks.append(mask)
    return masks


def generate_states(N: int, nelec: int, twoSz: int | None = None) -> list[int]:
    """Generate Fock states in canonical order (twoSz, D, state)."""
    if twoSz is None:
        states = []
        nup_min = max(0, nelec - N)
        nup_max = min(N, nelec)
        for nup in range(nup_min, nup_max + 1):
            ndown = nelec - nup
            up_masks = _generate_spin_occupation_masks(N, nup, 0)
            down_masks = _generate_spin_occupation_masks(N, ndown, 1)
            states.extend(up | down for up in up_masks for down in down_masks)
        return sort_states(states, N)

    if (nelec + twoSz) % 2 != 0:
        return []
    nup = (nelec + twoSz) // 2
    ndown = nelec - nup
    if nup < 0 or nup > N or ndown < 0 or ndown > N:
        return []
    up_masks = _generate_spin_occupation_masks(N, nup, 0)
    down_masks = _generate_spin_occupation_masks(N, ndown, 1)
    states = [up | down for up in up_masks for down in down_masks]
    return sort_states(states, N)


def group_states(states: list[int], N: int) -> dict[tuple[int, int], list[int]]:
    groups = {}
    for state in states:
        twoSz = calc_twoSz(state, N)
        double_occ = count_double_occ(state, N)
        groups.setdefault((twoSz, double_occ), []).append(state)
    return groups
