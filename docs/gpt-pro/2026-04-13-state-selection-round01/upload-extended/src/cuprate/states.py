"""Fock space state utilities for the single-band Hubbard model.

State encoding: each site holds one of [0, 1, -1, 2]
  0  = empty
  1  = spin-up
  -1 = spin-down
  2  = doubly occupied (up + down)
"""

import itertools

import numpy as np


# ============================================================
# Single-site / single-state helpers
# ============================================================

def calc_twoSz(state):
    """Twice the total z-magnetisation, 2*Sz = N_up - N_down (always integer)."""
    return sum(spin for spin in state if abs(spin) == 1)


def sum_elec(state, n):
    """Number of electrons on sites 0..n-1."""
    return sum(abs(state[i]) for i in range(n))


def is_half_filled(state):
    """True if every site is singly occupied (no double occ, no empty)."""
    return all(s in (-1, 1) for s in state)


def calc_double_occupation(state):
    """Count of doubly-occupied sites."""
    return state.count(2)


def sign_fermi(n):
    """Fermionic sign (-1)^n."""
    return 1 if n % 2 == 0 else -1


def sign_state(state, index):
    """Fermionic sign from electrons to the left of *index*."""
    return sign_fermi(sum_elec(state, index))


# ============================================================
# State-space generators
# ============================================================

def generate_all_states(n, s_values=None):
    """All product states on *n* sites.

    *s_values* defaults to [0, 1, -1, 2] (full Fock space).
    """
    values = [0, 1, -1, 2] if s_values is None else list(s_values)
    return list(itertools.product(values, repeat=n))


def generate_states(nsites, nelec, s_values=None):
    """States with a fixed electron number *nelec*."""
    if s_values is None:
        s_values = [0, 1, -1, 2]
    states = generate_all_states(nsites, s_values)
    return [s for s in states if sum_elec(s, nsites) == nelec]


def generate_states_twoSz(nsites, nelec, twoSz, s_values=None):
    """States with fixed electron number *nelec* and fixed 2*Sz = *twoSz* (integer)."""
    if s_values is None:
        s_values = [0, 1, -1, 2]
    states = generate_all_states(nsites, s_values)
    return [
        s for s in states
        if sum_elec(s, nsites) == nelec
        and calc_twoSz(s) == twoSz
    ]


# ============================================================
# State sorting
# ============================================================

def sort_by_twoSz(state_list):
    """Sort states by |2*Sz| ascending, then 2*Sz descending.

    Returns (sorted_list, counts_dict).
    """
    sorted_list = sorted(state_list, key=lambda s: (abs(calc_twoSz(s)), -calc_twoSz(s)))
    sorted_mags = [calc_twoSz(s) for s in sorted_list]
    counts = {mag: len(list(group)) for mag, group in itertools.groupby(sorted_mags)}
    return sorted_list, counts


def sort_by_double_occupation(states):
    """Sort states first by double-occupation count, then by 2*Sz."""
    states_by_double = {}
    for state in states:
        d = calc_double_occupation(state)
        states_by_double.setdefault(d, []).append(state)

    sorted_states = []
    for d in sorted(states_by_double.keys()):
        group, _ = sort_by_twoSz(states_by_double[d])
        sorted_states.extend(group)
    return sorted_states


# ============================================================
# Double-occupation matrix
# ============================================================

def calc_double_occupation_matrix(states):
    """Diagonal matrix of double-occupation counts."""
    diag = [calc_double_occupation(s) for s in states]
    return np.diag(diag)
