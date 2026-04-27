"""Analytical perturbation theory for half-filled bipartite Hubbard model.

Reference (intent): Headings, Hayden, Coldea, Perring,
"Magnon Spectra of Cuprates beyond Spin Wave Theory" — fill in the exact
coefficients from the paper here. The plot script imports these functions
unchanged; edit this file to update the reference curves.

Conventions:
- t, U > 0; J's returned in units of energy (same as t, U).
- All formulas use the standard 4th-order Hubbard-to-spin expansion at
  half filling. Replace with the paper's exact values when available.
"""

from __future__ import annotations


def J_NN(t: float, U: float) -> float:
    """Nearest-neighbor (vector (1,0)) Heisenberg coupling.

    Standard: J = 4t^2/U - 24 t^4/U^3 (4th-order correction).
    """
    return 4.0 * t**2 / U - 24.0 * t**4 / U**3


def J_NNN(t: float, U: float) -> float:
    """Diagonal NNN (vector (1,1)). Pure 4th order."""
    return 4.0 * t**4 / U**3


def J_3rd(t: float, U: float) -> float:
    """3rd-NN axial (vector (2,0)). Pure 4th order."""
    return 4.0 * t**4 / U**3


def Jc_edge(t: float, U: float) -> float:
    """Plaquette ring exchange — parallel-edge pairings.

    Both edge pairings ((S0 S1)(S2 S3) "vertical" and (S0 S2)(S1 S3)
    "horizontal") have equal magnitude and same sign (C4 symmetry).
    Standard ring-exchange coefficient: K = 80 t^4/U^3 (Takahashi 1977,
    MacDonald-Girvin-Yoshioka 1988).
    """
    return 80.0 * t**4 / U**3


def Jc_diagonal(t: float, U: float) -> float:
    """Plaquette ring exchange — diagonal "cross" pairing ((S0 S3)(S1 S2)).

    Same |K| as edges, opposite sign in the standard convention:
        H_ring = K [(S0S1)(S2S3) + (S0S2)(S1S3) - (S0S3)(S1S2)]
    Per the user, cuprate shows a small finite-t correction making the
    diagonal magnitude slightly different from the edge.
    """
    return -80.0 * t**4 / U**3
