"""Single-band Hubbard model — thin coordinator class.

Delegates to:
  hamiltonian.py  (02-HAMILTONIAN)
  sectors.py      (03-SYMMETRY_SECTORS)
  downfolding.py  (04-DOWNFOLDING)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cuprate import ATOL
from cuprate.states import (
    generate_states,
    generate_states_twoSz,
    is_half_filled,
    sort_by_double_occupation,
)
from cuprate import hamiltonian
from cuprate import sectors
from cuprate import downfolding


# ============================================================
# Data containers
# ============================================================

@dataclass
class SzSectors:
    """Per-Sz-sector data for block diagonalisation."""
    twoSz_list: list[int]
    states: list[list]
    dimspin: list[int]
    hams: Optional[list] = None
    eigvals: Optional[list] = None
    eigvecs: Optional[list] = None
    global_indices: Optional[list] = None


@dataclass
class S2Sectors:
    """Per-(Sz, S²)-sector data for block diagonalisation."""
    sector_list: list[tuple]       # [(twoSz, twoS, sz_sector_idx), ...]
    transforms: list               # unitary matrices per sector
    dimspin: list[int]
    basis_eigvals: Optional[list] = None
    basis_eigvecs: Optional[list] = None
    hams: Optional[list] = None
    eigvals: Optional[list] = None
    eigvecs: Optional[list] = None
    global_indices: Optional[list] = None


@dataclass
class S2Diagnostics:
    """S² expectation values and variance for all eigenstates."""
    basis_matrix: np.ndarray
    matrix: np.ndarray
    diag: np.ndarray
    error: np.ndarray


@dataclass
class DownfoldResult:
    """Effective Hamiltonian and selection diagnostics."""
    heff: np.ndarray
    t11m1: np.ndarray
    t11m1_norm: float
    selected_indices: np.ndarray
    selected_occupation: np.ndarray
    double_occ_expectation: np.ndarray
    overlap: Optional[float] = None


# ============================================================
# Hubbard model coordinator
# ============================================================

class HubbardModel:
    """Single-band Hubbard model on a finite cluster.

    Thin coordinator: delegates construction, sector decomposition,
    and downfolding to dedicated modules.
    """

    def __init__(self, N, U, t):
        self.N = N
        self.U = U
        self.t = t

        self.dimspin = 2 ** N
        self.tmp_dir = None

        # Input
        self.bonds: list = []
        self.hoppings: list = []
        self.states: list = []

        # Configuration
        self.mode: str = "full"
        self.result_kind: str = "spin_couplings"
        self.reconstruct_full: bool = False
        self.use_sz_blocks: bool = False
        self.use_S2_blocks: bool = False
        self.match_spin_sectors: bool = False
        self.select_mode: str = "full"
        self.target_twoSz: Optional[int] = None
        self.target_twoS: Optional[int] = None

        # Primary eigensystem
        self.ham: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

        # Sector data
        self.sz_sectors: Optional[SzSectors] = None
        self.S2_sectors: Optional[S2Sectors] = None

        # Results
        self.S2: Optional[S2Diagnostics] = None
        self.downfold: Optional[DownfoldResult] = None

    def clear(self):
        self.ham = None
        self.eigvals = None
        self.eigvecs = None
        self.bonds = []
        self.states = []
        self.hoppings = []
        self.dimspin = 0
        self.sz_sectors = None
        self.S2_sectors = None
        self.downfold = None
        self.S2 = None

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    def set_mode_spec(self, mode_spec):
        self.mode = mode_spec.mode
        self.result_kind = mode_spec.result_kind
        self.reconstruct_full = mode_spec.reconstruct_full
        self.target_twoSz = mode_spec.twoSz
        self.target_twoS = mode_spec.twoS
        self.use_sz_blocks = mode_spec.block_sz
        self.use_S2_blocks = mode_spec.block_S2

    def set_select_mode(self, select_mode):
        self.select_mode = select_mode if select_mode is not None else "full"

    def enable_match_spin_sectors(self):
        self.match_spin_sectors = True
        self.use_sz_blocks = True
        self.use_S2_blocks = True
        self.reconstruct_full = True

    # ----------------------------------------------------------
    # Bonds / hoppings
    # ----------------------------------------------------------

    def add_hopping_bonds(self, bonds, hopping):
        self.bonds.extend(bonds)
        self.hoppings.extend([hopping] * len(bonds))

    # ----------------------------------------------------------
    # State space
    # ----------------------------------------------------------

    def set_states(self, nsites, nelec, twoSz_set=None):
        s_values = [0, 1, -1, 2]
        full_states = sort_by_double_occupation(generate_states(nsites, nelec, s_values))

        if self.reconstruct_full or twoSz_set is None:
            self.states = full_states
        elif isinstance(twoSz_set, int):
            self.states = sort_by_double_occupation(
                generate_states_twoSz(nsites, nelec, twoSz_set, s_values)
            )
        else:
            selected = []
            for twoSz in twoSz_set:
                selected.extend(generate_states_twoSz(nsites, nelec, twoSz, s_values))
            self.states = sort_by_double_occupation(selected)

        self.dimspin = len([s for s in self.states if is_half_filled(s)])

        if self.use_sz_blocks or self.use_S2_blocks or twoSz_set is not None:
            if twoSz_set is None:
                twoSz_list = list(range(0 if nsites % 2 == 0 else 1, nsites + 1, 2))
            elif isinstance(twoSz_set, int):
                twoSz_list = [twoSz_set]
            else:
                twoSz_list = list(twoSz_set)

            sz_states = [
                sort_by_double_occupation(
                    generate_states_twoSz(nsites, nelec, twoSz, s_values)
                )
                for twoSz in twoSz_list
            ]
            sz_dimspin = [
                len([s for s in states if is_half_filled(s)])
                for states in sz_states
            ]
            self.sz_sectors = SzSectors(
                twoSz_list=twoSz_list,
                states=sz_states,
                dimspin=sz_dimspin,
            )

    # ----------------------------------------------------------
    # Hamiltonian construction  →  hamiltonian.py
    # ----------------------------------------------------------

    def calc_hamiltonian(self):
        if self.sz_sectors is None:
            self.ham = hamiltonian.build_hamiltonian(
                self.states, self.N, self.U, self.bonds, self.hoppings
            )
        else:
            sz = self.sz_sectors
            sz.hams = hamiltonian.build_hamiltonian_blocked(
                sz.states, self.N, self.U, self.bonds, self.hoppings
            )
            if self.S2_sectors is not None:
                s2s = self.S2_sectors
                s2s.hams = hamiltonian.build_hamiltonian_S2(
                    sz.hams, s2s.sector_list, s2s.transforms
                )

    # ----------------------------------------------------------
    # Diagonalisation  →  hamiltonian.py
    # ----------------------------------------------------------

    def solve(self):
        preserve_reconstructed_blocks = False
        if self.sz_sectors is None:
            self.eigvals, self.eigvecs = hamiltonian.diagonalise(self.ham)
        elif self.S2_sectors is not None:
            s2s = self.S2_sectors
            s2s.eigvals, s2s.eigvecs = hamiltonian.diagonalise_blocked(s2s.hams)
            if self.reconstruct_full:
                ev, evec, gi = sectors.reconstruct_from_S2(
                    self.states, self.sz_sectors, s2s
                )
                self.eigvals, self.eigvecs = ev, evec
                self.S2_sectors.global_indices = gi
                self.sz_sectors.global_indices = None
                preserve_reconstructed_blocks = True
            else:
                block_idx = sectors.find_fixed_S2_block_index(
                    s2s, self.target_twoSz, self.target_twoS
                )
                U = s2s.transforms[block_idx]
                self.eigvals = s2s.eigvals[block_idx]
                self.eigvecs = sectors.canonicalize_vector_phases(U @ s2s.eigvecs[block_idx])
                self.dimspin = s2s.dimspin[block_idx]
        else:
            sz = self.sz_sectors
            sz.eigvals, sz.eigvecs = hamiltonian.diagonalise_blocked(sz.hams)
            if self.reconstruct_full:
                ev, evec, gi = sectors.reconstruct_from_sz(self.states, sz)
                self.eigvals, self.eigvecs = ev, evec
                self.sz_sectors.global_indices = gi
                if self.S2_sectors is not None:
                    self.S2_sectors.global_indices = None
                preserve_reconstructed_blocks = True
            else:
                self.eigvals, self.eigvecs = hamiltonian.diagonalise(sz.hams[0])

        if preserve_reconstructed_blocks:
            self.eigvecs = sectors.canonicalize_vector_phases(self.eigvecs)
        else:
            self.eigvals, self.eigvecs = sectors.canonicalize_eigenpairs_with_S2(
                self.states,
                self.eigvals,
                self.eigvecs,
            )

    # ----------------------------------------------------------
    # S² block construction  →  sectors.py
    # ----------------------------------------------------------

    def load_blocks(self):
        if not self.use_S2_blocks:
            return
        if self.S2_sectors is not None and self.S2_sectors.basis_eigvals is not None:
            return
        sz = self.sz_sectors
        basis_eigvals = []
        basis_eigvecs = []
        for states in sz.states:
            eigvals, eigvecs = sectors.solve_S2_blocks(states)
            basis_eigvals.append(eigvals)
            basis_eigvecs.append(eigvecs)

        self.S2_sectors = S2Sectors(
            sector_list=[],
            transforms=[],
            dimspin=[],
            basis_eigvals=basis_eigvals,
            basis_eigvecs=basis_eigvecs,
        )

    # ----------------------------------------------------------
    # S² expectation values  →  sectors.py
    # ----------------------------------------------------------

    def calc_S2(self):
        basis_matrix = sectors.compute_S2_matrix(self.states)
        matrix = self.eigvecs.conj().T @ basis_matrix @ self.eigvecs
        s4_matrix = self.eigvecs.conj().T @ basis_matrix @ basis_matrix @ self.eigvecs
        diag = np.array([matrix[i][i].real for i in range(len(matrix))])
        error = np.array([s4_matrix[i][i] - matrix[i][i] ** 2 for i in range(len(matrix))])
        self.S2 = S2Diagnostics(
            basis_matrix=basis_matrix,
            matrix=matrix,
            diag=diag,
            error=error,
        )

    # ----------------------------------------------------------
    # Heff extraction  →  downfolding.py
    # ----------------------------------------------------------

    def _prepare_selection_blocks(self, params):
        double_occ = self.downfold.double_occ_expectation

        if params.match_spin_sectors:
            if self.mode not in ("full", "block_sz_full", "block_sz_s2_full"):
                raise RuntimeError(
                    "MATCH_SPIN_SECTORS is only supported for full, block_sz_full, and block_sz_s2_full."
                )
            global_indices = self.S2_sectors.global_indices
            dimspin_blocks = self._expanded_S2_dimspin()
        elif self.select_mode == "block" and self.use_S2_blocks:
            global_indices = self.S2_sectors.global_indices
            dimspin_blocks = self._expanded_S2_dimspin()
        elif self.select_mode == "block" and self.use_sz_blocks:
            global_indices = self.sz_sectors.global_indices
            dimspin_blocks = self._expanded_sz_dimspin()
        else:
            global_indices = [np.arange(len(self.eigvals))]
            dimspin_blocks = [self.dimspin]

        eigvals_blocks = [self.eigvals[idx] for idx in global_indices]
        eigvecs_blocks = [self.eigvecs[:, idx] for idx in global_indices]
        double_occ_blocks = [double_occ[idx] for idx in global_indices]

        return global_indices, eigvals_blocks, eigvecs_blocks, double_occ_blocks, dimspin_blocks

    def _expanded_sz_dimspin(self):
        sz = self.sz_sectors
        if sz.global_indices is None or sz.dimspin is None:
            return sz.dimspin
        if len(sz.global_indices) == len(sz.dimspin):
            return sz.dimspin
        expanded = []
        for twoSz, ds in zip(sz.twoSz_list, sz.dimspin):
            expanded.append(ds)
            if twoSz > 0:
                expanded.append(ds)
        if len(expanded) != len(sz.global_indices):
            raise RuntimeError(
                f"Expanded Sz dimensions mismatch: {len(expanded)} != {len(sz.global_indices)}"
            )
        return expanded

    def _expanded_S2_dimspin(self):
        sec = self.S2_sectors
        if sec.global_indices is None or sec.dimspin is None:
            return sec.dimspin
        if len(sec.global_indices) == len(sec.dimspin):
            return sec.dimspin
        expanded = []
        for (twoSz, _twoS, _idx_sz), ds in zip(sec.sector_list, sec.dimspin):
            expanded.append(ds)
            if twoSz > 0:
                expanded.append(ds)
        if len(expanded) != len(sec.global_indices):
            raise RuntimeError(
                f"Expanded Sz/S^2 dimensions mismatch: {len(expanded)} != {len(sec.global_indices)}"
            )
        return expanded

    def calc_heff_halffilled(self, params, params_cluster):
        double_occ_exp = downfolding.calc_double_occ_expectation(self.states, self.eigvecs)
        self.downfold = DownfoldResult(
            heff=None, t11m1=None, t11m1_norm=np.inf,
            selected_indices=None, selected_occupation=None,
            double_occ_expectation=double_occ_exp,
        )

        global_indices, eigvals_blocks, eigvecs_blocks, double_occ_blocks, dimspin_blocks = \
            self._prepare_selection_blocks(params)

        selected_indices, best_norm, overlap = downfolding.select_eigenstates(
            method=params.workflow,
            global_indices=global_indices,
            eigvals_blocks=eigvals_blocks,
            eigvecs_blocks=eigvecs_blocks,
            double_occ_blocks=double_occ_blocks,
            dimspin_blocks=dimspin_blocks,
            params=params,
            params_cluster=params_cluster,
            tmp_dir=self.tmp_dir,
            twoSz=self.target_twoSz,
            N=self.N,
            U=self.U,
            t=self.t,
        )

        heff, t11m1, t11m1_norm = downfolding.extract_heff(
            self.eigvals, self.eigvecs, selected_indices, self.dimspin
        )

        self.downfold = DownfoldResult(
            heff=heff,
            t11m1=t11m1,
            t11m1_norm=t11m1_norm,
            selected_indices=selected_indices,
            selected_occupation=double_occ_exp[selected_indices],
            double_occ_expectation=double_occ_exp,
            overlap=overlap,
        )

    # ----------------------------------------------------------
    # Spin-coupling fitting  →  downfolding.py
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Data I/O
    # ----------------------------------------------------------

    def save_data(self, base_filename):
        df = self.downfold
        s2 = self.S2
        np.save(f"{base_filename}_Heff.npy", np.asarray(df.heff))
        np.save(f"{base_filename}_T11m1.npy", np.asarray(df.t11m1))
        np.save(f"{base_filename}_t11_selected_indices.npy", np.asarray(df.selected_indices, dtype=int))
        np.save(f"{base_filename}_t11_selected_occupation.npy", np.asarray(df.selected_occupation))
        np.save(f"{base_filename}_double_occupation_expectation.npy", np.asarray(df.double_occ_expectation))
        np.save(f"{base_filename}_states.npy", np.asarray(self.states, dtype=int))
        np.save(f"{base_filename}_S2_diagonal.npy", np.column_stack([s2.diag, s2.error]))
        np.save(f"{base_filename}_S2_selected.npy",
                np.column_stack([s2.diag[df.selected_indices],
                                 s2.error[df.selected_indices]]))

    # ----------------------------------------------------------
    # Diagnostics  →  sectors.py
    # ----------------------------------------------------------

    def analyze_S2_transform_columns(self, atol=ATOL["loose"], check=True, print_out=True):
        if not self.use_S2_blocks:
            raise RuntimeError("S^2 block not enabled.")
        if self.S2_sectors is None or self.S2_sectors.sector_list is None:
            raise RuntimeError("S^2 transform matrices not constructed.")
        return sectors.analyze_S2_transform_columns(
            self.sz_sectors, self.S2_sectors, atol=atol, check=check, print_out=print_out
        )

    def summarize_S2_blocks(self, atol=ATOL["loose"], print_out=True,
                            print_do_values=True, print_do_sequence=False):
        if not self.use_S2_blocks:
            raise RuntimeError("S^2 block not enabled.")
        if self.S2_sectors is None or self.S2_sectors.sector_list is None:
            raise RuntimeError("S^2 transform matrices not constructed.")
        return sectors.summarize_S2_blocks(
            self.sz_sectors, self.S2_sectors, atol=atol,
            print_out=print_out, print_do_values=print_do_values,
            print_do_sequence=print_do_sequence,
        )
