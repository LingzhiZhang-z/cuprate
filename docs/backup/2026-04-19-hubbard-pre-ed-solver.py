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
    calc_fourS2_matrix,
    count_double_occ,
    generate_states,
    group_states,
    pure_spin_state_indices,
    spin_flip_state,
)
from cuprate.sectors import (
    build_S2_multiplets,
    build_S2_sectors,
    build_S2_transforms,
)
from cuprate import hamiltonian
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
    pure_spin_indices: list[np.ndarray]
    hams: Optional[list] = None
    eigvals: Optional[list] = None
    eigvecs: Optional[list] = None
    global_indices: Optional[list] = None
    spin_bases: Optional[list[np.ndarray]] = None


@dataclass
class S2Sectors:
    """Per-(Sz, S²)-sector data for block diagonalisation."""
    sector_list: list[tuple]       # [(twoSz, twoS, sz_sector_index), ...]
    transforms: list               # unitary matrices per sector
    double_occ_eigvals: list[np.ndarray]  # one D eigenvalue per sector column
    dimspin: list[int]
    hams: Optional[list] = None
    eigvals: Optional[list] = None
    eigvecs: Optional[list] = None
    global_indices: Optional[list] = None
    spin_bases: Optional[list[np.ndarray]] = None


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
        self.nelec: Optional[int] = None
        self.U = U
        self.t = t

        self.dimspin = 2 ** N
        self.tmp_dir = None

        # Input
        self.bonds: list = []
        self.hoppings: list = []
        self.states: list = []
        self.states_spin: Optional[list[int]] = None
        self.pure_spin_indices: Optional[np.ndarray] = None
        self.spin_basis: Optional[np.ndarray] = None

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
        self.nelec = None
        self.bonds = []
        self.states = []
        self.states_spin = None
        self.pure_spin_indices = None
        self.spin_basis = None
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

    def _selector_matrix(self, row_indices, nrows):
        basis = np.zeros((len(row_indices), nrows), dtype=complex)
        basis[np.arange(len(row_indices), dtype=int), row_indices] = 1.0
        return basis

    def _state_to_global(self):
        return {state: i for i, state in enumerate(self.states)}

    def _rows_from_states(self, state_to_global, states_local):
        return [state_to_global[state] for state in states_local]

    def _mirrored_rows(self, state_to_global, states_local):
        return [state_to_global[spin_flip_state(state, self.N)] for state in states_local]

    def _mirrored_vectors(self, states_local, local_vectors):
        signs = np.array([(-1) ** count_double_occ(state, self.N) for state in states_local], dtype=float)
        return signs[:, np.newaxis] * local_vectors

    def _update_spin_basis_from_states(self):
        indices = np.asarray(pure_spin_state_indices(self.states, self.N), dtype=int)
        self.pure_spin_indices = indices
        self.states_spin = [self.states[idx] for idx in indices]
        self.dimspin = len(indices)
        self.spin_basis = self._selector_matrix(indices, len(self.states))

    def _s2_spin_basis_columns(self, transform, double_occ_eigvals, dimspin):
        spin_cols = np.flatnonzero(np.abs(double_occ_eigvals) <= ATOL["loose"])
        if len(spin_cols) != dimspin:
            raise RuntimeError(
                f"Expected {dimspin} zero-double-occupation columns, got {len(spin_cols)}"
            )
        return transform[:, spin_cols]

    def _build_sz_spin_bases(self):
        sz = self.sz_sectors
        if sz is None:
            return
        if self.reconstruct_full:
            state_to_global = self._state_to_global()
            spin_bases = []
            for twoSz, states_local, pure_idx in zip(sz.twoSz_list, sz.states, sz.pure_spin_indices):
                rows = np.asarray(self._rows_from_states(state_to_global, states_local), dtype=int)[pure_idx]
                spin_bases.append(self._selector_matrix(rows, len(self.states)))
                if twoSz > 0:
                    pure_states_local = [states_local[i] for i in pure_idx]
                    mirrored_rows = np.asarray(self._mirrored_rows(state_to_global, pure_states_local), dtype=int)
                    spin_bases.append(self._selector_matrix(mirrored_rows, len(self.states)))
            sz.spin_bases = spin_bases
        else:
            sz.spin_bases = [
                self._selector_matrix(pure_idx, len(states_local))
                for states_local, pure_idx in zip(sz.states, sz.pure_spin_indices)
            ]

    def _build_s2_spin_bases(self):
        s2s = self.S2_sectors
        sz = self.sz_sectors
        if s2s is None or sz is None:
            return
        if not self.reconstruct_full:
            s2s.spin_bases = None
            return

        state_to_global = self._state_to_global()
        spin_bases = []
        for blk_idx, (twoSz, _twoS, sz_sector_index) in enumerate(s2s.sector_list):
            basis_local = self._s2_spin_basis_columns(
                s2s.transforms[blk_idx],
                s2s.double_occ_eigvals[blk_idx],
                s2s.dimspin[blk_idx],
            )
            basis_global = np.zeros((len(self.states), basis_local.shape[1]), dtype=complex)
            basis_global[np.asarray(self._rows_from_states(state_to_global, sz.states[sz_sector_index]), dtype=int), :] = basis_local
            spin_bases.append(basis_global.conj().T)
            if twoSz > 0:
                mirrored_rows = np.asarray(self._mirrored_rows(state_to_global, sz.states[sz_sector_index]), dtype=int)
                mirrored_basis = self._mirrored_vectors(sz.states[sz_sector_index], basis_local)
                basis_global_mirror = np.zeros((len(self.states), basis_local.shape[1]), dtype=complex)
                basis_global_mirror[mirrored_rows, :] = mirrored_basis
                spin_bases.append(basis_global_mirror.conj().T)
        s2s.spin_bases = spin_bases

    def set_states(self, nsites, nelec, twoSz_set=None):
        self.nelec = nelec
        full_states = generate_states(nsites, nelec)

        if self.reconstruct_full or twoSz_set is None:
            self.states = full_states
        elif isinstance(twoSz_set, int):
            self.states = generate_states(nsites, nelec, twoSz=twoSz_set)
        else:
            selected = []
            for twoSz in sorted(twoSz_set):
                selected.extend(generate_states(nsites, nelec, twoSz=twoSz))
            self.states = selected

        self._update_spin_basis_from_states()

        if self.use_sz_blocks or self.use_S2_blocks or twoSz_set is not None:
            if self.use_S2_blocks:
                twoSz_list = list(range(0 if nsites % 2 == 0 else 1, nsites + 1, 2))
            elif twoSz_set is None:
                twoSz_list = list(range(0 if nsites % 2 == 0 else 1, nsites + 1, 2))
            elif isinstance(twoSz_set, int):
                twoSz_list = [twoSz_set]
            else:
                twoSz_list = list(twoSz_set)

            twoSz_states = [
                generate_states(nsites, nelec, twoSz=twoSz)
                for twoSz in twoSz_list
            ]
            self.sz_sectors = SzSectors(
                twoSz_list=twoSz_list,
                states=twoSz_states,
                dimspin=[len(pure_spin_state_indices(states, nsites)) for states in twoSz_states],
                pure_spin_indices=[
                    np.asarray(pure_spin_state_indices(states, nsites), dtype=int)
                    for states in twoSz_states
                ],
            )
            self._build_sz_spin_bases()

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
                s2s.hams = [None] * len(s2s.sector_list)
                for idx, sector_spec in enumerate(s2s.sector_list):
                    sz_sector_index = sector_spec[-1]
                    U = s2s.transforms[idx]
                    s2s.hams[idx] = U.conj().T @ sz.hams[sz_sector_index] @ U

    # ----------------------------------------------------------
    # Diagonalisation  →  hamiltonian.py
    # ----------------------------------------------------------

    def solve(self):
        if self.sz_sectors is None:
            self.eigvals, self.eigvecs = hamiltonian.diagonalise(self.ham)
        elif self.S2_sectors is not None:
            s2s = self.S2_sectors
            s2s.eigvals, s2s.eigvecs = hamiltonian.diagonalise_blocked(s2s.hams)
            if self.reconstruct_full:
                ev, evec, gi = self._reconstruct_from_S2()
                self.eigvals, self.eigvecs = ev, evec
                self.S2_sectors.global_indices = gi
                self.sz_sectors.global_indices = None
            else:
                block_idx = self._find_fixed_S2_block_index(
                    self.target_twoSz, self.target_twoS
                )
                U = s2s.transforms[block_idx]
                self.eigvals = s2s.eigvals[block_idx]
                self.eigvecs = U @ s2s.eigvecs[block_idx]
                self.dimspin = s2s.dimspin[block_idx]
                self.spin_basis = self._s2_spin_basis_columns(
                    U,
                    s2s.double_occ_eigvals[block_idx],
                    self.dimspin,
                ).conj().T
                self.states_spin = None
                self.pure_spin_indices = None
        else:
            sz = self.sz_sectors
            sz.eigvals, sz.eigvecs = hamiltonian.diagonalise_blocked(sz.hams)
            if self.reconstruct_full:
                ev, evec, gi = self._reconstruct_from_sz()
                self.eigvals, self.eigvecs = ev, evec
                self.sz_sectors.global_indices = gi
                if self.S2_sectors is not None:
                    self.S2_sectors.global_indices = None
            else:
                self.eigvals, self.eigvecs = sz.eigvals[0], sz.eigvecs[0]

        if self.reconstruct_full and self.sz_sectors is not None:
            self._build_sz_spin_bases()
        if self.reconstruct_full and self.S2_sectors is not None:
            self._build_s2_spin_bases()

    # ----------------------------------------------------------
    # S² block construction  →  sectors.py
    # ----------------------------------------------------------

    def load_blocks(self):
        if not self.use_S2_blocks:
            return
        if self.S2_sectors is not None and self.S2_sectors.transforms:
            return

        sz = self.sz_sectors
        sz_states_union = []
        for block in sz.states:
            sz_states_union.extend(block)
        grouped = group_states(sz_states_union, self.N)

        _hw, multiplets = build_S2_multiplets(grouped, self.N)
        sector_blocks = build_S2_sectors(grouped, multiplets)
        transformers = build_S2_transforms(grouped, self.N, sector_blocks)

        # Per (twoSz, twoS), reconstruct column D labels in the same column
        # order that build_S2_transforms stacks: ascending D within each key.
        D_per_key: dict[tuple[int, int], list[int]] = {}
        for twoSz, twoS, D, _states_block, coeff in sector_blocks:
            D_per_key.setdefault((twoSz, twoS), []).extend([D] * coeff.shape[1])

        twoSz_to_sz_idx = {tz: i for i, tz in enumerate(sz.twoSz_list)}

        sector_list = []
        transforms = []
        double_occ_eigvals = []
        dimspin_list = []

        for (twoSz, twoS) in sorted(transformers.keys()):
            if twoSz not in twoSz_to_sz_idx:
                continue
            D_vec = np.asarray(D_per_key[(twoSz, twoS)], dtype=float)
            sector_list.append((twoSz, twoS, twoSz_to_sz_idx[twoSz]))
            transforms.append(transformers[(twoSz, twoS)])
            double_occ_eigvals.append(D_vec)
            dimspin_list.append(int(np.sum(np.abs(D_vec) <= ATOL["loose"])))

        self.S2_sectors = S2Sectors(
            sector_list=sector_list,
            transforms=transforms,
            double_occ_eigvals=double_occ_eigvals,
            dimspin=dimspin_list,
            spin_bases=None,
        )

    def _find_fixed_S2_block_index(self, target_twoSz, target_twoS):
        for i, (tz, ts, _) in enumerate(self.S2_sectors.sector_list):
            if tz == target_twoSz and ts == target_twoS:
                return i
        raise KeyError(
            f"No S² sector for (twoSz={target_twoSz}, twoS={target_twoS})"
        )

    def _reconstruct_from_sz(self):
        sz = self.sz_sectors
        state_to_global = self._state_to_global()

        all_eigvals = []
        all_eigvecs = []
        global_indices = []
        col_offset = 0
        dim_full = len(self.states)

        for idx, twoSz in enumerate(sz.twoSz_list):
            states_local = sz.states[idx]
            ev = sz.eigvals[idx]
            evec = sz.eigvecs[idx]
            dim_local = len(states_local)

            rows = np.asarray(self._rows_from_states(state_to_global, states_local), dtype=int)
            evec_full = np.zeros((dim_full, dim_local), dtype=complex)
            evec_full[rows, :] = evec
            all_eigvals.append(ev)
            all_eigvecs.append(evec_full)
            global_indices.append(np.arange(col_offset, col_offset + dim_local))
            col_offset += dim_local

            if twoSz > 0:
                mirrored_rows = np.asarray(self._mirrored_rows(state_to_global, states_local), dtype=int)
                evec_mirrored = np.zeros((dim_full, dim_local), dtype=complex)
                evec_mirrored[mirrored_rows, :] = self._mirrored_vectors(states_local, evec)
                all_eigvals.append(ev)
                all_eigvecs.append(evec_mirrored)
                global_indices.append(np.arange(col_offset, col_offset + dim_local))
                col_offset += dim_local

        return (
            np.concatenate(all_eigvals),
            np.concatenate(all_eigvecs, axis=1),
            global_indices,
        )

    def _reconstruct_from_S2(self):
        sz = self.sz_sectors
        s2s = self.S2_sectors
        state_to_global = self._state_to_global()

        all_eigvals = []
        all_eigvecs = []
        global_indices = []
        col_offset = 0
        dim_full = len(self.states)

        for blk_idx, (twoSz, _twoS, sz_idx) in enumerate(s2s.sector_list):
            states_local = sz.states[sz_idx]
            U = s2s.transforms[blk_idx]
            ev = s2s.eigvals[blk_idx]
            evec = s2s.eigvecs[blk_idx]
            evec_sz_basis = U @ evec
            dim_sector = evec.shape[1]

            rows = np.asarray(self._rows_from_states(state_to_global, states_local), dtype=int)
            evec_full = np.zeros((dim_full, dim_sector), dtype=complex)
            evec_full[rows, :] = evec_sz_basis
            all_eigvals.append(ev)
            all_eigvecs.append(evec_full)
            global_indices.append(np.arange(col_offset, col_offset + dim_sector))
            col_offset += dim_sector

            if twoSz > 0:
                mirrored_rows = np.asarray(self._mirrored_rows(state_to_global, states_local), dtype=int)
                evec_mirrored = np.zeros((dim_full, dim_sector), dtype=complex)
                evec_mirrored[mirrored_rows, :] = self._mirrored_vectors(states_local, evec_sz_basis)
                all_eigvals.append(ev)
                all_eigvecs.append(evec_mirrored)
                global_indices.append(np.arange(col_offset, col_offset + dim_sector))
                col_offset += dim_sector

        return (
            np.concatenate(all_eigvals),
            np.concatenate(all_eigvecs, axis=1),
            global_indices,
        )

    # ----------------------------------------------------------
    # S² expectation values  →  states.py
    # ----------------------------------------------------------

    def calc_S2(self):
        basis_fourS2 = calc_fourS2_matrix(self.states, self.N)
        matrix_fourS2 = self.eigvecs.conj().T @ basis_fourS2 @ self.eigvecs
        square_fourS2 = self.eigvecs.conj().T @ basis_fourS2 @ basis_fourS2 @ self.eigvecs
        basis_matrix = 0.25 * basis_fourS2
        matrix = 0.25 * matrix_fourS2
        diag = np.array([matrix[i][i].real for i in range(len(matrix))])
        error = 0.0625 * np.array(
            [square_fourS2[i][i] - matrix_fourS2[i][i] ** 2 for i in range(len(matrix_fourS2))]
        )
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
            spin_basis_blocks = self.S2_sectors.spin_bases
        elif self.select_mode == "block" and self.use_S2_blocks:
            global_indices = self.S2_sectors.global_indices
            dimspin_blocks = self._expanded_S2_dimspin()
            spin_basis_blocks = self.S2_sectors.spin_bases
        elif self.select_mode == "block" and self.use_sz_blocks:
            global_indices = self.sz_sectors.global_indices
            dimspin_blocks = self._expanded_sz_dimspin()
            spin_basis_blocks = self.sz_sectors.spin_bases
        else:
            global_indices = [np.arange(len(self.eigvals))]
            dimspin_blocks = [self.dimspin]
            spin_basis_blocks = [self.spin_basis]

        eigvals_blocks = [self.eigvals[idx] for idx in global_indices]
        eigvecs_blocks = [self.eigvecs[:, idx] for idx in global_indices]
        double_occ_blocks = [double_occ[idx] for idx in global_indices]

        return (
            global_indices,
            eigvals_blocks,
            eigvecs_blocks,
            double_occ_blocks,
            dimspin_blocks,
            spin_basis_blocks,
        )

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
        double_occ_exp = downfolding.calc_double_occ_expectation(self.states, self.eigvecs, self.N)
        self.downfold = DownfoldResult(
            heff=None, t11m1=None, t11m1_norm=np.inf,
            selected_indices=None, selected_occupation=None,
            double_occ_expectation=double_occ_exp,
        )

        global_indices, eigvals_blocks, eigvecs_blocks, double_occ_blocks, dimspin_blocks, spin_basis_blocks = \
            self._prepare_selection_blocks(params)

        selected_indices, best_norm, overlap = downfolding.select_eigenstates(
            method=params.workflow,
            global_indices=global_indices,
            eigvals_blocks=eigvals_blocks,
            eigvecs_blocks=eigvecs_blocks,
            double_occ_blocks=double_occ_blocks,
            dimspin_blocks=dimspin_blocks,
            spin_basis_blocks=spin_basis_blocks,
            params=params,
            params_cluster=params_cluster,
            tmp_dir=self.tmp_dir,
            twoSz=self.target_twoSz,
            N=self.N,
            U=self.U,
            t=self.t,
        )

        heff, t11m1, t11m1_norm = downfolding.extract_heff(
            self.eigvals, self.eigvecs, selected_indices, self.spin_basis
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
        np.save(
            f"{base_filename}_states.npy",
            np.asarray(self.states, dtype=int),
        )
        np.save(f"{base_filename}_S2_diagonal.npy", np.column_stack([s2.diag, s2.error]))
        np.save(f"{base_filename}_S2_selected.npy",
                np.column_stack([s2.diag[df.selected_indices],
                                 s2.error[df.selected_indices]]))
