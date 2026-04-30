"""State-manifold objects and operations built on solved eigensystems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.linalg

from cuprate import ATOL
from cuprate.paths import block_token
from cuprate.states import (
    calc_double_occupation_matrix,
    calc_fourS2_matrix,
    calc_twoSz,
    count_double_occ,
    spin_matrix,
)


__all__ = [
    "Block",
]


@dataclass
class Block:
    N: int
    nelec: int
    basis_states: list[int]
    ham: np.ndarray | None = None
    eigvals: np.ndarray | None = None
    eigvecs: np.ndarray | None = None
    twoSz: int | None = None
    twoS: int | None = None
    eta: int | None = None
    basis_transform: np.ndarray | None = None

    @staticmethod
    def _label(
        twoSz: int | None,
        twoS: int | None,
        eta: int | None = None,
    ) -> str:
        return block_token(twoSz, twoS, eta)

    def label(self) -> str:
        return self._label(self.twoSz, self.twoS, self.eta)

    def _error_context(self) -> str:
        return (
            f"block={self.label()} N={self.N} nelec={self.nelec} "
            f"twoSz={self.twoSz} twoS={self.twoS} eta={self.eta}"
        )

    def save(self, directory: str | Path) -> None:
        """Save block into `directory`: `{label}_data.npz` + `{label}_label.txt`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / self.label()

        arrays: dict[str, np.ndarray] = {}
        if self.eigvecs is not None:
            arrays["eigvecs"] = np.asarray(self.eigvecs)
        if self.basis_transform is not None:
            arrays["basis_transform"] = np.asarray(self.basis_transform)
        np.savez_compressed(f"{base}_data.npz", **arrays)

        states = list(self.basis_states)
        state_width = max((len(str(s)) for s in states), default=1)
        twoSz_str = "all" if self.twoSz is None else str(self.twoSz)
        twoS_str = "all" if self.twoS is None else str(self.twoS)
        eta_str = "all" if self.eta is None else str(self.eta)
        eigvals = np.real_if_close(np.asarray(self.eigvals)) if self.eigvals is not None else np.array([])

        header = f"{self.N} {self.nelec} {twoSz_str} {twoS_str} {eta_str} {len(states)} {len(eigvals)}"

        lines = [header]
        for i in range(0, len(states), 10):
            lines.append(" ".join(f"{s:>{state_width}d}" for s in states[i:i + 10]))
        lines.append("")
        for i in range(0, len(eigvals), 10):
            lines.append(" ".join(f"{float(v):24.15e}" for v in eigvals[i:i + 10]))
        Path(f"{base}_label.txt").write_text("\n".join(lines) + "\n")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        twoSz: int | None = None,
        twoS: int | None = None,
        eta: int | None = None,
    ) -> "Block":
        """Load block `{label}_data.npz` + `{label}_label.txt` from `directory`."""
        requested_twoSz = twoSz
        requested_twoS = twoS
        requested_eta = eta
        base = Path(directory) / cls._label(requested_twoSz, requested_twoS, requested_eta)
        corrupted = f"Cached block is corrupted: {base}"

        try:
            with np.load(f"{base}_data.npz") as data:
                eigvecs = data["eigvecs"] if "eigvecs" in data.files else None
                basis_transform = data["basis_transform"] if "basis_transform" in data.files else None

            text = Path(f"{base}_label.txt").read_text()
            lines = text.splitlines()
            header_tokens = lines[0].split()
            # Old header: N nelec twoSz twoS n_states n_eigvals          (6 tokens)
            # New header: N nelec twoSz twoS eta n_states n_eigvals      (7 tokens)
            if len(header_tokens) == 6:
                loaded_eta = None
                n_states = int(header_tokens[4])
                n_eigvals = int(header_tokens[5])
            elif len(header_tokens) == 7:
                loaded_eta = None if header_tokens[4] == "all" else int(header_tokens[4])
                n_states = int(header_tokens[5])
                n_eigvals = int(header_tokens[6])
            else:
                raise ValueError
            N = int(header_tokens[0])
            nelec = int(header_tokens[1])
            loaded_twoSz = None if header_tokens[2] == "all" else int(header_tokens[2])
            loaded_twoS = None if header_tokens[3] == "all" else int(header_tokens[3])
            loaded_label = cls._label(loaded_twoSz, loaded_twoS, loaded_eta)
            requested_label = cls._label(requested_twoSz, requested_twoS, requested_eta)
            body = " ".join(lines[1:]).split()
            basis_states = [int(x) for x in body[:n_states]]
            eigvals = (
                np.array([float(x) for x in body[n_states:n_states + n_eigvals]])
                if n_eigvals > 0 else None
            )
        except (OSError, KeyError, IndexError, ValueError):
            raise ValueError(corrupted) from None

        if loaded_label != requested_label or not _loaded_eigensystem_is_valid(
            eigvals=eigvals,
            eigvecs=eigvecs,
            basis_transform=basis_transform,
            n_states=len(basis_states),
            n_eigvals=n_eigvals,
        ):
            raise ValueError(corrupted)

        return cls(
            N=N,
            nelec=nelec,
            basis_states=basis_states,
            ham=None,
            eigvals=eigvals,
            eigvecs=eigvecs,
            twoSz=loaded_twoSz,
            twoS=loaded_twoS,
            eta=loaded_eta,
            basis_transform=basis_transform,
        )

    @classmethod
    def exists(
        cls,
        directory: str | Path,
        twoSz: int | None = None,
        twoS: int | None = None,
        eta: int | None = None,
    ) -> bool:
        base = Path(directory) / cls._label(twoSz, twoS, eta)
        return Path(f"{base}_data.npz").exists() and Path(f"{base}_label.txt").exists()

    def eigenstate_twoSz(self) -> np.ndarray:
        """Per-eigenstate 2·<Sz> (rounded to int)."""
        ev = self.eigvecs_fock
        op = np.diag([calc_twoSz(state, self.N) for state in self.basis_states]).astype(complex)
        diag = np.real(np.diag(ev.conj().T @ op @ ev))
        return np.rint(diag).astype(int)

    def eigenstate_twoS(self) -> np.ndarray:
        """Per-eigenstate 2·S derived from ⟨4·S²⟩ = (2S+1)² − 1 (rounded to int)."""
        ev = self.eigvecs_fock
        op = calc_fourS2_matrix(self.basis_states, self.N)
        diag = np.real(np.diag(ev.conj().T @ op @ ev))
        return np.rint(np.sqrt(np.maximum(diag + 1.0, 0.0)) - 1.0).astype(int)

    def eigenstate_D(self) -> np.ndarray:
        """Per-eigenstate ⟨double-occupation count⟩ (not rounded)."""
        ev = self.eigvecs_fock
        op = np.diag([count_double_occ(state, self.N) for state in self.basis_states]).astype(complex)
        return np.real(np.diag(ev.conj().T @ op @ ev))

    def set_hamiltonian(self, ham: np.ndarray) -> None:
        if self.basis_transform is None:
            self.ham = ham
        else:
            self.ham = self.basis_transform.conj().T @ ham @ self.basis_transform

    def solve(self, *, eigh: str = "lowmem") -> None:
        if self.ham is None:
            raise RuntimeError("Block.solve(): ham is None; call build_hamiltonians() on the model first")
        # ev is the low-workspace path; evd is faster but needs larger workspace.
        driver = "evd" if eigh == "fast" else "ev"
        ham = np.asfortranarray(self.ham)
        try:
            eigvals, eigvecs = scipy.linalg.eigh(
                ham,
                driver=driver,
                overwrite_a=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Diagonalization failed for {self._error_context()} "
                f"ham_shape={ham.shape} eigh={eigh}"
            ) from exc
        self.ham = None
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    @property
    def eigvecs_fock(self) -> np.ndarray:
        """Eigvecs lifted into Fock-basis row coordinates."""
        if self.basis_transform is None:
            return self.eigvecs
        return self.basis_transform @ self.eigvecs

    def spin_fock_rows(self, D=0) -> list[int]:
        return [i for i, state in enumerate(self.basis_states) if count_double_occ(state, self.N) == D]

    def spin_sector_columns(self, D=0) -> list[int]:
        spin_fock_rows = self.spin_fock_rows(D)
        if self.basis_transform is None:
            return spin_fock_rows
        U_spin = self.basis_transform[spin_fock_rows, :]
        keep = np.linalg.norm(U_spin, axis=0) > ATOL["tight"]
        return list(np.where(keep)[0])

    @property
    def spin_dim(self) -> int:
        """Dimension of the spin (D=0) subspace in this block's basis."""
        return len(self.spin_sector_columns())

    def t11_norm(self, selected: Sequence[int]) -> float:
        # ||U Σ U† - I||_F = ||σ - 1||_2 by unitary invariance of Frobenius norm.
        spin_sector_columns = self.spin_sector_columns()
        if len(selected) != len(spin_sector_columns):
            raise ValueError(
                f"t11_norm needs exactly spin_dim={len(spin_sector_columns)} selected "
                f"eigenstates, got {len(selected)}."
            )
        s_bd = self.eigvecs[np.ix_(spin_sector_columns, selected)]
        try:
            sigma = np.linalg.svd(s_bd, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"T11 SVD failed for {self._error_context()} "
                f"spin_dim={len(spin_sector_columns)} selected={list(map(int, selected))}"
            ) from exc
        return float(np.linalg.norm(sigma - 1.0))

    def double_occ_expectation(self) -> np.ndarray:
        dom = calc_double_occupation_matrix(self.basis_states, self.N)
        eigvecs = self.eigvecs_fock
        return np.real(np.diag(eigvecs.conj().T @ dom @ eigvecs))

    def selected_occ(self) -> list[int]:
        """Select spin_dim eigenstates with lowest double-occupation expectation."""
        spin_dim = self.spin_dim
        double_occ = self.double_occ_expectation()
        n = len(self.eigvals)
        order = np.lexsort((np.arange(n), np.real(self.eigvals), np.real(double_occ)))
        return order[:spin_dim].tolist()

    def selected_energy(self) -> list[int]:
        """Select spin_dim lowest-energy eigenstates, using occupation as tie-breaker."""
        spin_dim = self.spin_dim
        double_occ = self.double_occ_expectation()
        n = len(self.eigvals)
        order = np.lexsort((np.arange(n), np.real(double_occ), np.real(self.eigvals)))
        return order[:spin_dim].tolist()

    def _selection_pool(self, ratio: int):
        double_occ = self.double_occ_expectation()
        eigvals = self.eigvals
        sorted_indices = _argsort_double_occ(double_occ, eigvals)
        dimspin = self.spin_dim
        size_space = min(ratio * dimspin, len(double_occ))
        selected = sorted_indices[:dimspin].copy()
        candidates = sorted_indices[dimspin:size_space].copy()
        return double_occ, eigvals, selected, candidates

    def _greedy_swap(self, selected, candidates):
        # Cache spin_sector_columns and the spin-projected eigvec slice once;
        # score each trial via ||σ - 1|| with compute_uv=False to skip the
        # full SVD reconstruction. Batched / Gram-based scoring was tried and
        # was slower for spin_dim ≲ 32 (fancy-indexing + eigvalsh dispatch
        # overhead dominated). Stick with the simple per-trial SVD path.
        selected = selected.copy()
        candidates = candidates.copy()
        spin_cols = self.spin_sector_columns()
        eigvecs_spin = self.eigvecs[spin_cols, :]
        if not np.all(np.isfinite(eigvecs_spin)):
            raise RuntimeError(
                f"Greedy selection failed for {self._error_context()}: "
                "spin-projected eigvecs contain non-finite values"
            )

        svd_failures = 0
        first_svd_failure = None

        def _norm(sel, stage: str, selected_pos=None, candidate_pos=None) -> float | None:
            nonlocal svd_failures, first_svd_failure
            try:
                sigma = np.linalg.svd(eigvecs_spin[:, sel], compute_uv=False)
            except np.linalg.LinAlgError:
                svd_failures += 1
                if first_svd_failure is None:
                    first_svd_failure = {
                        "stage": stage,
                        "selected_pos": None if selected_pos is None else int(selected_pos),
                        "candidate_pos": None if candidate_pos is None else int(candidate_pos),
                    }
                return None
            return float(np.linalg.norm(sigma - 1.0))

        best_norm = _norm(selected, "initial")

        for selected_pos in range(len(selected)):
            best_candidate_pos = None
            best_candidate_norm = best_norm
            for candidate_pos in range(len(candidates)):
                selected[selected_pos], candidates[candidate_pos] = (
                    candidates[candidate_pos],
                    selected[selected_pos],
                )
                norm = _norm(selected, "swap", selected_pos, candidate_pos)
                selected[selected_pos], candidates[candidate_pos] = (
                    candidates[candidate_pos],
                    selected[selected_pos],
                )
                if norm is None:
                    continue
                if best_candidate_norm is None or norm < best_candidate_norm:
                    best_candidate_norm = norm
                    best_candidate_pos = candidate_pos

            if best_candidate_pos is not None:
                selected[selected_pos], candidates[best_candidate_pos] = (
                    candidates[best_candidate_pos],
                    selected[selected_pos],
                )
                best_norm = best_candidate_norm

        if best_norm is None:
            raise RuntimeError(
                f"Greedy selection failed for {self._error_context()}: "
                f"no valid SVD result after {svd_failures} failed SVD attempt(s)"
            )

        diagnostics = {
            "svd_failures": svd_failures,
            "first_svd_failure": first_svd_failure,
        }
        return selected, candidates, best_norm, diagnostics

    def selected_greedy(
        self,
        *,
        ratio: int = 5,
        selection_info_path=None,
        info_callback=None,
        return_info: bool = False,
    ):
        _, _, selected, candidates = self._selection_pool(ratio)
        try:
            initial_norm = self.t11_norm(selected)
        except RuntimeError as exc:
            initial_norm = None
            initial_t11_error = str(exc)
        else:
            initial_t11_error = None
        selected, _, best_norm, greedy_diagnostics = self._greedy_swap(selected, candidates)
        selected = selected.tolist()
        info = {
            "method": "greedy",
            "block": self.label(),
            "initial_norm": None if initial_norm is None else float(initial_norm),
            "best_norm": float(best_norm),
            "improved": None if initial_norm is None else best_norm < initial_norm,
            "initial_t11_error": initial_t11_error,
            **greedy_diagnostics,
        }
        record = {
            "block": self.label(),
            "event": "selection_block",
            "method": "greedy",
            "selected": selected,
            "info": info,
        }
        if info_callback is not None:
            info_callback(record)
        if selection_info_path is not None:
            selection_info_path = Path(selection_info_path)
            selection_info_path.parent.mkdir(parents=True, exist_ok=True)
            with selection_info_path.open("a") as info_file:
                info_file.write(json.dumps(record) + "\n")
                info_file.flush()
        if return_info:
            return selected, info
        return selected

    def selected_greedy_multi(
        self,
        *,
        ratio: int = 8,
        rand_frac: float = 0.10,
        ratio_rand_swap: int = 2,
        n_trials: int = 40,
        max_failures: int | None = 4,
        selection_info_path=None,
        info_callback=None,
        return_info: bool = False,
    ):
        double_occ, eigvals, selected, candidates = self._selection_pool(ratio)
        best_selected = None
        best_candidates = None
        best_norm = None
        svd_failures = 0
        svd_failed_trials = 0
        first_svd_failure = None
        try:
            best_selected, best_candidates, best_norm, greedy_diagnostics = self._greedy_swap(
                selected,
                candidates,
            )
            svd_failures += int(greedy_diagnostics["svd_failures"])
            first_svd_failure = greedy_diagnostics["first_svd_failure"]
        except RuntimeError as exc:
            initial_svd_error = str(exc)
        else:
            initial_svd_error = None
        initial_norm = None if best_norm is None else float(best_norm)
        failures = 0
        if max_failures is None:
            max_failures = n_trials

        trials = []
        stopped_by = "n_trials"
        for trial in range(n_trials):
            base_selected = selected if best_selected is None else best_selected
            base_candidates = candidates if best_candidates is None else best_candidates
            trial_selected, trial_candidates = _randomize_tail(
                base_selected,
                base_candidates,
                double_occ,
                eigvals,
                rand_frac,
                ratio_rand_swap,
            )
            try:
                cur_selected, cur_candidates, cur_norm, greedy_diagnostics = self._greedy_swap(
                    trial_selected,
                    trial_candidates,
                )
            except RuntimeError:
                svd_failed_trials += 1
                trial_info = {
                    "trial": trial + 1,
                    "norm": None,
                    "best_norm": None if best_norm is None else float(best_norm),
                    "improved": False,
                    "failures": failures,
                    "svd_failed": True,
                }
                trials.append(trial_info)
                record = {
                    "block": self.label(),
                    "event": "greedy_multi_trial",
                    "method": "greedy_multi",
                    **trial_info,
                }
                if info_callback is not None:
                    info_callback(record)
                if selection_info_path is not None:
                    selection_info_path = Path(selection_info_path)
                    selection_info_path.parent.mkdir(parents=True, exist_ok=True)
                    with selection_info_path.open("a") as info_file:
                        info_file.write(json.dumps(record) + "\n")
                        info_file.flush()
                continue
            svd_failures += int(greedy_diagnostics["svd_failures"])
            if first_svd_failure is None:
                first_svd_failure = greedy_diagnostics["first_svd_failure"]

            improved = best_norm is None or cur_norm < best_norm
            if improved:
                best_norm = cur_norm
                best_selected = cur_selected.copy()
                best_candidates = cur_candidates.copy()
                failures = 0
            else:
                failures += 1
                if failures >= max_failures:
                    stopped_by = "max_failures"
            trial_info = {
                "trial": trial + 1,
                "norm": float(cur_norm),
                "best_norm": float(best_norm),
                "improved": improved,
                "failures": failures,
                "svd_failed": False,
                "svd_failures": int(greedy_diagnostics["svd_failures"]),
            }
            trials.append(trial_info)
            record = {
                "block": self.label(),
                "event": "greedy_multi_trial",
                "method": "greedy_multi",
                **trial_info,
            }
            if info_callback is not None:
                info_callback(record)
            if selection_info_path is not None:
                selection_info_path = Path(selection_info_path)
                selection_info_path.parent.mkdir(parents=True, exist_ok=True)
                with selection_info_path.open("a") as info_file:
                    info_file.write(json.dumps(record) + "\n")
                    info_file.flush()
            if stopped_by == "max_failures":
                break

        if best_selected is None or best_candidates is None or best_norm is None:
            detail = f"; initial_error={initial_svd_error}" if initial_svd_error else ""
            raise RuntimeError(
                f"Greedy multi selection failed for {self._error_context()}: "
                f"no valid SVD result after initial pass and {len(trials)} trial(s){detail}"
            )

        selected_double_occ = double_occ[best_selected]
        sorted_by_occ = _argsort_double_occ(selected_double_occ, eigvals[best_selected])
        selected = best_selected[sorted_by_occ].tolist()
        info = {
            "method": "greedy_multi",
            "block": self.label(),
            "initial_norm": initial_norm,
            "best_norm": float(best_norm),
            "n_trials": n_trials,
            "n_trials_run": len(trials),
            "max_failures": max_failures,
            "stopped_by": stopped_by,
            "svd_failures": svd_failures,
            "svd_failed_trials": svd_failed_trials,
            "first_svd_failure": first_svd_failure,
            "trials": trials,
        }
        record = {
            "block": self.label(),
            "event": "selection_block",
            "method": "greedy_multi",
            "selected": selected,
            "info": info,
        }
        if info_callback is not None:
            info_callback(record)
        if selection_info_path is not None:
            selection_info_path = Path(selection_info_path)
            selection_info_path.parent.mkdir(parents=True, exist_ok=True)
            with selection_info_path.open("a") as info_file:
                info_file.write(json.dumps(record) + "\n")
                info_file.flush()
        if return_info:
            return selected, info
        return selected

    def selected_adiabatic(
        self,
        eigvecs_previous: np.ndarray,
        selected_previous: Sequence[int],
        return_info: bool = False,
    ):
        selected_previous = np.asarray(selected_previous, dtype=int)
        eigvecs_previous = eigvecs_previous[:, selected_previous]
        norms_matrix = np.abs(eigvecs_previous.conj().T @ self.eigvecs_fock) ** 2
        norms = np.sum(norms_matrix, axis=0)
        selected = np.argsort(norms)[-self.spin_dim:].tolist()
        info = {
            "t11m1_norm": self.t11_norm(selected),
            "overlap": float(np.sum(norms[selected])),
        }
        if return_info:
            return selected, info
        return selected

    def selected(self, method: str = "occ", return_info: bool = False, **kwargs):
        """Select eigenstates for this block and return their column indices."""
        if method == "occ":
            selected = self.selected_occ()
            info = {"t11m1_norm": self.t11_norm(selected), "overlap": None}
        elif method == "energy":
            selected = self.selected_energy()
            info = {"t11m1_norm": self.t11_norm(selected), "overlap": None}
        elif method == "greedy":
            selected, info = self.selected_greedy(return_info=True, **kwargs)
        elif method == "greedy_multi":
            selected, info = self.selected_greedy_multi(return_info=True, **kwargs)
        elif method == "adiabatic":
            selected, info = self.selected_adiabatic(return_info=True, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        if return_info:
            return selected, info
        return selected

    def downfold(self, selected_eigenstates: list[int]) -> tuple[np.ndarray, float]:
        """SVD downfold with caller-selected eigenstates -> (Heff, |T11 - I|)."""
        selected_eigenstates = [int(idx) for idx in selected_eigenstates]

        spin_sector_columns = self.spin_sector_columns()
        if len(spin_sector_columns) != len(selected_eigenstates):
            raise ValueError(
                f"downfold needs exactly spin_dim={len(spin_sector_columns)} selected "
                f"eigenstates, got {len(selected_eigenstates)}."
            )

        s_bd = self.eigvecs[np.ix_(spin_sector_columns, selected_eigenstates)]
        try:
            U, sigma, vh = np.linalg.svd(s_bd, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Downfold SVD failed for {self._error_context()} "
                f"spin_dim={len(spin_sector_columns)} selected={selected_eigenstates}"
            ) from exc
        lam = np.diag(self.eigvals[selected_eigenstates])
        heff = U @ vh @ lam @ vh.conj().T @ U.conj().T

        t11m1 = U @ np.diag(sigma) @ U.conj().T - np.eye(s_bd.shape[0])
        return heff, float(np.linalg.norm(t11m1.flatten()))

    def _spin_operators(self, bonds: list[Sequence[int]]) -> np.ndarray:
        """Columns of [I, S_bond1, S_bond2, ...] flattened on this block's D=0 basis."""
        spin_fock_rows = self.spin_fock_rows()
        spin_sector_columns = self.spin_sector_columns()
        if self.basis_transform is None:
            U_spin = np.eye(len(spin_fock_rows), dtype=np.float64)
        else:
            U_spin = self.basis_transform[np.ix_(spin_fock_rows, spin_sector_columns)]

        states = [self.basis_states[i] for i in spin_fock_rows]
        matrices = [np.eye(len(spin_sector_columns), dtype=np.float64)]
        for bond in bonds:
            matrices.append(U_spin.conj().T @ spin_matrix(states, bond) @ U_spin)
        return np.array([matrix.flatten() for matrix in matrices]).T


def _loaded_eigensystem_is_valid(
    *,
    eigvals: np.ndarray | None,
    eigvecs: np.ndarray | None,
    basis_transform: np.ndarray | None,
    n_states: int,
    n_eigvals: int,
) -> bool:
    try:
        if eigvecs is None or eigvals is None:
            return False
        if eigvecs.ndim != 2 or eigvecs.shape[0] != eigvecs.shape[1]:
            return False
        n = eigvecs.shape[0]
        if eigvecs.shape != (n, n) or eigvals.shape != (n,) or n_eigvals != n:
            return False
        if basis_transform is not None and basis_transform.shape != (n_states, n):
            return False
        for array in (eigvecs, eigvals, basis_transform):
            if array is not None and not np.all(np.isfinite(array)):
                return False

        gram = eigvecs.conj().T @ eigvecs
        identity = np.eye(n, dtype=gram.dtype)
        ortho_scale = max(1.0, float(n))
        if float(np.linalg.norm(gram - identity)) / ortho_scale > ATOL["loose"]:
            return False
        return True
    except (TypeError, ValueError, FloatingPointError):
        return False


def _argsort_double_occ(double_occ, eigvals=None):
    secondary = np.real(eigvals) if eigvals is not None else np.arange(len(double_occ), dtype=float)
    return np.lexsort((np.arange(len(double_occ), dtype=int), secondary, np.real(double_occ)))


def _randomize_tail(selected, candidates, double_occ, eigvals, rand_frac, ratio_rand_swap):
    n_random = max(8, int(len(selected) * rand_frac))
    if n_random <= 0 or len(selected) < n_random or len(candidates) < n_random:
        return selected.copy(), candidates.copy()

    selected = selected[_argsort_double_occ(double_occ[selected], eigvals[selected])].copy()
    candidates = candidates[_argsort_double_occ(double_occ[candidates], eigvals[candidates])].copy()

    n_candidate = min(n_random * ratio_rand_swap, len(candidates))
    pool = np.concatenate([selected[-n_random:], candidates[:n_candidate]])
    chosen = np.random.choice(len(pool), size=n_random, replace=False)
    remaining = np.setdiff1d(np.arange(len(pool)), chosen)

    trial_selected = selected.copy()
    trial_candidates = candidates.copy()
    trial_selected[-n_random:] = pool[chosen]
    trial_candidates[:n_candidate] = pool[remaining]
    return trial_selected, trial_candidates
