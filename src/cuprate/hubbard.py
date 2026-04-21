"""Minimal exact diagonalization for the single-band Hubbard model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from cuprate import ATOL
from cuprate.clusters import Cluster
from cuprate.downfolding import spin_fit, spin_matrix
from cuprate.hamiltonian import build_hamiltonian as build_hub_hamiltonian
from cuprate.hamiltonian import diagonalise
from cuprate.sectors import build_S2_multiplets, build_S2_sectors, build_S2_transforms
from cuprate.states import calc_fourS2_matrix, calc_twoSz, count_double_occ, generate_states, group_states

MODE_SINGLE = "single"
MODE_BY_SZ = "by_sz"
MODE_BY_SZ_S2 = "by_sz_s2"
MODE_ONE_SZ = "one_sz"
MODE_ONE_SZ_S2 = "one_sz_s2"
MODE_ONE_SZ_BY_S2 = "one_sz_by_s2"

MODE_ALIASES = {
    "full": MODE_SINGLE,
    "fixed_sz": MODE_ONE_SZ,
    "block_sz_full": MODE_BY_SZ,
    "fixed_sz_s2": MODE_ONE_SZ_S2,
    "block_sz_s2_full": MODE_BY_SZ_S2,
    "fixed_sz_s2_all": MODE_ONE_SZ_BY_S2,
}

BUCKET_FULL = "full"
BUCKET_SZ_BLOCK = "sz_block"
BUCKET_SZ_S2_BLOCK = "sz_s2_block"

MODE_ARGS = {
    MODE_SINGLE: (False, False),
    MODE_BY_SZ: (False, False),
    MODE_BY_SZ_S2: (False, False),
    MODE_ONE_SZ: (True, False),
    MODE_ONE_SZ_S2: (True, True),
    MODE_ONE_SZ_BY_S2: (True, False),
}

MODE_BUCKETS = {
    MODE_SINGLE: BUCKET_FULL,
    MODE_BY_SZ: BUCKET_SZ_BLOCK,
    MODE_ONE_SZ: BUCKET_SZ_BLOCK,
    MODE_BY_SZ_S2: BUCKET_SZ_S2_BLOCK,
    MODE_ONE_SZ_S2: BUCKET_SZ_S2_BLOCK,
    MODE_ONE_SZ_BY_S2: BUCKET_SZ_S2_BLOCK,
}


@dataclass
class Block:
    N: int
    nelec: int
    basis_states: list[int]
    ham: np.ndarray | None
    eigvals: np.ndarray
    eigvecs: np.ndarray
    twoSz: int | None = None
    twoS: int | None = None
    basis_transform: np.ndarray | None = None

    @staticmethod
    def _fmt_value(value: int | None) -> str:
        if value is None:
            return "all"
        return f"n{-value}" if value < 0 else str(value)

    @staticmethod
    def _label(twoSz: int | None, twoS: int | None) -> str:
        return f"twoSz_{Block._fmt_value(twoSz)}_twoS_{Block._fmt_value(twoS)}"

    def label(self) -> str:
        return self._label(self.twoSz, self.twoS)

    def save(self, directory: str | Path) -> None:
        """Save block into `directory`: `{label}_data.npz` + `{label}_label.txt`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / self.label()

        arrays: dict[str, np.ndarray] = {"eigvecs": np.asarray(self.eigvecs)}
        if self.ham is not None:
            arrays["ham"] = np.asarray(self.ham)
        if self.basis_transform is not None:
            arrays["basis_transform"] = np.asarray(self.basis_transform)
        np.savez_compressed(f"{base}_data.npz", **arrays)

        twoSz_str = "all" if self.twoSz is None else str(self.twoSz)
        twoS_str = "all" if self.twoS is None else str(self.twoS)
        eigvals = np.real_if_close(np.asarray(self.eigvals))
        states = list(self.basis_states)
        state_width = max((len(str(s)) for s in states), default=1)

        header = f"{self.N} {self.nelec} {twoSz_str} {twoS_str} {len(states)} {len(eigvals)}"

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
    ) -> "Block":
        """Load block `{label}_data.npz` + `{label}_label.txt` from `directory`."""
        base = Path(directory) / cls._label(twoSz, twoS)
        data = np.load(f"{base}_data.npz")
        eigvecs = data["eigvecs"]
        ham = data["ham"] if "ham" in data.files else None
        basis_transform = data["basis_transform"] if "basis_transform" in data.files else None

        tokens = Path(f"{base}_label.txt").read_text().split()
        N = int(tokens[0])
        nelec = int(tokens[1])
        twoSz = None if tokens[2] == "all" else int(tokens[2])
        twoS = None if tokens[3] == "all" else int(tokens[3])
        n_states, n_eigvals = int(tokens[4]), int(tokens[5])
        idx = 6
        basis_states = [int(x) for x in tokens[idx:idx + n_states]]
        idx += n_states
        eigvals = np.array([float(x) for x in tokens[idx:idx + n_eigvals]])

        return cls(
            N=N, nelec=nelec, basis_states=basis_states, 
            ham=ham, eigvals=eigvals, eigvecs=eigvecs,
            twoSz=twoSz, twoS=twoS,
            basis_transform=basis_transform,
        )

    @classmethod
    def exists(
        cls,
        directory: str | Path,
        twoSz: int | None = None,
        twoS: int | None = None,
    ) -> bool:
        base = Path(directory) / cls._label(twoSz, twoS)
        return Path(f"{base}_data.npz").exists() and Path(f"{base}_label.txt").exists()

    def quantum_numbers(self) -> list[dict[str, float | int]]:
        eigvecs_fock = self.eigvecs
        twoSz_op = np.diag([calc_twoSz(state, self.N) for state in self.basis_states]).astype(complex)
        double_occ_op = np.diag([count_double_occ(state, self.N) for state in self.basis_states]).astype(complex)
        fourS2_op = calc_fourS2_matrix(self.basis_states, self.N)

        twoSz_diag = np.real(np.diag(eigvecs_fock.conj().T @ twoSz_op @ eigvecs_fock))
        double_occ_diag = np.real(np.diag(eigvecs_fock.conj().T @ double_occ_op @ eigvecs_fock))
        fourS2_diag = np.real(np.diag(eigvecs_fock.conj().T @ fourS2_op @ eigvecs_fock))

        quantum_numbers = []
        for idx in range(len(self.eigvals)):
            twoSz_value = int(np.rint(twoSz_diag[idx]))
            twoS_float = np.sqrt(max(fourS2_diag[idx] + 1.0, 0.0)) - 1.0
            twoS_value = int(np.rint(twoS_float))
            quantum_numbers.append(
                {
                    "twoSz": twoSz_value,
                    "twoS": twoS_value,
                    "D": float(double_occ_diag[idx]),
                }
            )
        return quantum_numbers

    def spin_rows(self) -> list[int]:
        return [i for i, state in enumerate(self.basis_states) if count_double_occ(state, self.N) == 0]

    def spin_basis(self) -> np.ndarray:
        """Orthonormal D=0 basis inside this block, in D=0 row coordinates."""
        spin_rows = self.spin_rows()
        dimspin = len(spin_rows)
        if self.basis_transform is None:
            return np.eye(dimspin, dtype=complex)

        U_spin = self.basis_transform[spin_rows, :]
        left, sigma, _ = np.linalg.svd(U_spin, full_matrices=False)
        keep = sigma > ATOL["tight"]
        return left[:, keep].conj().T

    def spin_dim(self) -> int:
        return self.spin_basis().shape[0]

    def downfold(self, selected_eigenstates: list[int]) -> tuple[np.ndarray, float]:
        """SVD downfold with caller-selected eigenstates → (heff, t11m1_norm).

        Caller provides `selected_eigenstates`; returns `(heff, t11m1_norm)`
        where `t11m1_norm = |U Sigma U^dagger - I|`.
        """
        spin_rows = self.spin_rows()
        spin_basis = self.spin_basis()
        S_BD = spin_basis @ self.eigvecs[np.ix_(spin_rows, selected_eigenstates)]
        U, Sigma, VH = np.linalg.svd(S_BD, full_matrices=False)
        Lambda = np.diag(self.eigvals[selected_eigenstates])
        heff = U @ VH @ Lambda @ VH.conj().T @ U.conj().T

        t11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(spin_basis.shape[0])
        return heff, float(np.linalg.norm(t11m1.flatten()))

    def _spin_operators(self, bonds: list[Sequence[int]]) -> np.ndarray:
        """Columns of [I, S_bond₁, S_bond₂, …] flattened, on this block's D=0 basis.

        `bonds` is a flat list of bond specs — grouping (shared-J equivalence) is
        the caller's concern; Block just emits one column per bond.
        """
        spin_rows = self.spin_rows()
        spin_basis = self.spin_basis()
        states = [self.basis_states[i] for i in spin_rows]
        Ms = [np.eye(spin_basis.shape[0], dtype=complex)]
        for bond in bonds:
            Ms.append(spin_basis @ spin_matrix(states, bond) @ spin_basis.conj().T)
        return np.array([m.flatten() for m in Ms]).T

@dataclass
class Spectrum:
    N: int
    nelec: int
    mode: str
    blocks: list[Block]
    cluster: Cluster | None = None

    def merge_sector_s2(self) -> "Spectrum":
        """Collapse S² sub-blocks within each Sz → one Block per Sz sector."""
        if self.mode not in (MODE_BY_SZ_S2, MODE_ONE_SZ_BY_S2):
            raise ValueError(
                f"merge_sector_s2 only applies to mode in {{{MODE_BY_SZ_S2!r}, {MODE_ONE_SZ_BY_S2!r}}}, got {self.mode!r}."
            )

        by_sz: dict[int, list[Block]] = {}
        for b in self.blocks:
            by_sz.setdefault(b.twoSz, []).append(b)

        merged: list[Block] = []
        for twoSz in sorted(by_sz):
            group = sorted(by_sz[twoSz], key=lambda b: b.twoS)
            first = group[0]
            merged.append(Block(
                N=self.N,
                nelec=self.nelec,
                basis_states=first.basis_states,
                ham=None,
                eigvals=np.concatenate([b.eigvals for b in group]),
                eigvecs=np.concatenate([b.eigvecs for b in group], axis=1),
                twoSz=twoSz,
                twoS=None,
                basis_transform=None,
            ))
        new_mode = MODE_ONE_SZ if len(merged) == 1 else MODE_BY_SZ
        return Spectrum(self.N, self.nelec, new_mode, merged, cluster=self.cluster)

    def merge_sector_sz(self) -> "Spectrum":
        """Collapse Sz blocks → one full block-diagonal Block with (twoSz=None, twoS=None)."""
        if self.mode != MODE_BY_SZ:
            raise ValueError(
                f"merge_sector_sz only applies to mode == {MODE_BY_SZ!r}, got {self.mode!r}."
            )

        blocks = sorted(self.blocks, key=lambda b: b.twoSz)

        basis_states: list[int] = []
        for b in blocks:
            basis_states.extend(b.basis_states)

        total_rows = sum(len(b.basis_states) for b in blocks)
        total_cols = sum(b.eigvecs.shape[1] for b in blocks)
        eigvecs = np.zeros((total_rows, total_cols), dtype=complex)
        row = col = 0
        for b in blocks:
            r, c = b.eigvecs.shape
            eigvecs[row:row + r, col:col + c] = b.eigvecs
            row += r
            col += c

        eigvals = np.concatenate([b.eigvals for b in blocks])

        merged = Block(
            N=self.N,
            nelec=self.nelec,
            basis_states=basis_states,
            ham=None,
            eigvals=eigvals,
            eigvecs=eigvecs,
            twoSz=None,
            twoS=None,
            basis_transform=None,
        )
        return Spectrum(self.N, self.nelec, MODE_SINGLE, [merged], cluster=self.cluster)

    def _bond_groups(self) -> list[list[Sequence[int]]]:
        """Canonical bond generation: 2-site all distance classes, 4/6-site connected."""
        if self.cluster is None:
            raise ValueError("Spectrum.project needs a cluster to generate bond groups.")
        return (
            self.cluster.generate_bonds(N=2, is_connected=False)
            + self.cluster.generate_bonds(N=4, is_connected=True)
            + self.cluster.generate_bonds(N=6, is_connected=True)
        )

    def project(
        self,
        select: str = "occ",
        **select_kwargs,
    ) -> tuple[list, list, tuple[float, float, float], list[float]]:
        """Select eigenstates per block -> SVD downfold -> LS fit to spin couplings.

        Selection method dispatch via downfolding.select. Per block: call
        block.downfold(indices) for (heff, t11m1_norm) then build the spin-operator
        matrix on the D=0 basis. Stack across blocks into one LS problem.

        Returns (coeffs, bond_groups, metrics, t11m1_norms):
          - coeffs[0]: constant offset; coeffs[1:]: per-bond-group coefficient lists
          - bond_groups: same structure used by the caller
          - metrics: (rel_err, residual, r2) from spin_fit
          - t11m1_norms: one |T11 - I| norm per block
        """
        from cuprate.downfolding import select as select_eigenstates

        bond_groups = self._bond_groups()
        bonds = [bond for group in bond_groups for bond in group]
        per_block_indices = select_eigenstates(self, select, **select_kwargs)

        A_rows: list[np.ndarray] = []
        b_rows: list[np.ndarray] = []
        t11m1_norms: list[float] = []
        for block, indices in zip(self.blocks, per_block_indices):
            heff, t11 = block.downfold(indices)
            A_rows.append(block._spin_operators(bonds))
            b_rows.append(heff.flatten())
            t11m1_norms.append(t11)

        x, metrics = spin_fit(np.vstack(A_rows), np.concatenate(b_rows))

        coeffs: list = [x[0]]
        idx = 1
        for group in bond_groups:
            coeffs.append(list(x[idx:idx + len(group)]))
            idx += len(group)
        return coeffs, bond_groups, metrics, t11m1_norms



class HubbardModel:
    def __init__(
        self,
        cluster: Cluster,
        U: float | complex,
        t: float | complex,
        nelec: int | None = None,
        hoppings: Sequence[float | complex] | None = None,
    ):
        self.cluster = cluster
        self.N = cluster.N
        self.U = complex(U)
        self.t = complex(t)
        self.nelec = self.N if nelec is None else nelec
        self.bonds = [tuple(map(int, bond)) for bond in cluster.bonds]
        self.hoppings = (
            [complex(h) for h in hoppings] if hoppings else [self.t] * len(self.bonds)
        )

    def label(self) -> str:
        def fmt(v: complex) -> str:
            x = v.real
            return f"n{-x:.12g}" if x < 0 else f"{x:.12g}"
        return f"N_{self.N}_nelec_{self.nelec}_U_{fmt(self.U)}_t_{fmt(self.t)}"

    def build_fock_basis(self, twoSz: int | None = None) -> list[int]:
        return generate_states(self.N, self.nelec, twoSz=twoSz)

    def build_hamiltonian(self, basis_states: Sequence[int]) -> np.ndarray:
        return build_hub_hamiltonian(list(basis_states), self.N, self.U, self.bonds, self.hoppings)

    def _validate_mode(self, mode: str, twoSz: int | None, twoS: int | None) -> str:
        mode = MODE_ALIASES.get(mode.lower(), mode.lower())
        if mode not in MODE_ARGS:
            raise ValueError(f"Unsupported Hubbard solver mode: {mode}")
        needs_twoSz, needs_twoS = MODE_ARGS[mode]
        if (twoSz is not None) != needs_twoSz:
            verb = "requires" if needs_twoSz else "does not accept"
            raise ValueError(f"mode={mode} {verb} twoSz")
        if (twoS is not None) != needs_twoS:
            verb = "requires" if needs_twoS else "does not accept"
            raise ValueError(f"mode={mode} {verb} twoS")
        return mode

    def _twoSz_values(self) -> range:
        lo = max(-self.nelec, self.nelec - 2 * self.N)
        hi = min(self.nelec, 2 * self.N - self.nelec)
        return range(lo, hi + 1, 2)

    def _sz_block(self, twoSz: int) -> tuple[list[int], np.ndarray]:
        basis_states = self.build_fock_basis(twoSz=twoSz)
        if not basis_states:
            raise ValueError(f"No states exist for twoSz={twoSz} at N={self.N}, nelec={self.nelec}.")
        return basis_states, self.build_hamiltonian(basis_states)

    def _s2_transforms(self) -> dict[tuple[int, int], np.ndarray]:
        """Build algebraic S² sector transforms keyed by (twoSz, twoS)."""
        grouped_states = group_states(generate_states(self.N, self.nelec), self.N)
        _, multiplets = build_S2_multiplets(grouped_states, self.N)
        sector_blocks = build_S2_sectors(grouped_states, multiplets)
        return build_S2_transforms(grouped_states, self.N, sector_blocks)

    def _diag_block(
        self,
        basis_states: list[int],
        ham: np.ndarray,
        *,
        twoSz: int | None = None,
        twoS: int | None = None,
        basis_transform: np.ndarray | None = None,
    ) -> Block:
        U = np.eye(ham.shape[0]) if basis_transform is None else basis_transform

        ham_block = U.conj().T @ ham @ U
        eigvals, eigvecs = diagonalise(ham_block)
        eigvecs = U @ eigvecs

        # NOTE: when basis_transform is not None, ham stays in the S² block basis
        # while eigvecs is projected back to the Fock (Sz-block) basis — not co-basis.
        return Block(
            self.N, self.nelec, basis_states, ham_block, eigvals, eigvecs,
            twoSz, twoS, basis_transform,
        )

    def _block_keys(self, mode, twoSz, twoS, transforms):
        if mode == MODE_SINGLE:
            return [(None, None)]
        if mode == MODE_BY_SZ:
            return [(tsz, None) for tsz in self._twoSz_values()]
        if mode == MODE_ONE_SZ:
            return [(twoSz, None)]
        if mode == MODE_ONE_SZ_S2:
            if (twoSz, twoS) not in transforms:
                raise ValueError(f"No (twoSz, twoS)=({twoSz}, {twoS}) block exists at N={self.N}, nelec={self.nelec}.")
            return [(twoSz, twoS)]
        if mode == MODE_ONE_SZ_BY_S2:
            return sorted(k for k in transforms if k[0] == twoSz)
        return sorted(transforms)  # MODE_BY_SZ_S2

    def _build_block(self, key, transforms, get_sz):
        tsz, ts = key
        if tsz is None:
            basis_states = self.build_fock_basis()
            ham = self.build_hamiltonian(basis_states)
            return self._diag_block(basis_states, ham)
        basis_states, ham = get_sz(tsz)
        if ts is None:
            return self._diag_block(basis_states, ham, twoSz=tsz)
        U = transforms[key]
        return self._diag_block(basis_states, ham, twoSz=tsz, twoS=ts, basis_transform=U)

    def solve(self, mode=MODE_SINGLE, *, twoSz=None, twoS=None, cache_dir=None):
        mode = self._validate_mode(mode, twoSz, twoS)
        bucket = MODE_BUCKETS[mode]
        transforms = self._s2_transforms() if bucket == BUCKET_SZ_S2_BLOCK else None
        keys = self._block_keys(mode, twoSz, twoS, transforms)
        sz_memo: dict[int, tuple[list[int], np.ndarray]] = {}

        def get_sz(tsz):
            if tsz not in sz_memo:
                sz_memo[tsz] = self._sz_block(tsz)
            return sz_memo[tsz]

        cache = (
            Path(cache_dir) / self.label() / self.cluster.label() / bucket
            if cache_dir else None
        )
        blocks = []
        for key in keys:
            if cache and Block.exists(cache, *key):
                blocks.append(Block.load(cache, *key))
                continue
            block = self._build_block(key, transforms, get_sz)
            if cache:
                block.save(cache)
            blocks.append(block)
        return Spectrum(self.N, self.nelec, mode, blocks, cluster=self.cluster)
