"""Single-band Hubbard model construction and exact diagonalisation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import block_diag

from cuprate.clusters import Cluster
from cuprate.manifold import Block
from cuprate.paths import SCOPE_NONNEGATIVE, SCOPE_PM, ModeSpec, mode_spec
from cuprate.sectors import build_S2_multiplets, build_S2_sectors, build_S2_transforms
from cuprate.states import apply_hop, count_double_occ, generate_states, group_states


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
        self.selected_indices: list[list[int]] | None = None
        self.selection_info: list[dict] | None = None
        self.heff: list[np.ndarray] | None = None
        self.t11m1_norms: list[float] | None = None
        self.bond_groups: list[list[Sequence[int]]] | None = None
        self.coupling_coeffs: list | None = None
        self.fit_metrics: tuple[float, float, float] | None = None

    def build_fock_basis(self, twoSz: int | None = None) -> list[int]:
        return generate_states(self.N, self.nelec, twoSz=twoSz)

    def _build_hamiltonian_t(self, basis_states: Sequence[int]) -> np.ndarray:
        """Build the hopping matrix H_t on `basis_states`."""
        states = list(basis_states)
        state_to_idx = {state: idx for idx, state in enumerate(states)}
        dim = len(states)
        H_t = np.zeros((dim, dim), dtype=complex)
        for col, state in enumerate(states):
            for (site_i, site_j), hopping in zip(self.bonds, self.hoppings):
                for spin in ("up", "down"):
                    moved_state, sign = apply_hop(state, src=site_j, dst=site_i, spin=spin)
                    if moved_state is not None:
                        row = state_to_idx.get(moved_state)
                        if row is not None:
                            H_t[row, col] += -hopping * sign

                    moved_state, sign = apply_hop(state, src=site_i, dst=site_j, spin=spin)
                    if moved_state is not None:
                        row = state_to_idx.get(moved_state)
                        if row is not None:
                            H_t[row, col] += -hopping.conjugate() * sign
        return H_t

    def _build_hamiltonian_U(self, basis_states: Sequence[int]) -> np.ndarray:
        """Build the diagonal interaction matrix H_U on `basis_states`."""
        diag = [self.U * count_double_occ(state, self.N) for state in basis_states]
        return np.diag(np.asarray(diag, dtype=complex))

    def build_hamiltonian(self, basis_states: Sequence[int]) -> np.ndarray:
        """Build the full Hubbard Hamiltonian matrix on `basis_states`."""
        return self._build_hamiltonian_t(basis_states) + self._build_hamiltonian_U(basis_states)

    def _validate_mode(
        self,
        mode: str,
        twoSz: int | None,
        twoS: int | None,
        scope: str | None,
    ) -> ModeSpec:
        spec = mode_spec(mode, twoSz=twoSz, twoS=twoS, scope=scope)
        if spec.twoSz is not None:
            if abs(spec.twoSz) > self.N:
                raise ValueError("twoSz must satisfy |twoSz| <= N")
            if spec.twoSz % 2 != self.nelec % 2:
                raise ValueError("twoSz parity must match nelec")
        if spec.twoS is not None:
            if not (0 <= spec.twoS <= self.N):
                raise ValueError("twoS must satisfy 0 <= twoS <= N")
            if spec.twoS % 2 != self.nelec % 2:
                raise ValueError("twoS parity must match nelec")
            if abs(spec.twoSz) > spec.twoS:
                raise ValueError("twoSz and twoS must satisfy |twoSz| <= twoS")
        return spec

    def _twoSz_values(self) -> range:
        lo = max(-self.nelec, self.nelec - 2 * self.N)
        hi = min(self.nelec, 2 * self.N - self.nelec)
        return range(lo, hi + 1, 2)

    def _s2_transforms(self) -> dict[tuple[int, int], np.ndarray]:
        """Build algebraic S² sector transforms keyed by (twoSz, twoS)."""
        grouped_states = group_states(generate_states(self.N, self.nelec), self.N)
        _, multiplets = build_S2_multiplets(grouped_states, self.N)
        sector_blocks = build_S2_sectors(grouped_states, multiplets)
        return build_S2_transforms(grouped_states, self.N, sector_blocks)

    def _make_sector(
        self,
        twoSz: int | None,
        twoS: int | None,
        transforms: dict[tuple[int, int], np.ndarray] | None,
    ) -> Block:
        basis_states = self.build_fock_basis(twoSz=twoSz)
        if not basis_states:
            raise ValueError(
                f"No states exist for twoSz={twoSz} at N={self.N}, nelec={self.nelec}."
            )
        U = transforms[(twoSz, twoS)] if twoS is not None else None
        return Block(
            N=self.N,
            nelec=self.nelec,
            basis_states=basis_states,
            basis_transform=U,
            twoSz=twoSz,
            twoS=twoS,
        )

    def set_symmetry(
        self,
        mode: str,
        *,
        twoSz: int | None = None,
        twoS: int | None = None,
        scope: str | None = SCOPE_NONNEGATIVE,
    ) -> "HubbardModel":
        spec = self._validate_mode(mode, twoSz, twoS, scope)
        sz_scope, s2_scope = spec.twoSz_scope, spec.twoS_scope
        transforms = self._s2_transforms() if s2_scope != "none" else None
        all_twoSz = list(self._twoSz_values())
        scoped_twoSz = all_twoSz if spec.scope == SCOPE_PM else [value for value in all_twoSz if value >= 0]

        sz_iter = {
            "none": [None],
            "one": [spec.twoSz],
            "all": scoped_twoSz,
        }[sz_scope]

        if s2_scope == "none":
            keys = [(tsz, None) for tsz in sz_iter]
        elif s2_scope == "one":
            for tsz in sz_iter:
                if (tsz, spec.twoS) not in transforms:
                    raise ValueError(
                        f"No (twoSz, twoS)=({tsz}, {spec.twoS}) block exists "
                        f"at N={self.N}, nelec={self.nelec}."
                    )
            keys = [(tsz, spec.twoS) for tsz in sz_iter]
        else:
            sz_set = set(sz_iter)
            keys = sorted(k for k in transforms if k[0] in sz_set)

        self.blocks = [self._make_sector(tsz, ts, transforms) for tsz, ts in keys]
        self.mode = spec.mode
        self._mode_spec = spec
        self._transforms = transforms
        return self

    def build_hamiltonians(self) -> "HubbardModel":
        """Build and assign the Hamiltonian for each existing sector."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        raw_hams: dict[int | None, np.ndarray] = {}
        for sector in self.blocks:
            if sector.ham is not None:
                continue
            key = sector.twoSz
            if key not in raw_hams:
                raw_hams[key] = self.build_hamiltonian(sector.basis_states)
            sector.set_hamiltonian(raw_hams[key])
        return self

    def _cache_dir(self, cache_dir: str | Path) -> Path:
        return Path(cache_dir) / self.cluster.label()

    def load(self, cache_dir: str | Path) -> "HubbardModel":
        """Load every existing sector from disk; fail instead of partially solving."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        cache = self._cache_dir(cache_dir)
        keys = [(sector.twoSz, sector.twoS) for sector in self.blocks]
        missing = [key for key in keys if not Block.exists(cache, *key)]
        if missing:
            labels = ", ".join(f"(twoSz, twoS)={key}" for key in missing)
            raise FileNotFoundError(f"Missing cached Hubbard blocks in {cache}: {labels}")
        self.blocks = [Block.load(cache, twoSz, twoS) for twoSz, twoS in keys]
        return self

    def save(self, cache_dir: str | Path) -> "HubbardModel":
        """Save every solved sector to disk."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        cache = self._cache_dir(cache_dir)
        for sector in self.blocks:
            if sector.eigvals is None or sector.eigvecs is None:
                raise RuntimeError("call solve() first")
            sector.save(cache)
        return self

    def solve(
        self,
        *,
        cache_mode: str = "none",
        cache_dir: str | Path | None = None,
    ) -> "HubbardModel":
        """Solve sectors using cache_mode: none, load, save, or partial."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")

        if cache_mode not in {"none", "load", "save", "partial"}:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}")
        if cache_mode == "none" and cache_dir is not None:
            raise ValueError("cache_dir requires cache_mode='load', 'save', or 'partial'")
        if cache_mode != "none" and cache_dir is None:
            raise ValueError(f"cache_mode={cache_mode!r} requires cache_dir")
        if cache_mode == "load":
            return self.load(cache_dir)
        if cache_mode == "partial":
            cache = self._cache_dir(cache_dir)
            for idx, block in enumerate(self.blocks):
                if Block.exists(cache, block.twoSz, block.twoS):
                    self.blocks[idx] = Block.load(cache, block.twoSz, block.twoS)
                else:
                    if block.ham is None:
                        raise RuntimeError("call build_hamiltonians() first")
                    block.solve()
                    block.save(cache)
            return self

        cache = self._cache_dir(cache_dir) if cache_mode == "save" else None
        for block in self.blocks:
            if block.ham is None:
                raise RuntimeError("call build_hamiltonians() first")
            block.solve()
            if cache is not None:
                block.save(cache)
        return self

    def merge_by_s2(self) -> "HubbardModel":
        """Merge S² sectors with the same twoSz via block-diagonal sector matrices."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")

        by_twoSz: dict[int, list[Block]] = {}
        for sector in self.blocks:
            if sector.eigvals is None or sector.eigvecs is None:
                raise RuntimeError("call solve() first")
            if sector.ham is None:
                raise RuntimeError("merge_by_s2 requires solved sectors with ham set")
            if sector.twoSz is None:
                raise ValueError("merge_by_s2 requires sectors with twoSz set")
            if sector.twoS is None or sector.basis_transform is None:
                raise ValueError("merge_by_s2 requires S² sectors with twoS and basis_transform set")
            by_twoSz.setdefault(sector.twoSz, []).append(sector)

        merged: list[Block] = []
        for twoSz in sorted(by_twoSz):
            sectors = sorted(
                by_twoSz[twoSz],
                key=lambda sector: -1 if sector.twoS is None else sector.twoS,
            )
            first = sectors[0]
            if any(sector.basis_states != first.basis_states for sector in sectors):
                raise ValueError("merge_by_s2 requires matching fixed-twoSz Fock bases")

            U_merged = np.concatenate([sector.basis_transform for sector in sectors], axis=1)
            if U_merged.shape != (len(first.basis_states), len(first.basis_states)):
                raise ValueError("merge_by_s2 requires all S² sectors for each fixed twoSz")
            sym_ham = block_diag(*[sector.ham for sector in sectors])
            sym_eigvecs = block_diag(*[sector.eigvecs for sector in sectors])
            fock_ham = U_merged @ sym_ham @ U_merged.conj().T
            fock_eigvecs = U_merged @ sym_eigvecs

            merged.append(
                Block(
                    N=self.N,
                    nelec=self.nelec,
                    basis_states=first.basis_states,
                    ham=fock_ham,
                    eigvals=np.concatenate([sector.eigvals for sector in sectors]),
                    eigvecs=fock_eigvecs,
                    twoSz=twoSz,
                    twoS=None,
                    basis_transform=None,
                )
            )

        self.blocks = merged
        self.mode = "Sz"
        previous_spec = getattr(self, "_mode_spec", None)
        previous_scope = SCOPE_NONNEGATIVE if previous_spec is None else previous_spec.scope
        self._mode_spec = mode_spec(
            "Sz",
            twoSz=merged[0].twoSz if len(merged) == 1 else None,
            scope=previous_scope if len(merged) != 1 else SCOPE_NONNEGATIVE,
        )
        return self

    def merge_by_sz(self) -> "HubbardModel":
        """Merge current sectors into one Fock-coordinate block and overwrite self.blocks."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        spec = getattr(self, "_mode_spec", None)
        if spec is None or spec.mode != "Sz" or spec.twoSz_scope != "all":
            raise ValueError("merge_by_sz requires MODE=Sz without fixed twoSz")

        sectors = sorted(self.blocks, key=lambda sector: sector.twoSz)
        for sector in sectors:
            if sector.eigvals is None or sector.eigvecs is None:
                raise RuntimeError("call solve() first")
            if sector.ham is None:
                raise RuntimeError("merge_by_sz requires solved sectors with ham set")
            if sector.twoSz is None or sector.twoS is not None or sector.basis_transform is not None:
                raise ValueError("merge_by_sz requires fixed-twoSz blocks without S² transforms")
        actual_twoSz = {sector.twoSz for sector in sectors}
        expected_twoSz = set(self._twoSz_values())
        if actual_twoSz != expected_twoSz:
            raise ValueError("merge_by_sz requires the complete positive and negative twoSz set")

        basis_states = [state for sector in sectors for state in sector.basis_states]
        ham = block_diag(*[sector.ham for sector in sectors])
        eigvecs = block_diag(*[sector.eigvecs for sector in sectors])

        self.blocks = [
            Block(
                N=self.N,
                nelec=self.nelec,
                basis_states=basis_states,
                ham=ham,
                eigvals=np.concatenate([np.asarray(sector.eigvals) for sector in sectors]),
                eigvecs=eigvecs,
                twoSz=None,
                twoS=None,
                basis_transform=None,
            )
        ]
        self.mode = "full"
        self._mode_spec = mode_spec("full")
        return self

    def project(
        self,
        method: str = "occ",
        **select_kwargs,
    ) -> "HubbardModel":
        """Select eigenstates and downfold each current block."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        for block in self.blocks:
            if block.eigvals is None or block.eigvecs is None:
                raise RuntimeError("call solve() first")

        adiabatic_seeds = select_kwargs.pop("adiabatic_seeds", None)
        if method == "adiabatic" and adiabatic_seeds is None:
            raise ValueError("method='adiabatic' requires adiabatic_seeds")
        if method != "adiabatic" and adiabatic_seeds is not None:
            raise ValueError("adiabatic_seeds applies only to method='adiabatic'")

        self.selected_indices = []
        self.selection_info = []
        self.heff = []
        self.t11m1_norms = []
        for block in self.blocks:
            block_kwargs = dict(select_kwargs)
            if method == "adiabatic":
                seed = adiabatic_seeds.get(block.label())
                if seed is None:
                    raise ValueError(f"missing adiabatic seed for block {block.label()}")
                block_kwargs["eigvecs_previous"] = seed["eigvecs_previous"]
                block_kwargs["selected_previous"] = seed["selected_previous"]
            selected, selection_info = block.selected(
                method=method,
                return_info=True,
                **block_kwargs,
            )
            if method == "adiabatic":
                selection_info = dict(selection_info)
                selection_info["method"] = "adiabatic"
                selection_info["block"] = block.label()
                selection_info["adiabatic_seed"] = {
                    "block": seed["block"],
                    "artifact": seed["artifact"],
                    "selected_indices": [
                        int(idx) for idx in np.asarray(seed["selected_previous"], dtype=int)
                    ],
                }
            heff, t11m1_norm = block.downfold(selected)
            self.selected_indices.append(selected)
            self.selection_info.append(selection_info)
            self.heff.append(heff)
            self.t11m1_norms.append(t11m1_norm)
        return self

    def fit(
        self,
        bond_groups: list[list[Sequence[int]]] | None = None,
    ) -> "HubbardModel":
        """Fit projected Heff blocks to grouped spin-coupling operators."""
        if self.heff is None:
            raise RuntimeError("call project() first")
        if len(self.heff) != len(self.blocks):
            raise ValueError("fit requires one Heff per current block")

        if bond_groups is None:
            bond_groups = (
                self.cluster.generate_bonds(N=2, is_connected=False)
                + self.cluster.generate_bonds(N=4, is_connected=True)
                + self.cluster.generate_bonds(N=6, is_connected=True)
            )

        bonds = [bond for group in bond_groups for bond in group]
        A = np.vstack([block._spin_operators(bonds) for block in self.blocks])
        b = np.concatenate([heff.flatten() for heff in self.heff])
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        residual_vector = A @ x - b
        residual = float(np.linalg.norm(residual_vector))
        b_norm = float(np.linalg.norm(b))
        rel_err = residual / b_norm if b_norm > 0 else 0.0

        ss_res = float(np.real(np.vdot(residual_vector, residual_vector)))
        centered = b - np.mean(b)
        ss_tot = float(np.real(np.vdot(centered, centered)))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        coeffs: list = [x[0]]
        offset = 1
        for group in bond_groups:
            coeffs.append(list(x[offset:offset + len(group)]))
            offset += len(group)

        self.bond_groups = bond_groups
        self.coupling_coeffs = coeffs
        self.fit_metrics = (rel_err, residual, r2)
        return self
