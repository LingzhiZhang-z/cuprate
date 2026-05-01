"""Single-band Hubbard model construction and exact diagonalisation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from cuprate.clusters import Cluster
from cuprate.manifold import Block, DTransform
from cuprate.paths import (
    SCOPE_NONNEGATIVE,
    SCOPE_PM,
    ModeSpec,
    mode_spec,
)
from cuprate.sectors import (
    build_S2_multiplets,
    build_S2_sectors,
    build_S2eta0_sectors,
)
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
        self.U = U
        self.t = t
        self.nelec = self.N if nelec is None else nelec
        self.bonds = [tuple(map(int, bond)) for bond in cluster.bonds]
        self.hoppings = (
            list(hoppings) if hoppings else [self.t] * len(self.bonds)
        )
        self.selected_indices: list[list[int]] | None = None
        self.selection_info: list[dict] | None = None
        self.heff: list[np.ndarray] | None = None
        self.t11m1_norms: list[float] | None = None
        self.bond_groups: list[list[Sequence[int]]] | None = None
        self.coupling_coeffs: list | None = None
        self.fit_metrics: tuple[float, float, float] | None = None
        self.fit_metrics_per_block: list[dict[str, float]] | None = None

    def build_fock_basis(self, twoSz: int | None = None) -> list[int]:
        return generate_states(self.N, self.nelec, twoSz=twoSz)

    def _build_hamiltonian_t(self, basis_states: Sequence[int]) -> np.ndarray:
        """Build the hopping matrix H_t on `basis_states`."""
        states = list(basis_states)
        state_to_idx = {state: idx for idx, state in enumerate(states)}
        dim = len(states)
        H_t = np.zeros((dim, dim), dtype=np.float64)
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
        return np.diag(np.asarray(diag, dtype=np.float64))

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
        if spec.mode == "SzS2eta2" and self.nelec != self.N:
            raise ValueError("MODE=SzS2eta2 requires half filling: nelec == N")
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

    @staticmethod
    def _sector_blocks_by_key(sector_blocks, *, eta: bool):
        blocks_by_key = {}
        for block in sector_blocks:
            key = (block.twoSz, block.twoS, block.eta) if eta else (block.twoSz, block.twoS)
            blocks_by_key.setdefault(key, []).append(block)
        for blocks in blocks_by_key.values():
            blocks.sort(key=lambda block: block.D)
        return blocks_by_key

    def _s2_sector_blocks(self):
        """Build algebraic S² transform blocks keyed by (twoSz, twoS)."""
        grouped_states = group_states(generate_states(self.N, self.nelec), self.N)
        _, multiplets = build_S2_multiplets(grouped_states, self.N)
        sector_blocks = build_S2_sectors(grouped_states, multiplets)
        return self._sector_blocks_by_key(sector_blocks, eta=False)

    def _s2eta0_sector_blocks(self):
        """Build algebraic S²/eta=0 transform blocks keyed by (twoSz, twoS, eta)."""
        grouped_states = group_states(generate_states(self.N, self.nelec), self.N)
        _, multiplets = build_S2_multiplets(grouped_states, self.N)
        sector_blocks = build_S2_sectors(grouped_states, multiplets)
        eta0_sector_blocks = build_S2eta0_sectors(
            self.N,
            sector_blocks,
            self.cluster,
        )
        return self._sector_blocks_by_key(eta0_sector_blocks, eta=True)

    def _transforms_by_D(self, basis_states, sector_blocks) -> dict[int, DTransform]:
        basis_map = {state: idx for idx, state in enumerate(basis_states)}
        transforms: dict[int, DTransform] = {}
        col_start = 0
        for block in sorted(sector_blocks, key=lambda item: item.D):
            coeff = np.asarray(block.transform)
            col_stop = col_start + coeff.shape[1]
            if block.D in transforms:
                raise ValueError(
                    f"Duplicate D={block.D} transform block for "
                    f"(twoSz,twoS,eta)=({block.twoSz},{block.twoS},{block.eta})"
            )
            transforms[block.D] = DTransform(
                D=block.D,
                fock_rows=np.asarray(
                    [basis_map[state] for state in block.basis_states],
                    dtype=int,
                ),
                sector_columns=np.arange(col_start, col_stop, dtype=int),
                matrix=coeff,
            )
            col_start = col_stop
        return transforms

    def _make_sector(self, twoSz: int | None, twoS: int | None, eta: int | None = None, sector_blocks=None):
        basis_states = self.build_fock_basis(twoSz=twoSz)
        transforms_by_D = None
        if sector_blocks is not None:
            transforms_by_D = self._transforms_by_D(basis_states, sector_blocks)
        return Block(
            N=self.N,
            nelec=self.nelec,
            basis_states=basis_states,
            transforms_by_D=transforms_by_D,
            twoSz=twoSz,
            twoS=twoS,
            eta=eta,
        )

    def _block_labels(self, spec: ModeSpec):
        if spec.twoSz_scope == "none":
            twoSz_values = [None]
        elif spec.twoSz_scope == "one":
            twoSz_values = [spec.twoSz]
        else:
            twoSz_values = [
                twoSz for twoSz in self._twoSz_values()
                if spec.scope == SCOPE_PM or twoSz >= 0
            ]

        if spec.twoS_scope == "none":
            return [(twoSz, None, None) for twoSz in twoSz_values]

        if spec.twoS_scope == "one":
            twoS_values = [spec.twoS]
        else:
            twoS_values = [twoS for twoS in self._twoSz_values() if twoS >= 0]

        eta = 0 if spec.mode == "SzS2eta2" else None
        labels = []
        for twoSz in twoSz_values:
            for twoS in twoS_values:
                if twoS >= abs(twoSz):
                    labels.append((twoSz, twoS, eta))
        return labels

    def set_symmetry(
        self,
        mode: str,
        *,
        twoSz: int | None = None,
        twoS: int | None = None,
        scope: str | None = SCOPE_NONNEGATIVE,
    ) -> "HubbardModel":
        spec = self._validate_mode(mode, twoSz, twoS, scope)
        if spec.twoS_scope == "none":
            sector_blocks = None
        elif spec.mode == "SzS2eta2":
            sector_blocks = self._s2eta0_sector_blocks()
        else:
            sector_blocks = self._s2_sector_blocks()

        blocks = []
        for tsz, ts, eta in self._block_labels(spec):
            if ts is None:
                transform_blocks = None
            elif eta is None:
                transform_blocks = sector_blocks[(tsz, ts)]
            else:
                transform_blocks = sector_blocks[(tsz, ts, eta)]
            blocks.append(self._make_sector(tsz, ts, eta=eta, sector_blocks=transform_blocks))
        if not blocks:
            raise ValueError("No sectors exist for the requested symmetry selection")
        self.blocks = blocks
        self.mode = spec.mode
        self._mode_spec = spec
        return self

    def build_hamiltonians(self) -> "HubbardModel":
        """Build and assign the Hamiltonian for each existing sector."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        raw_hams: dict[int | None, np.ndarray] = {}
        for sector in self.blocks:
            key = sector.twoSz
            if key not in raw_hams:
                raw_hams[key] = self.build_hamiltonian(sector.basis_states)
            sector.set_hamiltonian(raw_hams[key])
        return self

    def _cache_dir(self, cache_dir: str | Path) -> Path:
        return Path(cache_dir) / self.cluster.label()

    def save_transforms(self, cache_dir: str | Path) -> "HubbardModel":
        """Save current block transforms to the cluster cache directory."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        cache = self._cache_dir(cache_dir)
        for block in self.blocks:
            block.save_transforms(cache)
        return self

    def load_transforms(self, cache_dir: str | Path) -> "HubbardModel":
        """Load current block transforms from the cluster cache directory."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        cache = self._cache_dir(cache_dir)
        for block in self.blocks:
            block.load_transforms(cache)
        return self

    def discard_transforms(self) -> "HubbardModel":
        """Discard non-spin transform matrices from all current blocks."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")
        for block in self.blocks:
            block.discard_transforms()
        return self

    def solve(
        self,
        *,
        cache_mode: str = "solve",
        cache_dir: str | Path | None = None,
        eigh: str = "lowmem",
    ) -> "HubbardModel":
        """Solve sectors using cache_mode: read, solve, or auto."""
        if not getattr(self, "blocks", None):
            raise RuntimeError("call set_symmetry() first")

        if cache_mode not in {"read", "solve", "auto"}:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}")
        if cache_dir is None:
            raise ValueError(f"cache_mode={cache_mode!r} requires cache_dir")
        cache = self._cache_dir(cache_dir)
        for block in self.blocks:
            if cache_mode == "read":
                block.load_transforms(cache)
                block.load(cache)
            elif cache_mode == "solve":
                block.save_transforms(cache)
                block.solve(eigh=eigh)
                block.save(cache)
            else:  # cache_mode == "auto"
                transforms_by_D = block.transforms_by_D
                try:
                    block.load_transforms(cache)
                    block.load(cache)
                except ValueError:
                    block.transforms_by_D = transforms_by_D
                    block.save_transforms(cache)
                    block.solve(eigh=eigh)
                    block.save(cache)
            block.discard_transforms()
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
        b_blocks = [heff.flatten() for heff in self.heff]
        b = np.concatenate(b_blocks)
        try:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Spin fit failed for cluster={self.cluster.label()} "
                f"N={self.N} n_blocks={len(self.blocks)} "
                f"A_shape={A.shape} b_shape={b.shape} "
                f"bond_groups={len(bond_groups)}"
            ) from exc

        residual_vector = A @ x - b
        residual = float(np.linalg.norm(residual_vector))
        b_norm = float(np.linalg.norm(b))
        rel_err = residual / b_norm if b_norm > 0 else 0.0

        ss_res = float(np.real(np.vdot(residual_vector, residual_vector)))
        centered = b - np.mean(b)
        ss_tot = float(np.real(np.vdot(centered, centered)))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        per_block_metrics: list[dict[str, float]] = []
        offset = 0
        for block_b in b_blocks:
            n = block_b.size
            res_i = residual_vector[offset:offset + n]
            res_i_norm = float(np.linalg.norm(res_i))
            b_i_norm = float(np.linalg.norm(block_b))
            per_block_metrics.append({
                "residual": res_i_norm,
                "relative_error": res_i_norm / b_i_norm if b_i_norm > 0 else 0.0,
            })
            offset += n

        coeffs: list = [x[0]]
        offset = 1
        for group in bond_groups:
            coeffs.append(list(x[offset:offset + len(group)]))
            offset += len(group)

        self.bond_groups = bond_groups
        self.coupling_coeffs = coeffs
        self.fit_metrics = (rel_err, residual, r2)
        self.fit_metrics_per_block = per_block_metrics
        return self
