"""Stage-1 runtime helpers built on the current ED core."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cuprate.downfolding import select as select_eigenstates
from cuprate.downfolding import spin_fit
from cuprate.hubbard import (
    Block,
    MODE_BY_SZ,
    MODE_BY_SZ_S2,
    MODE_ONE_SZ,
    MODE_ONE_SZ_S2,
    MODE_SINGLE,
    HubbardModel,
    Spectrum,
)
from cuprate.io import (
    MODE_BLOCK_SZ_FULL,
    MODE_BLOCK_SZ_S2_FULL,
    MODE_FIXED_SZ,
    MODE_FIXED_SZ_S2,
    MODE_FULL,
    PathSpec,
    RESULT_KIND_PROJECTION_ANALYSIS,
    RESULT_KIND_SPIN_COUPLINGS,
    Params,
    build_path_spec,
)
from cuprate.states import calc_double_occupation_matrix, site_code

from .reporting import save_projection_result, save_spin_coupling_result


_LEGACY_SITE_CODE = {0: 0, 1: 1, 2: -1, 3: 2}
_LEGACY_VALUE_ORDER = {0: 0, 1: 1, -1: 2, 2: 3}


@dataclass
class FamilyProjection:
    spectrum: Spectrum
    project_block: Block
    eigvals: np.ndarray
    eigvecs: np.ndarray
    states: np.ndarray
    double_occ: np.ndarray
    s2_diag: np.ndarray
    s2_selected: np.ndarray
    selected_indices: np.ndarray
    selected_occupation: np.ndarray
    heff: np.ndarray
    t11m1: np.ndarray
    t11m1_norm: float
    overlap: float | None


def _solver_mode(mode_spec) -> str:
    return {
        MODE_FULL: MODE_SINGLE,
        MODE_FIXED_SZ: MODE_ONE_SZ,
        MODE_BLOCK_SZ_FULL: MODE_BY_SZ,
        MODE_FIXED_SZ_S2: MODE_ONE_SZ_S2,
        MODE_BLOCK_SZ_S2_FULL: MODE_BY_SZ_S2,
    }[mode_spec.mode]


def _selection_method(params: Params) -> str:
    if params.workflow is None:
        return "occ"
    return {
        "single": "single",
        "multi": "multi",
        "adiabatic": "adiabatic",
    }.get(params.workflow, params.workflow)


def _legacy_state_row(state: int, N: int) -> tuple[int, ...]:
    return tuple(_LEGACY_SITE_CODE[site_code(state, site)] for site in range(N))


def _legacy_state_key(state: int, N: int) -> tuple[int, int, int, tuple[int, ...]]:
    row = _legacy_state_row(state, N)
    twoSz = sum(value for value in row if abs(value) == 1)
    return (
        row.count(2),
        abs(twoSz),
        -twoSz,
        tuple(_LEGACY_VALUE_ORDER[value] for value in row),
    )


def _legacy_state_array(basis_states: list[int], N: int) -> np.ndarray:
    return np.asarray([_legacy_state_row(state, N) for state in basis_states], dtype=int)


def _block_order_key(block: Block) -> tuple[int, int, int]:
    twoSz = 0 if block.twoSz is None else int(block.twoSz)
    twoS = -1 if block.twoS is None else int(block.twoS)
    return (abs(twoSz), twoS, -twoSz)


def _ordered_blocks(spectrum: Spectrum) -> list[Block]:
    return sorted(spectrum.blocks, key=_block_order_key)


def _reorder_block_legacy(block: Block) -> Block:
    basis_states = sorted(block.basis_states, key=lambda state: _legacy_state_key(state, block.N))
    old_index = {state: i for i, state in enumerate(block.basis_states)}
    perm = np.asarray([old_index[state] for state in basis_states], dtype=int)

    eigvecs = block.eigvecs[perm, :]
    basis_transform = None
    if block.basis_transform is not None:
        basis_transform = block.basis_transform[perm, :]

    return Block(
        N=block.N,
        nelec=block.nelec,
        basis_states=basis_states,
        ham=None,
        eigvals=block.eigvals.copy(),
        eigvecs=eigvecs,
        twoSz=block.twoSz,
        twoS=block.twoS,
        basis_transform=basis_transform,
    )


def _build_project_block(spectrum: Spectrum, reconstruct_full: bool) -> Block:
    ordered_blocks = _ordered_blocks(spectrum)
    if len(ordered_blocks) == 1 and not reconstruct_full:
        return _reorder_block_legacy(ordered_blocks[0])

    basis_states = sorted(
        {state for block in ordered_blocks for state in block.basis_states},
        key=lambda state: _legacy_state_key(state, spectrum.N),
    )
    state_to_row = {state: i for i, state in enumerate(basis_states)}

    total_cols = sum(block.eigvecs.shape[1] for block in ordered_blocks)
    eigvecs = np.zeros((len(basis_states), total_cols), dtype=complex)
    eigvals = []

    col0 = 0
    for block in ordered_blocks:
        cols = block.eigvecs.shape[1]
        rows = np.asarray([state_to_row[state] for state in block.basis_states], dtype=int)
        eigvecs[np.ix_(rows, np.arange(col0, col0 + cols, dtype=int))] = block.eigvecs
        eigvals.append(np.asarray(block.eigvals))
        col0 += cols

    return Block(
        N=spectrum.N,
        nelec=spectrum.nelec,
        basis_states=basis_states,
        ham=None,
        eigvals=np.concatenate(eigvals) if eigvals else np.array([], dtype=complex),
        eigvecs=eigvecs,
        twoSz=None,
        twoS=None,
        basis_transform=None,
    )


def _selection_spectrum(spectrum: Spectrum, project_block: Block, params: Params) -> Spectrum:
    if params.select == "block" and len(spectrum.blocks) > 1:
        blocks = [_reorder_block_legacy(block) for block in _ordered_blocks(spectrum)]
        return Spectrum(spectrum.N, spectrum.nelec, spectrum.mode, blocks, cluster=spectrum.cluster)
    return Spectrum(spectrum.N, spectrum.nelec, MODE_SINGLE, [project_block], cluster=spectrum.cluster)


def _flatten_selected_indices(selection_blocks: list[Block], selected_list: list[list[int]]) -> np.ndarray:
    indices = []
    offset = 0
    for block, selected in zip(selection_blocks, selected_list):
        indices.extend(offset + int(idx) for idx in selected)
        offset += len(block.eigvals)
    return np.asarray(indices, dtype=int)


def _evaluate_block(block: Block, selected_indices: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_indices = list(np.asarray(selected_indices, dtype=int))
    spin_rows = block.spin_rows()
    spin_basis = block.spin_basis()
    s_bd = spin_basis @ block.eigvecs[np.ix_(spin_rows, selected_indices)]
    U, sigma, vh = np.linalg.svd(s_bd, full_matrices=False)
    lam = np.diag(block.eigvals[selected_indices])
    heff = U @ vh @ lam @ vh.conj().T @ U.conj().T
    t11m1 = U @ np.diag(sigma) @ U.conj().T - np.eye(block.spin_dim())
    return heff, t11m1


def _double_occ_expectation(block: Block) -> np.ndarray:
    dom = calc_double_occupation_matrix(block.basis_states, block.N)
    return np.diag(block.eigvecs.conj().T @ dom @ block.eigvecs)


def _s2_diagonal(spectrum: Spectrum) -> np.ndarray:
    rows = []
    for block in _ordered_blocks(spectrum):
        for qnum in block.quantum_numbers():
            s2_value = 0.25 * qnum["twoS"] * (qnum["twoS"] + 2)
            rows.append([complex(s2_value, 0.0), 0.0 + 0.0j, 1.0 + 0.0j])
    return np.asarray(rows, dtype=complex)


def _adiabatic_seed_paths(params: Params, hole: int, class_idx: int) -> dict[str, str | bool]:
    if params.delta is None:
        raise ValueError("WORKFLOW=adiabatic requires DELTA=<previous step size>.")

    prefix = f"hole{hole}_class{class_idx}"

    previous_params = replace(params, t=params.t - params.delta, restart=params.restart)
    previous_spec = build_path_spec(previous_params)
    previous_eigvecs = Path(previous_spec.data_dir) / f"{prefix}_eigvecs.npy"
    previous_indices = Path(previous_spec.output_dir) / f"{prefix}_t11_selected_indices.npy"

    if previous_eigvecs.exists() and previous_indices.exists():
        return {
            "eigvecs": str(previous_eigvecs),
            "indices": str(previous_indices),
            "reuse_indices": False,
        }

    seed_params = replace(params, workflow=None, restart=False)
    seed_spec = build_path_spec(seed_params)
    seed_eigvecs = Path(seed_spec.data_dir) / f"{prefix}_eigvecs.npy"
    seed_indices = Path(seed_spec.output_dir) / f"{prefix}_t11_selected_indices.npy"

    if not seed_indices.exists():
        raise FileNotFoundError(f"Missing adiabatic seed selected indices: {seed_indices}")
    if not seed_eigvecs.exists():
        raise FileNotFoundError(f"Missing adiabatic seed eigvecs: {seed_eigvecs}")

    return {
        "eigvecs": str(seed_eigvecs),
        "indices": str(seed_indices),
        "reuse_indices": True,
    }


def _selection_kwargs(
    params: Params,
    spec: PathSpec,
    mode_spec,
    hole: int,
    class_idx: int,
) -> dict:
    method = _selection_method(params)
    if method == "single":
        return {"ratio": 8, "tmp_dir": spec.tmp_dir, "twoSz": mode_spec.twoSz}
    if method == "multi":
        return {
            "ratio": 8,
            "n_restarts": 40,
            "max_iters_rand": 4,
            "tmp_dir": spec.tmp_dir,
            "twoSz": mode_spec.twoSz,
        }
    if method == "adiabatic":
        return {"seed": _adiabatic_seed_paths(params, hole, class_idx)}
    return {}


def _fit_cluster_projection(
    project_block: Block,
    cluster,
    heff: np.ndarray,
) -> tuple[list, list[list[list[int]]], tuple[float, float, float]]:
    bond_groups = (
        cluster.generate_bonds(N=2, is_connected=False)
        + cluster.generate_bonds(N=4, is_connected=True)
        + cluster.generate_bonds(N=6, is_connected=True)
    )
    bonds = [bond for group in bond_groups for bond in group]
    A = project_block._spin_operators(bonds)
    x, metrics = spin_fit(A, heff.flatten())

    coeffs: list = [x[0]]
    idx = 1
    for group in bond_groups:
        coeffs.append(list(x[idx:idx + len(group)]))
        idx += len(group)
    return coeffs, bond_groups, metrics


def _save_family_arrays(
    spec: PathSpec,
    hole: int,
    class_idx: int,
    projection: FamilyProjection,
) -> None:
    prefix = f"hole{hole}_class{class_idx}"
    np.save(f"{spec.data_dir}/{prefix}_eigvals.npy", projection.eigvals)
    np.save(f"{spec.data_dir}/{prefix}_eigvecs.npy", projection.eigvecs)

    np.save(f"{spec.output_dir}/{prefix}_states.npy", projection.states)
    np.save(f"{spec.output_dir}/{prefix}_double_occupation_expectation.npy", projection.double_occ)
    np.save(f"{spec.output_dir}/{prefix}_S2_diagonal.npy", projection.s2_diag)
    np.save(f"{spec.output_dir}/{prefix}_S2_selected.npy", projection.s2_selected)
    np.save(f"{spec.output_dir}/{prefix}_t11_selected_indices.npy", projection.selected_indices)
    np.save(f"{spec.output_dir}/{prefix}_Heff.npy", projection.heff)
    np.save(f"{spec.output_dir}/{prefix}_T11m1.npy", projection.t11m1)
    np.save(
        f"{spec.output_dir}/{prefix}_t11_selected_occupation.npy",
        projection.selected_occupation,
    )


def cluster_process(
    cluster,
    params: Params,
    spec: PathSpec,
    mode_spec,
    hole: int,
    class_idx: int,
) -> FamilyProjection:
    if params.match_spin_sectors:
        raise NotImplementedError("MATCH_SPIN_SECTORS is not implemented on the runtime path yet.")

    model = HubbardModel(cluster, params.U, params.t)
    spectrum = model.solve(
        _solver_mode(mode_spec),
        twoSz=mode_spec.twoSz,
        twoS=mode_spec.twoS,
    )

    project_block = _build_project_block(spectrum, mode_spec.reconstruct_full)
    selection_spectrum = _selection_spectrum(spectrum, project_block, params)
    selected_list, selection_info = select_eigenstates(
        selection_spectrum,
        _selection_method(params),
        return_info=True,
        **_selection_kwargs(params, spec, mode_spec, hole, class_idx),
    )
    selected_indices = _flatten_selected_indices(selection_spectrum.blocks, selected_list)

    heff, t11m1 = _evaluate_block(project_block, selected_indices)
    double_occ = np.asarray(_double_occ_expectation(project_block), dtype=complex)
    s2_diag = _s2_diagonal(spectrum)
    s2_selected = (
        s2_diag[selected_indices]
        if len(selected_indices)
        else np.zeros((0, 3), dtype=complex)
    )

    return FamilyProjection(
        spectrum=spectrum,
        project_block=project_block,
        eigvals=np.asarray(project_block.eigvals),
        eigvecs=np.asarray(project_block.eigvecs),
        states=_legacy_state_array(project_block.basis_states, project_block.N),
        double_occ=double_occ,
        s2_diag=s2_diag,
        s2_selected=s2_selected,
        selected_indices=selected_indices,
        selected_occupation=double_occ[selected_indices],
        heff=heff,
        t11m1=t11m1,
        t11m1_norm=float(np.linalg.norm(t11m1.ravel())),
        overlap=selection_info["overlap"],
    )


def cluster_process_work_item(
    hole: int,
    class_idx: int,
    cluster_families,
    params: Params,
    spec: PathSpec,
    mode_spec,
    rank: int,
) -> list[dict]:
    cluster_family = cluster_families[(hole, class_idx)]
    projection = cluster_process(cluster_family[0], params, spec, mode_spec, hole, class_idx)
    _save_family_arrays(spec, hole, class_idx, projection)

    cluster_entries = []
    for cluster in cluster_family:
        t0_cluster = time.time()
        cluster_idx = int(cluster.cluster_idx)
        if mode_spec.result_kind == RESULT_KIND_SPIN_COUPLINGS:
            coeffs, bond_groups, metrics = _fit_cluster_projection(
                projection.project_block,
                cluster,
                projection.heff,
            )
            cluster_entries.append(
                save_spin_coupling_result(
                    spec.output_dir,
                    hole,
                    class_idx,
                    cluster_idx,
                    rank,
                    time.time() - t0_cluster,
                    cluster.sites,
                    bond_groups,
                    coeffs,
                    metrics,
                    projection.t11m1_norm,
                    projection.overlap,
                )
            )
        elif mode_spec.result_kind == RESULT_KIND_PROJECTION_ANALYSIS:
            cluster_entries.append(
                save_projection_result(
                    spec.output_dir,
                    hole,
                    class_idx,
                    cluster_idx,
                    rank,
                    time.time() - t0_cluster,
                    cluster.sites,
                    projection,
                )
            )
        else:
            raise ValueError(f"Unsupported result kind: {mode_spec.result_kind}")

    return cluster_entries
