"""Thin MPI workchain for single-cluster Hubbard projection and fitting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuprate.clusters import Cluster, ClusterSets
from cuprate.hubbard import HubbardModel
from cuprate.paths import (
    RESULTS_FILE,
    STAGE_MAIN,
    family_clusters_file,
    family_exchange_file,
    family_projection_file,
    main_data_dir,
    mode_token,
    parameter_token,
    workflow_dir,
    workflow_token,
)
from cuprate.mpi import comm, rank, size
from cuprate.operators import complex_json, operators_from_terms, terms_from_fit


RESULT_SCHEMA_VERSION = 5
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi", "adiabatic"}


@dataclass(frozen=True)
class WorkchainParams:
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
    workflow: str
    root: Path
    cache_mode: str = "none"
    ratio: int | None = None
    n_trials: int | None = None
    max_failures: int | None = None
    seed_results: Path | None = None


def run_workchain(params: WorkchainParams) -> dict[str, Any] | None:
    """Run the representative-reuse workchain and return results on rank 0."""
    if params.workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"workflow={params.workflow!r} is not supported in this workchain; "
            "use one of occ, energy, greedy, greedy_multi, adiabatic"
        )

    seed_context = _load_adiabatic_seed_context(params)
    output_dir = workflow_dir(
        params.root,
        STAGE_MAIN,
        params.N,
        params.N,
        params.U,
        params.t,
        params.mode,
        params.workflow,
        twoSz=params.twoSz,
        twoS=params.twoS,
    )
    exchanges_dir = output_dir / "exchanges"
    clusters_dir = output_dir / "clusters"
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    exchanges_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    families = _enumerate_families(params.N) if rank == 0 else None
    families = comm.bcast(families, root=0)

    local_entries: list[dict[str, Any]] = []
    for family in families[rank::size]:
        local_entries.append(
            _process_family(params, family, exchanges_dir, clusters_dir, artifacts_dir, seed_context)
        )

    gathered = comm.gather(local_entries, root=0)
    if rank != 0:
        return None

    entries = [entry for batch in gathered for entry in batch]
    entries.sort(key=lambda entry: (entry["hole"], entry["class_idx"]))

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": "spin_couplings",
        "complete_family_set": True,
        "run_params": _run_params_json(params, seed_context),
        "families": entries,
    }
    (output_dir / RESULTS_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _enumerate_families(N: int) -> list[tuple[tuple[int, int], list[Cluster]]]:
    by_family: dict[tuple[int, int], list[Cluster]] = {}
    for cluster in ClusterSets(N).generate().clusters:
        key = (int(cluster.hole), int(cluster.class_idx))
        by_family.setdefault(key, []).append(cluster)
    return [(key, by_family[key]) for key in sorted(by_family)]


def _process_family(
    params: WorkchainParams,
    family: tuple[tuple[int, int], list[Cluster]],
    exchanges_dir: Path,
    clusters_dir: Path,
    artifacts_dir: Path,
    seed_context: dict[str, Any] | None,
) -> dict[str, Any]:
    (hole, class_idx), members = family
    representative = members[0]
    family_start = time.perf_counter()

    model = HubbardModel(representative, params.U, params.t)
    model.set_symmetry(params.mode, twoSz=params.twoSz, twoS=params.twoS)
    model.build_hamiltonians()
    cache_dir = (
        main_data_dir(params.root, params.N, params.N, params.U, params.t, params.mode)
        if params.cache_mode != "none"
        else None
    )
    model.solve(cache_mode=params.cache_mode, cache_dir=cache_dir)

    select_kwargs = _selection_kwargs(params, artifacts_dir, hole, class_idx, model, seed_context)
    model.project(method=params.workflow, **select_kwargs)

    artifact_name = family_projection_file(hole, class_idx)
    artifact_path = artifacts_dir / artifact_name
    _write_projection_npz(artifact_path, model)
    artifact_relpath = str(Path("artifacts") / artifact_name)

    bond_groups = _fit_bond_groups(representative)
    model.fit(bond_groups=bond_groups)
    constant, terms = terms_from_fit(model.bond_groups, model.coupling_coeffs)
    elapsed = time.perf_counter() - family_start

    exchange_name = family_exchange_file(hole, class_idx)
    clusters_name = family_clusters_file(hole, class_idx)
    exchange_relpath = str(Path("exchanges") / exchange_name)
    clusters_relpath = str(Path("clusters") / clusters_name)

    exchange_payload = _exchange_payload(
        params=params,
        representative=representative,
        model=model,
        artifact=artifact_relpath,
        constant=constant,
        terms=terms,
        elapsed=elapsed,
    )
    clusters_payload = _clusters_payload(
        params=params,
        representative=representative,
        members=members,
    )
    (exchanges_dir / exchange_name).write_text(json.dumps(exchange_payload, indent=2) + "\n")
    (clusters_dir / clusters_name).write_text(json.dumps(clusters_payload, indent=2) + "\n")

    return {
        "hole": int(hole),
        "class_idx": int(class_idx),
        "exchange_file": exchange_relpath,
        "clusters_file": clusters_relpath,
    }


def _selection_kwargs(
    params: WorkchainParams,
    artifacts_dir: Path,
    hole: int,
    class_idx: int,
    model: HubbardModel,
    seed_context: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if params.workflow in {"greedy", "greedy_multi"}:
        log_path = artifacts_dir / f"hole{hole}_class{class_idx}_{params.workflow}.jsonl"
        log_path.write_text("")
        kwargs["selection_info_path"] = log_path
    if params.ratio is not None and params.workflow in {"greedy", "greedy_multi"}:
        kwargs["ratio"] = params.ratio
    if params.workflow == "greedy_multi":
        if params.n_trials is not None:
            kwargs["n_trials"] = params.n_trials
        if params.max_failures is not None:
            kwargs["max_failures"] = params.max_failures
    if params.workflow == "adiabatic":
        if seed_context is None:
            raise ValueError("workflow=adiabatic requires SEED_RESULTS")
        kwargs["adiabatic_seeds"] = _load_adiabatic_seed_map(
            seed_context,
            model,
            hole,
            class_idx,
        )
    return kwargs


def _write_projection_npz(path: Path, model: HubbardModel) -> None:
    arrays: dict[str, np.ndarray] = {
        "n_blocks": np.array(len(model.blocks), dtype=int),
        "block_labels": np.array([block.label() for block in model.blocks]),
        "twoSz": np.array(
            [np.nan if block.twoSz is None else float(block.twoSz) for block in model.blocks]
        ),
        "twoS": np.array(
            [np.nan if block.twoS is None else float(block.twoS) for block in model.blocks]
        ),
        "t11_minus_1_norm": np.array(model.t11m1_norms, dtype=float),
        "overlap": np.array(
            [
                np.nan if (overlap := _info_overlap(info)) is None else float(overlap)
                for info in model.selection_info
            ],
            dtype=float,
        ),
    }
    for idx, (block, heff, selected) in enumerate(
        zip(model.blocks, model.heff, model.selected_indices)
    ):
        arrays[f"block_{idx}_Heff"] = np.asarray(heff)
        arrays[f"block_{idx}_selected_indices"] = np.asarray(selected, dtype=int)
        arrays[f"block_{idx}_basis_states"] = np.asarray(block.basis_states, dtype=np.int64)
        arrays[f"block_{idx}_eigvecs_fock"] = np.asarray(block.eigvecs_fock)
    np.savez_compressed(path, **arrays)


def _fit_bond_groups(cluster: Cluster) -> list[list[Any]]:
    return (
        cluster.generate_bonds(N=2, is_connected=False)
        + cluster.generate_bonds(N=4, is_connected=True)
        + cluster.generate_bonds(N=6, is_connected=True)
    )


def _exchange_payload(
    *,
    params: WorkchainParams,
    representative: Cluster,
    model: HubbardModel,
    artifact: str,
    constant: complex,
    terms: dict,
    elapsed: float,
) -> dict[str, Any]:
    rel_err, residual, r2 = model.fit_metrics
    t11_norm = max(float(value) for value in model.t11m1_norms)
    overlap = _aggregate_overlap(model.selection_info)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": "spin_coupling_exchange",
        "N": int(params.N),
        "hole": int(representative.hole),
        "class_idx": int(representative.class_idx),
        "representative_cluster_idx": int(representative.cluster_idx),
        "representative_sites": [[int(x), int(y)] for x, y in representative.sites],
        "projection": {
            "method": params.workflow,
            "artifact": artifact,
            "blocks": _projection_blocks(model),
        },
        "operators": operators_from_terms(representative.sites, constant, terms),
        "fit": {
            "relative_error": float(rel_err),
            "residual": float(residual),
            "r_squared": float(r2),
            "t11_minus_1_norm": t11_norm,
            "overlap": overlap,
        },
        "metadata": {
            "rank": int(rank),
            "computation_time_s": float(elapsed),
        },
    }


def _clusters_payload(
    *,
    params: WorkchainParams,
    representative: Cluster,
    members: list[Cluster],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": "cluster_family_geometry",
        "N": int(params.N),
        "hole": int(representative.hole),
        "class_idx": int(representative.class_idx),
        "representative_cluster_idx": int(representative.cluster_idx),
        "representative_sites": [[int(x), int(y)] for x, y in representative.sites],
        "clusters": [_cluster_geometry_payload(member) for member in members],
    }


def _cluster_geometry_payload(cluster: Cluster) -> dict[str, Any]:
    return {
        "cluster_idx": int(cluster.cluster_idx),
        "sites": [[int(x), int(y)] for x, y in cluster.sites],
        "indices": list(range(cluster.N)),
    }


def _projection_blocks(model: HubbardModel) -> list[dict[str, Any]]:
    blocks = []
    for block, selected, t11_norm, info in zip(
        model.blocks,
        model.selected_indices,
        model.t11m1_norms,
        model.selection_info,
    ):
        overlap = _info_overlap(info)
        payload = {
            "block": block.label(),
            "twoSz": None if block.twoSz is None else int(block.twoSz),
            "twoS": None if block.twoS is None else int(block.twoS),
            "selected_indices": [int(idx) for idx in selected],
            "selected_state_count": int(len(selected)),
            "spin_dim": int(block.spin_dim),
            "heff_dimension": [int(block.spin_dim), int(block.spin_dim)],
            "t11_minus_1_norm": float(t11_norm),
            "overlap": None if overlap is None else float(overlap),
            "selection_info": _json_ready(info),
        }
        if "adiabatic_seed" in info:
            payload["adiabatic_seed"] = _json_ready(info["adiabatic_seed"])
        blocks.append(payload)
    return blocks


def _load_adiabatic_seed_context(params: WorkchainParams) -> dict[str, Any] | None:
    if params.workflow != "adiabatic":
        if params.seed_results is not None:
            raise ValueError("SEED_RESULTS applies only to workflow=adiabatic")
        return None
    if params.seed_results is None:
        raise ValueError("SEED_RESULTS is required when workflow=adiabatic")

    seed_path = Path(params.seed_results)
    if not seed_path.is_file():
        raise ValueError(f"SEED_RESULTS does not exist: {seed_path}")
    try:
        payload = json.loads(seed_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"SEED_RESULTS is not valid JSON: {seed_path}") from exc

    if payload.get("result_kind") != "spin_couplings":
        raise ValueError(f"SEED_RESULTS is not a spin_couplings results file: {seed_path}")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"SEED_RESULTS schema_version={schema_version} cannot seed adiabatic; "
            f"regenerate with schema_version={RESULT_SCHEMA_VERSION}"
        )
    run_params = payload.get("run_params")
    if not isinstance(run_params, dict):
        raise ValueError(f"SEED_RESULTS is missing run_params: {seed_path}")
    seed_workflow = run_params.get("workflow")
    if seed_workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"SEED_RESULTS has unsupported workflow={seed_workflow!r}")

    families = payload.get("families")
    if not isinstance(families, list):
        raise ValueError(f"SEED_RESULTS is missing families: {seed_path}")

    return {
        "results_path": seed_path,
        "results_dir": seed_path.parent,
        "schema_version": schema_version,
        "run_params": run_params,
        "workflow": seed_workflow,
        "families": families,
    }


def _load_adiabatic_seed_map(
    seed_context: dict[str, Any],
    model: HubbardModel,
    hole: int,
    class_idx: int,
) -> dict[str, dict[str, Any]]:
    seed_entry = _find_seed_entry(seed_context, hole, class_idx)
    projection = seed_entry.get("projection")
    if not isinstance(projection, dict):
        raise ValueError(f"seed entry hole={hole} class={class_idx} is missing projection")
    artifact = projection.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError(f"seed entry hole={hole} class={class_idx} is missing projection artifact")

    artifact_path = Path(seed_context["results_dir"]) / artifact
    if not artifact_path.is_file():
        raise ValueError(f"seed projection artifact does not exist: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        seed_map = _seed_map_from_npz(data, artifact, artifact_path)

    for block in model.blocks:
        label = block.label()
        if label not in seed_map:
            raise ValueError(
                f"seed projection artifact {artifact_path} is missing block {label}"
            )
        seed = seed_map[label]
        if seed["basis_states"] != list(block.basis_states):
            raise ValueError(
                f"seed basis mismatch for block {label} in {artifact_path}"
            )
        eigvecs_previous = seed["eigvecs_previous"]
        if eigvecs_previous.shape[0] != len(block.basis_states):
            raise ValueError(
                f"seed eigvec row count mismatch for block {label} in {artifact_path}"
            )
        selected_previous = np.asarray(seed["selected_previous"], dtype=int)
        if selected_previous.size != block.spin_dim:
            raise ValueError(
                f"seed selected count mismatch for block {label}: "
                f"expected {block.spin_dim}, got {selected_previous.size}"
            )
        if selected_previous.size and (
            selected_previous.min() < 0 or selected_previous.max() >= eigvecs_previous.shape[1]
        ):
            raise ValueError(
                f"seed selected indices are out of range for block {label} in {artifact_path}"
            )
    return seed_map


def _find_seed_entry(seed_context: dict[str, Any], hole: int, class_idx: int) -> dict[str, Any]:
    matches = [
        entry
        for entry in seed_context["families"]
        if int(entry.get("hole", -1)) == hole and int(entry.get("class_idx", -1)) == class_idx
    ]
    if not matches:
        raise ValueError(f"SEED_RESULTS has no family entry for hole={hole} class={class_idx}")
    exchange_file = matches[0].get("exchange_file")
    if not isinstance(exchange_file, str) or not exchange_file:
        raise ValueError(f"seed family hole={hole} class={class_idx} is missing exchange_file")
    exchange_path = Path(seed_context["results_dir"]) / exchange_file
    if not exchange_path.is_file():
        raise ValueError(f"seed exchange file does not exist: {exchange_path}")
    return json.loads(exchange_path.read_text())


def _seed_map_from_npz(
    data: np.lib.npyio.NpzFile,
    artifact: str,
    artifact_path: Path,
) -> dict[str, dict[str, Any]]:
    required = {"n_blocks", "block_labels"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"seed projection artifact {artifact_path} is missing {', '.join(missing)}")

    n_blocks = int(np.asarray(data["n_blocks"]).item())
    labels = [str(label) for label in np.asarray(data["block_labels"]).tolist()]
    if len(labels) != n_blocks:
        raise ValueError(f"seed projection artifact {artifact_path} has inconsistent block labels")

    seed_map: dict[str, dict[str, Any]] = {}
    for idx, label in enumerate(labels):
        keys = {
            "selected_previous": f"block_{idx}_selected_indices",
            "basis_states": f"block_{idx}_basis_states",
            "eigvecs_previous": f"block_{idx}_eigvecs_fock",
        }
        missing = sorted(key for key in keys.values() if key not in data.files)
        if missing:
            raise ValueError(
                f"seed projection artifact {artifact_path} block {label} is missing "
                f"{', '.join(missing)}"
            )
        seed_map[label] = {
            "block": label,
            "artifact": artifact,
            "selected_previous": np.asarray(data[keys["selected_previous"]], dtype=int),
            "basis_states": np.asarray(data[keys["basis_states"]], dtype=np.int64).tolist(),
            "eigvecs_previous": np.asarray(data[keys["eigvecs_previous"]]),
        }
    return seed_map


def _info_overlap(info: dict[str, Any]) -> float | None:
    overlap = info.get("overlap")
    if overlap is None:
        return None
    return float(overlap)


def _aggregate_overlap(infos: list[dict[str, Any]]) -> float | None:
    overlaps = [_info_overlap(info) for info in infos]
    if all(value is None for value in overlaps):
        return None
    return float(sum(value for value in overlaps if value is not None))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        return complex_json(value)
    return value


def _run_params_json(
    params: WorkchainParams,
    seed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "N": int(params.N),
        "nelec": int(params.N),
        "U": float(params.U),
        "T": float(params.t),
        "MODE": params.mode,
        "twoSz": None if params.twoSz is None else int(params.twoSz),
        "twoS": None if params.twoS is None else int(params.twoS),
        "workflow": params.workflow,
        "ROOT": str(params.root),
        "parameter_token": parameter_token(params.N, params.N, params.U, params.t),
        "mode_token": mode_token(params.mode, twoSz=params.twoSz, twoS=params.twoS),
        "workflow_token": workflow_token(params.workflow),
        "data_dir": str(
            main_data_dir(params.root, params.N, params.N, params.U, params.t, params.mode)
        ),
        "CACHE_MODE": params.cache_mode,
    }
    if params.ratio is not None:
        payload["RATIO"] = int(params.ratio)
    if params.n_trials is not None:
        payload["N_TRIALS"] = int(params.n_trials)
    if params.max_failures is not None:
        payload["MAX_FAILURES"] = int(params.max_failures)
    if params.seed_results is not None:
        payload["SEED_RESULTS"] = str(params.seed_results)
    if seed_context is not None:
        payload["adiabatic_seed"] = {
            "results": str(seed_context["results_path"]),
            "schema_version": int(seed_context["schema_version"]),
            "workflow": str(seed_context["workflow"]),
            "run_params": _json_ready(seed_context["run_params"]),
        }
    return payload
