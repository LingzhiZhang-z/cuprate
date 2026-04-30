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
from cuprate.io import (
    SPIN_COUPLINGS_SCHEMA_VERSION,
    write_family_outputs,
    write_main_manifest,
    write_projection_npz,
)
from cuprate.paths import (
    STAGE_MAIN,
    family_projection_file,
    main_data_dir,
    workflow_dir,
)
from cuprate.mpi import comm, rank, size
from cuprate.operators import terms_from_fit


SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi", "adiabatic"}


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def _print_progress(message: str) -> None:
    print(f"[main-progress] {message}", flush=True)


def _family_label(family: tuple[tuple[int, int], list[Cluster]]) -> str:
    (hole, class_idx), members = family
    return f"hole={hole} class={class_idx} members={len(members)}"


def _rank0_progress_line(done: int, total: int, started_at: float) -> str:
    elapsed = time.perf_counter() - started_at
    eta = 0.0 if done <= 0 else (elapsed / done) * (total - done)
    return (
        f"rank0_progress={done}/{total} "
        f"elapsed={_format_duration(elapsed)} "
        f"ETA={_format_duration(eta)}"
    )


@dataclass(frozen=True)
class WorkchainParams:
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
    scope: str
    workflow: str
    root: Path
    cache_mode: str = "save"
    ratio: int | None = None
    n_trials: int | None = None
    max_failures: int | None = None
    seed_results: Path | None = None
    merge: str = "none"
    merge_basis: str | None = None
    eigh: str = "lowmem"


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
        scope=params.scope,
        merge=params.merge,
        merge_basis=params.merge_basis,
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
    assert families is not None

    total_families = len(families)
    local_families = families[rank::size]
    workchain_start = time.perf_counter()
    if rank == 0:
        _print_progress(
            f"start N={params.N} U={params.U:g} T={params.t:g} "
            f"mode={params.mode} workflow={params.workflow} "
            f"families={total_families} mpi_size={size} "
            f"rank0_families={len(local_families)}"
        )

    local_entries: list[dict[str, Any]] = []
    for local_index, family in enumerate(local_families, start=1):
        label = _family_label(family)
        family_start = time.perf_counter()
        local_entries.append(
            _process_family(params, family, exchanges_dir, clusters_dir, artifacts_dir, seed_context)
        )
        family_elapsed = time.perf_counter() - family_start
        if rank == 0:
            _print_progress(
                f"{_rank0_progress_line(local_index, len(local_families), workchain_start)} "
                f"total_families={total_families} {label} "
                f"family_time={_format_duration(family_elapsed)}"
            )

    gathered = comm.gather(local_entries, root=0)
    if rank != 0:
        return None

    entries = [entry for batch in gathered for entry in batch]
    entries.sort(key=lambda entry: (entry["hole"], entry["class_idx"]))

    manifest = write_main_manifest(output_dir, params, entries, seed_context)
    _print_progress(
        f"done families={len(entries)}/{total_families} "
        f"total_time={_format_duration(time.perf_counter() - workchain_start)}"
    )
    return manifest


def _enumerate_families(N: int) -> list[tuple[tuple[int, int], list[Cluster]]]:
    by_family: dict[tuple[int, int], list[Cluster]] = {}
    for cluster in ClusterSets(N).generate().clusters:
        key = (int(cluster.hole), int(cluster.class_idx))
        by_family.setdefault(key, []).append(cluster)
    return [(key, by_family[key]) for key in sorted(by_family)]


def _family_error_context(params: WorkchainParams, representative: Cluster) -> str:
    return (
        f"family hole={int(representative.hole)} "
        f"class={int(representative.class_idx)} "
        f"representative={representative.label()} "
        f"cluster_idx={int(representative.cluster_idx)} "
        f"rank={rank} N={params.N} mode={params.mode} "
        f"workflow={params.workflow} cache_mode={params.cache_mode} "
        f"merge={params.merge} merge_basis={params.merge_basis}"
    )


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

    try:
        model = HubbardModel(representative, params.U, params.t)
        model.set_symmetry(params.mode, twoSz=params.twoSz, twoS=params.twoS, scope=params.scope)
        model.build_hamiltonians()
        cache_dir = (
            main_data_dir(params.root, params.N, params.N, params.U, params.t, params.mode)
            if params.cache_mode != "none"
            else None
        )
        model.solve(cache_mode=params.cache_mode, cache_dir=cache_dir, eigh=params.eigh)
        if params.merge == "Sz":
            model.merge_to_sz(params.merge_basis or "fock")
        # Cache save/load is complete; projection and fit do not need ham.
        for block in model.blocks:
            block.ham = None

        select_kwargs = _selection_kwargs(params, artifacts_dir, hole, class_idx, model, seed_context)
        model.project(method=params.workflow, **select_kwargs)

        artifact_name = family_projection_file(hole, class_idx)
        artifact_path = artifacts_dir / artifact_name
        write_projection_npz(artifact_path, model)
        # The artifact has captured eigvecs_fock; fit/output only need Heff and spin bases.
        for block in model.blocks:
            block.eigvals = None
            block.eigvecs = None
        artifact_relpath = str(Path("artifacts") / artifact_name)

        bond_groups = _fit_bond_groups(representative)
        model.fit(bond_groups=bond_groups)
        constant, terms = terms_from_fit(model.bond_groups, model.coupling_coeffs)
        elapsed = time.perf_counter() - family_start

        return write_family_outputs(
            exchanges_dir=exchanges_dir,
            clusters_dir=clusters_dir,
            params=params,
            representative=representative,
            members=members,
            model=model,
            artifact=artifact_relpath,
            constant=constant,
            terms=terms,
            elapsed=elapsed,
            rank=rank,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} [{_family_error_context(params, representative)}]") from exc
    except ValueError as exc:
        raise ValueError(f"{exc} [{_family_error_context(params, representative)}]") from exc


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


def _fit_bond_groups(cluster: Cluster) -> list[list[Any]]:
    return (
        cluster.generate_bonds(N=2, is_connected=False)
        + cluster.generate_bonds(N=4, is_connected=True)
        + cluster.generate_bonds(N=6, is_connected=True)
    )


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
    if schema_version != SPIN_COUPLINGS_SCHEMA_VERSION:
        raise ValueError(
            f"SEED_RESULTS schema_version={schema_version} cannot seed adiabatic; "
            f"regenerate with schema_version={SPIN_COUPLINGS_SCHEMA_VERSION}"
        )
    run_params = payload.get("run_params")
    if not isinstance(run_params, dict):
        raise ValueError(f"SEED_RESULTS is missing run_params: {seed_path}")
    seed_workflow = run_params.get("workflow")
    if seed_workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"SEED_RESULTS has unsupported workflow={seed_workflow!r}")
    seed_merge = run_params.get("MERGE", "none")
    seed_merge_basis = run_params.get("MERGE_BASIS")
    if seed_merge != params.merge or seed_merge_basis != params.merge_basis:
        raise ValueError(
            "SEED_RESULTS merge settings do not match current run: "
            f"seed MERGE={seed_merge!r} MERGE_BASIS={seed_merge_basis!r}; "
            f"current MERGE={params.merge!r} MERGE_BASIS={params.merge_basis!r}"
        )

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

    corrupted = (
        "Adiabatic seed artifact is invalid/corrupted: "
        f"results={seed_context['results_path']} artifact={artifact_path} "
        f"hole={hole} class={class_idx}"
    )
    try:
        with np.load(artifact_path, allow_pickle=False) as data:
            seed_map = _seed_map_from_npz(data, artifact, artifact_path)

        for block in model.blocks:
            label = block.label()
            if label not in seed_map:
                raise ValueError
            seed = seed_map[label]
            if seed["basis_states"] != list(block.basis_states):
                raise ValueError
            eigvecs_previous = seed["eigvecs_previous"]
            if eigvecs_previous.shape[0] != len(block.basis_states):
                raise ValueError
            selected_previous = np.asarray(seed["selected_previous"], dtype=int)
            if selected_previous.size != block.spin_dim:
                raise ValueError
            if selected_previous.size and (
                selected_previous.min() < 0 or selected_previous.max() >= eigvecs_previous.shape[1]
            ):
                raise ValueError
    except (OSError, KeyError, IndexError, ValueError):
        raise ValueError(corrupted) from None
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
    try:
        return json.loads(exchange_path.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"seed exchange file is invalid/corrupted: {exchange_path}") from None


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
