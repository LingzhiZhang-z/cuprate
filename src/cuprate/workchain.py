"""Thin MPI workchain for single-cluster Hubbard projection and fitting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cuprate.clusters import Cluster, ClusterSets
from cuprate.hubbard import HubbardModel
from cuprate.mpi import comm, rank, size


RESULT_SCHEMA_VERSION = 2
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi"}


@dataclass(frozen=True)
class WorkchainParams:
    N: int
    U: float
    t: float
    mode: str
    workflow: str
    output_dir: Path
    twoSz: int | None = None
    twoS: int | None = None
    cache_mode: str = "none"
    cache_dir: Path | None = None
    ratio: int | None = None
    n_trials: int | None = None
    max_failures: int | None = None


def run_workchain(params: WorkchainParams) -> dict[str, Any] | None:
    """Run the representative-reuse workchain and return results on rank 0."""
    if params.workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"workflow={params.workflow!r} is not supported in this workchain; "
            "use one of occ, energy, greedy, greedy_multi"
        )

    output_dir = Path(params.output_dir)
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    families = _enumerate_families(params.N) if rank == 0 else None
    families = comm.bcast(families, root=0)

    local_entries: list[dict[str, Any]] = []
    for family in families[rank::size]:
        local_entries.extend(_process_family(params, family, artifacts_dir))

    gathered = comm.gather(local_entries, root=0)
    if rank != 0:
        return None

    entries = [entry for batch in gathered for entry in batch]
    entries.sort(key=lambda entry: (entry["hole"], entry["class_idx"], entry["cluster_idx"]))

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": "spin_couplings",
        "run_params": _run_params_json(params),
        "clusters": entries,
    }
    for entry in entries:
        sidecar = (
            output_dir
            / f"hole{entry['hole']}_class{entry['class_idx']}_cluster{entry['cluster_idx']}_results.json"
        )
        sidecar.write_text(json.dumps(entry, indent=2) + "\n")
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
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
    artifacts_dir: Path,
) -> list[dict[str, Any]]:
    (hole, class_idx), members = family
    representative = members[0]
    family_start = time.perf_counter()

    model = HubbardModel(representative, params.U, params.t)
    model.set_symmetry(params.mode, twoSz=params.twoSz, twoS=params.twoS)
    model.build_hamiltonians()
    cache_dir = params.cache_dir if params.cache_mode != "none" else None
    model.solve(cache_mode=params.cache_mode, cache_dir=cache_dir)

    select_kwargs = _selection_kwargs(params, artifacts_dir, hole, class_idx)
    model.project(method=params.workflow, **select_kwargs)

    artifact_name = f"hole{hole}_class{class_idx}_projection.npz"
    artifact_path = artifacts_dir / artifact_name
    _write_projection_npz(artifact_path, model)
    artifact_relpath = str(Path("artifacts") / artifact_name)

    entries: list[dict[str, Any]] = []
    for member in members:
        bond_groups = (
            member.generate_bonds(N=2, is_connected=False)
            + member.generate_bonds(N=4, is_connected=True)
            + member.generate_bonds(N=6, is_connected=True)
        )
        model.fit(bond_groups=bond_groups)
        entries.append(
            _cluster_payload(
                params=params,
                cluster=member,
                model=model,
                artifact=artifact_relpath,
                elapsed=time.perf_counter() - family_start,
            )
        )
    return entries


def _selection_kwargs(
    params: WorkchainParams,
    artifacts_dir: Path,
    hole: int,
    class_idx: int,
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
    for idx, (heff, selected) in enumerate(zip(model.heff, model.selected_indices)):
        arrays[f"block_{idx}_Heff"] = np.asarray(heff)
        arrays[f"block_{idx}_selected_indices"] = np.asarray(selected, dtype=int)
    np.savez_compressed(path, **arrays)


def _cluster_payload(
    *,
    params: WorkchainParams,
    cluster: Cluster,
    model: HubbardModel,
    artifact: str,
    elapsed: float,
) -> dict[str, Any]:
    rel_err, residual, r2 = model.fit_metrics
    t11_norm = max(float(value) for value in model.t11m1_norms)
    overlap = _aggregate_overlap(model.selection_info)
    return {
        "hole": int(cluster.hole),
        "class_idx": int(cluster.class_idx),
        "cluster_idx": int(cluster.cluster_idx),
        "sites": [[int(x), int(y)] for x, y in cluster.sites],
        "projection": {
            "method": params.workflow,
            "artifact": artifact,
            "blocks": _projection_blocks(model),
        },
        "operators": _operators_json(cluster, model.bond_groups, model.coupling_coeffs),
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


def _projection_blocks(model: HubbardModel) -> list[dict[str, Any]]:
    blocks = []
    for block, selected, t11_norm, info in zip(
        model.blocks,
        model.selected_indices,
        model.t11m1_norms,
        model.selection_info,
    ):
        overlap = _info_overlap(info)
        blocks.append(
            {
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
        )
    return blocks


def _operators_json(
    cluster: Cluster,
    bond_groups: list[list[Sequence[int]]],
    coeffs: list[Any],
) -> dict[str, Any]:
    two_site: dict[tuple[int, int], list[dict[str, Any]]] = {}
    multi_site: dict[int, list[dict[str, Any]]] = {4: [], 6: []}

    for group, group_coeffs in zip(bond_groups, coeffs[1:]):
        if not group:
            continue
        arity = len(group[0])
        terms = [
            {
                "sites": [int(site) for site in term],
                "coefficient": _complex_json(coefficient),
            }
            for term, coefficient in zip(group, group_coeffs)
        ]
        if arity == 2:
            vector = _bond_vector(cluster, group[0])
            two_site.setdefault(vector, []).extend(terms)
        elif arity in multi_site:
            multi_site[arity].append({"terms": terms})
        else:
            raise ValueError(f"Unsupported operator arity: {arity}")

    groups: list[dict[str, Any]] = []
    for label_idx, vector in enumerate(sorted(two_site, key=_vector_sort_key), start=1):
        groups.append(
            {
                "arity": 2,
                "vector": [int(vector[0]), int(vector[1])],
                "label": f"J{label_idx}",
                "terms": two_site[vector],
            }
        )

    for arity, prefix in ((4, "K"), (6, "L")):
        for label_idx, group in enumerate(multi_site[arity], start=1):
            groups.append(
                {
                    "arity": arity,
                    "vector": None,
                    "label": f"{prefix}{label_idx}",
                    "terms": group["terms"],
                }
            )

    return {
        "constant_term": _complex_json(coeffs[0]),
        "groups": groups,
    }


def _bond_vector(cluster: Cluster, bond: Sequence[int]) -> tuple[int, int]:
    site1, site2 = int(bond[0]), int(bond[1])
    dx = abs(cluster.sites[site2][0] - cluster.sites[site1][0])
    dy = abs(cluster.sites[site2][1] - cluster.sites[site1][1])
    return tuple(sorted((dx, dy), reverse=True))


def _vector_sort_key(vector: tuple[int, int]) -> tuple[int, int, int]:
    dx, dy = vector
    return (dx * dx + dy * dy, dx, dy)


def _complex_json(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


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
        return _complex_json(value)
    return value


def _run_params_json(params: WorkchainParams) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "N": int(params.N),
        "U": float(params.U),
        "T": float(params.t),
        "MODE": params.mode,
        "workflow": params.workflow,
        "CACHE_MODE": params.cache_mode,
    }
    if params.twoSz is not None:
        payload["twoSz"] = int(params.twoSz)
    if params.twoS is not None:
        payload["twoS"] = int(params.twoS)
    if params.cache_dir is not None:
        payload["CACHE_DIR"] = str(params.cache_dir)
    if params.ratio is not None:
        payload["RATIO"] = int(params.ratio)
    if params.n_trials is not None:
        payload["N_TRIALS"] = int(params.n_trials)
    if params.max_failures is not None:
        payload["MAX_FAILURES"] = int(params.max_failures)
    return payload
