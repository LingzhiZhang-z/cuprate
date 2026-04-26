"""Runtime output payloads and sidecar rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cuprate import ATOL
from cuprate.operators import (
    complex_from_json,
    complex_json,
    operator_summary,
    operators_from_terms,
)
from cuprate.paths import (
    EMBED_CLUSTERS_DIR,
    EMBED_RESULTS_FILE,
    EMBED_SUMMARY_FILE,
    EMBED_TWO_SITE_FILE,
    LCE_RESULTS_FILE,
    LCE_SUMMARY_FILE,
    LCE_WEIGHTS_DIR,
    RESULTS_FILE,
    cluster_weight_file,
    family_clusters_file,
    family_exchange_file,
    main_data_dir,
    mode_token,
    parameter_token,
    seed_token,
    workflow_token,
)


SPIN_COUPLINGS_SCHEMA_VERSION = 5
LCE_SCHEMA_VERSION = 2
LCE_WEIGHT_SCHEMA_VERSION = 1
EMBED_SCHEMA_VERSION = 2


def write_main_manifest(
    output_dir: Path,
    params: Any,
    entries: list[dict[str, Any]],
    seed_context: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SPIN_COUPLINGS_SCHEMA_VERSION,
        "result_kind": "spin_couplings",
        "complete_family_set": True,
        "run_params": _run_params_json(params, seed_context),
        "families": entries,
    }
    _write_json(output_dir / RESULTS_FILE, payload)
    return payload


def write_projection_npz(path: Path, model: Any) -> None:
    arrays: dict[str, np.ndarray] = {
        "n_blocks": np.array(len(model.blocks), dtype=int),
        "block_labels": np.array([block.label() for block in model.blocks]),
        "twoSz": np.array(
            [np.nan if block.twoSz is None else float(block.twoSz) for block in model.blocks]
        ),
        "twoS": np.array(
            [np.nan if block.twoS is None else float(block.twoS) for block in model.blocks]
        ),
        "eta": np.array(
            [np.nan if block.eta is None else float(block.eta) for block in model.blocks]
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


def write_family_outputs(
    *,
    exchanges_dir: Path,
    clusters_dir: Path,
    params: Any,
    representative: Any,
    members: list[Any],
    model: Any,
    artifact: str,
    constant: complex,
    terms: dict,
    elapsed: float,
    rank: int,
) -> dict[str, Any]:
    hole = int(representative.hole)
    class_idx = int(representative.class_idx)
    exchange_name = family_exchange_file(hole, class_idx)
    clusters_name = family_clusters_file(hole, class_idx)
    exchange_relpath = str(Path("exchanges") / exchange_name)
    clusters_relpath = str(Path("clusters") / clusters_name)

    exchange_payload = _exchange_payload(
        params=params,
        representative=representative,
        model=model,
        artifact=artifact,
        constant=constant,
        terms=terms,
        elapsed=elapsed,
        rank=rank,
    )
    clusters_payload = _clusters_payload(
        params=params,
        representative=representative,
        members=members,
    )

    _write_json(exchanges_dir / exchange_name, exchange_payload)
    (exchanges_dir / Path(exchange_name).with_suffix(".txt").name).write_text(
        _exchange_text(exchange_payload) + "\n"
    )
    _write_json(clusters_dir / clusters_name, clusters_payload)
    (clusters_dir / Path(clusters_name).with_suffix(".txt").name).write_text(
        _clusters_text(clusters_payload) + "\n"
    )

    return {
        "hole": hole,
        "class_idx": class_idx,
        "exchange_file": exchange_relpath,
        "clusters_file": clusters_relpath,
    }


def write_lce_outputs(
    *,
    output_dir: Path,
    params: Any,
    records_by_n: dict[int, list[Any]],
    source_inputs: list[dict[str, Any]],
    seed_set_sha256: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / LCE_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    summary_entries = []
    for N in sorted(records_by_n):
        n_dir = weights_dir / f"N_{int(N)}"
        n_dir.mkdir(parents=True, exist_ok=True)
        for record in records_by_n[N]:
            weight_name = cluster_weight_file(record.hole, record.class_idx, record.cluster_idx)
            weight_text_name = Path(weight_name).with_suffix(".txt").name
            relative_weight_file = f"{LCE_WEIGHTS_DIR}/N_{int(N)}/{weight_name}"
            weight_payload = _lce_weight_payload(record)
            _write_json(n_dir / weight_name, weight_payload)
            (n_dir / weight_text_name).write_text(_lce_weight_text(weight_payload) + "\n")
            entry = {
                "N": int(record.N),
                "hole": int(record.hole),
                "class_idx": int(record.class_idx),
                "cluster_idx": int(record.cluster_idx),
                "weight_file": relative_weight_file,
            }
            entries.append(entry)
            summary_entries.append({**entry, "diagnostics": weight_payload["diagnostics"]})

    payload = {
        "schema_version": LCE_SCHEMA_VERSION,
        "result_kind": "lce_spin_couplings",
        "seed_set": Path(params.seed_set).stem,
        "seed_token": seed_token(params.seed_set),
        "seed_set_file": str(params.seed_set),
        "seed_set_sha256": seed_set_sha256,
        "source_inputs": _json_ready(source_inputs),
        "run_params": {
            "N_min": min(records_by_n),
            "N_max": max(records_by_n),
            "N": int(params.N),
            "nelec": int(params.N),
            "U": float(params.U),
            "T": float(params.t),
            "ROOT": str(params.root),
            "parameter_token": parameter_token(params.N, params.N, params.U, params.t),
            "seed_token": seed_token(params.seed_set),
        },
        "weights": entries,
    }
    _write_json(output_dir / LCE_RESULTS_FILE, payload)
    (output_dir / LCE_SUMMARY_FILE).write_text(_lce_summary_text(payload, summary_entries) + "\n")
    return payload


def write_embed_outputs(
    *,
    output_dir: Path,
    params: Any,
    source_lce_path: Path,
    lce_manifest: dict[str, Any],
    records: list[Any],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
    two_site_entries: list[dict[str, Any]],
    cluster_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir = output_dir / EMBED_CLUSTERS_DIR
    clusters_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / EMBED_TWO_SITE_FILE).write_text(
        _embed_two_site_text(source_lce_path, params.N, two_site_entries) + "\n"
    )

    cluster_files = []
    for output in cluster_outputs:
        file_name = str(output["file_name"])
        (clusters_dir / file_name).write_text(_embed_cluster_text(output) + "\n")
        cluster_files.append(
            {
                "N": int(output["N"]),
                "hole": int(output["hole"]),
                "class_idx": int(output["class_idx"]),
                "cluster_idx": int(output["cluster_idx"]),
                "file": f"{EMBED_CLUSTERS_DIR}/{file_name}",
            }
        )

    payload = _embed_manifest_payload(
        params,
        source_lce_path,
        lce_manifest,
        records,
        orientation_cache,
        cluster_files,
        two_site_entries,
    )
    _write_json(output_dir / EMBED_RESULTS_FILE, payload)
    (output_dir / EMBED_SUMMARY_FILE).write_text(
        _embed_summary_text(payload, source_lce_path, records, orientation_cache) + "\n"
    )
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _exchange_payload(
    *,
    params: Any,
    representative: Any,
    model: Any,
    artifact: str,
    constant: complex,
    terms: dict,
    elapsed: float,
    rank: int,
) -> dict[str, Any]:
    rel_err, residual, r2 = model.fit_metrics
    t11_norm = max(float(value) for value in model.t11m1_norms)
    overlap = _aggregate_overlap(model.selection_info)
    return {
        "schema_version": SPIN_COUPLINGS_SCHEMA_VERSION,
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
    params: Any,
    representative: Any,
    members: list[Any],
) -> dict[str, Any]:
    return {
        "schema_version": SPIN_COUPLINGS_SCHEMA_VERSION,
        "result_kind": "cluster_family_geometry",
        "N": int(params.N),
        "hole": int(representative.hole),
        "class_idx": int(representative.class_idx),
        "representative_cluster_idx": int(representative.cluster_idx),
        "representative_sites": [[int(x), int(y)] for x, y in representative.sites],
        "clusters": [_cluster_geometry_payload(member) for member in members],
    }


def _cluster_geometry_payload(cluster: Any) -> dict[str, Any]:
    return {
        "cluster_idx": int(cluster.cluster_idx),
        "sites": [[int(x), int(y)] for x, y in cluster.sites],
        "indices": list(range(cluster.N)),
    }


def _projection_blocks(model: Any) -> list[dict[str, Any]]:
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
            "eta": None if block.eta is None else int(block.eta),
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


def _run_params_json(
    params: Any,
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
        "SCOPE": params.scope,
        "workflow": params.workflow,
        "ROOT": str(params.root),
        "parameter_token": parameter_token(params.N, params.N, params.U, params.t),
        "mode_token": mode_token(
            params.mode,
            twoSz=params.twoSz,
            twoS=params.twoS,
            scope=params.scope,
        ),
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


def _exchange_text(payload: dict[str, Any]) -> str:
    sites = payload["representative_sites"]
    fit = payload["fit"]
    lines = [
        (
            "Family: "
            f"hole={payload['hole']} class={payload['class_idx']} "
            f"representative={payload['representative_cluster_idx']}"
        ),
        f"N={payload['N']}",
        f"Sites: {_sites_line(sites)}",
        "",
        "Projection:",
        f"  method={payload['projection']['method']} artifact={payload['projection']['artifact']}",
    ]
    for block in payload["projection"]["blocks"]:
        lines.append(
            "  "
            f"block={block['block']} "
            f"twoSz={_none_as_all(block['twoSz'])} "
            f"twoS={_none_as_all(block['twoS'])} "
            f"eta={_none_as_all(block['eta'])} "
            f"spin_dim={block['spin_dim']} "
            f"selected={block['selected_state_count']} "
            f"|T11-I|={_format_float(block['t11_minus_1_norm'])} "
            f"overlap={_format_optional_float(block['overlap'])}"
        )
        lines.append(f"    selected_indices={_int_list(block['selected_indices'])}")

    constant = complex_from_json(payload["operators"]["constant_term"])
    lines.extend(
        [
            "",
            "Fit:",
            (
                f"  R2={_format_float(fit['r_squared'])} "
                f"relative_error={_format_float(fit['relative_error'])} "
                f"residual={_format_float(fit['residual'])}"
            ),
            (
                f"  |T11-I|={_format_float(fit['t11_minus_1_norm'])} "
                f"overlap={_format_optional_float(fit['overlap'])}"
            ),
            f"  constant={_format_complex(constant)}",
            "",
            "Couplings:",
        ]
    )

    lines.extend(_operator_groups_text(sites, payload["operators"]))
    return "\n".join(lines)


def _clusters_text(payload: dict[str, Any]) -> str:
    lines = [
        (
            "Family: "
            f"hole={payload['hole']} class={payload['class_idx']} "
            f"representative={payload['representative_cluster_idx']}"
        ),
        f"N={payload['N']}",
        f"Representative sites: {_sites_line(payload['representative_sites'])}",
        "",
        "Clusters:",
    ]
    for cluster in payload["clusters"]:
        lines.append(f"  cluster {cluster['cluster_idx']}:")
        indexed_sites = sorted(
            zip(cluster["indices"], cluster["sites"]),
            key=lambda item: int(item[0]),
        )
        for index, site in indexed_sites:
            lines.append(f"    {int(index)} -> {_coord(site)}")
    return "\n".join(lines)


def _lce_weight_payload(record: Any) -> dict[str, Any]:
    net_summary = operator_summary(record.net_constant, record.net_terms)
    return {
        "schema_version": LCE_WEIGHT_SCHEMA_VERSION,
        "result_kind": "lce_cluster_weight",
        "N": int(record.N),
        "hole": int(record.hole),
        "class_idx": int(record.class_idx),
        "cluster_idx": int(record.cluster_idx),
        "sites": [[int(x), int(y)] for x, y in record.sites],
        "indices": list(range(record.N)),
        "raw_summary": operator_summary(record.raw_constant, record.raw_terms),
        "operators": operators_from_terms(record.sites, record.net_constant, record.net_terms),
        "diagnostics": {
            "subcluster_count": int(record.subcluster_count),
            "reconstruction_error": float(record.reconstruction_error),
            "net_term_count": int(len(record.net_terms)),
            "max_abs_net_coefficient": float(net_summary["max_abs_coefficient"]),
        },
    }


def _lce_weight_text(payload: dict[str, Any]) -> str:
    raw_summary = payload["raw_summary"]
    diagnostics = payload["diagnostics"]
    constant = complex_from_json(payload["operators"]["constant_term"])
    raw_constant = complex_from_json(raw_summary["constant_term"])
    lines = [
        (
            "LCE weight: "
            f"N={payload['N']} hole={payload['hole']} "
            f"class={payload['class_idx']} cluster={payload['cluster_idx']}"
        ),
        f"Sites: {_sites_line(payload['sites'])}",
        "",
        "Diagnostics:",
        f"  subclusters={diagnostics['subcluster_count']}",
        f"  reconstruction_error={_format_float(diagnostics['reconstruction_error'])}",
        f"  net_term_count={diagnostics['net_term_count']}",
        f"  max_abs_net_coefficient={_format_float(diagnostics['max_abs_net_coefficient'])}",
        "",
        "Raw summary:",
        f"  term_count={raw_summary['term_count']}",
        f"  constant={_format_complex(raw_constant)}",
        f"  max_abs_coefficient={_format_float(raw_summary['max_abs_coefficient'])}",
        "",
        "Net couplings:",
        f"  constant={_format_complex(constant)}",
        "",
        "Couplings:",
    ]
    lines.extend(_operator_groups_text(payload["sites"], payload["operators"]))
    return "\n".join(lines)


def _lce_summary_text(payload: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    lines = [
        "LCE summary",
        f"schema_version: {payload['schema_version']}",
        f"seed_set: {payload['seed_set']}",
        f"seed_token: {payload['seed_token']}",
        f"seed_set_file: {payload['seed_set_file']}",
        f"seed_set_sha256: {payload['seed_set_sha256']}",
        f"N range: {payload['run_params']['N_min']}..{payload['run_params']['N_max']}",
        f"source_inputs: {len(payload['source_inputs'])}",
        "",
        "Weights:",
    ]
    for entry in entries:
        diagnostics = entry["diagnostics"]
        lines.append(
            "N={N} hole={hole} class={class_idx} cluster={cluster_idx} "
            "weight_file={weight_file} max_net={max_net:.12e} recon_error={recon:.12e}".format(
                N=entry["N"],
                hole=entry["hole"],
                class_idx=entry["class_idx"],
                cluster_idx=entry["cluster_idx"],
                weight_file=entry["weight_file"],
                max_net=diagnostics["max_abs_net_coefficient"],
                recon=diagnostics["reconstruction_error"],
            )
        )
    return "\n".join(lines)


def _embed_two_site_text(
    source_lce_path: Path,
    Nmax: int,
    entries: list[dict[str, Any]],
) -> str:
    lines = [
        "# two-site embedded couplings",
        f"# source: {source_lce_path}",
        f"# Nmax: {Nmax}",
        "# columns: index vector real imag",
        "",
    ]
    for entry in entries:
        vector = entry["vector"]
        value = entry["value"]
        prefix = f"{entry['index']}  ({vector[0]},{vector[1]})"
        if value is None:
            lines.append(f"{prefix}  None")
        else:
            lines.append(f"{prefix}  {_format_complex_pair(value)}")
    return "\n".join(lines)


def _embed_cluster_text(output: dict[str, Any]) -> str:
    sites = output["sites"]
    lines = [
        "sites: " + " ".join(f"({int(x)},{int(y)})" for x, y in sites),
        "indices: " + " ".join(str(index) for index in range(len(sites))),
        "",
    ]
    for key, value in zip(output["pairings"], output["values"]):
        label = "".join(f"(S{int(i)} S{int(j)})" for i, j in key)
        if value is None:
            lines.append(f"{label}  None")
        else:
            lines.append(f"{label}  {_format_complex_pair(value)}")
    return "\n".join(lines)


def _embed_manifest_payload(
    params: Any,
    source_lce_path: Path,
    lce_manifest: dict[str, Any],
    records: list[Any],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
    cluster_files: list[dict[str, Any]],
    two_site_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster_file_counts: dict[str, int] = {}
    for entry in cluster_files:
        key = f"N{entry['N']}"
        cluster_file_counts[key] = cluster_file_counts.get(key, 0) + 1

    return {
        "schema_version": EMBED_SCHEMA_VERSION,
        "result_kind": "embedded_spin_couplings",
        "source_lce_manifest": str(source_lce_path),
        "source_lce_schema_version": int(lce_manifest.get("schema_version", 0)),
        "seed_set": lce_manifest.get("seed_set"),
        "seed_token": lce_manifest.get("seed_token"),
        "seed_set_file": lce_manifest.get("seed_set_file"),
        "seed_set_sha256": lce_manifest.get("seed_set_sha256"),
        "source_inputs": _json_ready(lce_manifest.get("source_inputs", [])),
        "run_params": {
            "U": float(params.U),
            "T": float(params.t),
            "N": int(params.N),
            "Nmax": int(params.N),
            "nelec": int(params.N),
            "ROOT": str(params.root),
            "parameter_token": parameter_token(params.N, params.N, params.U, params.t),
            "seed_token": seed_token(params.seed_set),
        },
        "two_site_file": EMBED_TWO_SITE_FILE,
        "cluster_files": cluster_files,
        "diagnostics": {
            "weights_read": int(len(records)),
            "two_site_candidate_count": int(len(two_site_entries)),
            "cluster_file_counts": cluster_file_counts,
            "orientation_counts": [
                {
                    "N": record.N,
                    "hole": record.hole,
                    "class_idx": record.class_idx,
                    "cluster_idx": record.cluster_idx,
                    "weight_file": record.weight_file,
                    "distinct_orientations": len(orientation_cache[record.identity()]),
                }
                for record in records
            ],
        },
    }


def _embed_summary_text(
    payload: dict[str, Any],
    source_lce_path: Path,
    records: list[Any],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
) -> str:
    run_params = payload["run_params"]
    diagnostics = payload["diagnostics"]
    lines = [
        "Embed summary",
        f"schema_version: {payload['schema_version']}",
        f"source_lce_manifest: {source_lce_path}",
        f"seed_set: {payload['seed_set']}",
        f"seed_token: {payload['seed_token']}",
        f"seed_set_file: {payload['seed_set_file']}",
        f"seed_set_sha256: {payload['seed_set_sha256']}",
        f"Nmax: {run_params['Nmax']}",
        f"U: {run_params['U']:.12g}",
        f"T: {run_params['T']:.12g}",
        f"weights_read: {diagnostics['weights_read']}",
        f"two_site_candidate_count: {diagnostics['two_site_candidate_count']}",
        "cluster_file_counts: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(diagnostics["cluster_file_counts"].items())
        ),
        "",
        "Distinct parent orientations:",
    ]
    for record in records:
        lines.append(
            "N={N} hole={hole} class={class_idx} cluster={cluster_idx} "
            "weight_file={weight_file} orientations={count}".format(
                N=record.N,
                hole=record.hole,
                class_idx=record.class_idx,
                cluster_idx=record.cluster_idx,
                weight_file=record.weight_file,
                count=len(orientation_cache[record.identity()]),
            )
        )
    return "\n".join(lines)


def _operator_groups_text(
    sites: list[list[int]],
    operators: dict[str, Any],
    *,
    indent: str = "  ",
) -> list[str]:
    groups = operators["groups"]
    if not groups:
        return [f"{indent}(none)"]

    lines = []
    for group in groups:
        if group["arity"] == 2:
            lines.append(f"{indent}{group['label']}  vector={_coord(group['vector'])}")
        else:
            support = group.get("support", [])
            lines.append(
                f"{indent}{group['label']}  support={_int_list(support)} "
                f"coords={_support_coords(sites, support)}"
            )
        for term in group["terms"]:
            coefficient = complex_from_json(term["coefficient"])
            lines.append(
                f"{indent}  "
                f"{_operator_text(term['key'])} "
                f"sites={_term_sites_text(term['key'])} "
                f"coords={_term_coords_text(sites, term['key'])} "
                f"coefficient={_format_complex(coefficient)}"
            )
    return lines


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


def _sites_line(sites: list[list[int]]) -> str:
    return "  ".join(f"{idx}:{_coord(site)}" for idx, site in enumerate(sites))


def _support_coords(sites: list[list[int]], support: list[int]) -> str:
    if not support:
        return "(none)"
    return " ".join(f"{int(site)}:{_coord(sites[int(site)])}" for site in support)


def _term_coords_text(sites: list[list[int]], key: list[list[int]]) -> str:
    return " ".join(f"{_coord(sites[int(i)])}-{_coord(sites[int(j)])}" for i, j in key)


def _term_sites_text(key: list[list[int]]) -> str:
    return " ".join(f"{int(i)}-{int(j)}" for i, j in key)


def _operator_text(key: list[list[int]]) -> str:
    return "".join(f"(S{int(i)}.S{int(j)})" for i, j in key)


def _coord(site: list[int]) -> str:
    return f"({int(site[0])},{int(site[1])})"


def _int_list(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _none_as_all(value: Any) -> str:
    return "all" if value is None else str(int(value))


def _format_optional_float(value: Any) -> str:
    return "none" if value is None else _format_float(value)


def _format_float(value: Any) -> str:
    return f"{float(value):.12e}"


def _format_complex(value: complex) -> str:
    value = complex(value)
    if abs(value.imag) <= ATOL["tight"]:
        return _format_float(value.real)
    sign = "+" if value.imag >= 0.0 else "-"
    return f"{_format_float(value.real)}{sign}{_format_float(abs(value.imag))}i"


def _format_complex_pair(value: complex) -> str:
    value = complex(value)
    return f"{value.real:.12e}  {value.imag:.12e}"
