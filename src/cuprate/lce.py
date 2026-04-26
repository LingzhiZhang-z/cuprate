"""Linked-cluster expansion for fitted spin-coupling JSON outputs."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from cuprate import ATOL
from cuprate.clusters import connected_subsets, graph_from_sites
from cuprate.paths import (
    LCE_RESULTS_FILE,
    LCE_SUMMARY_FILE,
    LCE_WEIGHTS_DIR,
    RESULTS_FILE,
    STAGE_LCE,
    STAGE_MAIN,
    cluster_weight_file,
    mode_spec,
    mode_token,
    parameter_token,
    workflow_dir,
    workflow_token,
)
from cuprate.operators import (
    OperatorKey,
    map_key,
    max_operator_difference,
    operator_summary,
    operators_from_terms,
    operators_to_terms,
)


LCE_SCHEMA_VERSION = 1
REQUIRED_KEYS = {"ROOT", "N", "U", "T", "MODE", "workflow"}
CANONICAL_KEYS = {
    "root": "ROOT",
    "n": "N",
    "u": "U",
    "t": "T",
    "mode": "MODE",
    "twosz": "twoSz",
    "twos": "twoS",
    "workflow": "workflow",
}
COMMON_RUN_PARAM_KEYS = ("U", "T", "MODE", "twoSz", "twoS", "workflow")
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi", "adiabatic"}


@dataclass(frozen=True)
class LCEParams:
    root: Path
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
    workflow: str


@dataclass
class ClusterRecord:
    N: int
    hole: int
    class_idx: int
    cluster_idx: int
    sites: tuple[tuple[int, int], ...]
    raw_constant: complex
    raw_terms: dict[OperatorKey, complex]
    net_constant: complex = 0.0
    net_terms: dict[OperatorKey, complex] = field(default_factory=dict)
    subcluster_count: int = 0
    reconstruction_error: float = 0.0

    def identity(self) -> tuple[int, int, int, int]:
        return (self.N, self.hole, self.class_idx, self.cluster_idx)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        params = parse_args(argv)
        run_lce(params)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def parse_args(argv: list[str]) -> LCEParams:
    raw: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"expected KEY=VALUE argument, got {arg!r}")
        key, value = arg.split("=", 1)
        canonical = CANONICAL_KEYS.get(key.lower())
        if canonical is None:
            raise ValueError(f"unknown parameter {key!r}")
        if canonical in raw:
            raise ValueError(f"duplicate parameter {canonical}")
        raw[canonical] = value

    missing = sorted(REQUIRED_KEYS - raw.keys())
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}")

    N = _parse_int(raw["N"], "N")
    U = _parse_float(raw["U"], "U")
    t = _parse_float(raw["T"], "T")
    twoSz = _parse_optional_int(raw, "twoSz")
    twoS = _parse_optional_int(raw, "twoS")
    spec = mode_spec(raw["MODE"], twoSz=twoSz, twoS=twoS)
    mode = spec.mode
    workflow = raw["workflow"].lower()
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"unsupported workflow={raw['workflow']!r}")
    _validate_mode_args(N, spec)
    return LCEParams(
        root=Path(raw["ROOT"]),
        N=N,
        U=U,
        t=t,
        mode=mode,
        twoSz=spec.twoSz,
        twoS=spec.twoS,
        workflow=workflow,
    )


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float") from exc


def _parse_optional_int(raw: dict[str, str], key: str) -> int | None:
    if key not in raw:
        return None
    return _parse_int(raw[key], key)


def _validate_mode_args(N: int, spec) -> None:
    if spec.twoSz is not None:
        if abs(spec.twoSz) > N:
            raise ValueError("twoSz must satisfy |twoSz| <= N")
        if spec.twoSz % 2 != N % 2:
            raise ValueError("twoSz parity must match N")
    if spec.twoS is not None:
        if not (0 <= spec.twoS <= N):
            raise ValueError("twoS must satisfy 0 <= twoS <= N")
        if spec.twoS % 2 != N % 2:
            raise ValueError("twoS parity must match N")
        if spec.twoSz is not None and abs(spec.twoSz) > spec.twoS:
            raise ValueError("twoSz and twoS must satisfy |twoSz| <= twoS")


def run_lce(params: LCEParams) -> dict[str, Any]:
    input_paths = _input_paths(params)
    records_by_n, common_params = _load_input_records(input_paths)
    _compute_lce(records_by_n)

    output_dir = workflow_dir(
        params.root,
        STAGE_LCE,
        params.N,
        params.N,
        params.U,
        params.t,
        params.mode,
        params.workflow,
        twoSz=params.twoSz,
        twoS=params.twoS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / LCE_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    summary_entries = []
    for N in sorted(records_by_n):
        for record in records_by_n[N]:
            weight_name = cluster_weight_file(record.hole, record.class_idx, record.cluster_idx)
            relative_weight_file = f"{LCE_WEIGHTS_DIR}/{weight_name}"
            weight_payload = _weight_payload(record)
            (weights_dir / weight_name).write_text(json.dumps(weight_payload, indent=2) + "\n")
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
        "source_inputs": [str(path) for path in input_paths],
        "run_params": {
            **common_params,
            "N_min": min(records_by_n),
            "N_max": max(records_by_n),
            "ROOT": str(params.root),
            "parameter_token": parameter_token(params.N, params.N, params.U, params.t),
            "mode_token": mode_token(params.mode, twoSz=params.twoSz, twoS=params.twoS),
            "workflow_token": workflow_token(params.workflow),
        },
        "weights": entries,
    }
    (output_dir / LCE_RESULTS_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / LCE_SUMMARY_FILE).write_text(_summary_text(payload, summary_entries) + "\n")
    return payload


def _input_paths(params: LCEParams) -> list[Path]:
    return [
        workflow_dir(
            params.root,
            STAGE_MAIN,
            N,
            N,
            params.U,
            params.t,
            params.mode,
            params.workflow,
            twoSz=params.twoSz,
            twoS=params.twoS,
        )
        / RESULTS_FILE
        for N in range(2, params.N + 1)
    ]


def _load_input_records(
    input_paths: list[Path],
) -> tuple[dict[int, list[ClusterRecord]], dict[str, Any]]:
    records_by_n: dict[int, list[ClusterRecord]] = {}
    common_params: dict[str, Any] | None = None

    for path in input_paths:
        payload = json.loads(path.read_text())
        if payload.get("result_kind") != "spin_couplings":
            raise ValueError(f"{path} is not a spin_couplings results file")
        if int(payload.get("schema_version", 0)) != 5:
            raise ValueError(f"{path} must use spin_couplings schema_version=5")
        if payload.get("complete_family_set", True) is not True:
            raise ValueError(f"{path} is a partial main output and cannot be used for LCE")

        run_params = payload["run_params"]
        N = int(run_params["N"])
        if N in records_by_n:
            raise ValueError(f"duplicate input for N={N}")
        current_common = {key: run_params.get(key) for key in COMMON_RUN_PARAM_KEYS}
        if common_params is None:
            common_params = current_common
        elif current_common != common_params:
            raise ValueError("all inputs must share U/T/MODE/twoSz/twoS/workflow")

        records = _records_from_manifest(path, N, payload)
        records.sort(key=lambda record: (record.hole, record.class_idx, record.cluster_idx))
        records_by_n[N] = records

    expected = list(range(2, max(records_by_n) + 1))
    if sorted(records_by_n) != expected:
        raise ValueError(f"main results must cover consecutive N values {expected}")
    return records_by_n, common_params or {}


def _records_from_manifest(path: Path, N: int, payload: dict[str, Any]) -> list[ClusterRecord]:
    records = []
    results_dir = path.parent
    families = payload.get("families")
    if not isinstance(families, list):
        raise ValueError(f"{path} is missing families")

    for family in families:
        exchange_file = family.get("exchange_file")
        clusters_file = family.get("clusters_file")
        if not isinstance(exchange_file, str) or not exchange_file:
            raise ValueError(f"{path} has a family entry missing exchange_file")
        if not isinstance(clusters_file, str) or not clusters_file:
            raise ValueError(f"{path} has a family entry missing clusters_file")

        exchange_path = results_dir / exchange_file
        clusters_path = results_dir / clusters_file
        exchange = json.loads(exchange_path.read_text())
        geometry = json.loads(clusters_path.read_text())

        hole = int(exchange["hole"])
        class_idx = int(exchange["class_idx"])
        if hole != int(geometry["hole"]) or class_idx != int(geometry["class_idx"]):
            raise ValueError(f"family metadata mismatch between {exchange_path} and {clusters_path}")

        constant, terms = operators_to_terms(exchange["operators"])
        for cluster_entry in geometry["clusters"]:
            records.append(
                ClusterRecord(
                    N=N,
                    hole=hole,
                    class_idx=class_idx,
                    cluster_idx=int(cluster_entry["cluster_idx"]),
                    sites=_cluster_sites_in_operator_order(cluster_entry, N),
                    raw_constant=constant,
                    raw_terms=terms,
                )
            )
    return records


def _cluster_sites_in_operator_order(
    cluster_entry: dict[str, Any],
    N: int,
) -> tuple[tuple[int, int], ...]:
    sites = [tuple(map(int, site)) for site in cluster_entry["sites"]]
    indices = [int(index) for index in cluster_entry["indices"]]
    if len(sites) != N or len(indices) != N:
        raise ValueError(f"cluster sites and indices must both have length N={N}")
    if sorted(indices) != list(range(N)):
        raise ValueError(f"cluster indices must be a permutation of 0..{N - 1}")
    ordered: list[tuple[int, int] | None] = [None] * N
    for site, index in zip(sites, indices):
        ordered[index] = site
    return tuple(site for site in ordered if site is not None)


def _compute_lce(records_by_n: dict[int, list[ClusterRecord]]) -> None:
    for N in sorted(records_by_n):
        for record in records_by_n[N]:
            net_constant = record.raw_constant
            net_terms = defaultdict(complex, record.raw_terms)
            subcluster_count = 0

            for subset in connected_subsets(record.sites, min_size=2, max_size=N - 1):
                child, index_mapping = _find_match(subset, record.sites, records_by_n[len(subset)])
                net_constant -= child.net_constant
                for key, coefficient in child.net_terms.items():
                    net_terms[map_key(key, index_mapping)] -= coefficient
                subcluster_count += 1

            record.net_constant = net_constant
            record.net_terms = dict(net_terms)
            record.subcluster_count = subcluster_count
            record.reconstruction_error = _reconstruction_error(record, records_by_n)
            if record.reconstruction_error > ATOL["loose"]:
                raise ValueError(
                    "LCE reconstruction failed for "
                    f"N={record.N} hole={record.hole} class={record.class_idx} "
                    f"cluster={record.cluster_idx}: {record.reconstruction_error:.6e}"
                )


def _find_match(
    parent_subset: tuple[int, ...],
    parent_sites: tuple[tuple[int, int], ...],
    candidates: list[ClusterRecord],
) -> tuple[ClusterRecord, dict[int, int]]:
    subgraph_sites = tuple(parent_sites[idx] for idx in parent_subset)
    subgraph = graph_from_sites(subgraph_sites)

    matches: list[tuple[tuple[int, int, int], tuple[int, ...], ClusterRecord, dict[int, int]]] = []
    for candidate in candidates:
        matcher = nx.algorithms.isomorphism.GraphMatcher(graph_from_sites(candidate.sites), subgraph)
        for mapping in matcher.isomorphisms_iter():
            index_mapping = {
                int(child_idx): int(parent_subset[subgraph_idx])
                for child_idx, subgraph_idx in mapping.items()
            }
            mapping_key = tuple(index_mapping[idx] for idx in range(candidate.N))
            matches.append(
                (
                    (candidate.hole, candidate.class_idx, candidate.cluster_idx),
                    mapping_key,
                    candidate,
                    index_mapping,
                )
            )

    if not matches:
        raise ValueError(f"no LCE subcluster match for parent subset {parent_subset}")
    _, _, candidate, index_mapping = min(matches, key=lambda item: (item[0], item[1]))
    return candidate, index_mapping


def _reconstruction_error(
    record: ClusterRecord,
    records_by_n: dict[int, list[ClusterRecord]],
) -> float:
    reconstructed_constant = 0.0j
    reconstructed_terms: dict[OperatorKey, complex] = defaultdict(complex)

    for subset in connected_subsets(record.sites, min_size=2, max_size=record.N):
        if len(subset) == record.N:
            child = record
            index_mapping = {idx: idx for idx in range(record.N)}
        else:
            child, index_mapping = _find_match(subset, record.sites, records_by_n[len(subset)])
        reconstructed_constant += child.net_constant
        for key, coefficient in child.net_terms.items():
            reconstructed_terms[map_key(key, index_mapping)] += coefficient

    return max_operator_difference(
        record.raw_constant,
        record.raw_terms,
        reconstructed_constant,
        dict(reconstructed_terms),
    )


def _weight_payload(record: ClusterRecord) -> dict[str, Any]:
    net_summary = operator_summary(record.net_constant, record.net_terms)
    return {
        "schema_version": LCE_SCHEMA_VERSION,
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


def _summary_text(payload: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    lines = [
        "LCE summary",
        f"schema_version: {payload['schema_version']}",
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


if __name__ == "__main__":
    raise SystemExit(main())
