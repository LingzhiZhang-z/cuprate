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
from cuprate.cli import COMMON_KEYS, parse_common_runtime, parse_key_values
from cuprate.clusters import connected_subsets, graph_from_sites
from cuprate.io import SPIN_COUPLINGS_SCHEMA_VERSION, write_lce_outputs
from cuprate.paths import (
    RESULTS_FILE,
    STAGE_LCE,
    STAGE_MAIN,
    workflow_dir,
)
from cuprate.operators import (
    OperatorKey,
    map_key,
    max_operator_difference,
    operators_to_terms,
)


COMMON_RUN_PARAM_KEYS = ("U", "T", "MODE", "twoSz", "twoS", "SCOPE", "workflow")


@dataclass(frozen=True)
class LCEParams:
    root: Path
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
    scope: str
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
    raw = parse_key_values(argv, COMMON_KEYS)
    common = parse_common_runtime(raw)
    return LCEParams(
        root=common.root,
        N=common.N,
        U=common.U,
        t=common.t,
        mode=common.mode,
        twoSz=common.twoSz,
        twoS=common.twoS,
        scope=common.scope,
        workflow=common.workflow,
    )


def run_lce(params: LCEParams) -> dict[str, Any]:
    input_paths = _input_paths(params)
    records_by_n, common_params = _load_input_records(input_paths, expected_scope=params.scope)
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
        scope=params.scope,
    )
    return write_lce_outputs(
        output_dir=output_dir,
        params=params,
        input_paths=input_paths,
        records_by_n=records_by_n,
        common_params=common_params,
    )


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
            scope=params.scope,
        )
        / RESULTS_FILE
        for N in range(2, params.N + 1)
    ]


def _load_input_records(
    input_paths: list[Path],
    *,
    expected_scope: str,
) -> tuple[dict[int, list[ClusterRecord]], dict[str, Any]]:
    records_by_n: dict[int, list[ClusterRecord]] = {}
    common_params: dict[str, Any] | None = None

    for path in input_paths:
        payload = json.loads(path.read_text())
        if payload.get("result_kind") != "spin_couplings":
            raise ValueError(f"{path} is not a spin_couplings results file")
        if int(payload.get("schema_version", 0)) != SPIN_COUPLINGS_SCHEMA_VERSION:
            raise ValueError(
                f"{path} must use spin_couplings "
                f"schema_version={SPIN_COUPLINGS_SCHEMA_VERSION}"
            )
        if payload.get("complete_family_set", True) is not True:
            raise ValueError(f"{path} is a partial main output and cannot be used for LCE")

        run_params = payload["run_params"]
        if run_params.get("SCOPE") != expected_scope:
            raise ValueError(
                f"{path} has SCOPE={run_params.get('SCOPE')!r}, expected {expected_scope!r}"
            )
        N = int(run_params["N"])
        if N in records_by_n:
            raise ValueError(f"duplicate input for N={N}")
        current_common = {key: run_params.get(key) for key in COMMON_RUN_PARAM_KEYS}
        if common_params is None:
            common_params = current_common
        elif current_common != common_params:
            raise ValueError("all inputs must share U/T/MODE/twoSz/twoS/SCOPE/workflow")

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


if __name__ == "__main__":
    raise SystemExit(main())
