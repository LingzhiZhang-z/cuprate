"""Target-driven embedding of LCE spin-coupling weights."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuprate.cli import COMMON_KEYS, parse_common_runtime, parse_key_values
from cuprate.clusters import Cluster, ClusterSets, POINT_GROUP_OPERATIONS, transform
from cuprate.io import LCE_SCHEMA_VERSION, LCE_WEIGHT_SCHEMA_VERSION, write_embed_outputs
from cuprate.operators import OperatorKey, canonical_operator_key, operators_to_terms
from cuprate.paths import (
    LCE_RESULTS_FILE,
    STAGE_EMBED,
    STAGE_LCE,
    embed_cluster_file,
    workflow_dir,
)


@dataclass(frozen=True)
class EmbedParams:
    root: Path
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
    scope: str
    workflow: str


@dataclass(frozen=True)
class WeightRecord:
    N: int
    hole: int
    class_idx: int
    cluster_idx: int
    weight_file: str
    sites: tuple[tuple[int, int], ...]
    terms: dict[OperatorKey, complex]

    def identity(self) -> tuple[int, int, int, int]:
        return (self.N, self.hole, self.class_idx, self.cluster_idx)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        params = parse_args(argv)
        run_embed(params)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def parse_args(argv: list[str]) -> EmbedParams:
    raw = parse_key_values(argv, COMMON_KEYS)
    common = parse_common_runtime(raw)
    return EmbedParams(
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


def run_embed(params: EmbedParams) -> dict[str, Any]:
    lce_path = _lce_manifest_path(params)
    lce_manifest, records = _load_lce_weights(lce_path)
    lce_scope = lce_manifest.get("run_params", {}).get("SCOPE")
    if lce_scope != params.scope:
        raise ValueError(f"{lce_path} has SCOPE={lce_scope!r}, expected {params.scope!r}")
    orientation_cache = {
        record.identity(): _distinct_parent_orientations(record.sites)
        for record in records
    }

    output_dir = workflow_dir(
        params.root,
        STAGE_EMBED,
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

    two_site_entries = _two_site_entries(params.N, records, orientation_cache)
    cluster_outputs = _multi_site_cluster_outputs(params.N, records, orientation_cache)
    return write_embed_outputs(
        output_dir=output_dir,
        params=params,
        source_lce_path=lce_path,
        lce_manifest=lce_manifest,
        records=records,
        orientation_cache=orientation_cache,
        two_site_entries=two_site_entries,
        cluster_outputs=cluster_outputs,
    )


def _lce_manifest_path(params: EmbedParams) -> Path:
    return (
        workflow_dir(
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
        / LCE_RESULTS_FILE
    )


def _load_lce_weights(path: Path) -> tuple[dict[str, Any], list[WeightRecord]]:
    if not path.exists():
        raise ValueError(f"missing LCE manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("result_kind") != "lce_spin_couplings":
        raise ValueError(f"{path} is not an lce_spin_couplings manifest")
    if int(manifest.get("schema_version", 0)) != LCE_SCHEMA_VERSION:
        raise ValueError(f"{path} must use lce schema_version={LCE_SCHEMA_VERSION}")

    weights = manifest.get("weights")
    if not isinstance(weights, list):
        raise ValueError(f"{path} is missing weights")

    records = []
    for entry in weights:
        weight_file = entry.get("weight_file")
        if not isinstance(weight_file, str) or not weight_file:
            raise ValueError(f"{path} has a weight entry missing weight_file")
        weight_path = path.parent / weight_file
        if not weight_path.exists():
            raise ValueError(f"missing LCE weight file: {weight_path}")
        weight = json.loads(weight_path.read_text())
        if weight.get("result_kind") != "lce_cluster_weight":
            raise ValueError(f"{weight_path} is not an lce_cluster_weight file")
        if int(weight.get("schema_version", 0)) != LCE_WEIGHT_SCHEMA_VERSION:
            raise ValueError(
                f"{weight_path} must use lce weight "
                f"schema_version={LCE_WEIGHT_SCHEMA_VERSION}"
            )

        _, terms = operators_to_terms(weight["operators"])
        records.append(
            WeightRecord(
                N=int(weight["N"]),
                hole=int(weight["hole"]),
                class_idx=int(weight["class_idx"]),
                cluster_idx=int(weight["cluster_idx"]),
                weight_file=weight_file,
                sites=_sites_in_operator_order(weight),
                terms=terms,
            )
        )
    records.sort(key=lambda record: record.identity())
    return manifest, records


def _sites_in_operator_order(payload: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    N = int(payload["N"])
    sites = [tuple(map(int, site)) for site in payload["sites"]]
    indices = [int(index) for index in payload["indices"]]
    if len(sites) != N or len(indices) != N:
        raise ValueError(f"sites and indices must both have length N={N}")
    if sorted(indices) != list(range(N)):
        raise ValueError(f"indices must be a permutation of 0..{N - 1}")
    ordered: list[tuple[int, int] | None] = [None] * N
    for site, index in zip(sites, indices):
        ordered[index] = site
    return tuple(site for site in ordered if site is not None)


def _distinct_parent_orientations(
    sites: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[int, int], ...]]:
    orientations = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for op in POINT_GROUP_OPERATIONS:
        oriented = transform(sites, op)
        min_x = min(x for x, y in oriented)
        min_y = min(y for x, y in oriented)
        normalized = tuple((int(x - min_x), int(y - min_y)) for x, y in oriented)
        signature = tuple(sorted(normalized))
        if signature not in seen:
            seen.add(signature)
            orientations.append(normalized)
    return orientations


def _two_site_entries(
    Nmax: int,
    records: list[WeightRecord],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
) -> list[dict[str, Any]]:
    entries = []
    for index, vector in enumerate(_two_site_vectors(Nmax)):
        dx, dy = vector
        value = _accumulate_target(
            ((0, 0), (dx, dy)),
            canonical_operator_key((0, 1)),
            records,
            orientation_cache,
        )
        entries.append({"index": index, "vector": vector, "value": value})
    return entries


def _two_site_vectors(Nmax: int) -> list[tuple[int, int]]:
    candidates = []
    for dx in range(1, Nmax):
        for dy in range(dx + 1):
            if dx + dy + 1 <= Nmax:
                candidates.append((dx, dy))
    return sorted(
        candidates,
        key=lambda v: (v[0] + v[1], v[0] * v[0] + v[1] * v[1], v[0], v[1]),
    )


def _multi_site_cluster_outputs(
    Nmax: int,
    records: list[WeightRecord],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
) -> list[dict[str, Any]]:
    outputs = []
    for N in (4, 6):
        if N > Nmax:
            continue
        clusters = sorted(
            ClusterSets(N).generate().clusters,
            key=lambda cluster: (cluster.hole, cluster.class_idx, cluster.cluster_idx),
        )
        for cluster in clusters:
            file_name = embed_cluster_file(
                N,
                cluster.hole,
                cluster.class_idx,
                cluster.cluster_idx,
            )
            pairings = _target_pairings(cluster)
            values = [
                _accumulate_target(cluster.sites, key, records, orientation_cache)
                for key in pairings
            ]
            outputs.append(
                {
                    "N": int(N),
                    "hole": int(cluster.hole),
                    "class_idx": int(cluster.class_idx),
                    "cluster_idx": int(cluster.cluster_idx),
                    "file_name": file_name,
                    "sites": cluster.sites,
                    "pairings": pairings,
                    "values": values,
                }
            )
    return outputs


def _target_pairings(cluster: Cluster) -> list[OperatorKey]:
    groups = cluster.generate_bonds(N=cluster.N, is_connected=False)
    if len(groups) != 1:
        raise ValueError("target cluster pairing generation must produce one full-support group")
    keys = [canonical_operator_key(term) for term in groups[0]]
    return sorted(keys, key=lambda key: _term_sort_key(cluster.sites, key))


def _accumulate_target(
    target_sites: tuple[tuple[int, int], ...],
    target_key: OperatorKey,
    records: list[WeightRecord],
    orientation_cache: dict[
        tuple[int, int, int, int],
        list[tuple[tuple[int, int], ...]],
    ],
) -> complex | None:
    target_arity = len(target_sites)
    value = 0.0j
    seen = False
    for record in records:
        if record.N < target_arity:
            continue
        orientations = orientation_cache[record.identity()]
        for oriented_sites in orientations:
            for key, coefficient in record.terms.items():
                if 2 * len(key) != target_arity:
                    continue
                if _matches_by_translation(oriented_sites, key, target_sites, target_key):
                    value += coefficient
                    seen = True
    return value if seen else None


def _matches_by_translation(
    oriented_sites: tuple[tuple[int, int], ...],
    term_key: OperatorKey,
    target_sites: tuple[tuple[int, int], ...],
    target_key: OperatorKey,
) -> bool:
    term_support = {oriented_sites[site] for pair in term_key for site in pair}
    if len(term_support) != len(target_sites):
        return False

    target_coords = tuple((int(x), int(y)) for x, y in target_sites)
    target_anchor = target_coords[0]
    for term_anchor in term_support:
        shift = (term_anchor[0] - target_anchor[0], term_anchor[1] - target_anchor[1])
        shifted_to_target_idx = {
            (x + shift[0], y + shift[1]): idx
            for idx, (x, y) in enumerate(target_coords)
        }
        if set(shifted_to_target_idx) != term_support:
            continue

        sites = []
        for pair in term_key:
            for site in pair:
                sites.append(shifted_to_target_idx[oriented_sites[site]])
        if canonical_operator_key(sites) == target_key:
            return True
    return False


def _term_sort_key(
    cluster_sites: tuple[tuple[int, int], ...],
    key: OperatorKey,
) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(sorted((_pair_descriptor(cluster_sites, pair) for pair in key)))


def _pair_descriptor(
    cluster_sites: tuple[tuple[int, int], ...],
    pair: tuple[int, int],
) -> tuple[int, int, int, int, int]:
    i, j = pair
    if i > j:
        i, j = j, i
    dx = int(cluster_sites[j][0]) - int(cluster_sites[i][0])
    dy = int(cluster_sites[j][1]) - int(cluster_sites[i][1])
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx * dx + dy * dy, dx, dy, i, j)


if __name__ == "__main__":
    raise SystemExit(main())
