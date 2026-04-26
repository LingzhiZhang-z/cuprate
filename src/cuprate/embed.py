"""Target-driven embedding of LCE spin-coupling weights."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuprate.clusters import Cluster, ClusterSets, POINT_GROUP_OPERATIONS, transform
from cuprate.operators import OperatorKey, canonical_operator_key, operators_to_terms
from cuprate.paths import (
    EMBED_CLUSTERS_DIR,
    EMBED_RESULTS_FILE,
    EMBED_SUMMARY_FILE,
    EMBED_TWO_SITE_FILE,
    LCE_RESULTS_FILE,
    STAGE_EMBED,
    STAGE_LCE,
    embed_cluster_file,
    mode_spec,
    mode_token,
    parameter_token,
    workflow_dir,
    workflow_token,
)


EMBED_SCHEMA_VERSION = 1
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
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi", "adiabatic"}


@dataclass(frozen=True)
class EmbedParams:
    root: Path
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
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
    workflow = raw["workflow"].lower()
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"unsupported workflow={raw['workflow']!r}")
    _validate_mode_args(N, spec)
    return EmbedParams(
        root=Path(raw["ROOT"]),
        N=N,
        U=U,
        t=t,
        mode=spec.mode,
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


def run_embed(params: EmbedParams) -> dict[str, Any]:
    lce_path = _lce_manifest_path(params)
    lce_manifest, records = _load_lce_weights(lce_path)
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
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir = output_dir / EMBED_CLUSTERS_DIR
    clusters_dir.mkdir(parents=True, exist_ok=True)

    two_site_entries = _two_site_entries(params.N, records, orientation_cache)
    (output_dir / EMBED_TWO_SITE_FILE).write_text(
        _two_site_text(lce_path, params.N, two_site_entries) + "\n"
    )

    cluster_files = _write_multi_site_clusters(clusters_dir, params.N, records, orientation_cache)
    payload = _manifest_payload(
        params,
        lce_path,
        lce_manifest,
        records,
        orientation_cache,
        cluster_files,
        two_site_entries,
    )
    (output_dir / EMBED_RESULTS_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / EMBED_SUMMARY_FILE).write_text(
        _summary_text(payload, lce_path, records, orientation_cache) + "\n"
    )
    return payload


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
        )
        / LCE_RESULTS_FILE
    )


def _load_lce_weights(path: Path) -> tuple[dict[str, Any], list[WeightRecord]]:
    if not path.exists():
        raise ValueError(f"missing LCE manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("result_kind") != "lce_spin_couplings":
        raise ValueError(f"{path} is not an lce_spin_couplings manifest")
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"{path} must use lce schema_version=1")

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
        if int(weight.get("schema_version", 0)) != 1:
            raise ValueError(f"{weight_path} must use lce weight schema_version=1")

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


def _write_multi_site_clusters(
    clusters_dir: Path,
    Nmax: int,
    records: list[WeightRecord],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
) -> list[dict[str, Any]]:
    cluster_files = []
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
            (clusters_dir / file_name).write_text(_cluster_text(cluster, pairings, values) + "\n")
            cluster_files.append(
                {
                    "N": int(N),
                    "hole": int(cluster.hole),
                    "class_idx": int(cluster.class_idx),
                    "cluster_idx": int(cluster.cluster_idx),
                    "file": f"{EMBED_CLUSTERS_DIR}/{file_name}",
                }
            )
    return cluster_files


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


def _two_site_text(
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
            lines.append(f"{prefix}  {_format_complex(value)}")
    return "\n".join(lines)


def _cluster_text(
    cluster: Cluster,
    pairings: list[OperatorKey],
    values: list[complex | None],
) -> str:
    lines = [
        "sites: " + " ".join(f"({x},{y})" for x, y in cluster.sites),
        "indices: " + " ".join(str(index) for index in range(cluster.N)),
        "",
    ]
    for key, value in zip(pairings, values):
        label = "".join(f"(S{i} S{j})" for i, j in key)
        if value is None:
            lines.append(f"{label}  None")
        else:
            lines.append(f"{label}  {_format_complex(value)}")
    return "\n".join(lines)


def _format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:.12e}  {value.imag:.12e}"


def _manifest_payload(
    params: EmbedParams,
    source_lce_path: Path,
    lce_manifest: dict[str, Any],
    records: list[WeightRecord],
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
        "run_params": {
            "U": float(params.U),
            "T": float(params.t),
            "N": int(params.N),
            "Nmax": int(params.N),
            "nelec": int(params.N),
            "MODE": params.mode,
            "twoSz": params.twoSz,
            "twoS": params.twoS,
            "workflow": params.workflow,
            "ROOT": str(params.root),
            "parameter_token": parameter_token(params.N, params.N, params.U, params.t),
            "mode_token": mode_token(params.mode, twoSz=params.twoSz, twoS=params.twoS),
            "workflow_token": workflow_token(params.workflow),
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


def _summary_text(
    payload: dict[str, Any],
    source_lce_path: Path,
    records: list[WeightRecord],
    orientation_cache: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]],
) -> str:
    run_params = payload["run_params"]
    diagnostics = payload["diagnostics"]
    lines = [
        "Embed summary",
        f"schema_version: {payload['schema_version']}",
        f"source_lce_manifest: {source_lce_path}",
        f"Nmax: {run_params['Nmax']}",
        f"U: {run_params['U']:.12g}",
        f"T: {run_params['T']:.12g}",
        f"MODE: {run_params['MODE']}",
        f"twoSz: {run_params['twoSz']}",
        f"twoS: {run_params['twoS']}",
        f"workflow: {run_params['workflow']}",
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
