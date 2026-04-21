from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Optional

import networkx as nx
import numpy as np

from cuprate.clusters import (
    canonical_form,
    count_holes,
    graph_from_sites,
)


# ============================================================
# CLI Helpers
# ============================================================

def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value {value!r}; expected one of "
        "'true', 'false', 'yes', 'no', 'on', 'off', '1', '0'"
    )


# ============================================================
# Modes & Params
# ============================================================

MODE_FULL = "full"
MODE_FIXED_SZ = "fixed_sz"
MODE_BLOCK_SZ_FULL = "block_sz_full"
MODE_FIXED_SZ_S2 = "fixed_sz_s2"
MODE_BLOCK_SZ_S2_FULL = "block_sz_s2_full"

RESULT_KIND_SPIN_COUPLINGS = "spin_couplings"
RESULT_KIND_PROJECTION_ANALYSIS = "projection_analysis"

VALID_MODES = {
    MODE_FULL,
    MODE_FIXED_SZ,
    MODE_BLOCK_SZ_FULL,
    MODE_FIXED_SZ_S2,
    MODE_BLOCK_SZ_S2_FULL,
}


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    twoSz: Optional[int]
    twoS: Optional[int]
    block_sz: bool
    block_S2: bool
    reconstruct_full: bool
    result_kind: str


def _validate_label_integer(value, label: str) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if value_int != value:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value_int


def resolve_mode_spec(params: Params) -> ModeSpec:
    mode = params.mode
    if mode is not None:
        mode = mode.lower()
        if mode == "all":
            mode = MODE_FULL
    else:
        if params.twoS is not None:
            mode = MODE_FIXED_SZ_S2
        elif params.twoSz is not None:
            mode = MODE_FIXED_SZ
        else:
            mode = MODE_FULL

    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported MODE={mode}")

    twoSz_value = None
    twoS_value = None
    if mode == MODE_FULL:
        if params.twoSz is not None or params.twoS is not None:
            raise ValueError("MODE=full does not accept twoSz or twoS")
    elif mode == MODE_FIXED_SZ:
        if params.twoSz is None:
            raise ValueError("MODE=fixed_sz requires twoSz")
        if params.twoS is not None:
            raise ValueError("MODE=fixed_sz does not accept twoS")
        twoSz_value = validate_twoSz(params.twoSz, params.N)
    elif mode == MODE_FIXED_SZ_S2:
        if params.twoSz is None or params.twoS is None:
            raise ValueError("MODE=fixed_sz_s2 requires both twoSz and twoS")
        twoSz_value = validate_twoSz(params.twoSz, params.N)
        twoS_value = validate_twoS(params.twoS, params.N)
        if abs(twoSz_value) > twoS_value:
            raise ValueError(
                f"(twoSz, twoS)=({twoSz_value}, {twoS_value}) violates |twoSz| <= twoS"
            )
    elif mode == MODE_BLOCK_SZ_FULL:
        if params.twoSz is not None or params.twoS is not None:
            raise ValueError("MODE=block_sz_full does not accept twoSz or twoS")
    elif mode == MODE_BLOCK_SZ_S2_FULL:
        if params.twoSz is not None or params.twoS is not None:
            raise ValueError("MODE=block_sz_s2_full does not accept twoSz or twoS")

    return ModeSpec(
        mode=mode,
        twoSz=twoSz_value,
        twoS=twoS_value,
        block_sz=mode in (MODE_BLOCK_SZ_FULL, MODE_BLOCK_SZ_S2_FULL),
        block_S2=mode in (MODE_FIXED_SZ_S2, MODE_BLOCK_SZ_S2_FULL),
        reconstruct_full=mode in (MODE_BLOCK_SZ_FULL, MODE_BLOCK_SZ_S2_FULL),
        result_kind=(
            RESULT_KIND_PROJECTION_ANALYSIS
            if mode == MODE_FIXED_SZ_S2
            else RESULT_KIND_SPIN_COUPLINGS
        ),
    )


@dataclass
class Params:
    # Physical parameters
    N: int = 3
    U: float = 1.0
    t: float = 0.1

    # Spin-sector parameters
    mode: Optional[str] = None
    twoSz: Optional[int] = None
    twoS: Optional[int] = None
    match_spin_sectors: bool = False

    # Workflow / selection parameters
    workflow: Optional[str] = None
    select: Optional[str] = None
    restart: bool = False

    # Adiabatic sweep parameters
    delta: Optional[float] = None

    # Embedding parameters
    Ncell: Optional[int] = None
    Ncut: Optional[int] = None


@dataclass(frozen=True)
class CliSpec:
    module_name: str
    defaults: Params
    allowed_keys: frozenset[str]
    help_lines: tuple[str, ...]
    examples: tuple[str, ...] = ()


COMMON_MODE_KEYS = frozenset({
    "MODE",
    "TWOSZ",
    "TWOS",
    "WORKFLOW",
    "RESTART",
    "MATCH_SPIN_SECTORS",
})

VALID_WORKFLOWS = frozenset({"occ", "energy", "single", "multi", "adiabatic"})
VALID_SELECT_MODES = frozenset({"block"})

MAIN_ALLOWED_KEYS = frozenset({
    "N",
    "U",
    "T",
    "SELECT",
    "DELTA",
    *COMMON_MODE_KEYS,
})

LCE_ALLOWED_KEYS = frozenset({
    "N",
    "U",
    "T",
    *COMMON_MODE_KEYS,
})

EMBED_ALLOWED_KEYS = frozenset({
    "U",
    "T",
    "NCELL",
    "NCUT",
    *COMMON_MODE_KEYS,
})


MAIN_CLI_SPEC = CliSpec(
    module_name="cuprate.main",
    defaults=Params(N=3, U=1.0, t=0.1),
    allowed_keys=MAIN_ALLOWED_KEYS,
    help_lines=(
        "N: number of sites (default: 3)",
        "U: Hubbard U parameter (default: 1.0)",
        "T: hopping parameter (default: 0.1)",
        "MODE: one of full, fixed_sz, block_sz_full, fixed_sz_s2, block_sz_s2_full",
        "twoSz / twoS: explicit fixed-sector quantum numbers",
        "WORKFLOW: occ, energy, single, multi, or adiabatic",
        "SELECT: block or default full-space selection",
        "RESTART: true/false",
        "MATCH_SPIN_SECTORS: true/false",
    ),
    examples=(
        "python -m cuprate.main N=4 U=3 T=0.2 MODE=full",
        "python -m cuprate.main N=4 MODE=fixed_sz_s2 twoSz=0 twoS=0",
    ),
)

LCE_CLI_SPEC = CliSpec(
    module_name="cuprate.lce",
    defaults=Params(N=3, U=6.0, t=1.0),
    allowed_keys=LCE_ALLOWED_KEYS,
    help_lines=(
        "N: maximum cluster size to process (default: 3)",
        "U: Hubbard U parameter (default: 6.0)",
        "T: hopping parameter (default: 1.0)",
        "MODE / twoSz / twoS / WORKFLOW / MATCH_SPIN_SECTORS:",
        "select the upstream Block run directory to read",
        "RESTART: include the _restart run suffix when locating upstream data",
    ),
    examples=(
        "python -m cuprate.lce N=4 U=3 T=0.2",
        "python -m cuprate.lce N=4 MODE=block_sz_s2_full WORKFLOW=multi",
    ),
)

EMBED_CLI_SPEC = CliSpec(
    module_name="cuprate.embed",
    defaults=Params(N=3, U=6.0, t=1.0, Ncell=8, Ncut=3),
    allowed_keys=EMBED_ALLOWED_KEYS,
    help_lines=(
        "U: Hubbard U parameter (default: 6.0)",
        "T: hopping parameter (default: 1.0)",
        "NCELL: square supercell edge length (default: 8)",
        "NCUT: maximum LCE cluster size to embed (default: 3)",
        "MODE / twoSz / twoS / WORKFLOW / MATCH_SPIN_SECTORS:",
        "select the upstream LCE run directory to read",
        "RESTART: include the _restart run suffix when locating upstream data",
    ),
    examples=(
        "python -m cuprate.embed NCELL=8 NCUT=4",
        "python -m cuprate.embed NCELL=10 NCUT=4 MODE=block_sz_full",
    ),
)


def _parse_workflow(value: str) -> str:
    workflow = value.strip().lower()
    if workflow not in VALID_WORKFLOWS:
        allowed = ", ".join(sorted(VALID_WORKFLOWS))
        raise ValueError(f"Invalid WORKFLOW={value!r}. Allowed values: {allowed}")
    return workflow


def _parse_select_mode(value: str) -> str:
    select_mode = value.strip().lower()
    if select_mode not in VALID_SELECT_MODES:
        allowed = ", ".join(sorted(VALID_SELECT_MODES))
        raise ValueError(f"Invalid SELECT={value!r}. Allowed values: {allowed}")
    return select_mode


def parse_cli_args(
    argv: Iterable[str],
    *,
    cli_spec: CliSpec,
) -> Params:
    argv_list = list(argv)
    params = replace(cli_spec.defaults)

    for arg in argv_list[1:]:
        if "=" not in arg:
            raise ValueError(f"Expected KEY=VALUE argument, got {arg!r}")
        key, value = arg.split("=", 1)
        key = key.upper()
        if key not in cli_spec.allowed_keys:
            allowed = ", ".join(sorted(cli_spec.allowed_keys))
            raise ValueError(
                f"{cli_spec.module_name} does not accept {key}. Allowed keys: {allowed}"
            )
        if key == "N":
            params.N = int(value)
        elif key == "U":
            params.U = float(value)
        elif key == "T":
            params.t = float(value)
        elif key == "MODE":
            params.mode = value.strip().lower()
        elif key == "TWOSZ":
            params.twoSz = int(value)
        elif key == "TWOS":
            params.twoS = int(value)
        elif key == "WORKFLOW":
            params.workflow = _parse_workflow(value)
        elif key == "SELECT":
            params.select = _parse_select_mode(value)
        elif key == "DELTA":
            params.delta = float(value)
        elif key == "RESTART":
            params.restart = parse_bool_arg(value)
        elif key == "NCELL":
            params.Ncell = int(value)
        elif key == "NCUT":
            params.Ncut = int(value)
        elif key == "MATCH_SPIN_SECTORS":
            params.match_spin_sectors = parse_bool_arg(value)

    if cli_spec is EMBED_CLI_SPEC:
        params.N = params.Ncut

    spec = resolve_mode_spec(params)
    params.mode = spec.mode
    params.twoSz = spec.twoSz
    params.twoS = spec.twoS

    return params


def print_cli_help(cli_spec: CliSpec) -> None:
    print(f"Usage: python -m {cli_spec.module_name} [KEY=VALUE]...")
    for line in cli_spec.help_lines:
        print(f"  {line}")
    if cli_spec.examples:
        print("\nExamples:")
        for example in cli_spec.examples:
            print(f"  {example}")


def parse_main_cli_args(argv: Optional[Iterable[str]] = None) -> Params:
    return parse_cli_args(sys.argv if argv is None else argv, cli_spec=MAIN_CLI_SPEC)


def parse_lce_cli_args(argv: Optional[Iterable[str]] = None) -> Params:
    return parse_cli_args(sys.argv if argv is None else argv, cli_spec=LCE_CLI_SPEC)


def parse_embed_cli_args(argv: Optional[Iterable[str]] = None) -> Params:
    return parse_cli_args(sys.argv if argv is None else argv, cli_spec=EMBED_CLI_SPEC)


# ============================================================
# Path Management
# ============================================================


def build_run_dirnames(params) -> tuple[str, str, str]:
    """Return (base_dir, run_dir, data_dir) using canonical runtime naming."""
    base_dir = f"U{params.U:.4f}_t{params.t:.4f}"

    if params.mode == MODE_FULL:
        spin_label = ""
    elif params.mode == MODE_FIXED_SZ:
        spin_label = f"_twoSz_{f'n{abs(params.twoSz)}' if params.twoSz < 0 else params.twoSz}"
    elif params.mode == MODE_FIXED_SZ_S2:
        spin_label = (
            f"_twoSz_{f'n{abs(params.twoSz)}' if params.twoSz < 0 else params.twoSz}"
            f"_twoS_{params.twoS}"
        )
    elif params.mode == MODE_BLOCK_SZ_FULL:
        spin_label = "_twoSz_all"
    elif params.mode == MODE_BLOCK_SZ_S2_FULL:
        spin_label = "_twoSz_all_twoS_all"
    else:
        raise ValueError(f"Unsupported mode for paths: {params.mode}")

    suffix_common = "_match_spin_sectors" if params.match_spin_sectors else ""
    workflow_suffix = f"_{params.workflow}" if params.workflow is not None else ""

    data_dir = f"N{params.N}{spin_label}{suffix_common}"
    run_dir = data_dir
    if workflow_suffix:
        run_dir = f"{run_dir}{workflow_suffix}"
    if params.restart:
        run_dir = f"{run_dir}_restart"

    return base_dir, run_dir, data_dir


def build_run_suffixes(params) -> tuple[str, str, str]:
    """Return (base_dir, run_suffix, data_suffix) for downstream directory scans."""
    base_dir, run_dir, data_dir = build_run_dirnames(params)
    prefix = f"N{params.N}"
    if not run_dir.startswith(prefix) or not data_dir.startswith(prefix):
        raise ValueError(f"Unexpected run/data directory names: {run_dir}, {data_dir}")
    return base_dir, run_dir[len(prefix):], data_dir[len(prefix):]


@dataclass
class PathSpec:
    """Centralized path configuration for a run."""

    work_dir: str
    base_dir: str
    data_dir: str
    run_dir: str
    output_dir: str
    tmp_dir: str


def build_path_spec(params) -> PathSpec:
    """Build the canonical directory layout for a run."""
    work_dir = "./Block"
    base_dir, run_dir, data_dir_name = build_run_dirnames(params)

    output_dir = f"{work_dir}/{base_dir}/{run_dir}"
    tmp_dir = f"{work_dir}/{base_dir}/{run_dir}/tmp"
    data_dir = f"{work_dir}/{base_dir}/{data_dir_name}"

    return PathSpec(
        work_dir=work_dir,
        base_dir=base_dir,
        run_dir=run_dir,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        data_dir=data_dir,
    )


def block_root_dir(base_dir: str) -> str:
    return os.path.join("Block", base_dir)


def lce_root_dir(base_dir: str) -> str:
    return os.path.join("data_transfer", "LCE", f"LCE_{base_dir}")


def embed_root_dir(base_dir: str) -> str:
    return os.path.join("data_transfer", "Embed", f"Embed_{base_dir}")


# ============================================================
# Results Serialization
# ============================================================

RESULT_SCHEMA_VERSION = 2
CONSOLIDATED_RESULTS_FILENAME = "results.json"
ANALYSIS_ONLY_ERROR = (
    "No spin-coupling section found in report. "
    "Analysis-only reports from MODE=fixed_sz_s2 cannot be used for LCE/embed."
)


def consolidated_results_path(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, CONSOLIDATED_RESULTS_FILENAME)


def serialize_complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def deserialize_complex(payload: dict[str, Any]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def serialize_cluster(cluster) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in cluster]


def deserialize_cluster(payload) -> list[tuple[int, int]]:
    return [tuple(map(int, point)) for point in payload]


@dataclass
class SpinCouplingTerms:
    constant: complex
    two_site: dict[tuple[int, int], list[list]]
    four_site: list[list]
    six_site: list[list]
    eight_site: list[list]


def _flat_terms(bonds, coefficients=None) -> list[list[Any]]:
    if coefficients is None:
        return [[*[int(idx) for idx in bond], 0.0 + 0.0j] for bond in bonds]
    return [
        [*[int(idx) for idx in bond], coefficient]
        for bond, coefficient in zip(bonds, coefficients)
    ]


def _serialize_terms(entries, arity: int) -> list[dict[str, Any]]:
    terms = []
    for entry in entries:
        sites = [int(idx) for idx in entry[:arity]]
        coefficient = entry[arity]
        terms.append(
            {
                "sites": sites,
                "coefficient": serialize_complex(coefficient),
            }
        )
    return terms


def _multi_site_label_prefix(arity: int) -> str:
    if arity == 4:
        return "K"
    if arity == 6:
        return "L"
    if arity == 8:
        return "M"
    raise ValueError(f"Unsupported multi-site operator arity: {arity}")


def _canonical_pair_descriptor(cluster, site1: int, site2: int) -> tuple[int, int, int, int, int]:
    site1, site2 = sorted((int(site1), int(site2)))
    dx = abs(cluster[site2][0] - cluster[site1][0])
    dy = abs(cluster[site2][1] - cluster[site1][1])
    dx, dy = sorted((dx, dy), reverse=True)
    return (dx * dx + dy * dy, dx, dy, site1, site2)


def _canonicalize_multi_site_entry(cluster, entry, arity: int) -> list[Any]:
    coefficient = entry[arity]
    pairs = []
    for idx in range(0, arity, 2):
        site1 = int(entry[idx])
        site2 = int(entry[idx + 1])
        descriptor = _canonical_pair_descriptor(cluster, site1, site2)
        pairs.append((descriptor, tuple(sorted((site1, site2)))))

    pairs.sort(key=lambda item: item[0])
    flat_sites = [site for _, pair in pairs for site in pair]
    return [*flat_sites, coefficient]


def _multi_site_group_key(cluster, entry, arity: int) -> tuple[Any, ...]:
    sites = sorted(set(int(idx) for idx in entry[:arity]))
    return tuple((cluster[idx][0], cluster[idx][1], idx) for idx in sites)


def _multi_site_term_key(cluster, entry, arity: int) -> tuple[Any, ...]:
    descriptors = []
    for idx in range(0, arity, 2):
        descriptors.append(_canonical_pair_descriptor(cluster, entry[idx], entry[idx + 1]))
    return tuple(sorted(descriptors))


def _serialize_multi_site_groups_from_flat_terms(cluster, entries, arity: int) -> list[dict[str, Any]]:
    groups = []
    prefix = _multi_site_label_prefix(arity)
    grouped_entries = {}
    for entry in entries:
        normalized_entry = _canonicalize_multi_site_entry(cluster, entry, arity)
        group_key = _multi_site_group_key(cluster, normalized_entry, arity)
        grouped_entries.setdefault(group_key, []).append(normalized_entry)

    for group_idx, group_key in enumerate(sorted(grouped_entries), start=1):
        group_entries = sorted(
            grouped_entries[group_key],
            key=lambda entry: _multi_site_term_key(cluster, entry, arity),
        )
        groups.append(
            {
                "arity": arity,
                "vector": None,
                "label": f"{prefix}{group_idx}",
                "terms": _serialize_terms(group_entries, arity),
            }
        )
    return groups


def _spin_coupling_terms_from_catalog_and_coeffs(
    spin_operator_catalog,
    coeffs=None,
) -> SpinCouplingTerms:
    zero = 0.0 + 0.0j
    constant = zero if coeffs is None else coeffs[0]

    two_site = {}
    coeff_idx = 1
    for vector, bond_group in spin_operator_catalog.two_site_classes:
        group_coeffs = None if coeffs is None else coeffs[coeff_idx]
        two_site[vector] = _flat_terms(bond_group, group_coeffs)
        coeff_idx += 1

    multi_site_terms = {4: [], 6: [], 8: []}
    for arity, bonds in (
        (4, spin_operator_catalog.four_site_terms),
        (6, spin_operator_catalog.six_site_terms),
    ):
        if bonds:
            group_coeffs = None if coeffs is None else coeffs[coeff_idx]
            multi_site_terms[arity] = _flat_terms(bonds, group_coeffs)
            coeff_idx += 1

    return SpinCouplingTerms(
        constant=constant,
        two_site=two_site,
        four_site=multi_site_terms[4],
        six_site=multi_site_terms[6],
        eight_site=multi_site_terms[8],
    )


def build_spin_coupling_artifact_from_coeffs(
    cluster,
    spin_operator_catalog,
    coeffs,
    error,
    t11_norm: float,
    overlap,
    *,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return build_spin_coupling_artifact_from_terms(
        cluster,
        _spin_coupling_terms_from_catalog_and_coeffs(spin_operator_catalog, coeffs),
        hole=hole,
        class_idx=class_idx,
        cluster_idx=cluster_idx,
        fit={
            "relative_error": float(error[0]),
            "residual": float(error[1]),
            "r_squared": float(error[2]),
            "t11_minus_1_norm": float(t11_norm),
            "overlap": None if overlap is None else float(overlap),
        },
        metadata={
            "rank": int(metadata["rank"]),
            "computation_time_s": float(metadata["computation_time_s"]),
        },
    )


def build_spin_coupling_terms_from_catalog(spin_operator_catalog) -> SpinCouplingTerms:
    return _spin_coupling_terms_from_catalog_and_coeffs(spin_operator_catalog)


def build_spin_coupling_artifact_from_terms(
    cluster,
    spin_coupling_terms: SpinCouplingTerms,
    *,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    fit: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    groups = []
    for bond_idx, (dx, dy) in enumerate(get_all_possible_vectors(cluster)):
        groups.append(
            {
                "arity": 2,
                "vector": [dx, dy],
                "label": f"J{bond_idx + 1}",
                "terms": _serialize_terms(spin_coupling_terms.two_site.get((dx, dy), []), 2),
            }
        )

    for arity, entries in (
        (4, spin_coupling_terms.four_site),
        (6, spin_coupling_terms.six_site),
        (8, spin_coupling_terms.eight_site),
    ):
        if entries:
            groups.extend(_serialize_multi_site_groups_from_flat_terms(cluster, entries, arity))

    payload = {
        "hole": int(hole),
        "class_idx": int(class_idx),
        "cluster_idx": int(cluster_idx),
        "sites": serialize_cluster(cluster),
        "operators": {
            "constant_term": serialize_complex(spin_coupling_terms.constant),
            "groups": groups,
        },
    }
    if fit is not None:
        payload["fit"] = fit
    if metadata:
        payload["metadata"] = metadata
    return payload


def build_projection_analysis_artifact(
    cluster,
    model,
    *,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    df = model.downfold
    return {
        "hole": int(hole),
        "class_idx": int(class_idx),
        "cluster_idx": int(cluster_idx),
        "sites": serialize_cluster(cluster),
        "projection": {
            "t11_minus_1_norm": float(df.t11m1_norm),
            "overlap": None if df.overlap is None else float(df.overlap),
            "selected_state_count": int(len(df.selected_indices)),
            "heff_dimension": [int(df.heff.shape[0]), int(df.heff.shape[1])],
            "selected_indices": [int(idx) for idx in df.selected_indices],
            "double_occupation_expectation": [
                float(value.real) for value in df.selected_occupation
            ],
        },
        "metadata": {
            "rank": int(metadata["rank"]),
            "computation_time_s": float(metadata["computation_time_s"]),
        },
    }


def build_consolidated_results(
    result_kind: str,
    run_params: dict[str, Any],
    cluster_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_entries = sorted(
        cluster_entries,
        key=lambda entry: (entry["hole"], entry["class_idx"], entry["cluster_idx"]),
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": result_kind,
        "run_params": run_params,
        "clusters": sorted_entries,
    }


def load_consolidated_results(run_output_dir: str) -> dict[str, Any]:
    path = consolidated_results_path(run_output_dir)
    with open(path, "r") as f:
        payload = json.load(f)
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported result schema version {payload['schema_version']}; expected {RESULT_SCHEMA_VERSION}"
        )
    return payload


def spin_coupling_terms_from_artifact(entry: dict[str, Any]) -> SpinCouplingTerms:
    if "operators" not in entry:
        raise ValueError(ANALYSIS_ONLY_ERROR)

    two_site = {}
    multi_site_terms = {4: [], 6: [], 8: []}

    for group in entry["operators"]["groups"]:
        terms = [
            [*map(int, term["sites"]), deserialize_complex(term["coefficient"])]
            for term in group["terms"]
        ]
        if group["arity"] == 2:
            if terms:
                two_site[tuple(map(int, group["vector"]))] = terms
        elif group["arity"] in multi_site_terms:
            multi_site_terms[group["arity"]].extend(terms)
        else:
            raise ValueError(f"Unsupported operator arity: {group['arity']}")

    return SpinCouplingTerms(
        constant=deserialize_complex(entry["operators"]["constant_term"]),
        two_site=two_site,
        four_site=multi_site_terms[4],
        six_site=multi_site_terms[6],
        eight_site=multi_site_terms[8],
    )


def match_cluster_in_catalog(target_cluster, cluster_catalog):
    """Return the matching catalog entry and site mapping for a target cluster."""
    matches = []

    target_canon = canonical_form(target_cluster)
    nsite = len(target_canon)
    hole = count_holes(target_canon)

    if nsite not in cluster_catalog or hole not in cluster_catalog[nsite]:
        raise ValueError(f"No catalog entries for cluster with N={nsite}, hole={hole}")

    clusters = cluster_catalog[nsite][hole]
    for class_idx in clusters:
        first_cluster = next(iter(clusters[class_idx].values()))
        if canonical_form(first_cluster) != target_canon:
            continue

        graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph_from_sites(first_cluster),
            graph_from_sites(target_cluster),
        )
        if not graph_matcher.is_isomorphic():
            continue

        coord_mapping = [graph_matcher.mapping[i] for i in range(len(first_cluster))]

        for rank_idx, ref_cluster in clusters[class_idx].items():
            if canonical_form(ref_cluster) == target_canon:
                matches.append((nsite, hole, class_idx, rank_idx, coord_mapping))

    if not matches:
        raise ValueError(f"No matching cluster found for cluster: {target_cluster}, {target_canon}")
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple matching clusters found for cluster: {target_cluster}, {target_canon}"
        )

    return matches[0]


def write_cluster_points(f, cluster):
    """Write cluster points information."""
    f.write("\n=== Cluster Points ===\n")
    for i, point in enumerate(cluster):
        f.write(f"Point {i}: {point}\n")


def get_all_possible_vectors(cluster, max_distance=None):
    """Get all possible bond vectors in the first quadrant below y=x."""
    if max_distance is None:
        max_distance = len(cluster)
    vectors = []
    for dx in range(max_distance):
        for dy in range(dx + 1):
            if dx == 0 and dy == 0:
                continue
            vectors.append((dx, dy))
    return sorted(vectors, key=lambda v: v[0] ** 2 + v[1] ** 2)


# ============================================================
# File I/O Utilities
# ============================================================

def validate_twoSz(twoSz, N):
    """Validate 2*Sz against the standard quantum-number constraints."""
    if twoSz is None:
        return None
    twoSz = _validate_label_integer(twoSz, "twoSz")
    if abs(twoSz) > N:
        raise ValueError(f"twoSz={twoSz} violates |twoSz| <= N={N}")
    if N % 2 != twoSz % 2:
        raise ValueError(f"twoSz={twoSz} parity does not match N={N}")
    return twoSz


def validate_twoS(twoS, N):
    """Validate 2*S against the standard quantum-number constraints."""
    if twoS is None:
        return None
    twoS = _validate_label_integer(twoS, "twoS")
    if twoS < 0 or twoS > N:
        raise ValueError(f"twoS={twoS} violates 0 <= twoS <= N={N}")
    if N % 2 != twoS % 2:
        raise ValueError(f"twoS={twoS} parity does not match N={N}")
    return twoS
def k4s(i, j, k, l):
    pairs = [tuple(sorted([i, j])), tuple(sorted([k, l]))]
    return tuple(sorted(pairs))


def k6s(i, j, k, l, m, n):
    pairs = [tuple(sorted([i, j])), tuple(sorted([k, l])), tuple(sorted([m, n]))]
    return tuple(sorted(pairs))


def k8s(i, j, k, l, m, n, a, b):
    pairs = [
        tuple(sorted([i, j])),
        tuple(sorted([k, l])),
        tuple(sorted([m, n])),
        tuple(sorted([a, b])),
    ]
    return tuple(sorted(pairs))


def load_spin_coupling_catalog(prefix_path: str, run_suffix: str, n_max: int):
    """Load the spin-coupling cluster catalog from consolidated result files."""
    clusters = {}
    for n in range(1, n_max + 1):
        n_dir = f"{prefix_path}/N{n}{run_suffix}"
        if not os.path.exists(n_dir):
            continue
        clusters[n] = {}
        results_path = consolidated_results_path(n_dir)
        if not os.path.exists(results_path):
            clusters.pop(n)
            continue

        payload = load_consolidated_results(n_dir)
        if payload["result_kind"] != RESULT_KIND_SPIN_COUPLINGS:
            clusters.pop(n)
            continue

        for entry in payload["clusters"]:
            hole = entry["hole"]
            class_idx = entry["class_idx"]
            variant_idx = entry["cluster_idx"]
            if hole not in clusters[n]:
                clusters[n][hole] = {}
            if class_idx not in clusters[n][hole]:
                clusters[n][hole][class_idx] = {}
            clusters[n][hole][class_idx][variant_idx] = deserialize_cluster(entry["sites"])

        if not clusters[n]:
            clusters.pop(n)

    if not clusters:
        raise FileNotFoundError(
            "No cluster result directories found with spin-coupling artifacts under "
            f"{prefix_path} for 1 <= N <= {n_max} with suffix '{run_suffix}'"
        )

    return clusters
