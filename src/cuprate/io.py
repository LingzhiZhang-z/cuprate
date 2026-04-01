from __future__ import annotations

import glob
import itertools
import json
import math
import os
import shutil
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from cuprate import clusters as cluster_square


canonical_form = cluster_square.canonical_form
count_holes = cluster_square.count_holes
graph_from_adj = cluster_square.graph_from_adj
generate_adjacency_matrix = cluster_square.generate_adjacency_matrix

EDGE_MATCHER = nx.algorithms.isomorphism.numerical_edge_match("weight", 0)


# ============================================================
# CLI Helpers
# ============================================================

def parse_bool_arg(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def iter_cli_assignments(argv: Iterable[str]) -> Iterator[tuple[str, str]]:
    for arg in list(argv)[1:]:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        yield key.upper(), value


# ============================================================
# Modes & Params
# ============================================================

MODE_FULL = "full"
MODE_FIXED_SZ = "fixed_sz"
MODE_BLOCK_SZ_FULL = "block_sz_full"
MODE_FIXED_SZ_S2 = "fixed_sz_s2"
MODE_BLOCK_SZS2_FULL = "block_szs2_full"

RESULT_KIND_SPIN_COUPLINGS = "spin_couplings"
RESULT_KIND_PROJECTION_ANALYSIS = "projection_analysis"

VALID_MODES = {
    MODE_FULL,
    MODE_FIXED_SZ,
    MODE_BLOCK_SZ_FULL,
    MODE_FIXED_SZ_S2,
    MODE_BLOCK_SZS2_FULL,
}


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    sz: Optional[float]
    sz_idx: Optional[int]
    s: Optional[float]
    s_idx: Optional[int]
    block_sz: bool
    block_s2: bool
    reconstruct_full: bool
    result_kind: str
    supports_spin_couplings: bool


def nonnegative_sz_values(nsites: int) -> list[float]:
    start = 0.0 if nsites % 2 == 0 else 0.5
    smax = nsites * 0.5
    return [start + idx for idx in range(int(round(smax - start)) + 1)]


def s_values_for_sz(nsites: int, sz: float) -> list[float]:
    smin = abs(sz)
    smax = nsites * 0.5
    return [smin + idx for idx in range(int(round(smax - smin)) + 1)]


def _normalize_half_integer(value: float) -> float:
    return round(float(value) * 2.0) / 2.0


def _value_from_index(
    values: list[float], index: Optional[int], label: str
) -> tuple[Optional[float], Optional[int]]:
    if index is None:
        return None, None
    if index < 0 or index >= len(values):
        raise ValueError(f"{label}_idx={index} is out of range for available values {values}")
    return values[index], index


def _index_from_value(
    values: list[float], value: Optional[float], label: str
) -> tuple[Optional[float], Optional[int]]:
    if value is None:
        return None, None
    normalized = _normalize_half_integer(value)
    for idx, candidate in enumerate(values):
        if abs(candidate - normalized) < 1e-8:
            return normalized, idx
    raise ValueError(f"{label}={value} is not compatible with available values {values}")


def resolve_mode_spec(params) -> ModeSpec:
    mode = params.get("mode")
    if mode is not None:
        mode = mode.lower()
    else:
        block = params.get("block")
        if block == "szs2":
            mode = MODE_BLOCK_SZS2_FULL
        elif block == "sz":
            mode = MODE_BLOCK_SZ_FULL
        elif params.get("s2_fix") and (params.get("s2") is not None or params.get("s") is not None):
            mode = MODE_FIXED_SZ_S2
        elif params.get("sz") is not None:
            mode = MODE_FIXED_SZ
        else:
            mode = MODE_FULL

    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported MODE={mode}")

    sz_values = nonnegative_sz_values(params["N"])
    sz_idx = params.get("sz_idx")
    sz_value = params.get("sz")

    if mode in (MODE_FIXED_SZ, MODE_FIXED_SZ_S2):
        if sz_idx is not None:
            sz_value, sz_idx = _value_from_index(sz_values, int(sz_idx), "sz")
        else:
            sz_value, sz_idx = _index_from_value(sz_values, sz_value, "sz")
            if sz_value is None:
                sz_value, sz_idx = sz_values[0], 0
    else:
        sz_value = None
        sz_idx = None

    s_idx = params.get("s_idx")
    s_value = params.get("s")
    legacy_s_idx = params.get("s2")

    if mode == MODE_FIXED_SZ_S2:
        allowed_s = s_values_for_sz(params["N"], sz_value)
        if s_idx is not None:
            s_value, s_idx = _value_from_index(allowed_s, int(s_idx), "s")
        elif legacy_s_idx is not None:
            s_value, s_idx = _value_from_index(allowed_s, int(legacy_s_idx), "s")
        else:
            s_value, s_idx = _index_from_value(allowed_s, s_value, "s")
            if s_value is None:
                s_value, s_idx = allowed_s[0], 0
    else:
        s_value = None
        s_idx = None

    return ModeSpec(
        mode=mode,
        sz=sz_value,
        sz_idx=sz_idx,
        s=s_value,
        s_idx=s_idx,
        block_sz=mode in (MODE_BLOCK_SZ_FULL, MODE_BLOCK_SZS2_FULL),
        block_s2=mode in (MODE_FIXED_SZ_S2, MODE_BLOCK_SZS2_FULL),
        reconstruct_full=mode in (MODE_BLOCK_SZ_FULL, MODE_BLOCK_SZS2_FULL),
        result_kind=(
            RESULT_KIND_PROJECTION_ANALYSIS
            if mode == MODE_FIXED_SZ_S2
            else RESULT_KIND_SPIN_COUPLINGS
        ),
        supports_spin_couplings=(mode != MODE_FIXED_SZ_S2),
    )


@dataclass
class Params:
    # 物理参数
    N: int = 3
    U: float = 1.0
    t: float = 0.1
    t2: Optional[float] = None
    t3: Optional[float] = None

    # 自旋参数
    mode: Optional[str] = None
    sz_idx: Optional[int] = None
    s_idx: Optional[int] = None
    sz: Optional[float] = None
    s2: Optional[float] = None
    s2_fix: bool = False
    s: Optional[float] = None
    result_kind: Optional[str] = None
    supports_spin_couplings: bool = True
    match_spin_sectors: bool = False

    # 计算类型
    type: Optional[str] = None
    block: Optional[str] = None
    select: Optional[str] = None
    restart: bool = False

    # 绝热参数
    delta: Optional[float] = None
    delta2: Optional[float] = None
    type_delta: Optional[str] = None

    # 嵌入参数（仅 embed 使用，可考虑移除）
    Ncell: Optional[int] = None
    Ncut: Optional[int] = None
    ratio: Optional[float] = None

    # 运行时注入（不通过 CLI）
    path_spec: Optional[object] = field(default=None, repr=False)
    result_dir: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with code that expects dict params."""
        return {
            "N": self.N,
            "U": self.U,
            "t": self.t,
            "t2": self.t2,
            "t3": self.t3,
            "mode": self.mode,
            "sz_idx": self.sz_idx,
            "s_idx": self.s_idx,
            "sz": self.sz,
            "s2": self.s2,
            "s2_fix": self.s2_fix,
            "s": self.s,
            "result_kind": self.result_kind,
            "supports_spin_couplings": self.supports_spin_couplings,
            "match_spin_sectors": self.match_spin_sectors,
            "type": self.type,
            "block": self.block,
            "select": self.select,
            "restart": self.restart,
            "delta": self.delta,
            "delta2": self.delta2,
            "type_delta": self.type_delta,
            "Ncell": self.Ncell,
            "Ncut": self.Ncut,
            "ratio": self.ratio,
            "path_spec": self.path_spec,
            "result_dir": self.result_dir,
        }

    def __getitem__(self, key: str):
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow dict-style assignment for backward compatibility."""
        setattr(self, key, value)

    def get(self, key: str, default=None):
        """Allow dict-style .get() for backward compatibility."""
        return getattr(self, key, default)


# ============================================================
# Path Management
# ============================================================

@dataclass
class PathSpec:
    """集中管理文件路径的配置结构"""

    work_dir: str
    base_dir: str
    data_dir: str
    run_dir: str
    output_dir: str
    restart_dir: str
    tmp_dir: str


def build_path_spec(params) -> PathSpec:
    """根据参数构建标准路径规范"""
    mode_spec = resolve_mode_spec(params)

    # 0. 构建 work_dir (工作目录)
    work_dir = "./Block"

    # 1. 构建 base_dir (物理参数层)
    base_dir = f"U{params['U']:.4f}_t{params['t']:.4f}"
    if params["t2"] is not None:
        base_dir = f"{base_dir}_tp{params['t2']:.4f}"

    # 2. 构建 run_dir (运行参数层)
    if mode_spec.mode == "full":
        spin_label = ""
    elif mode_spec.mode == "fixed_sz":
        spin_label = f"_sz{mode_spec.sz_idx}"
    elif mode_spec.mode == "fixed_sz_s2":
        spin_label = f"_sz{mode_spec.sz_idx}_s{mode_spec.s_idx}"
    elif mode_spec.mode == "block_sz_full":
        spin_label = "_block_sz"
    elif mode_spec.mode == "block_szs2_full":
        spin_label = "_block_szs2"
    else:
        raise ValueError(f"Unsupported mode for paths: {mode_spec.mode}")

    suffix_type = ""
    suffix_common = ""
    if params.get("match_spin_sectors"):
        suffix_common = "_match_spin_sectors"
    if params["type"] is not None:
        suffix_type = f"_{params['type']}"
    if params["restart"]:
        suffix_type = f"{suffix_type}_restart"
    run_dir = f"N{params['N']}{spin_label}{suffix_common}{suffix_type}"

    # 3. 组合完整路径
    output_dir = f"{work_dir}/{base_dir}/{run_dir}"
    restart_dir = output_dir

    # 4. 构建 tmp 前缀
    tmp_dir = f"{work_dir}/{base_dir}/{run_dir}/tmp"

    # 5. 构建 data_dir
    data_dir = f"{work_dir}/{base_dir}/N{params['N']}{spin_label}{suffix_common}"

    return PathSpec(
        work_dir=work_dir,
        base_dir=base_dir,
        run_dir=run_dir,
        output_dir=output_dir,
        restart_dir=restart_dir,
        tmp_dir=tmp_dir,
        data_dir=data_dir,
    )


def setup_params(U=1.0, t=0.1, t2=None, sz=None, s=None, type=None, restart=False):
    params = {}
    params["U"] = U
    params["t"] = t
    params["t2"] = t2
    params["sz"] = sz
    params["s"] = s
    params["type"] = type
    params["restart"] = restart
    return params


def setup_work_environment(params, rank=0) -> PathSpec:
    """Setup the working environment and parameters using PathSpec"""
    spec = build_path_spec(params)

    if rank == 0:
        tmp_parent = os.path.dirname(spec.tmp_dir)
        os.makedirs(tmp_parent, exist_ok=True)

        if not params["restart"]:
            os.makedirs(spec.output_dir, exist_ok=True)
            print(f"Working directory: {spec.output_dir}")
        else:
            os.makedirs(spec.restart_dir, exist_ok=True)
            print(f"Working directory (restart): {spec.restart_dir}")

    return spec


def filter_work_items(work_items_all, params, force_distribute=True):
    """
    Filter work items based on restart conditions.
    If restart is enabled and not force_distribute, it checks previous results.
    Good results are copied to the restart directory, and only unfinished/bad items are returned.
    """
    if not params["restart"] or force_distribute:
        return work_items_all

    work_items = []
    spec = params.get("path_spec")
    if spec:
        restart_dir = spec.restart_dir
    else:
        result_dir = params.get("result_dir", "")
        restart_dir = f"{result_dir}_restart"

    if spec:
        base_dir = f"{spec.work_dir}/{spec.base_dir}"
    else:
        base_dir = f"Block_U{params['U']:.4f}_t{params['t']:.4f}"
        if params.get("t2") is not None:
            base_dir = f"{base_dir}_tp{params['t2']:.4f}"

    restart_dir_abs = os.path.abspath(restart_dir)
    for idx, (hole, class_idx) in enumerate(work_items_all):
        pattern = f"{base_dir}/N{params['N']}*/hole{hole}_class{class_idx}_cluster0_results.txt"
        files = glob.glob(pattern)

        errors = []
        t11s = []
        for file in files:
            errors.append(read_key(file, "Relative") or 100000)
            t11s.append(read_key(file, "T11") or 100000)

        if not t11s:
            work_items.append((hole, class_idx))
            continue

        idx_selected = t11s.index(min(t11s))
        if not errors or errors[idx_selected] > 0.05:
            work_items.append((hole, class_idx))
        else:
            base_path = files[idx_selected].replace("_cluster0_results.txt", "")

            base_path_abs = os.path.abspath(base_path)
            if base_path_abs.startswith(restart_dir_abs + os.sep):
                continue

            for src in glob.glob(f"{base_path}*"):
                shutil.copy2(src, restart_dir)
            for suffix in ("_eigvals.npy", "_eigvecs.npy"):
                for path in glob.glob(f"{restart_dir}/hole{hole}_class{class_idx}*{suffix}"):
                    os.remove(path)
            write_file(f"{restart_dir}/restart_flag.txt", f"cp {base_path}* ")

    return work_items


def check_and_print_adiabatic_info(params):
    if params["type"] != "adiabatic":
        return
    if params["delta"] is None:
        raise ValueError("delta is not set")
    if params["delta2"] is None:
        params["delta2"] = 0.0

    t1_previous = params["t"] - params["delta"]
    t2_previous = params["t2"] - params["delta2"] if params["t2"] is not None else None

    params_previous = setup_params(
        U=params["U"],
        t=t1_previous,
        t2=t2_previous,
        sz=params["sz"],
        s=params["s"],
        type=params["type"],
        restart=params["restart"],
    )
    spec_previous = build_path_spec(params_previous)
    print(f"In the adiabatic process, read data from {spec_previous.data_dir}", flush=True)


def read_key(filename, key):
    """Read a specific key from a file"""
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(key):
                    tmp = line.split()
                    return float(tmp[-1])
    except FileNotFoundError:
        return None
    return None


def write_file(filename, strings):
    with open(filename, "a") as f:
        f.write(f"{strings}\n")


def load_array_compat(filename, text_dtype=None, allow_pickle=False):
    """Load either a real .npy file or a legacy text file with a .npy suffix."""
    try:
        return np.load(filename, allow_pickle=allow_pickle)
    except (FileNotFoundError, OSError, ValueError):
        try:
            return np.loadtxt(filename, dtype=text_dtype)
        except (FileNotFoundError, OSError, ValueError) as e:
            raise RuntimeError(f"Failed to load array from {filename}: {str(e)}")


def read_previous(filename):
    eigvals = load_array_compat(f"{filename}_eigvals.npy", text_dtype=float)
    eigvecs = load_array_compat(f"{filename}_eigvecs.npy", text_dtype=complex)
    try:
        selected_indices = load_array_compat(
            f"{filename}_t11_selected_indices.npy", text_dtype=int
        )
    except RuntimeError:
        selected_indices = load_array_compat(f"{filename}_selected_indices.npy", text_dtype=int)
    selected_indices = np.atleast_1d(selected_indices).astype(int)
    return eigvals, eigvecs, selected_indices


def setup_work_environment_previous(params):
    if params["delta"] is None:
        raise ValueError("delta is not set")
    if params["delta2"] is None:
        params["delta2"] = 0.0
    t1_previous = params["t"] - params["delta"]
    t2_previous = params["t2"] - params["delta2"] if params["t2"] is not None else None

    result_dir1 = f"Block_U{params['U']:.4f}_t{t1_previous:.4f}"
    if t2_previous is not None:
        result_dir1 = f"{result_dir1}_tp{t2_previous:.4f}"

    result_dir2 = f"N{params['N']}"
    result_dir3 = f"N{params['N']}"
    if params["sz"] is not None:
        result_dir2 = f"{result_dir2}_sz{params['sz']:.4f}"
        result_dir3 = f"{result_dir3}_sz{params['sz']:.4f}"
    if params["s2"] is not None:
        result_dir2 = f"{result_dir2}_s{params['s2']:.0f}"
        result_dir3 = f"{result_dir3}_s{params['s2']:.0f}"
    if params["type"] is not None:
        result_dir2 = f"{result_dir2}_{params['type']}"

    return result_dir1, result_dir2, result_dir3


def build_legacy_result_dirs(params) -> tuple[str, str, str]:
    base_dir = f"U{params['U']:.4f}_t{params['t']:.4f}"
    if params.get("t2") is not None:
        base_dir = f"{base_dir}_tp{params['t2']:.4f}"

    primary_suffix = ""
    fallback_suffix = ""
    if params.get("sz") is not None:
        primary_suffix = f"{primary_suffix}_sz{params['sz']:.4f}"
        fallback_suffix = f"{fallback_suffix}_sz{0.5 - params['sz']:.4f}"
    if params.get("s2") is not None:
        primary_suffix = f"{primary_suffix}_s{params['s2']:.0f}"
        fallback_suffix = f"{fallback_suffix}_s{params['s2']:.0f}"
    if params.get("match_spin_sectors"):
        primary_suffix = f"{primary_suffix}_match_spin_sectors"
        fallback_suffix = f"{fallback_suffix}_match_spin_sectors"
    if params.get("type") is not None:
        primary_suffix = f"{primary_suffix}_{params['type']}"
        fallback_suffix = f"{fallback_suffix}_{params['type']}"
    if params.get("restart"):
        primary_suffix = f"{primary_suffix}_restart"
        fallback_suffix = f"{fallback_suffix}_restart"
    return base_dir, primary_suffix, fallback_suffix


def cluster_result_report_path(
    root_dir: str,
    nsites: int,
    suffix: str,
    hole: int,
    class_idx: int,
    cluster_idx: int,
) -> str:
    return (
        f"{root_dir}/N{nsites}{suffix}/"
        f"hole{hole}_class{class_idx}_cluster{cluster_idx}_results.txt"
    )


def _candidate_suffixes(primary_suffix: str, fallback_suffix: str) -> list[str]:
    suffixes = [primary_suffix]
    if fallback_suffix != primary_suffix:
        suffixes.append(fallback_suffix)
    return suffixes


def _report_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(result_json_path(path))


def find_existing_cluster_report(
    root_dir: str,
    nsites: int,
    primary_suffix: str,
    fallback_suffix: str,
    *,
    hole: int,
    class_idx: int,
    cluster_idx: int,
) -> tuple[str, str]:
    for suffix in _candidate_suffixes(primary_suffix, fallback_suffix):
        report_path = cluster_result_report_path(root_dir, nsites, suffix, hole, class_idx, cluster_idx)
        if _report_exists(report_path):
            return report_path, suffix
    raise FileNotFoundError(
        "No result report found under "
        f"{root_dir} for N={nsites}, hole={hole}, class={class_idx}, cluster={cluster_idx} "
        f"with suffixes '{primary_suffix}' and '{fallback_suffix}'"
    )


def find_existing_run_dir(
    root_dir: str,
    nsites: int,
    primary_suffix: str,
    fallback_suffix: str,
) -> tuple[str, str]:
    for suffix in _candidate_suffixes(primary_suffix, fallback_suffix):
        run_dir = f"{root_dir}/N{nsites}{suffix}"
        if os.path.exists(run_dir):
            return run_dir, suffix
    raise FileNotFoundError(
        f"No run directory found under {root_dir} for N={nsites} "
        f"with suffixes '{primary_suffix}' and '{fallback_suffix}'"
    )


# ============================================================
# Results Serialization
# ============================================================

RESULT_SCHEMA_VERSION = 1
ANALYSIS_ONLY_ERROR = (
    "No spin-coupling section found in report. "
    "Analysis-only reports from MODE=fixed_sz_s2 cannot be used for LCE/embed."
)


def result_json_path(path: str) -> str:
    if path.endswith(".json"):
        return path
    if path.endswith(".txt"):
        return f"{path[:-4]}.json"
    return f"{path}_results.json"


def write_result_artifact(path: str, payload: dict[str, Any]) -> None:
    with open(result_json_path(path), "w") as f:
        json.dump(payload, f, indent=2)


def load_result_artifact(path: str) -> Optional[dict[str, Any]]:
    json_path = result_json_path(path)
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def serialize_complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def deserialize_complex(payload: dict[str, Any]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def serialize_cluster(cluster) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in cluster]


def deserialize_cluster(payload) -> list[tuple[int, int]]:
    return [tuple(map(int, point)) for point in payload]


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


def build_spin_coupling_artifact_from_coeffs(
    cluster,
    bonds,
    coeffs,
    error,
    t11_norm: float,
    overlap,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            coeffs_by_vector.setdefault((dx, dy), []).append((bond_group, coeffs[class_idx + 1]))

    groups = []
    for bond_idx, (dx, dy) in enumerate(get_all_possible_vectors(cluster)):
        terms = []
        for bond_group, bond_coeffs in coeffs_by_vector.get((dx, dy), []):
            for bond, coefficient in zip(bond_group, bond_coeffs):
                terms.append(
                    {
                        "sites": [int(bond[0]), int(bond[1])],
                        "coefficient": serialize_complex(coefficient),
                    }
                )
        groups.append({"arity": 2, "vector": [dx, dy], "label": f"J{bond_idx + 1}", "terms": terms})

    for arity in (4, 6, 8):
        terms = []
        for class_idx, bond_group in enumerate(bonds):
            if len(bond_group) > 0 and len(bond_group[0]) == arity:
                for bond, coefficient in zip(bond_group, coeffs[class_idx + 1]):
                    terms.append(
                        {
                            "sites": [int(idx) for idx in bond],
                            "coefficient": serialize_complex(coefficient),
                        }
                    )
        groups.append({"arity": arity, "terms": terms})

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": RESULT_KIND_SPIN_COUPLINGS,
        "cluster": serialize_cluster(cluster),
        "fit": {
            "relative_error": float(error[0]),
            "residual": float(error[1]),
            "r_squared": float(error[2]),
            "t11_minus_1_norm": float(t11_norm),
            "overlap": None if overlap is None else float(overlap),
        },
        "operators": {
            "constant_term": serialize_complex(coeffs[0]),
            "groups": groups,
        },
    }
    if metadata:
        payload.update(metadata)
    return payload


def build_spin_coupling_artifact_from_operator_list(
    cluster,
    operators,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    groups = []
    all_vectors = get_all_possible_vectors(cluster)
    two_site_group_count = len(all_vectors)

    for idx, (dx, dy) in enumerate(all_vectors):
        terms = []
        if idx + 1 < len(operators):
            for site1, site2, coefficient in operators[idx + 1]:
                terms.append(
                    {
                        "sites": [int(site1), int(site2)],
                        "coefficient": serialize_complex(coefficient),
                    }
                )
        groups.append({"arity": 2, "vector": [dx, dy], "label": f"J{idx + 1}", "terms": terms})

    for offset, arity in enumerate((4, 6, 8), start=two_site_group_count + 1):
        group_terms = operators[offset] if offset < len(operators) else []
        groups.append({"arity": arity, "terms": _serialize_terms(group_terms, arity)})

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": RESULT_KIND_SPIN_COUPLINGS,
        "cluster": serialize_cluster(cluster),
        "operators": {
            "constant_term": serialize_complex(operators[0]),
            "groups": groups,
        },
    }
    if metadata:
        payload.update(metadata)
    return payload


def build_projection_analysis_artifact(
    cluster, model, metadata: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": RESULT_KIND_PROJECTION_ANALYSIS,
        "cluster": serialize_cluster(cluster),
        "projection": {
            "t11_minus_1_norm": float(model.T11m1_norm),
            "overlap": None if model.overlap is None else float(model.overlap),
            "selected_state_count": int(len(model.t11_selected_indices)),
            "heff_dimension": [int(model.Heff.shape[0]), int(model.Heff.shape[1])],
            "selected_indices": [int(idx) for idx in model.t11_selected_indices],
            "double_occupation_expectation": [
                float(value.real) for value in model.t11_selected_occupation
            ],
        },
    }
    if metadata:
        payload.update(metadata)
    return payload


def artifact_to_operator_list(payload: dict[str, Any]):
    if payload.get("result_kind") != RESULT_KIND_SPIN_COUPLINGS:
        raise ValueError(ANALYSIS_ONLY_ERROR)

    operators = [deserialize_complex(payload["operators"]["constant_term"])]
    for group in payload["operators"]["groups"]:
        terms = []
        for term in group.get("terms", []):
            coefficient = deserialize_complex(term["coefficient"])
            terms.append([*map(int, term["sites"]), coefficient])
        operators.append(terms)
    return operators


# ============================================================
# Bond Utilities
# ============================================================

def find_bond_vector(bond, cluster):
    """Find the bond vector of a bond"""
    site1, site2 = bond
    x1, y1 = cluster[site1]
    x2, y2 = cluster[site2]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    return sorted([dx, dy], reverse=True)


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

def tune_sz(sz, N, rank=0):
    if sz is None:
        return None
    else:
        sz_new = round(sz * 2) / 2
        if abs(sz - sz_new) > 1e-6:
            if rank == 0:
                print(f"Warning: sz={sz} is not a half integer, set sz to {sz_new}")
        if N % 2 == 0:
            if int(sz_new * 2) % 2 != 0:
                sz_new = 0.0
                if rank == 0:
                    print(f"Warning: sz does not match the number of sites {N}, set sz to {sz_new}")
        elif N % 2 == 1:
            if int(sz_new * 2) % 2 == 0:
                sz_new = 0.5
                if rank == 0:
                    print(f"Warning: sz does not match the number of sites {N}, set sz to {sz_new}")
        else:
            if rank == 0:
                raise ValueError(f"N is not an integer, N={N}!")

        return sz_new


def savefile(filename, data, encoding="npy"):
    if encoding == "txt":
        np.savetxt(f"{filename}.txt", data, fmt="%.10f")
    elif encoding == "npy":
        np.save(f"{filename}.npy", data, allow_pickle=False)
    else:
        raise ValueError(f"Invalid encoding: {encoding}")


def loadfile(filename, encoding="npy", dtype=float, format=None):
    if encoding == "txt":
        data = np.loadtxt(f"{filename}.txt", dtype=dtype)
    elif encoding == "npy":
        data = np.load(f"{filename}.npy", allow_pickle=False)
    else:
        raise ValueError(f"Invalid encoding: {encoding}")

    if format == "1d":
        return np.atleast_1d(data)
    elif format == "2d":
        return np.atleast_2d(data)
    else:
        return data


# ============================================================
# Operator Key Functions
# ============================================================

def k2s(i, j):
    return tuple(sorted([i, j]))


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


def k6s_all(i, j, k, l, m, n):
    return tuple(sorted([i, j, k, l, m, n]))


# ============================================================
# Data Reading
# ============================================================

def create_dict(ref_dict):
    data = {}
    for nsites in ref_dict:
        data[nsites] = {}
        for hole in ref_dict[nsites]:
            data[nsites][hole] = {}
            for class_idx in ref_dict[nsites][hole]:
                data[nsites][hole][class_idx] = {}
                for rank in ref_dict[nsites][hole][class_idx]:
                    data[nsites][hole][class_idx][rank] = {}
                    data[nsites][hole][class_idx][rank]["subgraph"] = []
                    data[nsites][hole][class_idx][rank]["match"] = []
                    data[nsites][hole][class_idx][rank]["indices"] = []
                    data[nsites][hole][class_idx][rank]["coupling_original"] = []
                    data[nsites][hole][class_idx][rank]["coupling_net"] = []
    return data


def parse_filename_Block(filename: str) -> Tuple[int, int, int]:
    stem = filename
    for suffix in ("_results.json", "_results.txt", "results.json", "results.txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.replace("hole", "").replace("class", "").replace("cluster", "").rstrip("_").split("_")
    hole, class_idx, cluster = map(int, parts)
    return hole, class_idx, cluster


def read_cluster_file_Block(file_path: str) -> List[Tuple[int, int]]:
    cluster = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Point"):
                coords = line.split(":")[1].strip()
                x, y = map(int, coords.strip("()").split(","))
                cluster.append((x, y))
    return cluster


def _artifact_cluster_from_file(file_path: str):
    payload = load_result_artifact(file_path)
    if payload is None:
        return None
    if payload.get("result_kind") != "spin_couplings":
        return None
    return deserialize_cluster(payload["cluster"])


def read_coords_file_Block(prefix_path, suffix, suffix1, n_max):
    # 使用字典存储数据
    clusters = {}
    print(f"prefix_path: {prefix_path}")
    print(f"suffix: {suffix}")
    print(f"suffix1: {suffix1}")
    print(f"n_max: {n_max}")
    # 读取每个位点数的数据
    for n in range(1, n_max + 1):
        try:
            n_dir, _ = find_existing_run_dir(prefix_path, n, suffix, suffix1)
        except FileNotFoundError:
            continue
        print(f"Reading clusters with {n} sites from {n_dir}...")
        clusters[n] = {}
        files = sorted(os.listdir(n_dir))
        for file in files:
            if file.endswith("_results.json"):
                file_path = os.path.join(n_dir, file)
                cluster = _artifact_cluster_from_file(file_path)
                if cluster is None:
                    continue

                hole, class_idx, rank = parse_filename_Block(file)
                if hole not in clusters[n]:
                    clusters[n][hole] = {}
                if class_idx not in clusters[n][hole]:
                    clusters[n][hole][class_idx] = {}
                clusters[n][hole][class_idx][rank] = cluster

            if file.endswith("_results.txt"):
                file_path = os.path.join(n_dir, file)
                if load_result_artifact(file_path) is not None:
                    continue

                hole, class_idx, rank = parse_filename_Block(file)
                if hole not in clusters[n]:
                    clusters[n][hole] = {}
                if class_idx not in clusters[n][hole]:
                    clusters[n][hole][class_idx] = {}

                cluster = read_cluster_file_Block(file_path)
                clusters[n][hole][class_idx][rank] = cluster

        if not clusters[n]:
            clusters.pop(n)

    if not clusters:
        raise FileNotFoundError(
            "No cluster result directories found with spin-coupling artifacts under "
            f"{prefix_path} for 1 <= N <= {n_max} with suffixes '{suffix}' and '{suffix1}'"
        )

    # Print the structure of the data
    print("\nData structure summary:")
    for n in sorted(clusters.keys()):
        print(f"\n{n} sites:")
        for hole in sorted(clusters[n].keys()):
            print(f"  Hole {hole}:")
            for class_idx in sorted(clusters[n][hole].keys()):
                count = len(clusters[n][hole][class_idx])
                print(f"    Class {class_idx}: {count} clusters")

    return clusters


def read_operators(file_path: str):
    payload = load_result_artifact(file_path)
    if payload is not None:
        return artifact_to_operator_list(payload)
    raise ValueError(ANALYSIS_ONLY_ERROR)


def read_fit_metric(file_path: str, metric: str):
    payload = load_result_artifact(file_path)
    if payload is None:
        return None
    fit = payload.get("fit")
    if fit is None:
        return None
    return fit.get(metric)


def create_graph_from_cluster(cluster, t2=None):
    max_bond = 2 if t2 is not None else 1
    adj = generate_adjacency_matrix(cluster, max_bond)
    G = graph_from_adj(adj)
    return G


def find_cluster_match(target_cluster, ref_clusters, t2=None):
    matches = []

    # 创建目标团簇的图
    target_canon = canonical_form(target_cluster)
    target_graph = create_graph_from_cluster(target_cluster, t2)
    target_graph_canon = create_graph_from_cluster(target_canon, t2)
    nsite = len(target_canon)
    hole = count_holes(target_canon)

    # 判断ref_clusters是2D还是4D
    is_4d = False
    for idx in ref_clusters:
        if isinstance(ref_clusters[idx], dict) and any(
            isinstance(ref_clusters[idx][hole], dict) for hole in ref_clusters[idx]
        ):
            is_4d = True
            break

    if is_4d:
        clusters = ref_clusters[nsite][hole]
    else:
        clusters = ref_clusters

    # 对每个class，先用第一个团簇判断是否同构
    for class_idx in clusters:
        # 获取class的第一个团簇
        first_cluster = clusters[class_idx][0]

        # 创建参考团簇的图
        ref_graph = create_graph_from_cluster(first_cluster, t2)
        # 使用GraphMatcher判断是否同构
        graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
            ref_graph, target_graph, edge_match=EDGE_MATCHER
        )
        graph_matcher_canon = nx.algorithms.isomorphism.GraphMatcher(
            ref_graph, target_graph_canon, edge_match=EDGE_MATCHER
        )

        if graph_matcher.is_isomorphic() and graph_matcher_canon.is_isomorphic():
            # 获取同构映射
            mapping = graph_matcher.mapping
            mapping_canon = graph_matcher_canon.mapping
            # 使用同构映射找到坐标对应关系
            coord_mapping = []
            coord_mapping_canon = []
            for i in range(len(first_cluster)):
                coord_mapping.append(mapping[i])
                coord_mapping_canon.append(mapping_canon[i])

            # 遍历class中的所有团簇
            for rank in clusters[class_idx]:
                ref_cluster = clusters[class_idx][rank]

                # 验证坐标是否匹配
                is_valid = True
                for i, ref_coord in enumerate(ref_cluster):
                    target_coord = target_canon[coord_mapping_canon[i]]
                    if ref_coord != target_coord:
                        is_valid = False
                        break

                if is_valid:
                    if is_4d:
                        matches.append((nsite, hole, class_idx, rank, coord_mapping))
                    else:
                        matches.append((class_idx, rank, coord_mapping))

    if not matches:
        raise ValueError(f"No matching cluster found for cluster: {target_cluster}, {target_canon}")
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple matching clusters found for cluster: {target_cluster}, {target_canon}"
        )

    return matches[0]
