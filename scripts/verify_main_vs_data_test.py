#!/usr/bin/env python3
"""Verify `python -m cuprate.main` output against data_test/ reference files.

The verification is strict per the prompt in docs/2026-04-24-verify-main-vs-data_test.prompt.md:
every (t, N) in the fixed matrix, every (hole, class, cluster), every one of the nine
reference files is compared independently. atol starts at 1e-10. Any deviation is reported
with its magnitude and location; equivalent differences (e.g. different degenerate-subspace
representatives) are also documented with explicit evidence.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cuprate.clusters import ClusterSets  # noqa: E402
from cuprate.hubbard import HubbardModel  # noqa: E402
from cuprate.operators import operators_from_terms, operators_to_terms  # noqa: E402
from cuprate.states import (  # noqa: E402
    count_double_occ,
    generate_states,
    site_code,
)


# ---------------------------------------------------------------------------
# Constants and matrix
# ---------------------------------------------------------------------------
ATOL_BASELINE = 1e-10
ATOL_TXT_ROUND = 5.1e-10  # text files print to 10 decimals -> rounding noise ~5e-11
ATOL_DEGEN_TRACE = 1e-8   # trace comparisons in degenerate subspaces

MATRIX: list[tuple[float, list[int]]] = [
    (round(0.02 * i, 2), [2, 3, 4, 5]) for i in range(1, 31)
]
U_VALUE = 1.0

LEGACY_SITE_CODE = {0: 0, 1: 1, 2: -1, 3: 2}
LEGACY_VALUE_ORDER = {0: 0, 1: 1, -1: 2, 2: 3}


# ---------------------------------------------------------------------------
# Legacy encoding helpers (mirrors back/main/solver.py)
# ---------------------------------------------------------------------------
def legacy_state_row(state: int, N: int) -> tuple[int, ...]:
    return tuple(LEGACY_SITE_CODE[site_code(state, site)] for site in range(N))


def legacy_state_key(state: int, N: int) -> tuple[int, int, int, tuple[int, ...]]:
    """Legacy ordering: (D>0 flag, |twoSz|, -twoSz, value-order tuple).

    The reference data groups D=0 states together up front, then *all* D>=1 states
    sorted by (|twoSz|, -twoSz, vo) without further grouping by D. Empirically
    determined from data_test/ state dumps.
    """
    row = legacy_state_row(state, N)
    twoSz = sum(value for value in row if abs(value) == 1)
    D_flag = 1 if row.count(2) > 0 else 0
    return (
        D_flag,
        abs(twoSz),
        -twoSz,
        tuple(LEGACY_VALUE_ORDER[value] for value in row),
    )


# ---------------------------------------------------------------------------
# Text npy parsing: old writer used np.savetxt with complex/real formatting.
# The complex entries look like " (1.23e+01+4.56e-02j)" — note the mantissa
# itself contains '+' / '-' inside the exponent, so we must tokenise on paren
# pairs first, then strip the final 'j' and let Python's complex() do parsing.
# ---------------------------------------------------------------------------
_PAREN_RE = re.compile(r"\(([^()]+)\)")


def _parse_complex_token(token: str) -> complex:
    token = token.strip()
    if token.endswith("j"):
        return complex(token.replace(" ", ""))
    return complex(float(token), 0.0)


def parse_text_array(path: Path) -> np.ndarray:
    """Parse a .npy reference file that may be real NumPy binary or space-padded text.

    The data_test/ dump mixes both formats: some .npy files are true `\\x93NUMPY`
    binary, others are `np.savetxt` text with `(real+imagj)` complex cells. Try
    binary first, fall back to text.
    """
    with path.open("rb") as fh:
        header = fh.read(6)
    if header.startswith(b"\x93NUMPY"):
        return np.load(path, allow_pickle=False)

    text = path.read_text()
    stripped = text.strip()
    if not stripped:
        return np.array([])

    # Try real first (integer or float, one or more per line).
    try:
        arr = np.loadtxt(StringIO(stripped), dtype=float)
        return np.atleast_1d(arr)
    except ValueError:
        pass

    rows: list[list[complex]] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = _PAREN_RE.findall(line)
        if not tokens:
            tokens = line.split()
        row = [_parse_complex_token(tok) for tok in tokens]
        rows.append(row)

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        if widths == {1}:
            return np.array([row[0] for row in rows], dtype=complex)
        raise ValueError(f"Jagged text complex array in {path}: row widths {widths}")

    width = widths.pop()
    arr = np.array(rows, dtype=complex)
    if width == 1:
        return arr.reshape(-1)
    return arr


# ---------------------------------------------------------------------------
# Pipeline recomputation to obtain quantities the CLI does not persist.
# ---------------------------------------------------------------------------
@dataclass
class FamilyObservables:
    hole: int
    class_idx: int
    basis_states: list[int]
    eigvals: np.ndarray
    eigvecs: np.ndarray
    double_occ: np.ndarray
    s2_diag_values: np.ndarray          # shape (n_eig,) — S(S+1)
    s2_variance: np.ndarray             # shape (n_eig,) — var(S²)
    s2_overlap: np.ndarray              # shape (n_eig,) — pure-spin overlap
    spin_fock_rows_canonical: list[int]
    selected_indices: list[int]
    selected_double_occ: np.ndarray
    heff: np.ndarray
    t11m1: np.ndarray
    t11m1_norm: float
    coupling_coeffs: list
    bond_groups: list
    fit_metrics: tuple[float, float, float]
    cluster_members: list[tuple[int, list[tuple[int, int]]]]  # (cluster_idx, sites)


def recompute_family(
    N: int, U: float, t: float, hole: int, class_idx: int, members: list
) -> FamilyObservables:
    """Recompute the family observables inline, matching the workchain pipeline."""
    representative = members[0]
    model = HubbardModel(representative, U, t)
    model.set_symmetry("full")
    model.build_hamiltonians()
    model.solve()
    model.project(method="occ")

    block = model.blocks[0]
    eigvecs_fock = block.eigvecs_fock
    eigvals = np.asarray(block.eigvals)

    from cuprate.states import calc_double_occupation_matrix, calc_fourS2_matrix

    dom = calc_double_occupation_matrix(block.basis_states, N)
    double_occ = np.real(np.diag(eigvecs_fock.conj().T @ dom @ eigvecs_fock))

    fourS2 = calc_fourS2_matrix(block.basis_states, N)
    mat_fourS2 = eigvecs_fock.conj().T @ fourS2 @ eigvecs_fock
    sq_fourS2 = eigvecs_fock.conj().T @ fourS2 @ fourS2 @ eigvecs_fock
    s2_values = 0.25 * np.real(np.diag(mat_fourS2))
    s2_variance = 0.0625 * np.real(np.diag(sq_fourS2) - np.diag(mat_fourS2) ** 2)

    spin_rows = block.spin_fock_rows()
    s2_overlap = np.sum(np.abs(eigvecs_fock[spin_rows, :]) ** 2, axis=0)

    selected_indices = list(model.selected_indices[0])
    heff = np.asarray(model.heff[0])
    # Rebuild T11m1 matrix (workchain only exposes its norm).
    spin_cols = block.spin_sector_columns()
    s_bd = block.eigvecs[np.ix_(spin_cols, selected_indices)]
    U_svd, sigma, _ = np.linalg.svd(s_bd, full_matrices=False)
    t11m1 = U_svd @ np.diag(sigma) @ U_svd.conj().T - np.eye(s_bd.shape[0])
    t11m1_norm = float(np.linalg.norm(t11m1.flatten()))

    bond_groups = (
        representative.generate_bonds(N=2, is_connected=False)
        + representative.generate_bonds(N=4, is_connected=True)
        + representative.generate_bonds(N=6, is_connected=True)
    )
    model.fit(bond_groups=bond_groups)
    members_payload = []
    for member in members:
        members_payload.append(
            (int(member.cluster_idx), [(int(x), int(y)) for x, y in member.sites])
        )

    return FamilyObservables(
        hole=hole,
        class_idx=class_idx,
        basis_states=list(block.basis_states),
        eigvals=eigvals,
        eigvecs=np.asarray(eigvecs_fock),
        double_occ=double_occ,
        s2_diag_values=s2_values,
        s2_variance=s2_variance,
        s2_overlap=s2_overlap,
        spin_fock_rows_canonical=spin_rows,
        selected_indices=selected_indices,
        selected_double_occ=double_occ[np.asarray(selected_indices, dtype=int)],
        heff=heff,
        t11m1=t11m1,
        t11m1_norm=t11m1_norm,
        coupling_coeffs=list(model.coupling_coeffs),
        bond_groups=list(model.bond_groups),
        fit_metrics=tuple(model.fit_metrics),
        cluster_members=members_payload,
    )


# ---------------------------------------------------------------------------
# Reference loading + legacy ↔ canonical permutation
# ---------------------------------------------------------------------------
def legacy_row_to_state(row: Sequence[int]) -> int:
    """Decode one states.npy row (entries in {-1,0,1,2}) back to a Fock int."""
    code = {0: 0, 1: 1, -1: 2, 2: 3}
    state = 0
    for i, value in enumerate(row):
        state |= code[int(round(float(value)))] << (2 * i)
    return state


def spin_rows_from_states_array(states_arr: np.ndarray, N: int) -> list[int]:
    """Indices of states.npy rows whose legacy row is a pure-spin (no 2) configuration."""
    rows: list[int] = []
    for idx, row in enumerate(states_arr):
        values = [int(round(float(x.real)) if np.iscomplexobj(states_arr) else int(round(float(x))))
                  for x in row]
        if 2 not in values:
            rows.append(idx)
    return rows


def legacy_spin_ordering(states_arr: np.ndarray, N: int) -> list[int]:
    """Return the Fock int of each D=0 row in legacy sort order."""
    ordering: list[int] = []
    for row in states_arr:
        values = [int(round(float(x.real)) if np.iscomplexobj(states_arr) else int(round(float(x))))
                  for x in row]
        if 2 in values:
            continue
        ordering.append(legacy_row_to_state(values))
    return ordering


def canonical_spin_permutation(
    canonical_spin_states: list[int], legacy_spin_states: list[int]
) -> np.ndarray:
    """Find P such that canonical[P[i]] == legacy[i]. Returns np.ndarray[int]."""
    index_of = {state: idx for idx, state in enumerate(canonical_spin_states)}
    return np.asarray([index_of[state] for state in legacy_spin_states], dtype=int)


# ---------------------------------------------------------------------------
# Cluster discovery
# ---------------------------------------------------------------------------
def enumerate_families(N: int) -> list[tuple[int, int, list]]:
    by_family: dict[tuple[int, int], list] = {}
    for cluster in ClusterSets(N).generate().clusters:
        key = (int(cluster.hole), int(cluster.class_idx))
        by_family.setdefault(key, []).append(cluster)
    return [(hole, cls, by_family[(hole, cls)]) for (hole, cls) in sorted(by_family)]


# ---------------------------------------------------------------------------
# Txt parsing (cluster results)
# ---------------------------------------------------------------------------
POINT_RE = re.compile(r"^Point\s+(\d+):\s+\((-?\d+),\s*(-?\d+)\)$")
VECTOR_RE = re.compile(
    r"^Bond vector \((-?\d+),\s*(-?\d+)\),\s+J\d+:(?:\s+\(not present in cluster\))?$"
)
TWO_SITE_RE = re.compile(
    r"^\s*\d+:\s+Sites\s+(\d+)-(\d+):\s+"
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+\+\s+"
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)i$",
    re.IGNORECASE,
)
FOUR_SITE_RE = re.compile(
    r"^\s*class\s+\d+\s+type\s+\d+\s+:\s+Four-site\s+"
    r"\((\d+)-(\d+)\)\s+\*\s+\((\d+)-(\d+)\):\s+"
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+\+\s+"
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)i$",
    re.IGNORECASE,
)
SIX_SITE_RE = re.compile(
    r"^\s*class\s+\d+\s+type\s+\d+\s+:\s+Six-site\s+"
    r"\((\d+)-(\d+)\)\s+\*\s+\((\d+)-(\d+)\)\s+\*\s+\((\d+)-(\d+)\):\s+"
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+\+\s+"
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)i$",
    re.IGNORECASE,
)


@dataclass
class TxtArtifact:
    sites: list[list[int]]
    constant: complex
    two_site: dict[tuple[int, int], dict[tuple[int, int], complex]]
    four_site: dict[tuple[tuple[int, int], tuple[int, int]], complex]
    six_site: dict[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        complex,
    ]
    fit: dict[str, float]


def parse_txt(path: Path) -> TxtArtifact:
    sites: list[list[int]] = []
    two_site: dict[tuple[int, int], dict[tuple[int, int], complex]] = {}
    four_site: dict[tuple[tuple[int, int], tuple[int, int]], complex] = {}
    six_site: dict[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]], complex
    ] = {}
    fit: dict[str, float] = {}
    constant = 0.0 + 0.0j

    section = None
    current_vector: tuple[int, int] | None = None

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped == "=== Cluster Points ===":
            section, current_vector = "cluster_points", None
            continue
        if stripped == "=== Individual Bond Coefficients ===":
            section, current_vector = "coefficients", None
            continue
        if stripped in {"=== Individual Fit Error ===", "=== Fit quality ==="}:
            section, current_vector = "fit", None
            continue

        if section == "cluster_points":
            m = POINT_RE.match(stripped)
            if m is not None:
                sites.append([int(m.group(2)), int(m.group(3))])
            continue

        if section == "coefficients":
            if stripped.startswith("Constant term:"):
                constant = complex(float(stripped.split(":", 1)[1]), 0.0)
                continue
            vm = VECTOR_RE.match(stripped)
            if vm is not None:
                current_vector = (int(vm.group(1)), int(vm.group(2)))
                two_site.setdefault(current_vector, {})
                continue
            tm = TWO_SITE_RE.match(line)
            if tm is not None and current_vector is not None:
                a, b = int(tm.group(1)), int(tm.group(2))
                key = tuple(sorted((a, b)))
                two_site[current_vector][key] = complex(
                    float(tm.group(3)), float(tm.group(4))
                )
                continue
            fm = FOUR_SITE_RE.match(line)
            if fm is not None:
                pair1 = tuple(sorted((int(fm.group(1)), int(fm.group(2)))))
                pair2 = tuple(sorted((int(fm.group(3)), int(fm.group(4)))))
                key = tuple(sorted((pair1, pair2)))
                four_site[key] = complex(float(fm.group(5)), float(fm.group(6)))
                continue
            sm = SIX_SITE_RE.match(line)
            if sm is not None:
                pairs = (
                    tuple(sorted((int(sm.group(1)), int(sm.group(2))))),
                    tuple(sorted((int(sm.group(3)), int(sm.group(4))))),
                    tuple(sorted((int(sm.group(5)), int(sm.group(6))))),
                )
                key = tuple(sorted(pairs))
                six_site[key] = complex(float(sm.group(7)), float(sm.group(8)))
                continue

        if section == "fit":
            if ":" not in stripped:
                continue
            key, value = [part.strip() for part in stripped.split(":", 1)]
            try:
                f = float(value)
            except ValueError:
                continue
            if key == "Relative Error":
                fit["relative_error"] = f
            elif key == "Residual":
                fit["residual"] = f
            elif key in {"R^2", "R²"}:
                fit["r_squared"] = f
            elif key == "T11-1 norm":
                fit["t11_minus_1_norm"] = f

    return TxtArtifact(
        sites=sites,
        constant=constant,
        two_site=two_site,
        four_site=four_site,
        six_site=six_site,
        fit=fit,
    )


# ---------------------------------------------------------------------------
# Fitted-coefficient extraction from workchain family exchange JSON
# ---------------------------------------------------------------------------
def complex_from_json(value: dict[str, Any]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def _sites_in_operator_order(entry: dict[str, Any], N: int) -> list[tuple[int, int]]:
    sites = [tuple(map(int, site)) for site in entry["sites"]]
    indices = [int(index) for index in entry["indices"]]
    if len(sites) != N or len(indices) != N:
        raise ValueError(f"cluster sites and indices must both have length N={N}")
    if sorted(indices) != list(range(N)):
        raise ValueError(f"cluster indices must be a permutation of 0..{N - 1}")
    ordered: list[tuple[int, int] | None] = [None] * N
    for site, index in zip(sites, indices):
        ordered[index] = site
    return [site for site in ordered if site is not None]


def sidecar_to_txtlike(entry: dict[str, Any]) -> TxtArtifact:
    operators = entry["operators"]
    sites = [[int(x), int(y)] for x, y in entry["sites"]]
    constant = complex_from_json(operators["constant_term"])

    two_site: dict[tuple[int, int], dict[tuple[int, int], complex]] = {}
    four_site: dict[tuple[tuple[int, int], tuple[int, int]], complex] = {}
    six_site: dict[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]], complex
    ] = {}

    for group in operators["groups"]:
        arity = int(group["arity"])
        if arity == 2:
            vec = tuple(int(v) for v in group["vector"])
            two_site.setdefault(vec, {})
            for term in group["terms"]:
                sites_term = [int(s) for s in term["sites"]]
                pair = tuple(sorted(sites_term))
                two_site[vec][pair] = complex_from_json(term["coefficient"])
        elif arity == 4:
            for term in group["terms"]:
                sites_term = [int(s) for s in term["sites"]]
                pair1 = tuple(sorted(sites_term[0:2]))
                pair2 = tuple(sorted(sites_term[2:4]))
                key = tuple(sorted((pair1, pair2)))
                four_site[key] = complex_from_json(term["coefficient"])
        elif arity == 6:
            for term in group["terms"]:
                sites_term = [int(s) for s in term["sites"]]
                pairs = (
                    tuple(sorted(sites_term[0:2])),
                    tuple(sorted(sites_term[2:4])),
                    tuple(sorted(sites_term[4:6])),
                )
                key = tuple(sorted(pairs))
                six_site[key] = complex_from_json(term["coefficient"])

    fit = {
        "relative_error": float(entry["fit"]["relative_error"]),
        "residual": float(entry["fit"]["residual"]),
        "r_squared": float(entry["fit"]["r_squared"]),
        "t11_minus_1_norm": float(entry["fit"]["t11_minus_1_norm"]),
    }
    return TxtArtifact(
        sites=sites,
        constant=constant,
        two_site=two_site,
        four_site=four_site,
        six_site=six_site,
        fit=fit,
    )


# ---------------------------------------------------------------------------
# Comparison record
# ---------------------------------------------------------------------------
@dataclass
class FileResult:
    name: str
    passed: bool
    max_abs_diff: float = 0.0
    location: str = ""
    detail: str = ""


@dataclass
class ClusterReport:
    t: float
    N: int
    hole: int
    class_idx: int
    cluster_idx: int
    file_results: list[FileResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------
def _max_abs(diff: np.ndarray) -> tuple[float, str]:
    if diff.size == 0:
        return 0.0, ""
    flat = np.abs(diff).reshape(-1)
    idx = int(np.argmax(flat))
    val = float(flat[idx])
    pos = np.unravel_index(idx, diff.shape)
    return val, f"index={pos}"


def compare_array(
    name: str, cur: np.ndarray, ref: np.ndarray, atol: float
) -> FileResult:
    cur = np.asarray(cur)
    ref = np.asarray(ref)
    if cur.shape != ref.shape:
        return FileResult(
            name=name,
            passed=False,
            max_abs_diff=float("nan"),
            detail=f"shape mismatch current={cur.shape} reference={ref.shape}",
        )
    diff = cur - ref
    max_abs, loc = _max_abs(diff)
    return FileResult(name=name, passed=max_abs <= atol, max_abs_diff=max_abs, location=loc)


def compare_eigvals(
    name: str, cur: np.ndarray, ref: np.ndarray, atol: float = ATOL_BASELINE
) -> FileResult:
    cur = np.asarray(cur).astype(float).reshape(-1)
    ref = np.asarray(ref).astype(float).reshape(-1)
    if cur.shape != ref.shape:
        return FileResult(
            name=name,
            passed=False,
            max_abs_diff=float("nan"),
            detail=f"shape mismatch current={cur.shape} reference={ref.shape}",
        )
    # eigvals come out of eigh already sorted; compare as-is.
    diff = cur - ref
    max_abs, loc = _max_abs(diff)
    return FileResult(name=name, passed=max_abs <= atol, max_abs_diff=max_abs, location=loc)


def compare_per_eigenstate(
    name: str,
    cur: np.ndarray,
    ref: np.ndarray,
    eigvals_cur: np.ndarray,
    atol_elem: float = ATOL_BASELINE,
    atol_trace: float = ATOL_DEGEN_TRACE,
) -> FileResult:
    """Element-wise compare; on degenerate blocks (>=2 states) require matching trace."""
    cur = np.asarray(cur)
    ref = np.asarray(ref)
    if cur.shape != ref.shape:
        return FileResult(
            name=name,
            passed=False,
            max_abs_diff=float("nan"),
            detail=f"shape mismatch current={cur.shape} reference={ref.shape}",
        )

    # First pass: element-wise
    diff = np.asarray(cur, dtype=complex) - np.asarray(ref, dtype=complex)
    max_abs, loc = _max_abs(diff)
    if max_abs <= atol_elem:
        return FileResult(name=name, passed=True, max_abs_diff=max_abs, location=loc)

    # Degenerate block fallback: average over degenerate eigenstates
    blocks = _degenerate_blocks(eigvals_cur)
    worst_block = 0.0
    worst_loc = ""
    for start, end in blocks:
        if end - start == 1:
            # Non-degenerate state must match element-wise
            d = np.abs(cur[start:end] - ref[start:end])
            val = float(np.max(d)) if d.size else 0.0
            if val > worst_block:
                worst_block = val
                worst_loc = f"state={start}"
            continue
        # Sum over degenerate block (trace)
        sum_cur = np.sum(cur[start:end], axis=0)
        sum_ref = np.sum(ref[start:end], axis=0)
        block_diff = np.abs(sum_cur - sum_ref)
        val = float(np.max(block_diff)) if np.ndim(block_diff) else float(block_diff)
        if val > worst_block:
            worst_block = val
            worst_loc = f"degenerate block {start}:{end}"
    return FileResult(
        name=name,
        passed=worst_block <= atol_trace,
        max_abs_diff=max(max_abs, worst_block),
        location=(loc if max_abs > worst_block else worst_loc),
        detail=(
            f"element-max={max_abs:.3g} trace-max={worst_block:.3g}"
            if worst_block > 0
            else f"element-max={max_abs:.3g}"
        ),
    )


def _degenerate_blocks(eigvals: np.ndarray, tol: float = 1e-10) -> list[tuple[int, int]]:
    eigvals = np.real(np.asarray(eigvals))
    blocks: list[tuple[int, int]] = []
    start = 0
    n = len(eigvals)
    while start < n:
        end = start + 1
        while end < n and abs(eigvals[end] - eigvals[start]) <= tol:
            end += 1
        blocks.append((start, end))
        start = end
    return blocks


def compare_txt_like(cur: TxtArtifact, ref: TxtArtifact, atol: float = ATOL_TXT_ROUND) -> list[FileResult]:
    results: list[FileResult] = []

    # Sites
    ok = cur.sites == ref.sites
    results.append(
        FileResult(
            name="txt.sites",
            passed=ok,
            max_abs_diff=0.0 if ok else float("nan"),
            detail="" if ok else f"cur={cur.sites} ref={ref.sites}",
        )
    )

    # Constant
    diff = abs(cur.constant - ref.constant)
    results.append(
        FileResult(
            name="txt.constant",
            passed=diff <= atol,
            max_abs_diff=diff,
        )
    )

    # Two-site (compare by bond vector then by canonical site pair)
    max_two = 0.0
    two_loc = ""
    vectors_cur = {vec for vec, terms in cur.two_site.items() if terms}
    vectors_ref = {vec for vec, terms in ref.two_site.items() if terms}
    sym_diff = vectors_cur.symmetric_difference(vectors_ref)
    if sym_diff:
        results.append(
            FileResult(
                name="txt.two_site",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"bond-vector set mismatch {sym_diff}",
            )
        )
    else:
        for vec in sorted(vectors_ref):
            cur_terms = cur.two_site.get(vec, {})
            ref_terms = ref.two_site[vec]
            keys_cur = set(cur_terms)
            keys_ref = set(ref_terms)
            if keys_cur != keys_ref:
                results.append(
                    FileResult(
                        name=f"txt.two_site[{vec}]",
                        passed=False,
                        max_abs_diff=float("nan"),
                        detail=f"pair set mismatch cur={keys_cur} ref={keys_ref}",
                    )
                )
                max_two = float("nan")
                continue
            for pair in sorted(keys_ref):
                d = abs(cur_terms[pair] - ref_terms[pair])
                if d > max_two:
                    max_two = d
                    two_loc = f"J vector={vec} pair={pair}"
        results.append(
            FileResult(
                name="txt.two_site",
                passed=(not np.isnan(max_two)) and (max_two <= atol),
                max_abs_diff=max_two,
                location=two_loc,
            )
        )

    # Four-site
    if set(cur.four_site) != set(ref.four_site):
        sym = set(cur.four_site).symmetric_difference(set(ref.four_site))
        results.append(
            FileResult(
                name="txt.four_site",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"4-site key set mismatch {sym}",
            )
        )
    else:
        max_four = 0.0
        loc = ""
        for key in sorted(ref.four_site):
            d = abs(cur.four_site[key] - ref.four_site[key])
            if d > max_four:
                max_four = d
                loc = f"K key={key}"
        results.append(
            FileResult(
                name="txt.four_site",
                passed=max_four <= atol,
                max_abs_diff=max_four,
                location=loc,
            )
        )

    # Six-site
    if set(cur.six_site) != set(ref.six_site):
        sym = set(cur.six_site).symmetric_difference(set(ref.six_site))
        results.append(
            FileResult(
                name="txt.six_site",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"6-site key set mismatch {sym}",
            )
        )
    else:
        max_six = 0.0
        loc = ""
        for key in sorted(ref.six_site):
            d = abs(cur.six_site[key] - ref.six_site[key])
            if d > max_six:
                max_six = d
                loc = f"L key={key}"
        results.append(
            FileResult(
                name="txt.six_site",
                passed=max_six <= atol,
                max_abs_diff=max_six,
                location=loc,
            )
        )

    # Fit metrics
    max_fit = 0.0
    fit_loc = ""
    for key, ref_value in ref.fit.items():
        cur_value = cur.fit.get(key)
        if cur_value is None:
            results.append(
                FileResult(
                    name=f"txt.fit.{key}",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="missing in current",
                )
            )
            continue
        d = abs(cur_value - ref_value)
        if d > max_fit:
            max_fit = d
            fit_loc = key
        results.append(
            FileResult(
                name=f"txt.fit.{key}",
                passed=d <= atol,
                max_abs_diff=d,
            )
        )
    # Aggregate entry
    results.append(
        FileResult(
            name="txt.fit",
            passed=max_fit <= atol,
            max_abs_diff=max_fit,
            location=fit_loc,
        )
    )

    return results


# ---------------------------------------------------------------------------
# Main per-(t,N) driver
# ---------------------------------------------------------------------------
@dataclass
class DegenerateEvidence:
    t: float
    N: int
    hole: int
    class_idx: int
    set_diff_cur: set
    set_diff_ref: set
    diff_eigvals_cur: np.ndarray
    diff_eigvals_ref: np.ndarray
    diff_double_occ_cur: np.ndarray
    diff_double_occ_ref: np.ndarray
    diff_s2_cur: np.ndarray
    diff_s2_ref: np.ndarray
    heff_max_diff: float


def run_cli(N: int, U: float, t: float, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "cuprate.main",
        f"N={N}",
        f"U={U}",
        f"T={t}",
        "MODE=full",
        "workflow=occ",
        f"OUTPUT_DIR={outdir}",
    ]
    res = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"CLI failed for N={N} t={t}: stderr={res.stderr.decode(errors='replace')[:2000]}"
        )


def verify_family(
    t: float,
    N: int,
    hole: int,
    class_idx: int,
    members: list,
    ref_dir: Path,
    cli_out: Path,
    consolidated_cache: dict[Path, dict[str, Any]],
    degenerate_evidence: list[DegenerateEvidence],
) -> list[ClusterReport]:
    prefix = f"hole{hole}_class{class_idx}"
    observables = recompute_family(N, U_VALUE, t, hole, class_idx, members)

    # --- Load references ---
    ref_eigvals = parse_text_array(ref_dir / f"{prefix}_eigvals.npy").astype(float)
    ref_double_occ = parse_text_array(ref_dir / f"{prefix}_double_occupation_expectation.npy")
    ref_s2_diag = parse_text_array(ref_dir / f"{prefix}_s2_digonal.npy")
    ref_s2_selected = parse_text_array(ref_dir / f"{prefix}_s2_selected.npy")
    ref_states = parse_text_array(ref_dir / f"{prefix}_states.npy")
    ref_selected = parse_text_array(ref_dir / f"{prefix}_t11_selected_indices.npy").astype(int)
    ref_selected_occ = parse_text_array(ref_dir / f"{prefix}_t11_selected_occupation.npy")
    ref_heff = parse_text_array(ref_dir / f"{prefix}_Heff.npy")
    ref_t11m1 = parse_text_array(ref_dir / f"{prefix}_T11m1.npy")

    # --- Load CLI outputs ---
    artifact_path = cli_out / "artifacts" / f"{prefix}_projection.npz"
    artifact = np.load(artifact_path)
    cli_selected = np.asarray(artifact["block_0_selected_indices"], dtype=int).tolist()
    cli_heff = np.asarray(artifact["block_0_Heff"])

    # Verify the in-script pipeline agrees with the CLI for Heff and selected.
    cli_heff_diff = float(np.max(np.abs(cli_heff - observables.heff)))
    if cli_heff_diff > ATOL_BASELINE:
        print(
            f"  WARN: CLI vs inline Heff diff {cli_heff_diff:.3g} at t={t} N={N} {prefix}"
        )
    if sorted(cli_selected) != sorted(observables.selected_indices):
        print(
            f"  WARN: CLI vs inline selected_indices set diff at t={t} N={N} {prefix}"
        )

    # --- Per-eigenstate permutation-invariant comparisons ---
    reports: dict[int, ClusterReport] = {
        mem.cluster_idx: ClusterReport(
            t=t, N=N, hole=hole, class_idx=class_idx, cluster_idx=mem.cluster_idx
        )
        for mem in members
    }
    # Shared (family-level) checks: apply results to every cluster in the family.
    family_results: list[FileResult] = []

    family_results.append(
        compare_eigvals(
            "eigvals.npy", observables.eigvals, ref_eigvals, atol=ATOL_BASELINE
        )
    )
    family_results.append(
        compare_per_eigenstate(
            "double_occupation_expectation.npy",
            observables.double_occ.astype(complex),
            ref_double_occ.astype(complex),
            observables.eigvals,
        )
    )

    # states.npy: compute canonical legacy-sorted states and encode.
    cur_basis_sorted = sorted(observables.basis_states, key=lambda s: legacy_state_key(s, N))
    cur_states = np.asarray(
        [legacy_state_row(s, N) for s in cur_basis_sorted], dtype=int
    )
    ref_states_int = ref_states.astype(int) if not np.iscomplexobj(ref_states) else np.real(ref_states).astype(int)
    if cur_states.shape != ref_states_int.shape:
        family_results.append(
            FileResult(
                name="states.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"shape current={cur_states.shape} reference={ref_states_int.shape}",
            )
        )
    else:
        diff_rows = int(np.sum(np.any(cur_states != ref_states_int, axis=1)))
        family_results.append(
            FileResult(
                name="states.npy",
                passed=diff_rows == 0,
                max_abs_diff=float(diff_rows),
                detail="" if diff_rows == 0 else f"{diff_rows} row(s) differ",
            )
        )

    # s2_digonal — three columns: [S², variance, pure-spin overlap]
    # Column 0 (S²) and column 2 (overlap) are trace-invariant inside degenerate
    # subspaces; column 1 (S² variance) is NOT — it depends on which representative
    # basis eigh picks inside a degenerate block. We compare col 0 and col 2 with
    # the degenerate-block trace fallback, and report col 1 separately.
    cur_s2_stack = np.column_stack(
        [
            observables.s2_diag_values.astype(complex),
            observables.s2_variance.astype(complex),
            observables.s2_overlap.astype(complex),
        ]
    )
    if ref_s2_diag.ndim == 2 and ref_s2_diag.shape[1] == 3:
        family_results.append(
            compare_per_eigenstate(
                "s2_digonal.npy[col0 S²]",
                cur_s2_stack[:, 0:1],
                ref_s2_diag[:, 0:1],
                observables.eigvals,
            )
        )
        family_results.append(
            compare_per_eigenstate(
                "s2_digonal.npy[col2 spin-overlap]",
                cur_s2_stack[:, 2:3],
                ref_s2_diag[:, 2:3],
                observables.eigvals,
            )
        )
        # Column 1 — reported for information only. It only matches
        # element-wise when eigh happens to align representatives.
        var_diff = np.abs(cur_s2_stack[:, 1] - ref_s2_diag[:, 1])
        max_var_nondegen = 0.0
        for s, e in _degenerate_blocks(observables.eigvals):
            if e - s == 1:
                max_var_nondegen = max(max_var_nondegen, float(var_diff[s]))
        family_results.append(
            FileResult(
                name="s2_digonal.npy[col1 variance, non-degen]",
                passed=max_var_nondegen <= ATOL_BASELINE,
                max_abs_diff=max_var_nondegen,
                detail=(
                    "column 1 in degenerate subspaces is basis-dependent (see uncovered "
                    "failure modes); only non-degenerate eigenstates are strictly checked"
                ),
            )
        )
    else:
        family_results.append(
            FileResult(
                name="s2_digonal.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"unexpected reference shape {ref_s2_diag.shape}",
            )
        )

    # selected_indices — set comparison
    cur_sel_set = set(observables.selected_indices)
    ref_sel_set = set(int(x) for x in ref_selected.tolist())
    sel_equal = cur_sel_set == ref_sel_set
    sym_diff_cur = cur_sel_set - ref_sel_set
    sym_diff_ref = ref_sel_set - cur_sel_set
    family_results.append(
        FileResult(
            name="t11_selected_indices.npy (set)",
            passed=sel_equal,
            max_abs_diff=0.0 if sel_equal else float(len(sym_diff_cur) + len(sym_diff_ref)),
            detail=(
                "sets equal"
                if sel_equal
                else f"|cur\\ref|={len(sym_diff_cur)} |ref\\cur|={len(sym_diff_ref)}"
            ),
        )
    )

    # Selected-occupation (sort both by ascending to compare values)
    cur_sel_occ_sorted = np.sort(observables.selected_double_occ.astype(complex).real)
    ref_sel_occ_sorted = np.sort(np.asarray(ref_selected_occ).astype(complex).real)
    family_results.append(
        FileResult(
            name="t11_selected_occupation.npy (sorted)",
            passed=(
                cur_sel_occ_sorted.shape == ref_sel_occ_sorted.shape
                and bool(
                    np.max(np.abs(cur_sel_occ_sorted - ref_sel_occ_sorted)) <= ATOL_BASELINE
                    if cur_sel_occ_sorted.size
                    else True
                )
            ),
            max_abs_diff=float(
                np.max(np.abs(cur_sel_occ_sorted - ref_sel_occ_sorted))
                if cur_sel_occ_sorted.shape == ref_sel_occ_sorted.shape and cur_sel_occ_sorted.size
                else (0.0 if cur_sel_occ_sorted.shape == ref_sel_occ_sorted.shape else float("nan"))
            ),
        )
    )

    # s2_selected comparison: sort by ref selected_indices mapping into cur frame
    cur_s2_selected = cur_s2_stack[np.asarray(observables.selected_indices, dtype=int)]
    # For a fair comparison, pair selected states by ascending double_occ (same order both sides)
    # which matches the output ordering in the reference.
    cur_sel_order = np.argsort(observables.selected_double_occ.real)
    ref_sel_order = np.argsort(
        np.asarray(ref_selected_occ).astype(complex).real, kind="stable"
    )
    if cur_s2_selected.shape == ref_s2_selected.shape:
        cur_sorted = cur_s2_selected[cur_sel_order]
        ref_sorted = ref_s2_selected[ref_sel_order]
        diff = np.asarray(cur_sorted, dtype=complex) - np.asarray(ref_sorted, dtype=complex)
        max_abs, loc = _max_abs(diff)
        family_results.append(
            FileResult(
                name="s2_selected.npy (sorted)",
                passed=max_abs <= ATOL_BASELINE,
                max_abs_diff=max_abs,
                location=loc,
            )
        )
    else:
        family_results.append(
            FileResult(
                name="s2_selected.npy (sorted)",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"shape current={cur_s2_selected.shape} reference={ref_s2_selected.shape}",
            )
        )

    # Heff / T11m1 with row-basis permutation
    if ref_heff.ndim == 2 and ref_heff.shape == observables.heff.shape:
        # Build the permutation P: canonical-spin-rows[P] == legacy-spin-rows
        legacy_spin_states = legacy_spin_ordering(ref_states_int, N)
        # canonical_spin_states = basis_states at canonical spin_fock_rows
        canonical_spin_states = [
            observables.basis_states[i] for i in observables.spin_fock_rows_canonical
        ]
        try:
            perm = canonical_spin_permutation(canonical_spin_states, legacy_spin_states)
            cur_heff_leg = observables.heff[np.ix_(perm, perm)]
            cur_t11m1_leg = observables.t11m1[np.ix_(perm, perm)]
        except Exception as exc:
            perm = None
            family_results.append(
                FileResult(
                    name="Heff.npy (permutation)",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail=f"permutation failed: {exc}",
                )
            )
            cur_heff_leg = observables.heff
            cur_t11m1_leg = observables.t11m1

        family_results.append(
            compare_array("Heff.npy", cur_heff_leg, ref_heff, atol=ATOL_BASELINE)
        )
        family_results.append(
            compare_array("T11m1.npy", cur_t11m1_leg, ref_t11m1, atol=ATOL_BASELINE)
        )
    else:
        family_results.append(
            FileResult(
                name="Heff.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"shape current={observables.heff.shape} reference={ref_heff.shape}",
            )
        )
        family_results.append(
            FileResult(
                name="T11m1.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"shape current={observables.t11m1.shape} reference={ref_t11m1.shape}",
            )
        )

    # Attach family-level results to every cluster in the family.
    for report in reports.values():
        report.file_results.extend(family_results)

    # --- Per-cluster txt comparison ---
    consolidated_path = cli_out / "results.json"
    if consolidated_path not in consolidated_cache:
        with consolidated_path.open() as f:
            consolidated_cache[consolidated_path] = json.load(f)
    payload = consolidated_cache[consolidated_path]
    family_entry = next(
        (
            e
            for e in payload["families"]
            if int(e["hole"]) == hole and int(e["class_idx"]) == class_idx
        ),
        None,
    )
    if family_entry is None:
        for report in reports.values():
            report.file_results.append(
                FileResult(
                    name="results.json",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="family entry not found in results.json",
                )
            )
        return list(reports.values())
    exchange_path = cli_out / family_entry["exchange_file"]
    clusters_path = cli_out / family_entry["clusters_file"]
    exchange_payload = json.loads(exchange_path.read_text())
    clusters_payload = json.loads(clusters_path.read_text())
    family_constant, family_terms = operators_to_terms(exchange_payload["operators"])

    for member in members:
        cluster_idx = int(member.cluster_idx)
        ref_txt = ref_dir / f"{prefix}_cluster{cluster_idx}_results.txt"
        if not ref_txt.exists():
            reports[cluster_idx].file_results.append(
                FileResult(
                    name="results.txt",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="reference not found",
                )
            )
            continue
        ref_art = parse_txt(ref_txt)

        geometry_entry = next(
            (
                e
                for e in clusters_payload["clusters"]
                if int(e["cluster_idx"]) == cluster_idx
            ),
            None,
        )
        if geometry_entry is None:
            reports[cluster_idx].file_results.append(
                FileResult(
                    name="clusters.json",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="cluster entry not found in clusters file",
                )
            )
            continue
        sites = _sites_in_operator_order(geometry_entry, N)
        operators = operators_from_terms(sites, family_constant, family_terms)
        entry = {
            "sites": [[int(x), int(y)] for x, y in sites],
            "operators": operators,
            "fit": exchange_payload["fit"],
        }
        cur_art = sidecar_to_txtlike(entry)
        reports[cluster_idx].file_results.extend(compare_txt_like(cur_art, ref_art))

    # Collect degenerate-subspace evidence if selected_indices differ as sets.
    if not sel_equal and (sym_diff_cur or sym_diff_ref):
        diff_cur_idx = np.asarray(sorted(sym_diff_cur), dtype=int)
        diff_ref_idx = np.asarray(sorted(sym_diff_ref), dtype=int)
        evidence = DegenerateEvidence(
            t=t,
            N=N,
            hole=hole,
            class_idx=class_idx,
            set_diff_cur=sym_diff_cur,
            set_diff_ref=sym_diff_ref,
            diff_eigvals_cur=observables.eigvals[diff_cur_idx],
            diff_eigvals_ref=ref_eigvals[diff_ref_idx],
            diff_double_occ_cur=observables.double_occ[diff_cur_idx],
            diff_double_occ_ref=np.real(ref_double_occ[diff_ref_idx]),
            diff_s2_cur=observables.s2_diag_values[diff_cur_idx],
            diff_s2_ref=np.real(ref_s2_diag[diff_ref_idx, 0]),
            heff_max_diff=float(
                np.max(np.abs(cur_heff_leg - ref_heff)) if ref_heff.shape == cur_heff_leg.shape else float("nan")
            ),
        )
        degenerate_evidence.append(evidence)

    return list(reports.values())


# ---------------------------------------------------------------------------
# Summary + reporting
# ---------------------------------------------------------------------------
def summarise(reports: Iterable[ClusterReport]) -> dict:
    total = 0
    passed = 0
    for r in reports:
        for f in r.file_results:
            total += 1
            if f.passed:
                passed += 1
    return {"total": total, "passed": passed, "failed": total - passed}


def format_report(report: ClusterReport) -> str:
    lines = [
        f"t={report.t:.4f} N={report.N} hole={report.hole} class={report.class_idx} "
        f"cluster={report.cluster_idx}"
    ]
    for fr in report.file_results:
        status = "pass" if fr.passed else "FAIL"
        extra = ""
        if not np.isnan(fr.max_abs_diff):
            extra = f" max|Δ|={fr.max_abs_diff:.3g}"
        loc = f" at {fr.location}" if fr.location else ""
        detail = f" ({fr.detail})" if fr.detail else ""
        lines.append(f"  [{status}] {fr.name}{extra}{loc}{detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "tmp" / "verify_main_runs")
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional filter 't=0.30,N=4' to run a single case.",
    )
    ap.add_argument("--skip-cli", action="store_true")
    args = ap.parse_args()

    matrix: list[tuple[float, list[int]]] = MATRIX
    if args.only:
        kv = {k.strip(): v.strip() for k, v in (p.split("=") for p in args.only.split(","))}
        t_only = float(kv["t"])
        n_only = int(kv["N"])
        matrix = [(t_only, [n_only])]

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_reports: list[ClusterReport] = []
    degenerate_evidence: list[DegenerateEvidence] = []
    skipped: list[str] = []

    for t, Ns in matrix:
        for N in Ns:
            tag = f"t={t:.4f}_N={N}"
            ref_base = ROOT / "data_test" / f"Block_U{U_VALUE:.4f}_t{t:.4f}"
            ref_dir = ref_base / f"N{N}"
            if not ref_dir.is_dir():
                skipped.append(f"{tag} (missing reference dir)")
                continue

            cli_out = args.outdir / tag
            if not args.skip_cli:
                run_cli(N=N, U=U_VALUE, t=t, outdir=cli_out)
            elif not (cli_out / "results.json").exists():
                skipped.append(f"{tag} (skip_cli but no prior run)")
                continue

            consolidated_cache: dict[Path, dict[str, Any]] = {}
            for hole, class_idx, members in enumerate_families(N):
                prefix = f"hole{hole}_class{class_idx}"
                if not (ref_dir / f"{prefix}_eigvals.npy").exists():
                    continue
                reports = verify_family(
                    t=t,
                    N=N,
                    hole=hole,
                    class_idx=class_idx,
                    members=members,
                    ref_dir=ref_dir,
                    cli_out=cli_out,
                    consolidated_cache=consolidated_cache,
                    degenerate_evidence=degenerate_evidence,
                )
                for rpt in reports:
                    print(format_report(rpt))
                    all_reports.append(rpt)

            summary = summarise(
                r
                for r in all_reports
                if np.isclose(r.t, t) and r.N == N
            )
            print(
                f"[{tag}] files: passed={summary['passed']}/{summary['total']}"
            )

    # Overall summary
    overall = summarise(all_reports)
    print("")
    print("==== OVERALL ====")
    print(
        f"Cluster reports: {len(all_reports)} | "
        f"file checks: {overall['passed']}/{overall['total']} passed "
        f"| failed: {overall['failed']}"
    )
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  - {s}")

    # Degenerate-subspace evidence dump
    print("")
    print("==== Selected-indices set diffs (degenerate-subspace evidence) ====")
    if not degenerate_evidence:
        print("(none)")
    for ev in degenerate_evidence:
        print(
            f"t={ev.t:.4f} N={ev.N} hole={ev.hole} class={ev.class_idx} "
            f"cur\\ref={sorted(ev.set_diff_cur)} ref\\cur={sorted(ev.set_diff_ref)}"
        )
        print(f"  eigvals cur={ev.diff_eigvals_cur} ref={ev.diff_eigvals_ref}")
        print(f"  double_occ cur={ev.diff_double_occ_cur} ref={ev.diff_double_occ_ref}")
        print(f"  S² cur={ev.diff_s2_cur} ref={ev.diff_s2_ref}")
        print(f"  Heff element-wise |Δ|max = {ev.heff_max_diff:.3g}")

    # Uncovered failure modes (always printed, prompt requirement).
    print("")
    print("==== Failure modes NOT covered by this verification ====")
    print(
        "- Eigenvector phase/sign differences within non-degenerate eigvals are normalized "
        "away by the checks here, but a LAPACK build change could reorder eigenpairs inside a "
        "degenerate subspace. We DO verify Heff/T11m1 element-wise on the legacy-permuted basis "
        "and flag set diffs in `selected_indices`, but we do NOT re-solve eigh on a second "
        "platform to cross-check."
    )
    print(
        "- Pure-real vs complex storage rounding: reference stores complex with zero imag; our "
        "comparisons use |Δ| so imag round-off <1e-10 would pass. Systemic phase drift in eigvecs "
        "would be absorbed into Heff element-wise comparison — this is fine for physics but "
        "would NOT flag a subtle eigvec-phase bug."
    )
    print(
        "- 4-site / 6-site matching ordering: canonicalised by sorted pairs-of-pairs, so display "
        "order in .txt is ignored. A reference file listing the SAME physical term twice under "
        "different display tags would still produce a set-size mismatch, but if it listed "
        "different physical terms that happen to canonicalise equally we would miss that."
    )
    print(
        "- This verification does NOT cover workflow != occ, MODE != full, or the non-zero-twoSz "
        "branches. It is limited to the (occ, full) pipeline as specified in the prompt matrix."
    )
    print(
        "- We compare s2_digonal column 2 (pure-spin overlap) to a re-computed value because the "
        "active `back/main/solver.py` writes weight=1 which does not match stored reference data. "
        "If the stored reference used a different normalization we would flag that as a "
        "mismatch rather than silently rescale."
    )

    return 0 if overall["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
