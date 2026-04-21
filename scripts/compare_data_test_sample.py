#!/usr/bin/env python3
"""Randomly sample regression cases from data_test and compare against current outputs."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


ATOL_TIGHT = 1e-10
ATOL_LOOSE = 1e-6
ATOL_DERIVED = 1e-9

PREFIX_RE = re.compile(r"^(hole\d+_class\d+)_Heff\.npy$")
VARIANT_RE = re.compile(r"^(hole\d+_class\d+)_cluster(\d+)_results\.txt$")
POINT_RE = re.compile(r"^Point\s+(\d+):\s+\((-?\d+),\s*(-?\d+)\)$")
VECTOR_RE = re.compile(r"^Bond vector \((-?\d+),\s*(-?\d+)\),\s+J\d+:(?:\s+\(not present in cluster\))?$")
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


@dataclass(frozen=True)
class ReferenceCase:
    ref_base_dir: Path
    ref_run_dir: Path
    ref_data_dir: Path
    current_base_dir: Path
    current_run_dir: Path
    current_data_dir: Path
    prefix: str
    variant_indices: tuple[int, ...]
    workflow: str | None

    @property
    def label(self) -> str:
        return f"{self.current_base_dir.name}/{self.current_run_dir.name}/{self.prefix}"


@dataclass
class ReferenceArtifact:
    sites: list[list[int]]
    constant: complex
    two_site: dict[tuple[int, int], list[list[Any]]]
    four_site: list[list[Any]]
    six_site: list[list[Any]]
    fit: dict[str, float]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Randomly sample current results and compare them against data_test.",
    )
    parser.add_argument("--reference-root", type=Path, default=root / "data_test")
    parser.add_argument("--current-root", type=Path, default=root / "Block")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sample-scope",
        choices=("available", "reference"),
        default="available",
        help="available: only sample cases with current outputs present; reference: sample all reference cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.reference_root.exists():
        print(f"Reference root not found: {args.reference_root}", file=sys.stderr)
        return 2

    all_cases = collect_reference_cases(args.reference_root, args.current_root)
    if not all_cases:
        print("No reference cases found under data_test.", file=sys.stderr)
        return 2

    if args.sample_scope == "available":
        population = [case for case in all_cases if case_is_available(case)]
    else:
        population = all_cases

    if not population:
        print(
            "No comparable cases found. Current output directories are missing for all reference cases.",
            file=sys.stderr,
        )
        return 2

    sample_size = min(args.samples, len(population))
    selected = rng.sample(population, sample_size)

    print(
        f"Sampling {sample_size} cases from {len(population)} eligible cases "
        f"(total reference cases: {len(all_cases)})."
    )
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    consolidated_cache: dict[Path, dict[str, Any]] = {}
    failures: list[tuple[str, list[str]]] = []
    passes = 0

    for case in selected:
        errors = compare_case(case, consolidated_cache)
        if errors:
            failures.append((case.label, errors))
            print(f"FAIL {case.label}")
            for error in errors:
                print(f"  - {error}")
        else:
            passes += 1
            print(f"PASS {case.label}")

    print("")
    print(f"Summary: {passes} passed, {len(failures)} failed.")
    return 1 if failures else 0


def collect_reference_cases(reference_root: Path, current_root: Path) -> list[ReferenceCase]:
    cases: list[ReferenceCase] = []
    for ref_base_dir in sorted(reference_root.glob("Block_*")):
        if not ref_base_dir.is_dir():
            continue
        current_base_dir = current_root / ref_base_dir.name.removeprefix("Block_")
        for ref_run_dir in sorted(ref_base_dir.iterdir()):
            if not ref_run_dir.is_dir():
                continue
            run_info = old_run_dir_to_current(ref_run_dir.name)
            prefix_to_variants: dict[str, list[int]] = {}
            for txt_path in ref_run_dir.glob("hole*_class*_cluster*_results.txt"):
                match = VARIANT_RE.match(txt_path.name)
                if match is None:
                    continue
                prefix_to_variants.setdefault(match.group(1), []).append(int(match.group(2)))

            for heff_path in ref_run_dir.glob("hole*_class*_Heff.npy"):
                match = PREFIX_RE.match(heff_path.name)
                if match is None:
                    continue
                prefix = match.group(1)
                variant_indices = tuple(sorted(prefix_to_variants.get(prefix, [])))
                cases.append(
                    ReferenceCase(
                        ref_base_dir=ref_base_dir,
                        ref_run_dir=ref_run_dir,
                        ref_data_dir=ref_base_dir / run_info["ref_data_dir"],
                        current_base_dir=current_base_dir,
                        current_run_dir=current_base_dir / run_info["run_dir"],
                        current_data_dir=current_base_dir / run_info["data_dir"],
                        prefix=prefix,
                        variant_indices=variant_indices,
                        workflow=run_info["workflow"],
                    )
                )
    return cases


def old_run_dir_to_current(old_run_dir: str) -> dict[str, str | None]:
    parts = old_run_dir.split("_")
    if not parts or not parts[0].startswith("N"):
        raise ValueError(f"Unsupported old run directory name: {old_run_dir}")

    n = int(parts[0][1:])
    workflow = None
    sz = None
    has_restart = False

    for token in parts[1:]:
        if token.startswith("sz"):
            sz = token[2:]
        elif token in {"multi", "adiabatic"}:
            workflow = token
        elif token == "restart":
            has_restart = True
        elif token == "":
            continue
        else:
            raise ValueError(f"Unsupported old run directory name: {old_run_dir}")

    data_dir = f"N{n}"
    if sz is not None:
        twoSz = int(round(2 * float(sz)))
        data_dir += f"_twoSz_{format_signed_label(twoSz)}"

    suffix = ""
    if workflow is not None:
        suffix += f"_{workflow}"
    if has_restart:
        suffix += "_restart"

    ref_data_dir = parts[0]
    if sz is not None:
        ref_data_dir += f"_sz{float(sz):.4f}"

    return {
        "ref_data_dir": ref_data_dir,
        "data_dir": data_dir,
        "run_dir": f"{data_dir}{suffix}",
        "workflow": workflow,
    }


def case_is_available(case: ReferenceCase) -> bool:
    if not case.current_run_dir.exists() or not case.current_data_dir.exists():
        return False
    required = [
        case.current_run_dir / f"{case.prefix}_states.npy",
        case.current_run_dir / f"{case.prefix}_double_occupation_expectation.npy",
        case.current_run_dir / f"{case.prefix}_S2_diagonal.npy",
        case.current_run_dir / f"{case.prefix}_S2_selected.npy",
        case.current_run_dir / f"{case.prefix}_t11_selected_indices.npy",
        case.current_run_dir / f"{case.prefix}_Heff.npy",
        case.current_run_dir / f"{case.prefix}_T11m1.npy",
        case.current_run_dir / f"{case.prefix}_t11_selected_occupation.npy",
        case.current_data_dir / f"{case.prefix}_eigvals.npy",
    ]
    if any(not path.exists() for path in required):
        return False
    return any(
        (case.current_run_dir / f"{case.prefix}_cluster{variant_idx}_results.json").exists()
        or (case.current_run_dir / "results.json").exists()
        for variant_idx in case.variant_indices
    )


def format_signed_label(value: int) -> str:
    return f"n{abs(value)}" if value < 0 else str(value)


def compare_case(case: ReferenceCase, consolidated_cache: dict[Path, dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    if not case.current_run_dir.exists():
        return [f"missing current run directory: {case.current_run_dir}"]
    if not case.current_data_dir.exists():
        return [f"missing current data directory: {case.current_data_dir}"]

    selected_match = compare_arrays(case, errors)

    for variant_idx in case.variant_indices:
        compare_variant_artifact(case, variant_idx, selected_match, consolidated_cache, errors)

    return errors


def compare_arrays(case: ReferenceCase, errors: list[str]) -> bool:
    deterministic = case.workflow is None

    ref_eigvals = load_npy(case.ref_run_dir / f"{case.prefix}_eigvals.npy")
    if ref_eigvals is None:
        ref_eigvals = load_npy(case.ref_data_dir / f"{case.prefix}_eigvals.npy")

    cur_eigvals = load_npy(case.current_data_dir / f"{case.prefix}_eigvals.npy")
    if cur_eigvals is None:
        errors.append("missing current eigvals.npy")
    elif ref_eigvals is not None:
        add_allclose_error(
            errors,
            np.sort(cur_eigvals),
            np.sort(ref_eigvals),
            ATOL_TIGHT,
            "eigvals mismatch",
        )

    ref_states = load_npy(case.ref_run_dir / f"{case.prefix}_states.npy")
    cur_states = load_npy(case.current_run_dir / f"{case.prefix}_states.npy")
    if cur_states is None:
        errors.append("missing current states.npy")
    elif canonical_state_rows(cur_states) != canonical_state_rows(ref_states):
        errors.append("states mismatch")

    ref_do = load_npy(case.ref_run_dir / f"{case.prefix}_double_occupation_expectation.npy")
    cur_do = load_npy(case.current_run_dir / f"{case.prefix}_double_occupation_expectation.npy")
    if cur_do is None:
        errors.append("missing current double_occupation_expectation.npy")
    else:
        add_allclose_error(errors, cur_do, ref_do, ATOL_DERIVED, "double_occupation_expectation mismatch")

    ref_s2 = load_npy(case.ref_run_dir / f"{case.prefix}_s2_digonal.npy")
    cur_s2 = load_npy(case.current_run_dir / f"{case.prefix}_S2_diagonal.npy")
    if cur_s2 is None:
        errors.append("missing current S2_diagonal.npy")
    else:
        ref_s2 = normalize_s2_columns(ref_s2)
        cur_s2 = normalize_s2_columns(cur_s2)
        compare_s2_diagnostics(
            errors,
            cur_s2,
            ref_s2,
            cur_eigvals,
            ref_eigvals,
            "S2_diagonal mismatch",
        )

    ref_s2_sel = load_npy(case.ref_run_dir / f"{case.prefix}_s2_selected.npy")
    cur_s2_sel = load_npy(case.current_run_dir / f"{case.prefix}_S2_selected.npy")
    if cur_s2_sel is None:
        errors.append("missing current S2_selected.npy")
    else:
        ref_s2_sel = normalize_s2_columns(ref_s2_sel)
        cur_s2_sel = normalize_s2_columns(cur_s2_sel)
        add_allclose_error(errors, cur_s2_sel, ref_s2_sel, ATOL_LOOSE, "S2_selected mismatch")

    ref_indices = load_npy(case.ref_run_dir / f"{case.prefix}_t11_selected_indices.npy")
    cur_indices = load_npy(case.current_run_dir / f"{case.prefix}_t11_selected_indices.npy")
    if cur_indices is None:
        errors.append("missing current t11_selected_indices.npy")
        return False

    ref_indices = np.atleast_1d(ref_indices).astype(int)
    cur_indices = np.atleast_1d(cur_indices).astype(int)
    selected_match = np.array_equal(np.sort(cur_indices), np.sort(ref_indices))

    if deterministic:
        if not selected_match:
            errors.append(
                f"selected indices mismatch: current={sorted(cur_indices.tolist())}, "
                f"reference={sorted(ref_indices.tolist())}"
            )
            return False
    else:
        ref_t11 = load_npy(case.ref_run_dir / f"{case.prefix}_T11m1.npy")
        cur_t11 = load_npy(case.current_run_dir / f"{case.prefix}_T11m1.npy")
        if cur_t11 is None:
            errors.append("missing current T11m1.npy")
            return False
        cur_t11_norm = float(np.linalg.norm(np.asarray(cur_t11).ravel()))
        ref_t11_norm = float(np.linalg.norm(np.asarray(ref_t11).ravel()))
        if cur_t11_norm > ref_t11_norm + ATOL_LOOSE:
            errors.append(
                f"T11 norm regression: current={cur_t11_norm:.10g}, reference={ref_t11_norm:.10g}"
            )
        if not selected_match:
            return False

    ref_heff = load_npy(case.ref_run_dir / f"{case.prefix}_Heff.npy")
    cur_heff = load_npy(case.current_run_dir / f"{case.prefix}_Heff.npy")
    if cur_heff is None:
        errors.append("missing current Heff.npy")
    else:
        add_allclose_error(errors, cur_heff, ref_heff, ATOL_LOOSE, "Heff mismatch")

    ref_t11 = load_npy(case.ref_run_dir / f"{case.prefix}_T11m1.npy")
    cur_t11 = load_npy(case.current_run_dir / f"{case.prefix}_T11m1.npy")
    if cur_t11 is None:
        errors.append("missing current T11m1.npy")
    else:
        add_allclose_error(errors, cur_t11, ref_t11, ATOL_LOOSE, "T11m1 mismatch")

    ref_sel_occ = load_npy(case.ref_run_dir / f"{case.prefix}_t11_selected_occupation.npy")
    cur_sel_occ = load_npy(case.current_run_dir / f"{case.prefix}_t11_selected_occupation.npy")
    if cur_sel_occ is None:
        errors.append("missing current t11_selected_occupation.npy")
    else:
        add_allclose_error(
            errors,
            cur_sel_occ,
            ref_sel_occ,
            ATOL_DERIVED,
            "selected occupation mismatch",
        )

    return True


def compare_variant_artifact(
    case: ReferenceCase,
    variant_idx: int,
    selected_match: bool,
    consolidated_cache: dict[Path, dict[str, Any]],
    errors: list[str],
) -> None:
    ref_txt = case.ref_run_dir / f"{case.prefix}_cluster{variant_idx}_results.txt"
    if not ref_txt.exists():
        errors.append(f"missing reference text artifact for cluster {variant_idx}")
        return

    ref_artifact = parse_reference_results(ref_txt)
    current_artifact = load_current_artifact(case, variant_idx, consolidated_cache)
    if current_artifact is None:
        errors.append(f"missing current artifact for cluster {variant_idx}")
        return

    current_sites = [[int(x), int(y)] for x, y in current_artifact["sites"]]
    if current_sites != ref_artifact.sites:
        errors.append(f"cluster {variant_idx}: site ordering mismatch")

    if not selected_match:
        return

    current_terms = artifact_to_terms(current_artifact)
    if not complex_close(current_terms["constant"], ref_artifact.constant, ATOL_LOOSE):
        errors.append(f"cluster {variant_idx}: constant term mismatch")

    compare_term_groups(
        current_terms["two_site"],
        ref_artifact.two_site,
        ATOL_LOOSE,
        f"cluster {variant_idx}: two-site terms",
        errors,
    )
    compare_term_list_by_key(
        current_terms["four_site"],
        ref_artifact.four_site,
        ATOL_LOOSE,
        f"cluster {variant_idx}: four-site terms",
        errors,
    )
    compare_term_list_by_key(
        current_terms["six_site"],
        ref_artifact.six_site,
        ATOL_LOOSE,
        f"cluster {variant_idx}: six-site terms",
        errors,
    )

    fit = current_artifact.get("fit")
    if fit is None:
        errors.append(f"cluster {variant_idx}: missing fit block in current artifact")
        return

    for key, ref_value in ref_artifact.fit.items():
        current_value = float(fit[key])
        if abs(current_value - ref_value) > ATOL_LOOSE:
            errors.append(
                f"cluster {variant_idx}: fit metric {key} mismatch "
                f"(current={current_value:.10g}, reference={ref_value:.10g})"
            )


def parse_reference_results(path: Path) -> ReferenceArtifact:
    sites: list[list[int]] = []
    two_site: dict[tuple[int, int], list[list[Any]]] = {}
    four_site: list[list[Any]] = []
    six_site: list[list[Any]] = []
    fit: dict[str, float] = {}
    constant = 0.0 + 0.0j

    section = None
    current_vector: tuple[int, int] | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped == "=== Cluster Points ===":
            section = "cluster_points"
            current_vector = None
            continue
        if stripped == "=== Individual Bond Coefficients ===":
            section = "coefficients"
            current_vector = None
            continue
        if stripped in {"=== Individual Fit Error ===", "=== Fit quality ==="}:
            section = "fit"
            current_vector = None
            continue

        if section == "cluster_points":
            match = POINT_RE.match(stripped)
            if match is not None:
                sites.append([int(match.group(2)), int(match.group(3))])
            continue

        if section == "coefficients":
            if stripped.startswith("Constant term:"):
                constant = complex(float(stripped.split(":", 1)[1]), 0.0)
                continue

            vector_match = VECTOR_RE.match(stripped)
            if vector_match is not None:
                current_vector = (int(vector_match.group(1)), int(vector_match.group(2)))
                two_site.setdefault(current_vector, [])
                continue

            two_site_match = TWO_SITE_RE.match(line)
            if two_site_match is not None and current_vector is not None:
                two_site[current_vector].append(
                    [
                        int(two_site_match.group(1)),
                        int(two_site_match.group(2)),
                        complex(float(two_site_match.group(3)), float(two_site_match.group(4))),
                    ]
                )
                continue

            four_match = FOUR_SITE_RE.match(line)
            if four_match is not None:
                four_site.append(
                    [
                        int(four_match.group(1)),
                        int(four_match.group(2)),
                        int(four_match.group(3)),
                        int(four_match.group(4)),
                        complex(float(four_match.group(5)), float(four_match.group(6))),
                    ]
                )
                continue

            six_match = SIX_SITE_RE.match(line)
            if six_match is not None:
                six_site.append(
                    [
                        int(six_match.group(1)),
                        int(six_match.group(2)),
                        int(six_match.group(3)),
                        int(six_match.group(4)),
                        int(six_match.group(5)),
                        int(six_match.group(6)),
                        complex(float(six_match.group(7)), float(six_match.group(8))),
                    ]
                )
                continue

        if section == "fit":
            if ":" not in stripped:
                continue
            key, value = [part.strip() for part in stripped.split(":", 1)]
            if key == "Relative Error":
                fit["relative_error"] = float(value)
            elif key == "Residual":
                fit["residual"] = float(value)
            elif key in {"R^2", "R²"}:
                fit["r_squared"] = float(value)
            elif key == "T11-1 norm":
                fit["t11_minus_1_norm"] = float(value)

    return ReferenceArtifact(
        sites=sites,
        constant=constant,
        two_site=two_site,
        four_site=four_site,
        six_site=six_site,
        fit=fit,
    )


def load_current_artifact(
    case: ReferenceCase,
    variant_idx: int,
    consolidated_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any] | None:
    sidecar_path = case.current_run_dir / f"{case.prefix}_cluster{variant_idx}_results.json"
    if sidecar_path.exists():
        with sidecar_path.open("r") as f:
            return json.load(f)

    consolidated_path = case.current_run_dir / "results.json"
    if not consolidated_path.exists():
        return None

    payload = consolidated_cache.get(consolidated_path)
    if payload is None:
        with consolidated_path.open("r") as f:
            payload = json.load(f)
        consolidated_cache[consolidated_path] = payload

    prefix_match = re.match(r"^hole(\d+)_class(\d+)$", case.prefix)
    if prefix_match is None:
        return None
    hole = int(prefix_match.group(1))
    class_idx = int(prefix_match.group(2))

    for entry in payload.get("clusters", []):
        if (
            int(entry["hole"]) == hole
            and int(entry["class_idx"]) == class_idx
            and int(entry["cluster_idx"]) == variant_idx
        ):
            return entry
    return None


def artifact_to_terms(entry: dict[str, Any]) -> dict[str, Any]:
    operators = entry["operators"]
    terms = {
        "constant": deserialize_complex(operators["constant_term"]),
        "two_site": {},
        "four_site": [],
        "six_site": [],
    }
    for group in operators["groups"]:
        group_terms = [
            [*map(int, item["sites"]), deserialize_complex(item["coefficient"])]
            for item in group["terms"]
        ]
        arity = int(group["arity"])
        if arity == 2:
            terms["two_site"][tuple(map(int, group["vector"]))] = group_terms
        elif arity == 4:
            terms["four_site"].extend(group_terms)
        elif arity == 6:
            terms["six_site"].extend(group_terms)
    return terms


def deserialize_complex(payload: dict[str, Any]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def compare_term_groups(
    current: dict[tuple[int, int], list[list[Any]]],
    reference: dict[tuple[int, int], list[list[Any]]],
    atol: float,
    label: str,
    errors: list[str],
) -> None:
    current_nonempty = {key: value for key, value in current.items() if value}
    reference_nonempty = {key: value for key, value in reference.items() if value}
    if set(current_nonempty) != set(reference_nonempty):
        errors.append(f"{label} vectors mismatch")
        return
    for vector in sorted(reference_nonempty):
        compare_term_list_by_key(
            current_nonempty[vector],
            reference_nonempty[vector],
            atol,
            f"{label} vector {vector}",
            errors,
        )


def compare_term_list_by_key(
    current: list[list[Any]],
    reference: list[list[Any]],
    atol: float,
    label: str,
    errors: list[str],
) -> None:
    current_map = {canonical_term_key(term): term[-1] for term in current}
    reference_map = {canonical_term_key(term): term[-1] for term in reference}
    if len(current_map) != len(current) or len(reference_map) != len(reference):
        errors.append(f"{label} contains duplicate operator keys")
        return
    if set(current_map) != set(reference_map):
        errors.append(f"{label} site tuple mismatch")
        return
    for key in sorted(reference_map):
        if not complex_close(current_map[key], reference_map[key], atol):
            errors.append(f"{label} coefficient mismatch")
            return


def load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "pickled" not in str(exc):
            return load_text_array(path)
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            return load_text_array(path)


def load_text_array(path: Path) -> np.ndarray:
    text = path.read_text().strip()
    if not text:
        return np.array([])
    for dtype in (float, complex):
        try:
            return np.loadtxt(StringIO(text), dtype=dtype)
        except ValueError:
            continue
    raise ValueError(f"Unsupported text array format: {path}")


def normalize_s2_columns(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, :2]
    return array


def compare_s2_diagnostics(
    errors: list[str],
    current: np.ndarray,
    reference: np.ndarray,
    current_eigvals: np.ndarray | None,
    reference_eigvals: np.ndarray | None,
    label: str,
) -> None:
    if current.shape != reference.shape:
        errors.append(f"{label}: shape mismatch current={current.shape}, reference={reference.shape}")
        return
    if current_eigvals is None or reference_eigvals is None:
        add_allclose_error(errors, current, reference, ATOL_LOOSE, label)
        return

    for start, end in degenerate_blocks(reference_eigvals):
        ref_block = reference[start:end]
        cur_block = current[start:end]
        if end - start == 1:
            if not np.allclose(cur_block, ref_block, atol=ATOL_LOOSE, rtol=0.0):
                diff = float(np.max(np.abs(cur_block - ref_block)))
                errors.append(f"{label}: max abs diff = {diff:.10g}")
                return
            continue

        ref_trace = np.sum(ref_block[:, 0])
        cur_trace = np.sum(cur_block[:, 0])
        if abs(cur_trace - ref_trace) > ATOL_LOOSE:
            errors.append(
                f"{label}: degenerate-block trace mismatch at rows {start}:{end} "
                f"(current={cur_trace:.10g}, reference={ref_trace:.10g})"
            )
            return


def canonical_state_rows(array: np.ndarray) -> list[tuple[int, ...]]:
    rows = np.asarray(array)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return sorted(tuple(int(round(float(x.real))) for x in row) for row in rows)


def canonical_term_key(term: list[Any]) -> tuple[Any, ...]:
    sites = [int(x) for x in term[:-1]]
    if len(sites) == 2:
        return tuple(sorted(sites))
    pairs = [tuple(sorted(sites[i:i + 2])) for i in range(0, len(sites), 2)]
    return tuple(sorted(pairs))


def degenerate_blocks(eigvals: np.ndarray) -> list[tuple[int, int]]:
    eigvals = np.asarray(eigvals)
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < len(eigvals):
        end = start + 1
        while end < len(eigvals) and abs(eigvals[end] - eigvals[start]) <= ATOL_TIGHT:
            end += 1
        blocks.append((start, end))
        start = end
    return blocks


def add_allclose_error(
    errors: list[str],
    current: np.ndarray,
    reference: np.ndarray,
    atol: float,
    label: str,
) -> None:
    if current.shape != reference.shape:
        errors.append(f"{label}: shape mismatch current={current.shape}, reference={reference.shape}")
        return
    if not np.allclose(current, reference, atol=atol, rtol=0.0):
        max_diff = float(np.max(np.abs(current - reference)))
        errors.append(f"{label}: max abs diff = {max_diff:.10g}")


def complex_close(current: complex, reference: complex, atol: float) -> bool:
    return abs(current - reference) <= atol


if __name__ == "__main__":
    raise SystemExit(main())
