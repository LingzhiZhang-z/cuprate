"""CLI entry point for `python -m cuprate.main`."""

from __future__ import annotations

import sys
from pathlib import Path

from cuprate.workchain import WorkchainParams, run_workchain


CANONICAL_KEYS = {
    "n": "N",
    "u": "U",
    "t": "T",
    "mode": "MODE",
    "workflow": "workflow",
    "output_dir": "OUTPUT_DIR",
    "twosz": "twoSz",
    "twos": "twoS",
    "cache_mode": "CACHE_MODE",
    "cache_dir": "CACHE_DIR",
    "ratio": "RATIO",
    "n_trials": "N_TRIALS",
    "max_failures": "MAX_FAILURES",
}

REQUIRED_KEYS = {"N", "U", "T", "MODE", "workflow", "OUTPUT_DIR"}
CANONICAL_MODES = {"full", "fixed_sz", "block_sz_full", "fixed_sz_s2", "block_sz_s2_full"}
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi"}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        params = parse_args(argv)
        run_workchain(params)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def parse_args(argv: list[str]) -> WorkchainParams:
    raw: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"expected KEY=VALUE argument, got {arg!r}")
        key, value = arg.split("=", 1)
        if not key:
            raise ValueError(f"empty key in argument {arg!r}")
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
    mode = raw["MODE"].lower()
    mode = "full" if mode == "all" else mode
    if mode not in CANONICAL_MODES:
        raise ValueError(f"unsupported MODE={raw['MODE']!r}")

    workflow = raw["workflow"].lower()
    if workflow == "adiabatic":
        raise ValueError("workflow='adiabatic' is not supported in this first workchain")
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"unsupported workflow={raw['workflow']!r}")

    twoSz = _parse_optional_int(raw, "twoSz")
    twoS = _parse_optional_int(raw, "twoS")
    _validate_sector_args(N, mode, twoSz, twoS)

    cache_mode = raw.get("CACHE_MODE", "none").lower()
    if cache_mode not in {"none", "load", "save", "partial"}:
        raise ValueError(f"unsupported CACHE_MODE={cache_mode!r}")
    cache_dir = Path(raw["CACHE_DIR"]) if "CACHE_DIR" in raw else None
    if cache_mode == "none" and cache_dir is not None:
        raise ValueError("CACHE_DIR is forbidden when CACHE_MODE=none")
    if cache_mode != "none" and cache_dir is None:
        raise ValueError(f"CACHE_DIR is required when CACHE_MODE={cache_mode}")

    ratio = _parse_optional_int(raw, "RATIO")
    n_trials = _parse_optional_int(raw, "N_TRIALS")
    max_failures = _parse_optional_int(raw, "MAX_FAILURES")
    for key, value in (("RATIO", ratio), ("N_TRIALS", n_trials), ("MAX_FAILURES", max_failures)):
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if workflow not in {"greedy", "greedy_multi"} and ratio is not None:
        raise ValueError("RATIO applies only to workflow=greedy or workflow=greedy_multi")
    if workflow != "greedy_multi" and (n_trials is not None or max_failures is not None):
        raise ValueError("N_TRIALS and MAX_FAILURES apply only to workflow=greedy_multi")

    return WorkchainParams(
        N=N,
        U=U,
        t=t,
        mode=mode,
        workflow=workflow,
        output_dir=Path(raw["OUTPUT_DIR"]),
        twoSz=twoSz,
        twoS=twoS,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        ratio=ratio,
        n_trials=n_trials,
        max_failures=max_failures,
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


def _validate_sector_args(N: int, mode: str, twoSz: int | None, twoS: int | None) -> None:
    needs_twoSz = mode in {"fixed_sz", "fixed_sz_s2"}
    needs_twoS = mode == "fixed_sz_s2"
    if needs_twoSz and twoSz is None:
        raise ValueError(f"MODE={mode} requires twoSz")
    if not needs_twoSz and twoSz is not None:
        raise ValueError(f"MODE={mode} does not accept twoSz")
    if needs_twoS and twoS is None:
        raise ValueError(f"MODE={mode} requires twoS")
    if not needs_twoS and twoS is not None:
        raise ValueError(f"MODE={mode} does not accept twoS")

    if twoSz is not None:
        if abs(twoSz) > N:
            raise ValueError("twoSz must satisfy |twoSz| <= N")
        if twoSz % 2 != N % 2:
            raise ValueError("twoSz parity must match N")
    if twoS is not None:
        if not (0 <= twoS <= N):
            raise ValueError("twoS must satisfy 0 <= twoS <= N")
        if twoS % 2 != N % 2:
            raise ValueError("twoS parity must match N")
        if abs(twoSz) > twoS:
            raise ValueError("twoSz and twoS must satisfy |twoSz| <= twoS")


if __name__ == "__main__":
    raise SystemExit(main())
