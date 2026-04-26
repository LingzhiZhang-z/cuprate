"""Shared KEY=VALUE parsing for runtime entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuprate.paths import ModeSpec, mode_spec


COMMON_KEYS = {
    "root": "ROOT",
    "n": "N",
    "u": "U",
    "t": "T",
    "mode": "MODE",
    "twosz": "twoSz",
    "twos": "twoS",
    "scope": "SCOPE",
    "workflow": "workflow",
}
REQUIRED_COMMON_KEYS = {"N", "U", "T"}
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi", "adiabatic"}


@dataclass(frozen=True)
class RuntimeArgs:
    root: Path
    N: int
    U: float
    t: float
    mode: str
    twoSz: int | None
    twoS: int | None
    scope: str
    workflow: str


def parse_key_values(argv: list[str], canonical_keys: dict[str, str]) -> dict[str, str]:
    raw: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"expected KEY=VALUE argument, got {arg!r}")
        key, value = arg.split("=", 1)
        if not key:
            raise ValueError(f"empty key in argument {arg!r}")
        canonical = canonical_keys.get(key.lower())
        if canonical is None:
            raise ValueError(f"unknown parameter {key!r}")
        if canonical in raw:
            raise ValueError(f"duplicate parameter {canonical}")
        raw[canonical] = value
    return raw


def parse_common_runtime(raw: dict[str, str]) -> RuntimeArgs:
    missing = sorted(REQUIRED_COMMON_KEYS - raw.keys())
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}")

    N = parse_int(raw["N"], "N")
    U = parse_float(raw["U"], "U")
    t = parse_float(raw["T"], "T")
    twoSz = parse_optional_int(raw, "twoSz")
    twoS = parse_optional_int(raw, "twoS")
    spec = mode_spec(raw.get("MODE", "full"), twoSz=twoSz, twoS=twoS, scope=raw.get("SCOPE"))
    _validate_mode_args(N, spec)

    workflow = raw.get("workflow", "occ").lower()
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"unsupported workflow={workflow!r}")

    return RuntimeArgs(
        root=Path(raw.get("ROOT", "results")),
        N=N,
        U=U,
        t=t,
        mode=spec.mode,
        twoSz=spec.twoSz,
        twoS=spec.twoS,
        scope=spec.scope,
        workflow=workflow,
    )


def parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float") from exc


def parse_optional_int(raw: dict[str, str], key: str) -> int | None:
    if key not in raw:
        return None
    return parse_int(raw[key], key)


def _validate_mode_args(N: int, spec: ModeSpec) -> None:
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
