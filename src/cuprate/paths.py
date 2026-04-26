"""Canonical runtime path and filename helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STAGE_MAIN = "block_main"
STAGE_LCE = "block_lce"
STAGE_EMBED = "block_embed"

DATA_FULL = "DATA"
DATA_TWOSZ = "DATA_twoSz"
DATA_TWOSZ_TWOS = "DATA_twoSz_twoS"
DATA_TWOSZ_TWOS_ETA0 = "DATA_twoSz_twoS_eta_0"

SCOPE_NONNEGATIVE = "nonnegative"
SCOPE_PM = "pm"
SUPPORTED_SCOPES = {SCOPE_NONNEGATIVE, SCOPE_PM}

RESULTS_FILE = "results.json"
LCE_RESULTS_FILE = "lce_results.json"
LCE_SUMMARY_FILE = "lce_summary.txt"
LCE_WEIGHTS_DIR = "weights"
EMBED_RESULTS_FILE = "embed_results.json"
EMBED_SUMMARY_FILE = "embed_summary.txt"
EMBED_TWO_SITE_FILE = "two_site.txt"
EMBED_CLUSTERS_DIR = "clusters"


def int_token(value: int) -> str:
    value = int(value)
    return f"n{-value}" if value < 0 else str(value)


def parse_int_token(value: str) -> int:
    if value.startswith("n"):
        return -int(value[1:])
    return int(value)


def cluster_token(hole: int, class_idx: int, cluster_idx: int | None = None) -> str:
    label = f"hole{int(hole)}_class{int(class_idx)}"
    if cluster_idx is not None:
        label += f"_idx{int(cluster_idx)}"
    return label


def block_token(
    twoSz: int | None,
    twoS: int | None = None,
    eta: int | None = None,
) -> str:
    if twoSz is None:
        if twoS is not None or eta is not None:
            raise ValueError("twoS/eta requires twoSz in block token")
        return "full"
    if eta is not None and twoS is None:
        raise ValueError("eta requires twoS in block token")
    label = f"twoSz_{int_token(twoSz)}"
    if twoS is not None:
        label += f"_twoS_{int_token(twoS)}"
    if eta is not None:
        label += f"_eta_{int_token(eta)}"
    return label


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    token: str
    twoSz_scope: str
    twoS_scope: str
    twoSz: int | None = None
    twoS: int | None = None
    scope: str = SCOPE_NONNEGATIVE


def parameter_token(N: int, nelec: int, U: float, t: float) -> str:
    return f"N_{int(N)}_nelec_{int(nelec)}_U_{float(U):.4f}_t_{float(t):.4f}"


def canonical_mode(mode: str) -> str:
    key = mode.strip().lower()
    if key == "full":
        return "full"
    if key == "sz":
        return "Sz"
    if key == "szs2":
        return "SzS2"
    if key == "szs2eta2":
        return "SzS2eta2"
    raise ValueError(f"unsupported MODE={mode!r}")


def canonical_scope(scope: str | None = None) -> str:
    value = SCOPE_NONNEGATIVE if scope is None else scope.strip().lower()
    if value not in SUPPORTED_SCOPES:
        raise ValueError(f"unsupported SCOPE={scope!r}")
    return value


def mode_spec(
    mode: str,
    twoSz: int | None = None,
    twoS: int | None = None,
    scope: str | None = None,
) -> ModeSpec:
    mode = canonical_mode(mode)
    scope = canonical_scope(scope)
    if mode == "full":
        if twoSz is not None or twoS is not None:
            raise ValueError("MODE=full does not accept twoSz or twoS")
        if scope == SCOPE_PM:
            raise ValueError(
                "SCOPE=pm applies only to MODE=Sz, MODE=SzS2, "
                "or MODE=SzS2eta2 without fixed twoSz"
            )
        return ModeSpec(mode, "mode_full", "none", "none", scope=scope)

    if mode == "Sz":
        if twoS is not None:
            raise ValueError("MODE=Sz does not accept twoS")
        if twoSz is None:
            token = "mode_twoSz_pm" if scope == SCOPE_PM else "mode_twoSz"
            return ModeSpec(mode, token, "all", "none", scope=scope)
        if scope == SCOPE_PM:
            raise ValueError("SCOPE=pm applies only when twoSz is not fixed")
        return ModeSpec(
            mode,
            f"mode_twoSz_{int_token(twoSz)}",
            "one",
            "none",
            twoSz=twoSz,
            scope=scope,
        )

    suffix = "_eta_0" if mode == "SzS2eta2" else ""

    if twoS is not None and twoSz is None:
        raise ValueError(f"MODE={mode} requires twoSz when twoS is set")
    if twoSz is None:
        token = "mode_twoSz_pm_twoS" if scope == SCOPE_PM else "mode_twoSz_twoS"
        token += suffix
        return ModeSpec(mode, token, "all", "all", scope=scope)
    if scope == SCOPE_PM:
        raise ValueError("SCOPE=pm applies only when twoSz is not fixed")
    if twoS is None:
        return ModeSpec(
            mode,
            f"mode_twoSz_{int_token(twoSz)}_twoS{suffix}",
            "one",
            "all",
            twoSz=twoSz,
            scope=scope,
        )
    if twoS < 0:
        raise ValueError("twoS must be non-negative")
    return ModeSpec(
        mode,
        f"mode_twoSz_{int_token(twoSz)}_twoS_{int_token(twoS)}{suffix}",
        "one",
        "one",
        twoSz=twoSz,
        twoS=twoS,
        scope=scope,
    )


def mode_token(
    mode: str,
    twoSz: int | None = None,
    twoS: int | None = None,
    scope: str | None = None,
) -> str:
    return mode_spec(mode, twoSz=twoSz, twoS=twoS, scope=scope).token


def data_dir_name(mode: str) -> str:
    mode = canonical_mode(mode)
    if mode == "full":
        return DATA_FULL
    if mode == "Sz":
        return DATA_TWOSZ
    if mode == "SzS2eta2":
        return DATA_TWOSZ_TWOS_ETA0
    return DATA_TWOSZ_TWOS


def workflow_token(workflow: str) -> str:
    return f"workflow_{workflow.lower()}"


def seed_token(seed_set: str | Path) -> str:
    stem = Path(seed_set).stem
    if not stem:
        raise ValueError("SEED_SET must have a non-empty file stem")
    return f"seed_{stem}"


def stage_parameter_dir(
    root: str | Path,
    stage: str,
    N: int,
    nelec: int,
    U: float,
    t: float,
) -> Path:
    return Path(root) / stage / parameter_token(N, nelec, U, t)


def main_data_dir(
    root: str | Path,
    N: int,
    nelec: int,
    U: float,
    t: float,
    mode: str,
) -> Path:
    return stage_parameter_dir(root, STAGE_MAIN, N, nelec, U, t) / data_dir_name(mode)


def workflow_dir(
    root: str | Path,
    stage: str,
    N: int,
    nelec: int,
    U: float,
    t: float,
    mode: str,
    workflow: str,
    twoSz: int | None = None,
    twoS: int | None = None,
    scope: str | None = None,
) -> Path:
    return (
        stage_parameter_dir(root, stage, N, nelec, U, t)
        / mode_token(mode, twoSz=twoSz, twoS=twoS, scope=scope)
        / workflow_token(workflow)
    )


def seed_stage_dir(
    root: str | Path,
    stage: str,
    N: int,
    nelec: int,
    U: float,
    t: float,
    seed_set: str | Path,
) -> Path:
    return stage_parameter_dir(root, stage, N, nelec, U, t) / seed_token(seed_set)


def family_exchange_file(hole: int, class_idx: int) -> str:
    return f"{cluster_token(hole, class_idx)}_exchange.json"


def family_clusters_file(hole: int, class_idx: int) -> str:
    return f"{cluster_token(hole, class_idx)}_clusters.json"


def family_projection_file(hole: int, class_idx: int) -> str:
    return f"{cluster_token(hole, class_idx)}_projection.npz"


def cluster_weight_file(hole: int, class_idx: int, cluster_idx: int) -> str:
    return f"{cluster_token(hole, class_idx, cluster_idx)}.json"


def embed_cluster_file(N: int, hole: int, class_idx: int, cluster_idx: int) -> str:
    return f"N{int(N)}_{cluster_token(hole, class_idx, cluster_idx)}.txt"
