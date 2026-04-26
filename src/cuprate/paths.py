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


def block_token(twoSz: int | None, twoS: int | None = None) -> str:
    if twoSz is None:
        if twoS is not None:
            raise ValueError("twoS requires twoSz in block token")
        return "full"
    label = f"twoSz_{int_token(twoSz)}"
    if twoS is not None:
        label += f"_twoS_{int_token(twoS)}"
    return label


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    token: str
    twoSz_scope: str
    twoS_scope: str
    twoSz: int | None = None
    twoS: int | None = None


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
    raise ValueError(f"unsupported MODE={mode!r}")


def mode_spec(
    mode: str,
    twoSz: int | None = None,
    twoS: int | None = None,
) -> ModeSpec:
    mode = canonical_mode(mode)
    if mode == "full":
        if twoSz is not None or twoS is not None:
            raise ValueError("MODE=full does not accept twoSz or twoS")
        return ModeSpec(mode, "mode_full", "none", "none")

    if mode == "Sz":
        if twoS is not None:
            raise ValueError("MODE=Sz does not accept twoS")
        if twoSz is None:
            return ModeSpec(mode, "mode_twoSz", "all", "none")
        return ModeSpec(mode, f"mode_twoSz_{int_token(twoSz)}", "one", "none", twoSz=twoSz)

    if twoS is not None and twoSz is None:
        raise ValueError("MODE=SzS2 requires twoSz when twoS is set")
    if twoSz is None:
        return ModeSpec(mode, "mode_twoSz_twoS", "all", "all")
    if twoS is None:
        return ModeSpec(
            mode,
            f"mode_twoSz_{int_token(twoSz)}_twoS",
            "one",
            "all",
            twoSz=twoSz,
        )
    if twoS < 0:
        raise ValueError("twoS must be non-negative")
    return ModeSpec(
        mode,
        f"mode_twoSz_{int_token(twoSz)}_twoS_{int_token(twoS)}",
        "one",
        "one",
        twoSz=twoSz,
        twoS=twoS,
    )


def mode_token(
    mode: str,
    twoSz: int | None = None,
    twoS: int | None = None,
) -> str:
    return mode_spec(mode, twoSz=twoSz, twoS=twoS).token


def data_dir_name(mode: str) -> str:
    mode = canonical_mode(mode)
    if mode == "full":
        return DATA_FULL
    if mode == "Sz":
        return DATA_TWOSZ
    return DATA_TWOSZ_TWOS


def workflow_token(workflow: str) -> str:
    return f"workflow_{workflow.lower()}"


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
) -> Path:
    return (
        stage_parameter_dir(root, stage, N, nelec, U, t)
        / mode_token(mode, twoSz=twoSz, twoS=twoS)
        / workflow_token(workflow)
    )


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
