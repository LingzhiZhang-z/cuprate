import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.io import Params, build_path_spec
from cuprate.main import parse_arguments


def test_parse_mode_and_sector_indices(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cuprate", "N=4", "MODE=fixed_sz_s2", "SZ_IDX=1", "S_IDX=0"],
    )

    params = parse_arguments(rank=0)

    assert params.mode == "fixed_sz_s2"
    assert params.sz_idx == 1
    assert params.s_idx == 0


def test_parse_block_mode_to_new_mode_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cuprate", "N=5", "BLOCK=szs2"],
    )

    params = parse_arguments(rank=0)

    assert params.mode == "block_szs2_full"


def test_parse_fixed_sz_s2_marks_analysis_result_kind(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cuprate", "N=4", "MODE=fixed_sz_s2", "SZ_IDX=0", "S_IDX=0"],
    )

    params = parse_arguments(rank=0)

    assert params.result_kind == "projection_analysis"
    assert params.supports_spin_couplings is False


def test_parse_match_spin_sectors_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cuprate", "N=4", "MODE=full", "MATCH_SPIN_SECTORS=true"],
    )

    params = parse_arguments(rank=0)

    assert params.match_spin_sectors is True


def test_build_path_spec_appends_match_spin_sectors_suffix():
    params = Params(N=4, U=1.0, t=0.1, mode="full", match_spin_sectors=True)

    spec = build_path_spec(params)

    assert spec.run_dir == "N4_match_spin_sectors"
