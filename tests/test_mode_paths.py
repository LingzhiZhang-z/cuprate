import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.io import Params, build_path_spec


def test_fixed_sz_path_uses_index_label():
    params = Params(N=4, U=1.0, t=0.1, mode="fixed_sz", sz_idx=0)

    spec = build_path_spec(params)

    assert spec.data_dir.endswith("Block/U1.0000_t0.1000/N4_sz0")
    assert spec.run_dir == "N4_sz0"


def test_fixed_sz_s2_path_uses_double_index_label():
    params = Params(N=4, U=1.0, t=0.1, mode="fixed_sz_s2", sz_idx=1, s_idx=0)

    spec = build_path_spec(params)

    assert spec.data_dir.endswith("Block/U1.0000_t0.1000/N4_sz1_s0")
    assert spec.run_dir == "N4_sz1_s0"


def test_block_full_paths_use_mode_suffixes():
    params_sz = Params(N=5, U=1.0, t=0.1, mode="block_sz_full")
    params_szs2 = Params(N=5, U=1.0, t=0.1, mode="block_szs2_full")

    spec_sz = build_path_spec(params_sz)
    spec_szs2 = build_path_spec(params_szs2)

    assert spec_sz.run_dir == "N5_block_sz"
    assert spec_szs2.run_dir == "N5_block_szs2"
