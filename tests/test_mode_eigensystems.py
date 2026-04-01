import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.hubbard import Hubbard_SingleBand
from cuprate.io import Params, resolve_mode_spec


def make_mode_model(N, U, t, mode, sz_idx=None, s_idx=None):
    params = Params(N=N, U=U, t=t, mode=mode, sz_idx=sz_idx, s_idx=s_idx)
    spec = resolve_mode_spec(params)

    model = Hubbard_SingleBand(N, U, t)
    model.set_mode_spec(spec)
    bonds = [(i, i + 1) for i in range(N - 1)]
    hoppings = [t] * len(bonds)
    model.set_bonds(bonds, hoppings)

    model.set_states(nsites=N, nelec=N, sz_set=spec.sz)

    if mode in ("block_szs2_full", "fixed_sz_s2"):
        model.save_blocks(N)
        model.load_blocks(N)
        model.construct_transform_matrix(N)

    model.calc_hamiltonian()
    model.solve()
    return model, spec


def test_block_sz_modes_only_use_nonnegative_sz_sectors():
    model, _ = make_mode_model(5, 4.0, 1.0, "block_sz_full")

    assert model.sz_list == [0.5, 1.5, 2.5]


def test_fixed_sz_s2_keeps_only_the_selected_s_block():
    model, spec = make_mode_model(4, 4.0, 1.0, "fixed_sz_s2", sz_idx=0, s_idx=0)

    expected_dim = model.szs2_Us[spec.s_idx].shape[1]

    assert model.eigvecs.shape[1] == expected_dim
    assert model.eigvals.shape[0] == expected_dim


def test_block_szs2_full_matches_full_spectrum():
    full, _ = make_mode_model(4, 4.0, 1.0, "full")
    blocked, _ = make_mode_model(4, 4.0, 1.0, "block_szs2_full")

    np.testing.assert_allclose(full.eigvals, blocked.eigvals, atol=1e-10)


def test_fixed_sz_s2_mode_spec_is_analysis_only():
    _, spec = make_mode_model(4, 4.0, 1.0, "fixed_sz_s2", sz_idx=0, s_idx=0)

    assert spec.result_kind == "projection_analysis"
    assert spec.supports_spin_couplings is False


def test_full_mode_spec_keeps_spin_coupling_result_kind():
    _, spec = make_mode_model(4, 4.0, 1.0, "full")

    assert spec.result_kind == "spin_couplings"
    assert spec.supports_spin_couplings is True
