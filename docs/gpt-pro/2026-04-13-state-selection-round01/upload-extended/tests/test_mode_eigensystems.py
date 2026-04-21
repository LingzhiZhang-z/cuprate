import numpy as np

from conftest import make_chain_model
from cuprate.sectors import find_fixed_S2_block_index


def test_block_sz_modes_only_use_nonnegative_sz_sectors():
    model, _ = make_chain_model(5, 4.0, 1.0, "block_sz_full")

    # twoSz_list contains integer 2*Sz values; for N=5 half-integer spins: [1, 3, 5]
    assert model.sz_sectors.twoSz_list == [1, 3, 5]


def test_fixed_sz_s2_keeps_only_the_selected_s_block():
    model, spec = make_chain_model(4, 4.0, 1.0, "fixed_sz_s2", twoSz=0, twoS=0)

    block_idx = find_fixed_S2_block_index(model.S2_sectors, spec.twoSz, spec.twoS)
    expected_dim = model.S2_sectors.transforms[block_idx].shape[1]

    assert model.eigvecs.shape[1] == expected_dim
    assert model.eigvals.shape[0] == expected_dim


def test_block_szs2_full_matches_full_spectrum():
    full, _ = make_chain_model(4, 4.0, 1.0, "full")
    blocked, _ = make_chain_model(4, 4.0, 1.0, "block_sz_s2_full")

    np.testing.assert_allclose(full.eigvals, blocked.eigvals, atol=1e-10)


def test_fixed_sz_s2_mode_spec_is_analysis_only():
    _, spec = make_chain_model(4, 4.0, 1.0, "fixed_sz_s2", twoSz=0, twoS=0)

    assert spec.result_kind == "projection_analysis"


def test_full_mode_spec_keeps_spin_coupling_result_kind():
    _, spec = make_chain_model(4, 4.0, 1.0, "full")

    assert spec.result_kind == "spin_couplings"
