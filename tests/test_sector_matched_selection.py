import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.hubbard import Hubbard_SingleBand, total_mag
from cuprate.io import Params, resolve_mode_spec
from cuprate.spin import comb_safe


def _round_half(value: float) -> float:
    return round(float(value) * 2.0) / 2.0


def _spin_from_s2(s2_value: float) -> float:
    return _round_half((-1.0 + np.sqrt(1.0 + 4.0 * float(s2_value))) / 2.0)


def _expected_spin_sector_counts(nsites: int) -> Counter:
    start_s = 0.0 if nsites % 2 == 0 else 0.5
    counts = Counter()
    for offset in range(int(nsites * 0.5 - start_s) + 1):
        s_value = start_s + offset
        multiplicity = comb_safe(nsites, int(round(nsites * 0.5 - s_value))) - comb_safe(
            nsites, int(round(nsites * 0.5 - s_value - 1.0))
        )
        for step in range(nsites + 1):
            sz_value = -nsites * 0.5 + step
            if abs(sz_value) <= s_value + 1e-8:
                counts[(_round_half(sz_value), _round_half(s_value))] = multiplicity
    return counts


def _selected_spin_sector_counts(model: Hubbard_SingleBand) -> Counter:
    sz_basis = np.array([total_mag(state) * 0.5 for state in model.states], dtype=float)
    sz_diag = np.array(
        [
            np.real(np.vdot(model.eigvecs[:, idx], sz_basis * model.eigvecs[:, idx]))
            for idx in range(model.eigvecs.shape[1])
        ]
    )
    return Counter(
        (_round_half(sz_diag[idx]), _spin_from_s2(model.S2_diag[idx]))
        for idx in model.t11_selected_indices
    )


def _make_selected_model(mode: str) -> Hubbard_SingleBand:
    params = Params(N=4, U=1.0, t=1.0, mode=mode, match_spin_sectors=True)
    spec = resolve_mode_spec(params)

    model = Hubbard_SingleBand(params.N, params.U, params.t)
    model.set_mode_spec(spec)
    if params.match_spin_sectors and mode in ("full", "block_sz_full"):
        model.enable_match_spin_sectors()
    model.set_bonds([(i, i + 1) for i in range(params.N - 1)], [params.t] * (params.N - 1))
    model.set_states(nsites=params.N, nelec=params.N, sz_set=spec.sz)

    if model.if_block_szs2:
        model.save_blocks(params.N)
        model.load_blocks(params.N)
        model.construct_transform_matrix(params.N)

    model.calc_hamiltonian()
    model.solve()
    model.calc_s2()
    model.calc_heff_halffilled(params, {"hole": 0, "class_idx": 0})
    return model


def test_full_mode_match_spin_sectors_respects_spin_sector_multiplicities():
    model = _make_selected_model("full")

    assert _selected_spin_sector_counts(model) == _expected_spin_sector_counts(model.N)


def test_block_sz_full_match_spin_sectors_respects_spin_sector_multiplicities():
    model = _make_selected_model("block_sz_full")

    assert _selected_spin_sector_counts(model) == _expected_spin_sector_counts(model.N)
