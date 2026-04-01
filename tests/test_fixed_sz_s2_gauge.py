import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.hubbard import Hubbard_SingleBand
from cuprate.io import Params, resolve_mode_spec


def build_fixed_sz_s2_model(N=4, U=4.0, t=1.0, sz_idx=0, s_idx=0):
    params = Params(N=N, U=U, t=t, mode="fixed_sz_s2", sz_idx=sz_idx, s_idx=s_idx)
    spec = resolve_mode_spec(params)

    model = Hubbard_SingleBand(N, U, t)
    model.set_mode_spec(spec)
    model.set_bonds([(i, i + 1) for i in range(N - 1)], [t] * (N - 1))
    model.set_states(nsites=N, nelec=N, sz_set=spec.sz)
    model.save_blocks(N)
    model.load_blocks(N)
    model.construct_transform_matrix(N)
    model.calc_hamiltonian()
    model.solve()
    return model


def assert_positive_real_pivots(matrix):
    for col in range(matrix.shape[1]):
        vec = matrix[:, col]
        pivot = np.argmax(np.abs(vec))
        value = vec[pivot]
        assert abs(value.imag) < 1e-10
        assert value.real > 0


def test_fixed_sz_s2_transform_columns_have_canonical_phase():
    model = build_fixed_sz_s2_model()

    for transform in model.szs2_Us:
        assert_positive_real_pivots(transform)


def test_fixed_sz_s2_output_vectors_have_canonical_phase():
    model = build_fixed_sz_s2_model()

    assert_positive_real_pivots(model.eigvecs)
