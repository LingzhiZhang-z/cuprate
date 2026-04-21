"""Shared test fixtures for the cuprate test suite."""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.hubbard import HubbardModel
from cuprate.io import Params, resolve_mode_spec
from cuprate import sectors


def make_chain_model(N, U, t, mode="full", twoSz=None, twoS=None, match_spin_sectors=False):
    """Build a HubbardModel for a simple chain of N sites."""
    params = Params(N=N, U=U, t=t, mode=mode, twoSz=twoSz, twoS=twoS,
                    match_spin_sectors=match_spin_sectors)
    mode_spec = resolve_mode_spec(params)

    model = HubbardModel(N, U, t)
    model.set_mode_spec(mode_spec)

    if match_spin_sectors and mode in ("full", "block_sz_full"):
        model.enable_match_spin_sectors()

    bonds = [(i, i + 1) for i in range(N - 1)]
    model.add_hopping_bonds(bonds, t)
    model.set_states(nsites=N, nelec=N, twoSz_set=mode_spec.twoSz)

    if model.use_S2_blocks:
        model.load_blocks()
        sectors.construct_transform_matrix(N, model.sz_sectors, model.S2_sectors)

    model.calc_hamiltonian()
    model.solve()
    return model, mode_spec
