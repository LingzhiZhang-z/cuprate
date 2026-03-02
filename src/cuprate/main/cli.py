import sys
from cuprate.main.params import Params
from cuprate.core.states import tune_sz

DEFAULT_N = 3
DEFAULT_U = 1.0
DEFAULT_T = 0.1


def parse_arguments(rank: int = 0) -> Params:
    """Parse command-line arguments and return a Params dataclass."""
    params = Params(N=DEFAULT_N, U=DEFAULT_U, t=DEFAULT_T)

    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, value = arg.split('=', 1)
        key = key.upper()
        if key == 'N':
            params.N = int(value)
        elif key == 'U':
            params.U = float(value)
        elif key == 'T':
            params.t = float(value)
        elif key == 'T2':
            params.t2 = float(value)
        elif key == 'T3':
            params.t3 = float(value)
        elif key == 'SZ':
            params.sz = float(value)
        elif key == 'S2':
            params.s2 = float(value)
        elif key == 'S2_FIX':
            params.s2_fix = bool(value)
        elif key == 'TYPE':
            params.type = value
        elif key == 'BLOCK':
            params.block = value
        elif key == 'DELTA':
            params.delta = float(value)
        elif key == 'DELTA2':
            params.delta2 = float(value)
        elif key == 'RESTART':
            params.restart = bool(value)
        elif key == 'TYPE_DELTA':
            params.type_delta = value
        elif key == 'NCELL':
            params.Ncell = int(value)
        elif key == 'NCUT':
            params.Ncut = int(value)
        elif key == 'RATIO':
            params.ratio = float(value)

    # 统一 sz（若未指定则自动调节/选择）
    params.sz = tune_sz(params.sz, params.N, rank=rank)
    return params
