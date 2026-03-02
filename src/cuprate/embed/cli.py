import sys
from cuprate.core.states import tune_sz

DEFAULT_N = 3
DEFAULT_U = 6.0
DEFAULT_T = 1.0


def parse_arguments(rank=0):
    """Parse command-line arguments for embed module. Returns a dict."""
    params = {}
    params['N'] = DEFAULT_N
    params['U'] = DEFAULT_U
    params['t'] = DEFAULT_T
    params['t2'] = None
    params['t3'] = None
    params['sz'] = None
    params['s2'] = None
    params['s2_fix'] = False
    params['type'] = None
    params['delta'] = None
    params['delta2'] = None
    params['restart'] = False
    params['type_delta'] = None

    params['Ncell'] = None
    params['Ncut'] = None
    params['ratio'] = None

    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, value = arg.split('=', 1)
        key = key.upper()
        if key == 'N':
            params['N'] = int(value)
        elif key == 'U':
            params['U'] = float(value)
        elif key == 'T':
            params['t'] = float(value)
        elif key == 'T2':
            params['t2'] = float(value)
        elif key == 'T3':
            params['t3'] = float(value)
        elif key == 'SZ':
            params['sz'] = float(value)
        elif key == 'S2':
            params['s2'] = float(value)
        elif key == 'S2_FIX':
            params['s2_fix'] = bool(value)
        elif key == 'TYPE':
            params['type'] = value
        elif key == 'DELTA':
            params['delta'] = float(value)
        elif key == 'DELTA2':
            params['delta2'] = float(value)
        elif key == 'RESTART':
            params['restart'] = bool(value)
        elif key == 'TYPE_DELTA':
            params['type_delta'] = value
        elif key == 'NCELL':
            params['Ncell'] = int(value)
        elif key == 'NCUT':
            params['Ncut'] = int(value)
        elif key == 'RATIO':
            params['ratio'] = float(value)

    # 统一 sz（embed 需要 tune_sz 调用）
    params['sz'] = tune_sz(params['sz'], params['N'], rank=rank)
    return params
