from math import comb


def comb_safe(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def analyze_space(N: int) -> None:
    Smax = N * 0.5
    dim = comb(2*N, N)
    dim_eff = 2**N
    n_half = N * 0.5

    Nups = [N - i for i in range(N + 1)]
    Ndos = [i for i in range(N + 1)]
    sz_states = []
    sz_s2_states = []
    for idx_sz in range(N + 1):
        sz = (Nups[idx_sz] - Ndos[idx_sz]) * 0.5
        dim_sz = comb_safe(N, Nups[idx_sz]) * comb_safe(N, Ndos[idx_sz])
        dim_eff_sz = comb_safe(N, Nups[idx_sz])

        sum_dim_sz_s = 0
        sum_dim_eff_sz_s = 0
        for idx_s in range(int(Smax - abs(sz)) + 1):
            s = abs(sz) + idx_s
            s2 = s * (s + 1)
            dim_sz_s = comb_safe(N, int(n_half + s)) * comb_safe(N, int(n_half - s)) - comb_safe(N, int(n_half + s + 1)) * comb_safe(N, int(n_half - s - 1))
            dim_eff_sz_s = comb_safe(N, int(n_half - s)) - comb_safe(N, int(n_half - s - 1))
            sz_s2_states.append((sz, s, s2, dim_sz_s, dim_eff_sz_s))
            sum_dim_sz_s += dim_sz_s
            sum_dim_eff_sz_s += dim_eff_sz_s
        sz_states.append((sz, dim_sz, dim_eff_sz, sum_dim_sz_s, sum_dim_eff_sz_s))

    print(f"Total dimension: {dim}")
    print(f"Total effective dimension: {dim_eff}")
    print("Sz sectors (dim, dim_eff):")
    for sz, dim_sz, dim_eff_sz, sum_dim_sz_s, sum_dim_eff_sz_s in sz_states:
        print(f"  sz={sz:.1f}: dim={dim_sz}, dim_eff={dim_eff_sz}")
    print("Sz, S^2 sectors (dim, dim_eff):")
    for sz, s, s2, dim_sz_s, dim_eff_sz_s in sz_s2_states:
        print(f"  sz={sz:.1f}, s={s:.1f}, s2={s2:.2f}: dim={dim_sz_s}, dim_eff={dim_eff_sz_s}")
