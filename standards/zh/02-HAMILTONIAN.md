# 02-哈密顿量

Fock 态编码、基矢构建与单带 Hubbard 哈密顿量。

## 1) 局域格点字母表 (MUST)

MUST:
- 每个格点由两位 `(n_{i\uparrow}, n_{i\downarrow})` 编码。
- 导出的局域格点代码固定为：
  - `0` = 空
  - `1` = 自旋上（单占据）
  - `2` = 自旋下（单占据）
  - `3` = 双占据（上 + 下）

Code form:
```
code = (state >> (2 * i)) & 3   # 0=空, 1=上, 2=下, 3=双占据
```

Validation:
- 任何其他局域符号均非法。

## 2) 全局多体态 (MUST)

MUST:
- $N$ 个格点上的多体基矢态是一个 Python `int`。
- 比特按规范费米子顺序交错排列：
  $c^\dagger_{0\uparrow} c^\dagger_{0\downarrow} c^\dagger_{1\uparrow} c^\dagger_{1\downarrow} \cdots
  c^\dagger_{(N-1)\uparrow} c^\dagger_{(N-1)\downarrow}$。

Code form:
```
state: int
bit 2i   = n_{i\uparrow}
bit 2i+1 = n_{i\downarrow}
```

Validation:
- 不允许在位置 `2 * N` 及以上出现占据比特。

## 3) 电子数与磁化 (MUST)

MUST:
- 每格点电子数：$n_i = n_{i\uparrow} + n_{i\downarrow}$。
- 总电子数：$N_e = \sum_i (n_{i\uparrow} + n_{i\downarrow})$。
- 总磁化（$S_z$ 的两倍）为 `twoSz = N_up - N_down`。

Math:
$$
N_e = \sum_i (n_{i\uparrow} + n_{i\downarrow}), \qquad
S_z = \frac{1}{2}\left(\sum_i n_{i\uparrow} - \sum_i n_{i\downarrow}\right).
$$

Code form:
```python
count_electrons(state) = state.bit_count()
calc_twoSz(state, N) = (state & up_mask(N)).bit_count() - (state & down_mask(N)).bit_count()
```

Validation:
- 半填充时：$N_e = N$。

## 4) 双占据 (MUST)

MUST:
- 双占据计数为满足 `n_{i\uparrow} = n_{i\downarrow} = 1` 的格点数。

Math:
$$
D(|s\rangle) = \sum_i \mathbf{1}_{n_{i\uparrow}=1}\mathbf{1}_{n_{i\downarrow}=1}.
$$

Code form:
```python
count_double_occ(state, N) = (state & ((state >> 1) & up_mask(N))).bit_count()
```

## 5) 费米子符号约定 (MUST)

MUST:
- 在轨道比特索引 `bit_idx` 上产生或湮灭的费米子符号为
  $(-1)^{n_{\text{below}}}$，其中 $n_{\text{below}}$ 是更小比特索引处
  已占据轨道的个数。
- 对双占据格点的自旋下轨道不需要代码中的特殊分支：
  因为比特 `2i`（上）天然位于 `2i+1`（下）之前，额外的负号会自动出现。

Math:
$$
\text{sign}(\text{state}, \text{bit\_idx}) =
(-1)^{\#\{\text{比特索引} < \text{bit\_idx}\text{ 的已占据轨道}\}}.
$$

Code form:
```python
sign_below(state, bit_idx) = 1 if (state & ((1 << bit_idx) - 1)).bit_count() % 2 == 0 else -1
# apply_hop 先用 sign_below(state, src_bit)，翻转 src，再对新态调用 sign_below(new_state, dst_bit)
```

Validation:
- 跳跃矩阵元必须满足 $H_{ij} = H_{ji}^*$（厄米性）。

## 6) 基矢排序 (MUST)

MUST:
- 态按三元组
  `(twoSz(state), double_occ(state), state)` 的升序进行规范排序。
- 这一规范顺序同时用于完整基和 `generate_states(...)` 构造出的任何受限基。

Code form:
```python
states = sort_states(generate_states(N, N_e), N)
```

Validation:
- 排序后的列表必须先按 `twoSz` 非降序，再按 $D$ 非降序，最后按整数 `state` 非降序。
- `is_pure_spin_state(state, N)` 当且仅当所有格点都为单占据时返回 `True`。

## 7) 半填充纯自旋子空间 (MUST)

MUST:
- 纯自旋子空间由所有格点均为单占据的态组成
  （`site_code(state, i)` 为 `1` 或 `2`，即 $D = 0$）。
- 半填充（$N_e = N$）时，该子空间维度为 $2^N$。
- 在规范 Fock 基中，这些态通过
  `pure_spin_state_indices(states, N)` 显式标识，而不是依赖连续前缀行号。

Math:
$$
\mathcal{H}_{\text{spin}}(C) =
\{|s\rangle : (n_{i\uparrow}, n_{i\downarrow}) \in \{(1,0),(0,1)\},\; \forall i\},
\qquad
\dim \mathcal{H}_{\text{spin}} = 2^N.
$$

Code form:
```python
pure_spin_idx = pure_spin_state_indices(states, N)
dimspin = len(pure_spin_idx)
states_spin = [states[i] for i in pure_spin_idx]
```

Validation:
- 半填充时 `dimspin == 2**N`。

## 8) 模型定义 (MUST)

MUST:
- 代码求解有限团簇上半填充（$N_e = N$）的单带 Hubbard 模型。

Math:
$$
H = -\sum_{\langle ij\rangle, \sigma} t_{ij}
\left(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\right)
+ U \sum_i n_{i\uparrow} n_{i\downarrow}.
$$

Index:
- $t_{ij}$：格点 $i$ 和 $j$ 之间的跳跃振幅（实数或复数）。
  按键类型赋值：$t$ 为 NN，$t_2$ 为 NNN，$t_3$ 为第三近邻。
- $U$：在位库仑排斥。
- $\sigma \in \{\uparrow, \downarrow\}$。

## 9) 跳跃赋值 (MUST)

MUST:
- 生产工作流只在最近邻键上赋予跳跃。
- 键列表来自 `Cluster.bonds`。

Code form:
```python
model = HubbardModel(cluster, U, t)
# model.bonds 从 cluster.bonds 初始化
```

## 10) 哈密顿量矩阵元 (MUST)

MUST:
- 完整矩阵元为 $\langle s_1 | H | s_2 \rangle = H_t + H_U$。
- **在位 U 项**（对角）：
  当 $|s_1\rangle = |s_2\rangle$ 时 $H_U = U \cdot D(|s\rangle)$，否则为 $0$。
- **跳跃项**：对每个存储的键 $(i, j)$ 和每个自旋通道 $\sigma$，
  作用 $t_{ij} c^\dagger_{i\sigma} c_{j\sigma}$ 与
  $t_{ij}^* c^\dagger_{j\sigma} c_{i\sigma}$ 到 $|s_2\rangle$ 上。
  若结果态等于 $|s_1\rangle$，则累加对应带符号贡献。

Math:
$$
\langle s_1 | H_t | s_2 \rangle =
-\sum_{(i,j)\in\text{bonds}} \sum_{\sigma}
\left[
t_{ij}\langle s_1|c^\dagger_{i\sigma} c_{j\sigma}|s_2\rangle +
t_{ij}^*\langle s_1|c^\dagger_{j\sigma} c_{i\sigma}|s_2\rangle
\right].
$$

Code form:
```python
H_t = build_hamiltonian_t(states, bonds, hoppings)
H_U = build_hamiltonian_U(states, N, U)
H = H_t + H_U
```

Validation:
- $H$ 必须是厄米的。
- 纯自旋态（$D = 0$）的对角元必须为零（无双占据 → 无 $U$ 项，无自跳跃）。

## 11) 对角化 (MUST)

MUST:
- 用 `scipy.linalg.eigh` 对角化哈密顿量（假设厄米）。
- `EIGH=lowmem` 使用 LAPACK driver `ev`。
- `EIGH=fast` 使用 LAPACK driver `evd`。
- 传给 `eigh` 的矩阵是 Fortran-contiguous，并允许被覆盖。

Code form:
```python
eigvals, eigvecs = scipy.linalg.eigh(
    H,
    driver="ev",
    overwrite_a=True,
    check_finite=False,
)
```

Validation:
- 本征值必须为实数。
- 本征向量必须正交归一。
- 基态能量必须随 $U/t$ 减小而降低（金属极限）。
