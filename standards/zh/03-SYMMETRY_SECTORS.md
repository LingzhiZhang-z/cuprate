# 03-对称性扇区

$S_z$ 分块、固定 $S_z$ 上的 $S^2$ 对角化，以及可选的全谱重构。

## 1) $S_z$ 扇区分解 (MUST)

MUST:
- 半填充 $N$ 个格点时，$S_z$ 取值从 $-N/2$ 到 $N/2$，步长为整数
  （$N$ 为奇数时为半整数步长）。
- 仅计算非负 $S_z$ 值；负扇区通过自旋翻转对称性获得。
- 每个 $S_z$ 扇区的维度为 $\binom{N}{N_\uparrow} \cdot \binom{N}{N_\downarrow}$，
  其中 $N_\uparrow = N/2 + S_z$，$N_\downarrow = N/2 - S_z$。
- 运行时输入和路径名使用规范整数标签 `twoSz = 2 S_z` 和 `twoS = 2 S`。

Code form:
```python
twoSz_list = nonneg_twoSz_values(N)  # [0, 2, ..., N] 或 [1, 3, ..., N]
sz_states[idx] = generate_states(N, N, twoSz=twoSz)
```

## 2) Fock 基中的 `fourS2 = 4 S^2` 矩阵 (MUST)

MUST:
- 实现中在每个 $S_z$ 扇区的 Fock 基里直接构造 `fourS2 = 4 * S^2` 矩阵。
- 对角元：每个单占据格点贡献 `3`；成对项中
  局域代码 `1` 取局域 `twoSz_i = +1`，局域代码 `2` 取 `twoSz_i = -1`，其余取 `0`。
- 非对角元：仅当 $|s_1\rangle$ 和 $|s_2\rangle$ 恰好相差一个自旋翻转对
  $(i, j)$，即这两个格点上的局域代码发生 `1 ↔ 2` 交换且其余所有格点完全相同时非零，
  每对贡献 `4`。

Math:
$$
4\hat{S}^2 = 4\sum_i \hat{s}_i(\hat{s}_i + 1) + 8\sum_{i<j} S_z^i S_z^j + 4\sum_{i<j}\left(\frac{1}{2}S_+^i S_-^j + \frac{1}{2}S_-^i S_+^j\right).
$$

Code form:
```python
fourS2_matrix = calc_fourS2_matrix(basis, N)  # basis = 某个 Sz 扇区中的态
```

## 3) 固定 $S_z$ 上的 $(S_z, S)$ 扇区构造 (MUST)

MUST:
- 主实现通过总自旋升降算符 `S_plus` 和 `S_minus` 来构造 $(S_z, S)$ 扇区，
  而不是在每个扇区里都整体对角化完整 `fourS2` 矩阵。
- 实现直接消费 `generate_states(...)` 给出的固定 `twoSz` 基底；
  不会在 sector 构造层里重新生成或逐态重新校验。
- 在每个固定 `twoSz` 的基底内部，先按双占据 `D` 分组。
- 对每个非负 `twoS` 和固定 `D`，最高权空间就是 `(twoSz = twoS, D)` 这个块上
  `S_plus` 的右零空间。
- 同一个 `(twoS, D)` multiplet 的其余列通过在同一个 `D` 块里反复作用 `S_minus`，
  并除以标准 SU(2) 降阶系数来得到。
- 对每个 `(twoSz, twoS)` 扇区，不同 `D` 块得到的列按 `D` 递增拼接，
  因而 `D=0` 的 pure-spin 列天然排在最前面。
- 实现直接返回显式的 $(twoSz, twoS)$ 扇区元数据、每个扇区的变换矩阵，
  每个扇区列对应的显式双占据本征值，以及该扇区的 pure-spin 维度。

Code form:
```python
sector_list, transforms, double_occ_eigvals, dimspin = build_S2_sectors(twoSz_list, sz_states, N)
```

Validation:
- 最高权列必须满足 `S_plus @ U_hw = 0`。
- 每个变换矩阵都必须满足 `U.conj().T @ fourS2 @ U = twoS * (twoS + 2) * I`。

## 4) 到 $(S_z, S)$ 基的酉变换 (MUST)

MUST:
- 对每个 $(S_z, S)$ 扇区，最高权基连同反复降阶得到的列向量一起，
  构成从 $S_z$ Fock 基到 $(S_z, S)$ 本征基的酉变换 $U_{S^2}$。
- $(S_z, S)$ 基中的哈密顿量为 $H_{S^2} = U_{S^2}^\dagger H_{S_z} U_{S^2}$。

Code form:
```python
sector_list, transforms, dimspin = build_S2_sectors(twoSz_list, sz_states, N)
for idx, (_twoSz, _twoS, sz_sector_index) in enumerate(sector_list):
    U = transforms[idx]
    H_S2[idx] = U.conj().T @ H_Sz[sz_sector_index] @ U
```

## 5) 全谱重构 (MUST)

MUST:
- 在 `block_sz_full` 或 `block_sz_s2_full` 模式下，扇区本征系统被组装为全谱。
- 对 $S_z > 0$，负 $S_z$ 扇区通过自旋翻转获得：
  $|s'\rangle = $ 翻转所有单占据自旋，本征向量每个基矢态获得符号 $(-1)^D$。
- 全局索引记录重构本征向量矩阵的哪一列属于哪个扇区块。

Code form:
```python
reconstruct_from_sz(...)  # 用于 block_sz_full
reconstruct_from_S2(...)  # 用于 block_sz_s2_full
```

Validation:
- 重构的本征值数量必须等于完整 Hilbert 空间维度 $\binom{2N}{N}$。

## 6) 五种对角化模式 (MUST)

MUST:
- 代码支持恰好五种模式：

| 模式 | 分块 | 重构 | 自旋耦合 |
|------|------|------|----------|
| `full` | 无 | 不适用 | 是 |
| `fixed_sz` | 单个 $S_z$ | 否 | 是 |
| `block_sz_full` | 所有 $S_z$ | 是 | 是 |
| `fixed_sz_s2` | 单个 $(S_z, S)$ | 否 | 否（仅投影分析） |
| `block_sz_s2_full` | 所有 $(S_z, S)$ | 是 | 是 |

- `fixed_sz_s2` 不产生自旋耦合，因为单个总自旋扇区
  无法确定唯一的 SU(2) 不变耦合常数。它仅报告投影诊断。

Code form:
```python
ModeSpec = resolve_mode_spec(params)
# ModeSpec.result_kind == "projection_analysis" 仅对 fixed_sz_s2 成立
```
