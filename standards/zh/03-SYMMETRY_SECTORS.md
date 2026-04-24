# 03-对称性扇区

$S_z$ 分块、固定 $S_z$ 上的 $S^2$ 对角化，以及可选的全谱重构。

## 1) $S_z$ 扇区分解 (MUST)

MUST:
- 半填充 $N$ 个格点时，$S_z$ 取值从 $-N/2$ 到 $N/2$，步长为整数
  （$N$ 为奇数时为半整数步长）。
- 当请求 all-`Sz` 模式时，直接构造所有合法的 `twoSz` 扇区。
- 每个 $S_z$ 扇区的维度为 $\binom{N}{N_\uparrow} \cdot \binom{N}{N_\downarrow}$，
  其中 $N_\uparrow = N/2 + S_z$，$N_\downarrow = N/2 - S_z$。
- 运行时输入和路径名使用规范整数标签 `twoSz = 2 S_z` 和 `twoS = 2 S`。

Code form:
```python
twoSz_values = range(-N, N + 1, 2)
states = generate_states(N, N, twoSz=twoSz)
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
- 实现返回显式的 `S2SectorBlock` 记录。每条记录是一个
  `(twoSz, twoS, D)` 变换块，包含固定 `D` 的基态和该块的变换列。

Code form:
```python
grouped_states = group_states(generate_states(N, N), N)
_hw, multiplets = build_S2_multiplets(grouped_states, N)
sector_blocks = build_S2_sectors(grouped_states, multiplets)
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
transforms = build_S2_transforms(grouped_states, N, sector_blocks)
U = transforms[(twoSz, twoS)]
H_S2 = U.conj().T @ H_Sz @ U
```

## 5) 全谱重构 (MUST)

MUST:
- 扇区本征系统保持为 `Block` 对象，直到调用者显式合并它们。
- `HubbardModel.merge_by_s2()` 合并同一 `twoSz` 下的所有 `twoS` 扇区：
  先构造块对角的扇区矩阵，再变换回固定 `twoSz` 的 Fock 基。
- `HubbardModel.merge_by_sz()` 将所有固定 `twoSz` 的 Fock 坐标块合并成一个完整 Fock 坐标块。

Code form:
```python
model.merge_by_s2()  # block_sz_s2_full -> block_sz_full frame
model.merge_by_sz()  # block_sz_full -> full frame
```

Validation:
- 重构的本征值数量必须等于完整 Hilbert 空间维度 $\binom{2N}{N}$。

## 6) 五种对角化模式 (MUST)

MUST:
- 代码支持恰好五种模式：

| 模式 | 可选合并前的块 | 可选重构 |
|------|----------------|----------|
| `full` | 一个完整 Fock 块 | 不适用 |
| `fixed_sz` | 一个固定 `twoSz` 块 | 否 |
| `block_sz_full` | 所有固定 `twoSz` 块 | `merge_by_sz()` |
| `fixed_sz_s2` | 一个固定 `(twoSz,twoS)` 块 | 否 |
| `block_sz_s2_full` | 所有固定 `(twoSz,twoS)` 块 | 先 `merge_by_s2()`，再 `merge_by_sz()` |

- Projection 和 spin fitting 都在当前 `Block` 坐标框架内逐块执行。
  `fixed_sz_s2` 本身不禁止拟合；调用者负责选择能回答目标物理问题的块框架。

Code form:
```python
model.set_symmetry(mode, twoSz=twoSz, twoS=twoS)
```
