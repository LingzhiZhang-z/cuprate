# 03-对称性扇区

$S_z$ 分块、固定 $S_z$ 上的 $S^2$ 对角化，以及可选的全谱重构。

## 1) $S_z$ 扇区分解 (MUST)

MUST:
- 半填充 $N$ 个格点时，$S_z$ 取值从 $-N/2$ 到 $N/2$，步长为整数
  （$N$ 为奇数时为半整数步长）。
- all-`twoSz` 模式使用 `SCOPE` 选择扇区范围：`SCOPE=nonnegative`
  构造合法的 `twoSz >= 0` 扇区，`SCOPE=pm` 构造所有合法的正负
  `twoSz` 扇区。
- 每个 $S_z$ 扇区的维度为 $\binom{N}{N_\uparrow} \cdot \binom{N}{N_\downarrow}$，
  其中 $N_\uparrow = N/2 + S_z$，$N_\downarrow = N/2 - S_z$。
- 运行时输入和路径名使用规范整数标签 `twoSz = 2 S_z` 和 `twoS = 2 S`。

Code form:
```python
twoSz_values = [twoSz for twoSz in range(-N, N + 1, 2) if scope == "pm" or twoSz >= 0]
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
- Runtime `Block` 将 transform 存为显式 fixed-`D` 小块，而不是长期保存一个
  dense `basis_transform`。
- `build_hamiltonians()` 消费所有 fixed-`D` transform 小块来组装完整的
  symmetry-sector Hamiltonian，然后只保留 `D=0` transform matrix 以及
  projection/fitting 需要的 `D` column metadata。

Code form:
```python
blocks = blocks_by_sector[(twoSz, twoS)]  # 每个 D 一个 S2SectorBlock
H_S2[D1, D2] = U_D1.conj().T @ H_Sz[D1, D2] @ U_D2
U_spin = blocks_by_D[0].transform
```

## 5) 合并到固定 Sz 块 (MUST)

MUST:
- 扇区本征系统保持为 `Block` 对象，直到调用者显式合并它们。
- `HubbardModel.merge_to_sz(merge_basis)` 将一个固定 `twoSz` 下所有选中的
  `twoS` 扇区合并成一个固定 `twoSz` 块。
- `merge_basis="fock"` 将本征矢存储在固定 `twoSz` 的 Fock 行坐标中，并令
  `basis_transform=None`。
- `merge_basis="block"` 保留块坐标下的本征矢，并将各 sector transform 按列
  拼接成 `basis_transform`。
- 生产运行时只在固定 `twoSz` 且没有固定 `twoS` 的 `MODE=SzS2` 和
  `MODE=SzS2eta2` 运行中支持 `MERGE=Sz`。
- 生产运行时不提供 Sz 到 full 的重构。

Code form:
```python
model.merge_to_sz("fock")   # eigvecs 位于 fixed-twoSz Fock 行
model.merge_to_sz("block")  # eigvecs 保持在 merged block 坐标
```

Validation:
- 合并后的 `SzS2` block label 不包含 `twoS`。
- 合并后的 `SzS2eta2` block label 不包含 `twoS`，但保留 `eta=0`。
- 固定 `twoS`、`MODE=Sz` 或 `MODE=full` 下使用 `MERGE=Sz` 必须失败。

## 6) 对角化模式 (MUST)

MUST:
- 代码支持恰好四种 `MODE` 值，以及可选 block selector：

| CLI 输入 | 可选合并前的块 | 可选合并 |
|----------|----------------|----------|
| `MODE=full` | 一个完整 Fock 块 | 不适用 |
| `MODE=Sz twoSz=<value>` | 一个固定 `twoSz` 块 | 否 |
| `MODE=Sz` | `twoSz >= 0` 的固定 `twoSz` 块 | 否 |
| `MODE=Sz SCOPE=pm` | 所有固定 `twoSz` 块 | 否 |
| `MODE=SzS2 twoSz=<value> twoS=<value>` | 一个固定 `(twoSz,twoS)` 块 | 否 |
| `MODE=SzS2 twoSz=<value>` | 固定 `twoSz` 下的所有 `twoS` 块 | 可选 `MERGE=Sz` |
| `MODE=SzS2` | `twoSz >= 0` 的固定 `(twoSz,twoS)` 块 | 不做 full 重构 |
| `MODE=SzS2 SCOPE=pm` | 所有固定 `(twoSz,twoS)` 块 | 不做 full 重构 |
| `MODE=SzS2eta2 twoSz=<value> twoS=<value>` | 一个固定 `(twoSz,twoS,eta=0)` 块 | 否 |
| `MODE=SzS2eta2 twoSz=<value>` | 固定 `twoSz` 下所有 `twoS` 块，且均细分到 `eta=0` | 可选 `MERGE=Sz` |
| `MODE=SzS2eta2` | 默认 scope 的固定 `(twoSz,twoS,eta=0)` 块 | 不做 full 重构 |
| `MODE=SzS2eta2 SCOPE=pm` | 所有正负 `twoSz` 下的固定 `(twoSz,twoS,eta=0)` 块 | 非零 eta 扇区支持前不做 full 重构 |

- Projection 和 spin fitting 都在当前 `Block` 坐标框架内逐块执行。
  S2 分辨的模式本身不禁止拟合；调用者负责选择能回答目标物理问题的块框架。
- 固定 `twoSz` 和 `twoS` 数值是独立可选 selector，不编码在 `MODE` 里。
- `SCOPE` 只在没有显式指定 `twoSz` 时生效。

Code form:
```python
model.set_symmetry("SzS2", twoSz=0, twoS=0, scope="nonnegative")
```
