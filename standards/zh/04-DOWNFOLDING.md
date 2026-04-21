# 04-降维

本征态选择、$T_{11}$ 投影度量、SVD 降维至 $H_{\text{eff}}$ 与自旋耦合拟合。

## 1) 选择目标 (MUST)

MUST:
- 从完整本征系统中选择 $d_{\text{spin}}$ 个本征态，使其对纯自旋子空间
  的投影尽可能接近酉映射。
- 在分块模式下，选择按块操作，使用每块的 $d_{\text{spin}}$ 值。

## 2) $T_{11}$ 投影度量 (MUST)

MUST:
- 设 $\Psi_{\text{sel}}$ 为选定本征向量矩阵（全空间行数，$d_{\text{spin}}$ 列）。
- 自旋投影矩阵为 $S_{BD} = P_{\text{spin}} \Psi_{\text{sel}}$
  其中 $P_{\text{spin}}$ 是显式的纯自旋基矩阵。
- SVD 分解：$S_{BD} = U \Sigma V^\dagger$。
- 投影质量度量为：

Math:
$$
T_{11} = U \Sigma U^\dagger, \qquad
\Delta_{T_{11}} = \|T_{11} - I\|_F.
$$

Code form:
```python
S_BD = spin_basis @ eigvecs_selected
U, Sigma, VH = np.linalg.svd(S_BD, full_matrices=False)
T11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(dimspin)
norm  = np.linalg.norm(T11m1.flatten())
```

实现规则：
- 对 `full`、`fixed_sz`、`block_sz_full`，`spin_basis` 由
  `pure_spin_state_indices(states, N)` 构造出的行选择矩阵给出。
- 对 `fixed_sz_s2`，`spin_basis` 是所选 $(S_z, S)$ 扇区内部
  $D = 0$ 模型空间的显式基。

Validation:
- 小的 $\Delta_{T_{11}}$ 意味着所选流形紧密张成自旋子空间。
- $\Delta_{T_{11}} = 0$ 表示完美投影。

## 3) 选择策略 (MUST)

MUST:
- 实现了五种策略：

| 方法 | 键 | 描述 |
|------|-----|------|
| 按占据 | `occ` | 每块选择 $\langle D \rangle$ 最低的 $d_{\text{spin}}$ 个态 |
| 按能量 | `energy` | 每块选择能量最低的 $d_{\text{spin}}$ 个态 |
| 贪心交换 | `single` | 从按占据排序的初始猜测出发，尝试所有逐对交换以最小化 $\Delta_{T_{11}}$ |
| 多重启动 | `multi` | 贪心 + 对高 $D$ 尾部随机扰动，多次重启 |
| 绝热 | `adiabatic` | 最大化与前一参数点的重叠 $|\langle\psi_{\text{prev}}|\psi_{\text{curr}}\rangle|^2$ |

- 默认（`None`）：先尝试按占据，若 $\Delta_{T_{11}} = \infty$ 则回退到贪心。
- 贪心/多重启动的候选池为按 $\langle D \rangle$ 排序的 `ratio * dimspin` 个态。
- 在绝热扫描的首个参数点，如果前一参数点的绝热结果不存在，
  则用同参数的基线运行（`workflow=None`）作为绝热初始种子。

Code form:
```python
selected_indices, best_norm, overlap = select_eigenstates(method, ...)
```

## 4) 双占据期望值 (MUST)

MUST:
- 选择前，计算本征基中双占据算符的对角期望值：

Math:
$$
\langle D \rangle_a = \langle \psi_a | \hat{D} | \psi_a \rangle,
\qquad
\hat{D} = \text{diag}(D(|s_0\rangle), D(|s_1\rangle), \ldots).
$$

Code form:
```python
dom = calc_double_occupation_matrix(states)  # 对角矩阵
double_occ = np.diag(eigvecs.conj().T @ dom @ eigvecs)
```

## 5) 有效哈密顿量 $H_{\text{eff}}$ (MUST)

MUST:
- 给定规则 2 中的 SVD 和选定本征值 $\Lambda = \text{diag}(E_{\text{sel}})$：

Math:
$$
H_{\text{eff}} = U V^\dagger \Lambda (U V^\dagger)^\dagger = U V^\dagger \Lambda V U^\dagger.
$$

Code form:
```python
Lambda = np.diag(eigvals[selected_indices])
Heff   = U @ VH @ Lambda @ VH.conj().T @ U.conj().T
```

Validation:
- $H_{\text{eff}}$ 在容差内必须是厄米的。
- $H_{\text{eff}}$ 的谱必须等于选定的本征值。

## 6) 自旋算符基 (MUST)

MUST:
- 用于拟合的自旋算符基包括：
  1. 恒等算符 $I$（常数能量偏移 $E_0$）。
  2. 两格点 Heisenberg：$\mathbf{S}_i \cdot \mathbf{S}_j = S_z^i S_z^j + \frac{1}{2}(S_+^i S_-^j + S_-^i S_+^j)$。
  3. 四格点双二次：$(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)$。
  4. 六格点：$(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)(\mathbf{S}_m \cdot \mathbf{S}_n)$。
- 所有算符都在 `states_spin` 上构建；`states_spin` 是显式的纯自旋基态列表。
- $S_+$、$S_-$ 定义在格点编码上：
  $S_+(|-1\rangle) = |+1\rangle$，$S_-(|+1\rangle) = |-1\rangle$，其余为零
  （$s = 0$ 或 $s = 2$ 的格点被湮灭）。
- 多格点算符（四格点、六格点……）必须按 `08-OPERATOR_OUTPUT.md` §3
  定义的**规范配对排序**枚举。这保证拟合系数向量具有几何确定的含义：
  对于给定格点组，第 $k$ 个系数总对应相同的物理配对模式，与格点编号无关。

Code form:
```python
spin_matrix_J_ij(states, [i, j])       # S_i · S_j
spin_matrix_JJ_ij(states, [i,j,k,l])   # (S_i·S_j)(S_k·S_l)
spin_matrix_JJJ_ij(states, [i,j,k,l,m,n])  # 三重积
```

## 7) 最小二乘拟合 (MUST)

MUST:
- 将 $H_{\text{eff}}$ 展平为向量 $b$，每个自旋算符展平为矩阵 $A$ 的列向量。
- 通过正规方程求解 $A x = b$：$x = (A^T A)^{-1} A^T b$，以 `lstsq` 为后备。
- 报告拟合质量：相对误差、残差范数和 $R^2$。

Math:
$$
\min_x \|Ax - b\|, \qquad
R^2 = 1 - \frac{\|Ax - b\|^2}{\|b - \bar{b}\|^2}.
$$

Code form:
```python
coeffs, error = model.calc_spin_coeff(bonds)
# error = (relative_error, residual_norm, R_squared)
```

## 8) 解释规则 (MUST)

MUST:
- 小的拟合残差本身并不能验证自旋映射。
- $\Delta_{T_{11}}$ 和拟合诊断必须结合解释。
- 若 $\Delta_{T_{11}}$ 较大，则无论自旋算符对 $H_{\text{eff}}$ 的拟合多好，
  $H_{\text{eff}}$ 都可能无法忠实代表低能物理。
