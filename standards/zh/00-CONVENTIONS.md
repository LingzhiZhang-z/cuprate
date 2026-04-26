# 00-约定

全部标准的书写约定、符号定义与数值容差。

## 1) 适用范围 (MUST)

MUST:
- 本约定适用于 `./standards/en/` 和 `./standards/zh/` 下的全部文件。
- 英文标准为权威版本；中文文件仅为翻译。

## 2) 逐条结构 (MUST)

每条非平凡规则应按以下顺序撰写：
1. `MUST` 条目（硬约束）。
2. `Math:` 块，使用 LaTeX（`$$ ... $$`）。
3. `Code form:` 单行 ASCII 表达式或短代码块。
4. `Index:` 显式符号含义。
5. `Validation:` 最小化校验。

## 3) 公式风格 (MUST)

- 优先使用独立公式块，而非冗长的行内公式。
- 在局部重新定义关键符号，而非依赖远处章节。

## 4) 代码映射 (MUST)

- 每个核心公式必须包含一行 `Code form`。
- `Code form` 必须是面向实现的纯文本。

## 5) 符号表 (MUST)

| 符号 | 含义 |
|------|------|
| $N$ | 团簇中格点数 |
| $N_e$ | 总电子数（半填充时 $N_e = N$） |
| $U$ | 在位库仑排斥 |
| $t, t_2, t_3$ | 最近邻、次近邻、第三近邻跳跃振幅 |
| $s_i$ | 局域格点占据 $\in \{0, 1, -1, 2\}$ |
| $S_z$ | 总自旋 z 分量 $= \frac{1}{2}(N_\uparrow - N_\downarrow)$ |
| $S$ | 总自旋量子数，由 $\hat{S}^2 = S(S+1)$ 确定 |
| $D$ | 双占据格点数 |
| $d_{\text{spin}}$ | 纯自旋子空间维度 $= 2^N$ |
| $d(S)$ | $(S_z, S)$ 扇区维度 $= \binom{N}{N/2-S} - \binom{N}{N/2-S-1}$ |
| $H_{\text{eff}}$ | SVD 降维得到的有效自旋哈密顿量 |
| $T_{11}$ | 投影质量度量 $= U\Sigma U^\dagger$ |
| $W(C)$ | 团簇 $C$ 的 LCE 净贡献 |

## 6) 规范命名 (MUST)

MUST:
- `twoSz`（$= 2S_z$）和 `twoS`（$= 2S$）是**整数标签**，
  用于 CLI 参数、路径名、文件名和索引比较。
  其目的是消除浮点歧义：半整数量子数如 $S_z = 1/2$ 无法用浮点数精确表示，
  这会在字符串格式化、文件查找和相等判断中引发不可预测的行为。
- 实际物理运算仍使用半整数值 $S_z = twoSz / 2$ 和 $S = twoS / 2$。
  整数形式仅用于标签。
- `S2`（$= S(S+1)$）是计算得到的浮点数，不是整数标签。
- 规范的运行时工作流参数名为 `workflow`；规范的 all-`twoSz`
  扇区范围参数名为 `SCOPE`。
- 标准和实现中不得引入替代规范名称，如
  `ssq`、`s_squared`、`sz` 或 `s`。
- 唯一允许的 CLI `MODE` 拼写为 `full`、`Sz` 和 `SzS2`。
- `MODE` 只选择对角化分块层级；可选 CLI 参数 `twoSz` 和 `twoS`
  在该层级内选择具体 block 子集。
- `SCOPE` 只为 all-`twoSz` 的 `MODE=Sz` 和 `MODE=SzS2` 运行选择扇区范围。
  合法值为 `nonnegative` 和 `pm`。
- 规范的运行时路径标记为：
  - `N_<N>_nelec_<nelec>_U_<U:.4f>_t_<T:.4f>`
  - `twoSz_<value>`（负值使用 `n` 前缀：`twoSz_n1` 表示 $-1$）
  - `twoS_<value>`（`twoS` 始终非负，无需 `n` 前缀）
  - `mode_full`
  - `mode_twoSz`
  - `mode_twoSz_pm`
  - `mode_twoSz_<value>`
  - `mode_twoSz_twoS`
  - `mode_twoSz_pm_twoS`
  - `mode_twoSz_<value>_twoS`
  - `mode_twoSz_<value>_twoS_<value>`
- 路径里的 `mode_*` token 是输出目录名，不是 CLI `MODE` 输入值。
- 遗留 CLI 名称如 `fixed_sz`、`block_sz_full`、`fixed_sz_s2`、
  `fixed_sz_s2_all`、`block_sz_s2_full`、`fixed_sz_ssq`、
  `block_sz_ssq_full`、`_sz...`、`_s...`、`SZ`、`S`、`S2`、`SZ_IDX`、
  `S_IDX` 和 `TYPE` 不属于生产 CLI 标准。

Code form:
```text
canonical physics names: twoSz, twoS, S2
canonical workflow key: workflow
canonical all-twoSz scope key: SCOPE
canonical CLI modes: full, Sz, SzS2
canonical path tokens: N_<N>_nelec_<nelec>_U_<U:.4f>_t_<T:.4f>, twoSz_<value>, twoS_<value>, mode_*
negative value encoding: n prefix (e.g. twoSz_n1 = twoSz = -1)
```

## 7) 量子数取值规则 (MUST)

MUST:
- 代码假设半填充：$N_e = N$（电子数等于格点数）。
  半填充下 $N_\uparrow + N_\downarrow = N$，因此
  $S_z = \tfrac{1}{2}(N_\uparrow - N_\downarrow)$ 的取值范围为
  $\{-N/2, -N/2+1, \dots, N/2\}$，即 `twoSz` ∈ $[-N, N]$，步长为 2。
- `twoSz` 和 `twoS` 为整数。
- `twoSz` 可以为负。
- `twoS` 必须满足 `0 <= twoS <= N`。
- `twoSz` 和 `twoS` 必须满足 `|twoSz| <= twoS`。
- `twoSz` 和 `twoS` 必须与 `N` 具有相同的奇偶性。
- 非法输入必须报错；标准不允许静默修正。

Math:
$$
N_e = N, \qquad N_\uparrow + N_\downarrow = N, \qquad S_z = \tfrac{1}{2}(N_\uparrow - N_\downarrow).
$$
$$
N \bmod 2 = twoSz \bmod 2 = twoS \bmod 2, \qquad |twoSz| \le twoS \le N.
$$

Code form:
```python
assert isinstance(twoSz, int)
assert isinstance(twoS, int)
assert N % 2 == twoSz % 2 == twoS % 2
assert abs(twoSz) <= twoS <= N
```

Validation:
- 偶数 `N` 只允许偶数 `twoSz` 和偶数 `twoS`。
- 奇数 `N` 只允许奇数 `twoSz` 和奇数 `twoS`。
- 示例：
  - `N=4`：合法 `(twoSz, twoS)` 包括 `(0, 0)`、`(0, 2)`、`(-2, 2)`、`(2, 4)`。
  - `N=5`：合法 `(twoSz, twoS)` 包括 `(-1, 1)`、`(1, 3)`、`(-3, 5)`。

## 8) 数值容差 (MUST)

MUST:
- 所有容差必须定义在单一参数表中（代码中的 `ATOL` 字典），
  并从该处引用。禁止硬编码魔数。

| 键 | 值 | 用途 |
|----|------|------|
| `tight` | `1e-10` | 本征值简并分组、相位规范化 |
| `loose` | `1e-6` | $S^2$/$S_z$ 校验、一般浮点比较 |

Code form:
```python
ATOL = {
    "tight": 1e-10,
    "loose": 1e-6,
}
```

## 9) 工程克制 (MUST)

本代码是科研计算代码，不是生产服务。代码必须保持精简，信任调用方。

MUST:
- 仅在系统边界做一次校验（CLI 入口、文件 I/O、用户输入）；
  内部辅助函数必须信任调用方传入的值。
- 不要在每一层重复同一个检查。一旦前置条件已被建立，下游代码视其为已知。
- 不要为了"保险"添加 `try/except`。只有存在具体的恢复路径时才捕获异常。
- 不要为科研工作流不会产生的输入添加回退分支。如果某分支在既定流程中
  不可能被触达，删除它，而不是作为兜底脚手架保留。
- 不要引入可选参数、特性开关或配置项，除非当前规范或工作流明确需要。
- 不要仅为抽象而把简单操作包装成辅助类/辅助函数。一次性代码保持内联。
- 对具有物理意义的量，直接失败（抛异常、触发断言）优于静默修正、默认填充或截断。

Code form:
```text
boundary validation: CLI parsing, file readers, top-level entry points
internal functions: trust arguments, no re-validation
error handling: raise on invalid state, do not mask
new abstractions: only when a real second caller exists
```
