# Cuprate 科学标准

## 权威性
代码实现必须遵循 `./standards/en/` 下的英文标准。
`./standards/zh/` 下的文件仅为翻译。
若中英文存在冲突，以英文为准。

## 组织原则
这些标准按物理流水线组织：每个文件覆盖计算的一个阶段，
从团簇几何到超胞嵌入。它们不按当前 Python 模块布局组织。

## 标准列表

| # | 文件 | 范围 |
|---|------|------|
| 00 | `00-CONVENTIONS.md` | 书写规则、符号表、命名约定、数值容差 |
| 01 | `01-CLUSTERS.md` | 方格子枚举、加权图分类、键类型、多格点模式 |
| 02 | `02-HAMILTONIAN.md` | Fock 态编码、基矢构建、单带 Hubbard 哈密顿量、对角化 |
| 03 | `03-SYMMETRY_SECTORS.md` | $S_z$ 分块、$S^2$ 基矢变换、谱重构、五种模式 |
| 04 | `04-DOWNFOLDING.md` | 本征态选择、$T_{11}$、SVD 降维至 $H_{\text{eff}}$、自旋耦合拟合 |
| 05 | `05-LCE_AND_EMBEDDING.md` | 链接团簇展开、超胞嵌入 |
| 06 | `06-RUNTIME.md` | Workchain、solve cache、派生输出归属 |
| 07 | `07-TESTING.md` | 回归测试策略、参考数据、命名转换 |
| 08 | `08-OPERATOR_OUTPUT.md` | 自旋耦合算符、规范配对排序、输出格式 |

## 代码 ↔ 标准映射

| 代码模块 | 主要标准 |
|----------|----------|
| `clusters.py` | 01-CLUSTERS |
| `states.py` | 02-HAMILTONIAN |
| `hubbard.py` | 02-HAMILTONIAN, 03-SYMMETRY_SECTORS, 04-DOWNFOLDING |
| `sectors.py` | 03-SYMMETRY_SECTORS |
| `manifold.py` | 04-DOWNFOLDING |
| `mpi.py` | 06-RUNTIME |

`src/cuprate/back/` 包含旧 workchain/LCE/embed 代码，仅供参考。
它不是 active 模块布局，也不得定义生产兼容性要求。

## 建议阅读顺序
1. `00-CONVENTIONS.md`
2. `01-CLUSTERS.md`
3. `02-HAMILTONIAN.md`
4. `03-SYMMETRY_SECTORS.md`
5. `04-DOWNFOLDING.md`
6. `05-LCE_AND_EMBEDDING.md`
7. `06-RUNTIME.md`
8. `07-TESTING.md`
9. `08-OPERATOR_OUTPUT.md`
