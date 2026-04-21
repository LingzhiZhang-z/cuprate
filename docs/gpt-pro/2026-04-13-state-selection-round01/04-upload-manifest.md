# 上传清单

## 最小上传集

第一轮先传这些，不要贪多：

- `standards/en/02-HAMILTONIAN.md`
- `standards/en/03-SYMMETRY_SECTORS.md`
- `standards/en/04-DOWNFOLDING.md`
- `src/cuprate/downfolding.py`
- `src/cuprate/hubbard.py`
- `src/cuprate/sectors.py`
- `src/cuprate/states.py`
- `tests/test_sector_matched_selection.py`
- `tests/test_mode_eigensystems.py`
- `tests/test_sz0_output_and_projection.py`
- `tests/conftest.py`
- `docs/gpt-pro/2026-04-13-state-selection-round01/03-review-context.md`

## 扩展上传集

只有当 GPT Pro 明确说还需要更多运行时或流程上下文时，再补这些：

- `src/cuprate/io.py`
- `src/cuprate/main/solver.py`
- `src/cuprate/main/workflow.py`
- `src/cuprate/hamiltonian.py`
- `tests/test_mode_routing.py`
- `tests/test_block_diag.py`

## 可选的定量结果上传集

如果第二轮要问 fit error 和选态的关系，再补这些：

- 按 `08-selection-fit-summary-template.csv` 填好的一份结果表
- 一份简短的 markdown 说明，列出你怀疑有问题的参数点
- 小型表格或图，至少比较这些量：
  `workflow`、`T11`、fit relative error、fit residual、selected-state 统计

## 第一轮不要上传

- 整个仓库
- 大的二进制结果文件，除非 GPT Pro 明确要
- 和这次问题无关的 LCE / embed 文件
