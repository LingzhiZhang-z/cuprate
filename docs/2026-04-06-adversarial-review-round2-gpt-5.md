## CRITICAL (must fix before merge)

- Tracked code currently depends on untracked modules under `src/`. `git ls-files src/cuprate/downfolding.py src/cuprate/hamiltonian.py src/cuprate/sectors.py src/cuprate/states.py src/cuprate/hubbard.py src/cuprate/main/__main__.py` returns only `src/cuprate/hubbard.py` and `src/cuprate/main/__main__.py` as tracked, while `git ls-files --others --exclude-standard src` lists `src/cuprate/downfolding.py`, `src/cuprate/hamiltonian.py`, `src/cuprate/sectors.py`, and `src/cuprate/states.py` as untracked. Those untracked modules are imported by tracked code at `src/cuprate/hubbard.py:18`, `src/cuprate/hubbard.py:24`, `src/cuprate/hubbard.py:25`, and `src/cuprate/hubbard.py:26`. If merged as-is, the tracked tree is incomplete.

## WARNING (should fix soon)

- The authoritative English standards still describe removed further-neighbor hopping and deleted bond-classification APIs. Specific lines:
  `standards/en/01-CLUSTERS.md:10`
  `standards/en/01-CLUSTERS.md:11`
  `standards/en/01-CLUSTERS.md:12`
  `standards/en/01-CLUSTERS.md:13`
  `standards/en/01-CLUSTERS.md:32`
  `standards/en/01-CLUSTERS.md:54`
  `standards/en/01-CLUSTERS.md:61`
  `standards/en/01-CLUSTERS.md:76`
  `standards/en/01-CLUSTERS.md:87`
  `standards/en/01-CLUSTERS.md:92`
  `standards/en/01-CLUSTERS.md:93`
  `standards/en/01-CLUSTERS.md:101`
  `standards/en/02-HAMILTONIAN.md:147`
  `standards/en/02-HAMILTONIAN.md:154`
  `standards/en/02-HAMILTONIAN.md:156`
  `standards/en/02-HAMILTONIAN.md:163`
  `standards/en/02-HAMILTONIAN.md:164`
  `standards/en/06-RUNTIME.md:36`
  `standards/en/06-RUNTIME.md:75`

- Additional stale references remain elsewhere in standards: `standards/en/00-CONVENTIONS.md:37`, `standards/en/05-LCE_AND_EMBEDDING.md:28`, and the corresponding Chinese translations listed in the scan section.

## INFO (noted)

- Import consistency for deleted bond helpers is clean in code: the exact import scan found no `src/` or `tests/` imports of `canonical_bond_type`, `canonical_bond`, `get_min_lexicographic_bond`, `_matching_two_site_bond_group`, `nnn_bonds`, or `third_neighbor`. `src/cuprate/hubbard.py:17-30` also does not import `cuprate.mpi`; the only `cuprate.mpi` imports found were `src/cuprate/main/__main__.py:11` and `src/cuprate/downfolding.py:17`.

- The simplified S2 flow is implemented as requested. `src/cuprate/hubbard.py:310-312` makes `save_blocks()` a no-op. `src/cuprate/hubbard.py:314-331` makes `load_blocks()` compute the S2 basis directly in memory via `sectors.solve_S2_blocks(states)` with no cache/file access. In `src/cuprate/main/__main__.py:655-660`, `cluster_process()` orders the calls as `set_states()` -> `load_blocks()` -> `construct_transform_matrix()`. `cluster_process()` does not call `save_blocks()`, but since `save_blocks()` is a no-op this has no behavioral effect.

- The NN-only hopping application path is internally consistent. `src/cuprate/clusters.py:261-273` sorts two-site groups by bond length, so NN `(1,0)` groups come before longer pairs. `src/cuprate/clusters.py:388-397` appends those two-site groups before any four-site or six-site groups. `src/cuprate/main/__main__.py:152-156` then applies the first two-site group only. Direct runtime check on `cluster = [(0,0), (1,0), (2,0)]` returned `[[ (0, 1), (1, 2) ], [ (0, 2) ]]`, so the first two-site group is the NN set.

## Test results

Command run:

```text
python -m pytest tests/ -x -q
```

Full output:

```text
..........................................                               [100%]
42 passed in 5.96s
```

## Remaining references scan

Exact-term scan command:

```text
rg -n -P "\b(?:t2|t3|max_bond|delta2|NNN|NNNN|nnn_bonds|third_neighbor|canonical_bond_type|canonical_bond|rotate_90|get_min_lexicographic_bond|_matching_two_site_bond_group)\b|next-nearest" src tests standards
```

Exact matches:

```text
standards/en/02-HAMILTONIAN.md:147:  Assigned by bond type: $t$ for NN, $t_2$ for NNN, $t_3$ for 3rd neighbor.
standards/en/02-HAMILTONIAN.md:154:- Bonds are classified by `canonical_bond_type(cluster, pair)`:
standards/en/02-HAMILTONIAN.md:156:  - Type 2 (NNN): bond vector $[1, 1]$ → hopping $t_2$.
standards/en/02-HAMILTONIAN.md:163:model.set_bonds_by_class(nnn_bonds, params["t2"])    # skipped if t2 is None
standards/en/02-HAMILTONIAN.md:164:model.set_bonds_by_class(third_bonds, params["t3"])   # skipped if t3 is None
standards/en/00-CONVENTIONS.md:37:| $t, t_2, t_3$ | NN, NNN, 3rd-neighbor hopping amplitudes |
standards/en/01-CLUSTERS.md:10:- Connectivity is determined by `max_bond`:
standards/en/01-CLUSTERS.md:11:  - `max_bond=1`: NN only — $(1,0)$ family.
standards/en/01-CLUSTERS.md:12:  - `max_bond=2`: NN + NNN — adds $(1,1)$ family.
standards/en/01-CLUSTERS.md:13:  - `max_bond=3`: NN + NNN + 3rd — adds $(2,0)$ family.
standards/en/01-CLUSTERS.md:32:generate_clusters(N, max_bond)          # DFS enumeration
standards/en/01-CLUSTERS.md:54:  - `2` = NNN bond (distance $\sqrt{2}$)
standards/en/01-CLUSTERS.md:61:adj = generate_adjacency_matrix(sites, max_bond)
standards/en/01-CLUSTERS.md:76:classify_isomorphic_clusters(clusters, max_bond)
standards/en/01-CLUSTERS.md:87:- `generate_bonds(cluster, max_bond)` returns a list of `max_bond` sublists.
standards/en/01-CLUSTERS.md:92:bonds = generate_bonds(cluster, max_bond)
standards/en/01-CLUSTERS.md:93:# bonds[0] = NN pairs, bonds[1] = NNN pairs, bonds[2] = 3rd-neighbor pairs
standards/en/01-CLUSTERS.md:101:- This produces bond classes for the spin-coupling fit (e.g. all NN bonds, all NNN bonds, etc.).
standards/en/06-RUNTIME.md:36:  where `base_dir = U{U:.4f}_t{t:.4f}[_tp{t2:.4f}]`
standards/en/06-RUNTIME.md:75:  computed from `t_previous = t - delta`, `t2_previous = t2 - delta2`.
standards/en/05-LCE_AND_EMBEDDING.md:28:match = find_cluster_match(subgraph, clusters, t2)
standards/zh/02-HAMILTONIAN.md:149:  按键类型赋值：$t$ 为 NN，$t_2$ 为 NNN，$t_3$ 为第三近邻。
standards/zh/02-HAMILTONIAN.md:156:- 键由 `canonical_bond_type(cluster, pair)` 分类：
standards/zh/02-HAMILTONIAN.md:158:  - 类型 2（NNN）：键向量 $[1, 1]$ → 跳跃 $t_2$。
standards/zh/02-HAMILTONIAN.md:165:model.set_bonds_by_class(nnn_bonds, params["t2"])    # t2 为 None 时跳过
standards/zh/02-HAMILTONIAN.md:166:model.set_bonds_by_class(third_bonds, params["t3"])   # t3 为 None 时跳过
standards/zh/06-RUNTIME.md:36:  其中 `base_dir = U{U:.4f}_t{t:.4f}[_tp{t2:.4f}]`，
standards/zh/06-RUNTIME.md:75:  前一参数由 `t_previous = t - delta`、`t2_previous = t2 - delta2` 计算。
standards/zh/01-CLUSTERS.md:9:- 连通性由 `max_bond` 确定：
standards/zh/01-CLUSTERS.md:10:  - `max_bond=1`：仅最近邻（NN）— $(1,0)$ 族。
standards/zh/01-CLUSTERS.md:11:  - `max_bond=2`：NN + 次近邻（NNN）— 增加 $(1,1)$ 族。
standards/zh/01-CLUSTERS.md:12:  - `max_bond=3`：NN + NNN + 第三近邻 — 增加 $(2,0)$ 族。
standards/zh/01-CLUSTERS.md:31:generate_clusters(N, max_bond)          # DFS 枚举
standards/zh/01-CLUSTERS.md:53:  - `2` = NNN 键（距离 $\sqrt{2}$）
standards/zh/01-CLUSTERS.md:60:adj = generate_adjacency_matrix(sites, max_bond)
standards/zh/01-CLUSTERS.md:75:classify_isomorphic_clusters(clusters, max_bond)
standards/zh/01-CLUSTERS.md:86:- `generate_bonds(cluster, max_bond)` 返回 `max_bond` 个子列表。
standards/zh/01-CLUSTERS.md:91:bonds = generate_bonds(cluster, max_bond)
standards/zh/01-CLUSTERS.md:92:# bonds[0] = NN 对, bonds[1] = NNN 对, bonds[2] = 第三近邻对
standards/zh/01-CLUSTERS.md:101:- 这为自旋耦合拟合产生键类别（如所有 NN 键、所有 NNN 键等）。
standards/zh/05-LCE_AND_EMBEDDING.md:28:match = find_cluster_match(subgraph, clusters, t2)
```

No exact-term matches were found under `src/` or `tests/`.
