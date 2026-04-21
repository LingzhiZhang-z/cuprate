# Symmetry Sectors Rewrite Plan

> For Opus: read the current code and standards first, then rewrite the sector logic from the physics picture below. Do not preserve the current structure just because it already exists.

## Status

Current `sectors.py` is unsatisfactory. The present code is not simple enough, not clear enough, and not easy enough to read.

## Goal

Rewrite the symmetry-sector construction so the main logic is physically obvious, short, and easy to audit.

Target:
- the core sector code should be within about 200 lines total
- the main flow should be readable top-to-bottom without jumping through many helpers
- no compatibility layer
- no preserving old local naming just because it existed before

## Physics Picture

Only follow this picture:

1. Generate all Fock states for fixed `N` and `nelec`.
2. Sort once with the canonical basis order already used by the project.
3. Group states by `(twoSz, D)`.
4. For each non-negative `twoS`, use the block `(twoSz = twoS, D)` and construct `S_plus`.
5. The null space of that `S_plus` block is the highest-weight space for `(twoS, D)`.
6. Use `S_minus` within the same `D` block to generate the remaining states of the multiplet at lower `twoSz`.
7. For each `(twoSz, twoS)` sector, concatenate columns from different `D` in ascending `D`, so `D = 0` comes first.

This is the whole physical idea. Do not add extra abstraction unless it is clearly necessary.

## Implementation Rules

- Prefer a small number of direct functions.
- The expected main functions are roughly:
  - one grouping function
  - one highest-weight construction function
  - one lowering/assembly function
- `build_S2_sectors(...)` should be a thin wrapper over that logic.
- Reconstruction helpers may remain separate, but the core sector construction must stay compact.
- Do not design around old interfaces.
- If `hubbard.py` or nearby code must change to fit the cleaner structure, change it.

## What Not To Do

- Do not keep old data-flow shapes just to avoid touching callers.
- Do not add tie-break machinery unless it is physically required.
- Do not over-factor simple loops into many tiny helpers.
- Do not describe implementation through old `sz` naming if `twoSz` naming is clearer.

## Acceptance

The rewrite is acceptable only if all of the following are true:

- The code follows the physics picture above directly.
- The core sector code is concise and easy to read.
- The resulting transforms still satisfy the expected `fourS2` projection property.
- `fixed_sz_s2` and `block_sz_s2_full` still work.
- The new code is simpler than the current code, not just different.
