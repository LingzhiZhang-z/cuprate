# Standards Naming Plan

**Goal:** Update the authoritative English standards so naming, CLI grammar, and path grammar are unambiguous before any implementation work.

**Scope:** Standards-only pass. Do not modify `src/` or `tests/` in this step.

## Steps

1. Update `standards/en/00-CONVENTIONS.md` with canonical naming rules:
   - canonical physics names: `twoSz`, `twoS`, `S2`
   - canonical runtime workflow name: `workflow`
   - canonical mode spellings
   - integer/parity/ordering constraints for `twoSz` and `twoS`
2. Update `standards/en/07-RUNTIME.md` with canonical runtime grammar:
   - CLI accepts `twoSz` / `twoS` keys case-insensitively
   - CLI accepts `workflow` case-insensitively
   - canonical `MODE` spellings remain `full`, `fixed_sz`, `block_sz_full`, `fixed_sz_s2`, `block_sz_s2_full`
   - `MODE=all` is accepted as an input alias for `MODE=full`
   - path grammar uses `twoSz_<value>`, `twoS_<value>`, `twoSz_all`, `twoSz_all_twoS_all`
3. Verify the updated standards are internally consistent and identify any remaining standards files that still carry obsolete names.

## Non-Goals

- No compatibility layer design beyond the approved exception `MODE=all -> full`
- No code changes
- No test updates
