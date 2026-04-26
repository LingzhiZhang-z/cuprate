# Cuprate Scientific Standards

## Authority
Code implementation MUST follow the English standards under `./standards/en/`.
Files under `./standards/zh/` are translations only.
If EN and ZH conflict, EN is authoritative.

## Organizing Principle
These standards are organized by the physics pipeline: each file covers one
stage of the computation, from cluster geometry through LCE and embed output.
They are not organized by the current Python module layout.

## Standards

| # | File | Scope |
|---|------|-------|
| 00 | `00-CONVENTIONS.md` | Writing rules, symbol table, naming conventions, numerical tolerances |
| 01 | `01-CLUSTERS.md` | Square-lattice enumeration, weighted-graph classification, bond types, multi-site patterns |
| 02 | `02-HAMILTONIAN.md` | Fock-state encoding, basis construction, single-band Hubbard Hamiltonian, diagonalisation |
| 03 | `03-SYMMETRY_SECTORS.md` | $S_z$ blocking, $S^2$ basis transform, spectrum reconstruction, diagonalisation modes |
| 04 | `04-DOWNFOLDING.md` | Eigenstate selection, $T_{11}$, $H_{\text{eff}}$ via SVD, spin-coupling fit |
| 05 | `05-LCE_AND_EMBEDDING.md` | Linked-cluster expansion, embed output |
| 06 | `06-RUNTIME.md` | Workchain, solve cache, derived-output ownership |
| 07 | `07-TESTING.md` | Regression testing strategy, reference data, naming translation |
| 08 | `08-OPERATOR_OUTPUT.md` | Spin-coupling operators, canonical pairing order, output formats |

## Code ↔ Standards Mapping

| Code module | Primary standard |
|-------------|-----------------|
| `clusters.py` | 01-CLUSTERS |
| `states.py` | 02-HAMILTONIAN |
| `hubbard.py` | 02-HAMILTONIAN, 03-SYMMETRY_SECTORS, 04-DOWNFOLDING |
| `sectors.py` | 03-SYMMETRY_SECTORS |
| `manifold.py` | 04-DOWNFOLDING |
| `mpi.py` | 06-RUNTIME |
| `workchain.py` | 06-RUNTIME, 08-OPERATOR_OUTPUT |
| `operators.py` | 08-OPERATOR_OUTPUT, 05-LCE_AND_EMBEDDING |
| `lce.py` | 05-LCE_AND_EMBEDDING, 06-RUNTIME |
| `embed.py` | 05-LCE_AND_EMBEDDING, 06-RUNTIME |

`src/cuprate/back/` contains old workchain/LCE/embed code for reference only.
It is not the active module layout and must not define production compatibility
requirements.

## Required Reading Order
1. `00-CONVENTIONS.md`
2. `01-CLUSTERS.md`
3. `02-HAMILTONIAN.md`
4. `03-SYMMETRY_SECTORS.md`
5. `04-DOWNFOLDING.md`
6. `05-LCE_AND_EMBEDDING.md`
7. `06-RUNTIME.md`
8. `07-TESTING.md`
9. `08-OPERATOR_OUTPUT.md`
