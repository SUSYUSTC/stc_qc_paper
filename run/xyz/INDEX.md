# run/xyz — Geometry Description

Molecular / periodic-system geometries (`.xyz`) and lattice (`.lattice`) files used in the calculations.

| Subdirectory | Contents |
|--------------|----------|
| `water/` | Water-cluster geometries (`waterN_min.xyz`), used for STC-CCSD scaling and unbiasedness studies |
| `benchmarking/` | Geometries of the molecular benchmark set (vs. DLPNO / exact) |
| `ISOL24/` | Geometries of the 22 systems in the ISOL24 reaction set |
| `cell/` | `.xyz` files of periodic systems (e.g. Si-doped diamond) together with the corresponding `.lattice` lattice files (stored as pairs) |
| `lattice/` | H-hBN and PAH supercell geometries (used for local-orbital and solid wall-time figures) |
| `benzene/` | Benzene single-molecule geometry `benzene.xyz` (used for unbiasedness verification) |

Note: Each `.lattice` file is paired with its corresponding `.xyz` and is used together with the `--pbc` option of the driver script.
