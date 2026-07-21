# figures_and_results — Directory Overview

An archive of all plotting scripts, processed data, generated figures, and input assets used in the paper.

## Directory Structure

- `scripts/` — plotting scripts for the main-text figures, plus shared modules (`config.py` / `libformat.py`)
- `data/` — processed data files for plotting (no extension; plain-text numeric tables)
- `figures/` — main-text generated figures (scaling / benchmarking / isol24 / locality / solid / unbiaseness)
- `assets/` — input image assets for plotting (e.g. the H-hBN supercell schematic `hBN_3x5.png`)
- `diagrams/` — method schematics (tensor contraction / (T), etc.), generated as `diagram_*.png` by `plot_diagram*.py`
- `results_benchmarking/` — data and MAE computation scripts for molecular benchmarks (vs. DLPNO / exact)
- `SI/` — scripts and data for supplementary-information figures (convergence, canonical/Haar scaling)

## scripts/ Description

| File | Purpose | Reads data | Output figure |
|------|---------|------------|---------------|
| `config.py` | Color/font configuration + path constants `DATA_DIR`/`FIG_DIR`/`ASSET_DIR` | — | — |
| `libformat.py` | Plotting helper functions (log-scale, etc.) | — | — |
| `main_plot_scaling.py` | Figure: `N_sample` and wall-time scaling of water clusters with system size | `Nsample_water`, `stats_water` | `scaling.png` |
| `main_plot_benchmarking.py` | Molecular benchmark: error/wall-time vs. DLPNO & exact | `stats_all` | `benchmarking.png` |
| `main_plot_isol24.py` | ISOL24 reaction-set benchmark | `stats_isol24` | `isol24.png`, `isol24.pdf` |
| `main_plot_locality.py` | Local-orbital performance (PAH / H-hBN) | `data_locality` (+ `hBN_3x5.png`) | `locality.png` |
| `main_plot_solid.py` | Wall-time scaling for periodic Si-doped diamond | (hard-coded values) | `solid.png` |
| `main_plot_unbiaseness.py` | Unbiasedness verification on benzene (CCSD / (T) error distributions) | `benzene_energy_CCSD_tz`, `benzene_energy_pt_tz` | `unbiaseness.png` |

**Run convention**: The data/image/asset paths inside the scripts are resolved via `config.data()` / `config.fig()` / `config.asset()` to absolute paths relative to the script location, so the files are found correctly regardless of the working directory from which the script is run.

## data/ Data Files

| File | Purpose |
|------|---------|
| `Nsample_water` | Number of samples required for water clusters to reach the target error |
| `stats_water` | Wall-time per realization for water clusters |
| `stats_all` | Error/wall-time summary for the molecular benchmark set |
| `stats_isol24` | Error/wall-time for the ISOL24 reaction set |
| `data_locality` | `N_sample` and wall-time for PAH / H-hBN supercells of different sizes |
| `benzene_energy_CCSD_tz` | STC-CCSD energy samples for benzene at cc-pVTZ |
| `benzene_energy_pt_tz` | STC-(T) energy samples for benzene at cc-pVTZ |
