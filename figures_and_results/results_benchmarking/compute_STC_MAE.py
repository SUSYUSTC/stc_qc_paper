import numpy as np
from pathlib import Path

systems = np.loadtxt("stats_DF_exact", skiprows=1, usecols=0, dtype=str)
e_corr = np.loadtxt("stats_DF_exact", skiprows=1, usecols=2)
ref = dict(zip(systems, e_corr))

kcalmol = 0.628

for p in sorted(Path("STC_Ecorr").iterdir()):
    if not p.is_file() or p.name not in ref:
        continue
    stc_energies = np.loadtxt(p)
    mae = np.mean(np.abs(stc_energies - ref[p.name]))
    print(p.name, mae * 1000 * kcalmol)
