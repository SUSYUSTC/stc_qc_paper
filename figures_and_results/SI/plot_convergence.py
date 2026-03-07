import numpy as np
import matplotlib as mpl
mpl.rc('font', size=14)
import matplotlib.pyplot as plt

# water dimer, cc-pvdz, CCD
stats_40k = np.loadtxt("./convergence_40k_stats")[0:20]
stats_80k = np.loadtxt("./convergence_80k_stats")[0:20]
stats_160k = np.loadtxt("./convergence_160k_stats")[0:20]

stats_40k = stats_40k[stats_40k[:, 1] < 0.1]

E_ref = -0.427307629

for stats, label in zip([stats_40k, stats_160k], ['40k samples', '160k samples']):
    y, yerr = stats.T
    plt.errorbar(np.arange(len(y)), y, yerr=yerr, label=label)
plt.axhline(E_ref, color='k', linestyle='--', label='Exact energy')
plt.legend()

plt.xlabel('STC-CCSD iteration')
plt.ylabel('Energy ($E_h$)')
plt.title('STC-CCSD energy by iteration')
plt.tight_layout()
plt.savefig('convergence_plot_SI.png')
#plt.show()
plt.close()
