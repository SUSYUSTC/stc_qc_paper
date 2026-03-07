import numpy as np
import config
import matplotlib as mpl
fs = 16
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt
import libformat
ns_CCSD, ns_pt, t_sample, t_STC, t_DLPNO, error_DLPNO = np.loadtxt("data_locality", skiprows=1, usecols=(5, 6, 7, 8, 9, 10)).T
error_DLPNO /= 0.6275
xs = [0, 1, 2, 3]
xticks = ['1x13', '2x8', '3x5', '4x4']
get_ratio = lambda vec: vec / vec[0]
s = 60


def plot_scaling(x, y, label, fit_slice=slice(None), **kwargs):
    plt.plot(x, y, label=label, **kwargs)
    plt.scatter(x, y, s=s, **kwargs)
    a, b = np.polyfit(np.log10(x[fit_slice]), np.log10(y[fit_slice]), 1)
    y_fit = 10**(a * np.log10(x) + b)
    label = f'$N^{{{a:.2f}}}$'
    plt.plot(x, y_fit, '--', label=label, **kwargs)


def plot(*args, label=None, linestyle=None, **kwargs):
    plt.plot(*args, label=label, linestyle=linestyle, **kwargs)
    plt.scatter(*args, s=s, **kwargs)


ns = np.array([4, 8, 12, 18, 24])
xlabel = ['1x2x2', '2x2x2', '2x2x3', '2x3x3', '2x3x4']
time_STC = np.array([0.227, 0.73, 2.65, 11.0, 52.5])
time_exact = np.array([0.295, 5.80, 48.6, 560, 5364])
plot_scaling(ns, time_STC, 'STC-CCSD(T)', c=config.color_STC, fit_slice=slice(1, None))
plot(ns, time_exact, label='Exact-CCSD(T)', c=config.color_exact)
plt.legend(frameon=False)
plt.xscale('log')
plt.yscale('log')
libformat.set_log_ticks(plt.gca().xaxis)
plt.xticks(ns, xlabel)
plt.gca().xaxis.minorticks_off()
plt.xlabel('Supercell size')
plt.title('Total computation time (min)')

plt.tight_layout()
plt.savefig("solid.png")
#plt.show()
