import matplotlib
matplotlib.rc('font', size=14)
import matplotlib.pyplot as plt
import numpy as np
import libformat


def plot(x, y, label, fit_slice=slice(None), **kwargs):
    plt.plot(x, y, label=label, **kwargs)
    plt.scatter(x, y, **kwargs)
    a, b = np.polyfit(np.log10(x[fit_slice]), np.log10(y[fit_slice]), 1)
    y_fit = 10**(a * np.log10(x) + b)
    label = f'fit: $N^{{{a:.2f}}}$'
    plt.plot(x, y_fit, '--', label=label, **kwargs)


def raw_plot(x, y, label, **kwargs):
    plt.plot(x, y, label=label, **kwargs)
    plt.scatter(x, y, **kwargs)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ns = np.loadtxt("./canonical_norm", usecols=0, dtype=int)
E, fake_L1, true_L1 = np.loadtxt("./canonical_norm", usecols=[1, 2, 3]).T
_, fake_L1_haar, true_L1_haar = np.loadtxt("./haar_random_norm", usecols=[1, 2, 3]).T

plt.sca(ax1)
plot(ns, fake_L1_haar**2 - E**2, 'Haar-random, opt', c='C1', fit_slice=slice(2, None))
plot(ns, true_L1_haar**2 - E**2, 'Haar-random, LB', c='C3', fit_slice=slice(2, None))
raw_plot(ns, fake_L1**2 - E**2, 'Canonical, opt', c='C0')
raw_plot(ns, true_L1**2 - E**2, 'Canonical, LB', c='C2')
plt.ylim(1e-5, 2)

plt.legend(frameon=False)
plt.xscale('log')
plt.xticks(ns)
libformat.set_log_ticks(plt.gca().xaxis)
plt.yscale('log')
plt.xlabel('Number of water molecules')
plt.title('Variance of $\\mathcal{C}(V, T, T)$')

ns = np.loadtxt("./canonical_norm2", usecols=0, dtype=int)
E, fake_L1, true_L1 = np.loadtxt("./canonical_norm2", usecols=[1, 2, 3]).T
_, fake_L1_haar, true_L1_haar = np.loadtxt("./haar_random_norm2", usecols=[1, 2, 3]).T

plt.sca(ax2)
plot(ns, fake_L1_haar**2 - E**2, 'Haar-random, opt', c='C1', fit_slice=slice(2, None))
plot(ns, true_L1_haar**2 - E**2, 'Haar-random, LB', c='C3', fit_slice=slice(2, None))
raw_plot(ns, fake_L1**2 - E**2, 'Canonical, opt', c='C0')
raw_plot(ns, true_L1**2 - E**2, 'Canonical, LB', c='C2')
plt.ylim(1e-6, 0.5)

plt.legend(frameon=False)
plt.xscale('log')
plt.xticks(ns)
libformat.set_log_ticks(plt.gca().xaxis)
plt.yscale('log')
plt.xlabel('Number of water molecules')
plt.title('Variance of $\\mathcal{C}(V, T, T, T)$')

plt.tight_layout()
plt.savefig('Haar-canonical-scaling_SI.png')
plt.close()
