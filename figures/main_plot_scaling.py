import numpy as np
import matplotlib as mpl
import config
import libformat
fs = 16
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt


def plot(x, y, label, s=None, fit_slice=slice(None), **kwargs):
    plt.plot(x, y, label=label, **kwargs)
    plt.scatter(x, y, s=s, **kwargs)
    a, b = np.polyfit(np.log10(x[fit_slice]), np.log10(y[fit_slice]), 1)
    y_fit = 10**(a * np.log10(x) + b)
    label = f'$N^{{{a:.2f}}}$'
    plt.plot(x, y_fit, '--', label=label, **kwargs)


def raw_plot(x, y, label, s=None, **kwargs):
    plt.scatter(x, y, s=s, **kwargs)
    plt.plot(x, y, label=label, **kwargs)


def CCSD_flops_pyscf(nocc, nvir):
    '''
    copy from https://github.com/pyscf/pyscf/blob/master/pyscf/cc/ccsd.py#L1659
    '''
    return (nocc**3*nvir**2*2 + nocc**2*nvir**3*2 +     # Ftilde
            nocc**4*nvir*2 * 2 + nocc**4*nvir**2*2 +    # Wijkl
            nocc*nvir**4*2 * 2 +                        # Wabcd
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 + nocc**3*nvir**3*2 +
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # Wiabj
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # t1
            nocc**3*nvir**2*2 * 2 + nocc**4*nvir**2*2 +
            nocc*(nocc+1)/2*nvir**4*2 +                 # vvvv
            nocc**2*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 2 +     # t2
            nocc**3*nvir**3*2 +
            nocc**3*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 4)      # Wiabj


ns, nsamples_local, nsamples_local_critical, nsamples_canonical, nsamples_canonical_critical, nsamples_pt= np.loadtxt("./Nsample_water").T

lw = 2.0
s1 = 30
s2 = 80
alpha = 0.5
legend_opts = dict(handlelength=1.0, handletextpad=0.3, columnspacing=0.8, handleheight=2.2, labelspacing=0)
# error = 0.2mH
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

label_nsample = '$N_{\\text{sample}}^{\\epsilon}$'
label_ncritical = '$N_{\\text{critical}}$'

plt.sca(ax1)
plot(ns, nsamples_local, f'Local {label_nsample}', fit_slice=slice(4, None), color=config.color_1, linewidth=lw)
plot(ns, nsamples_local_critical, f'Local {label_ncritical}', fit_slice=slice(7, None), color=config.color_1, linewidth=lw, alpha=alpha)

plot(ns[0:-2], nsamples_canonical[0:-2], f'Can. {label_nsample}', fit_slice=slice(4, None), color=config.color_2, linewidth=lw)
plot(ns[0:-2], nsamples_canonical_critical[0:-2], f'Can. {label_ncritical}', fit_slice=slice(0, None), color=config.color_2, linewidth=lw, alpha=alpha)

handles, labels = ax1.get_legend_handles_labels()

# desired order by index
order = list(range(len(labels))[::2]) + list(range(len(labels))[1::2])

leg = plt.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, ncols=2, loc='lower right', bbox_to_anchor=(1.03, -0.06), **legend_opts)
plt.xscale('log')
plt.gca().xaxis.minorticks_off()
libformat.set_log_ticks(plt.gca().xaxis)
plt.xticks([2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30])
plt.xlim(1.8, 36)
plt.yscale('log')
plt.ylim(6e3, 1e11)
plt.xlabel('$N_{\\text{mol}}$')
plt.title('$N_\\text{sample}^{\\epsilon}$ and $N_\\text{critical}$ of STC-CCSD', fontsize=fs)

from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
ms = MarkerStyle('x')
path = ms.get_path().transformed(
    Affine2D().rotate_deg(20)   # rotate by 20 degrees
)

nocc = 5
nvir = 8
lw = 1.0
plt.sca(ax2)
plot(ns, nsamples_local, 'STC-CCSD', color=config.color_STC, fit_slice=slice(4, None), linewidth=lw, marker='d', s=s1)
plot(ns, nsamples_pt * ns * (5 + 8) / 2 * 36, 'STC-(T)', color=config.color_STC, fit_slice=slice(0, None), linewidth=lw, marker=path, s=s2)
#raw_plot(ns, ns**6 * nocc**2 * nvir**4, 'exact CCSD (leading)', color='C3')
raw_plot(ns, CCSD_flops_pyscf(ns * nocc, ns * nvir), 'Exact CCSD', color=config.color_exact, linewidth=lw, marker='d', s=s1)
plt.plot([2, 3], [1e10, 1e10], c='white', label='  ')  # spacer
raw_plot(ns, ns**7 * nocc**3 * nvir**3 * (nocc + nvir) / 6, 'Exact (T)', color=config.color_exact, linewidth=lw, marker=path, s=s2)
plt.plot([2, 3], [1e10, 1e10], c='white', label='  ')  # spacer

plt.xscale('log')
plt.gca().xaxis.minorticks_off()
libformat.set_log_ticks(plt.gca().xaxis)
plt.xticks([2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30])
plt.yscale('log')
plt.xlabel('$N_{\\text{mol}}$')
plt.title('Total FLOP', fontsize=fs)

handles, labels = ax2.get_legend_handles_labels()
order = list(range(len(labels))[::2]) + list(range(len(labels))[1::2])
plt.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0.02, 0.95), **legend_opts)


plt.sca(ax3)
lw = 2.0
ns, t_Rov, t_mix, t_exact = np.loadtxt("./stats_water").T

raw_plot(ns, t_mix, 'STC-CCSD(T), pure impl', c=config.color_STC, linestyle='--', linewidth=lw)
raw_plot(ns, t_Rov, 'STC-CCSD(T), optimized impl', c=config.color_STC, linestyle='-', linewidth=lw)
raw_plot(ns, t_exact, 'Exact CCSD(T)', c=config.color_exact, linewidth=lw)

plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.95), **legend_opts)
plt.xscale('log')
plt.gca().xaxis.minorticks_off()
libformat.set_log_ticks(plt.gca().xaxis)
plt.xticks(ns)
plt.yscale('log')
plt.xlabel('$N_{\\text{mol}}$')
plt.title('Total computation time (min)', fontsize=fs)

for ax, lab in zip([ax1, ax2, ax3], ['A', 'B', 'C']):
    ax.text(
        0.02, 0.97, lab,
        transform=ax.transAxes, fontsize=18,
        fontweight="bold",
        va="top",
        ha="left"
    )

plt.tight_layout()
plt.savefig("scaling.png")
plt.close()
#plt.show()
