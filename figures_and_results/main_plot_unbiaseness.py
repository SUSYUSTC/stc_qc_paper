import numpy as np
import config
import matplotlib as mpl
fs = 20
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
legend_opts = dict(handlelength=1.2, handletextpad=0.4, columnspacing=0.8)


def gauss(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)


E_CCSD = np.loadtxt("./benzene_energy_CCSD_tz")
E_pt = np.loadtxt("./benzene_energy_pt_tz")

CCSD_ref = -1.0710897
error_CCSD = E_CCSD - CCSD_ref
CCSD_mean = error_CCSD.mean()
CCSD_std = error_CCSD.std()
CCSD_target_error = 0.25e-3

pt_ref = -0.0533265
error_pt = E_pt - pt_ref
pt_mean = error_pt.mean()
pt_std = error_pt.std()
pt_target_error = 0.2e-3

lw = 2.5

print(CCSD_mean, CCSD_std)
print(pt_mean, pt_std)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

plt.sca(ax1)
xlim = (-1, 1)
bins = np.linspace(xlim[0], xlim[1], 21, endpoint=True)
plt.hist(error_CCSD * 1000, bins=bins, color=config.color_STC)
ylim = (0, 425)
plt.plot([CCSD_mean * 1000, CCSD_mean * 1000], ylim, 'k', label='mean', linestyle='--', linewidth=lw)
plt.fill_betweenx((0, 0), -CCSD_std * 0.02 * 1000, CCSD_std * 0.02 * 1000, color='gray', alpha=0.5, label='SEM')

x = np.linspace(-1, 1, 101, endpoint=True)
density = gauss(x / 1000 / CCSD_target_error) * len(E_CCSD) * (xlim[1] - xlim[0]) / 20 / (1000 * CCSD_target_error)
plt.plot(x, density, color='k', label='$N(0, \\epsilon^2)$', linewidth=lw)

plt.xlim((-1.2, 1))
plt.ylim(ylim)
plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.02, 1), **legend_opts)
plt.xlabel('Energy error (m$E_h$)')
plt.ylabel('Distribution (normalized)')
plt.xticks([-0.5, 0, 0.5])
plt.yticks([])
plt.title('STC-CCSD')

ax1_ins = inset_axes(
    ax1,
    width="25%",
    height="25%",
    loc = 'upper left',
    bbox_to_anchor=(0.06, 0, 1, 1),
    bbox_transform=ax1.transAxes,
)

rect, line1, line2 = mark_inset(
    ax1, ax1_ins,
    loc1=1, loc2=3,      # which corners to connect
    fc="none",           # transparent box
    ec="0.4",            # edge color
    lw=1.5,
)
rect.set_linestyle('-')
for c in [line1, line2]:
    c.set_linestyle((0, (1, 1)))

plt.sca(ax1_ins)
plt.plot([CCSD_mean * 1000, CCSD_mean * 1000], ylim, 'k', label='mean', linestyle='--', linewidth=2)
plt.fill_betweenx((0, ylim[1]/5), (CCSD_mean - CCSD_std * 0.02) * 1000, (CCSD_mean + CCSD_std * 0.02) * 1000, color='gray', alpha=0.5, label='$\\pm \\frac{\\sigma}{\\sqrt{n}}$')
plt.ylim((0, ylim[1]/10))
plt.yticks([])
plt.xlim(-0.02, 0.02)
plt.xticks([-0.02, 0, 0.02], ['-0.02', '0', '0.02'])

plt.sca(ax2)
xlim = (-0.8, 0.8)
bins = np.linspace(xlim[0], xlim[1], 21, endpoint=True)
plt.hist(error_pt * 1000, bins=bins, color=config.color_STC)
ylim = (0, 425)
plt.plot([pt_mean * 1000, pt_mean * 1000], ylim, 'k', label='mean', linestyle='--', linewidth=lw)
plt.fill_betweenx((0, 0), -pt_std * 0.02 * 1000, pt_std * 0.02 * 1000, color='gray', alpha=0.5, label='$\\pm \\frac{\\sigma}{\\sqrt{n}}$')

x = np.linspace(-0.8, 0.8, 101, endpoint=True)
density = gauss(x / 1000 / pt_target_error) * len(E_pt) * (xlim[1] - xlim[0]) / 20 / (1000 * pt_target_error)
plt.plot(x, density, color='k', linewidth=lw)

plt.xlim((-1, 0.8))
plt.ylim(ylim)
#plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.05, 1))
plt.xlabel('Energy error (m$E_h$)')
plt.xticks([-0.5, 0, 0.5])
plt.yticks([])
plt.title('STC-(T)')

ax2_ins = inset_axes(
    ax2,
    width="25%",
    height="25%",
    loc = 'upper left',
    bbox_to_anchor=(0.06, 0, 1, 1),
    bbox_transform=ax2.transAxes,
)

rect, line1, line2 = mark_inset(
    ax2, ax2_ins,
    loc1=1, loc2=3,      # which corners to connect
    fc="none",           # transparent box
    ec="0.4",            # edge color
    lw=1.5,
)
rect.set_linestyle('-')
for c in [line1, line2]:
    c.set_linestyle((0, (1, 1)))


plt.sca(ax2_ins)
plt.plot([pt_mean * 1000, pt_mean * 1000], ylim, 'k', linestyle='--', linewidth=2)
plt.fill_betweenx((0, ylim[1]/5), (pt_mean - pt_std * 0.02) * 1000, (pt_mean + pt_std * 0.02) * 1000, color='gray', alpha=0.5, label='$\\pm \\frac{\\sigma}{\\sqrt{n}}$')
plt.ylim((0, ylim[1]/10))
plt.yticks([])
plt.xlim(-0.02, 0.02)
plt.xticks([-0.02, 0, 0.02], ['-0.02', '0', '0.02'])

for ax, lab in zip([ax1, ax2], ['A', 'B']):
    ax.text(
        0.05, 0.4, lab,
        transform=ax.transAxes, fontsize=24,
        fontweight="bold",
        va="top",
        ha="left"
    )

plt.tight_layout()
#plt.show()
plt.savefig("unbiaseness.png")
plt.close()
