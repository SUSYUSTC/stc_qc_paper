import numpy as np
import config
import matplotlib as mpl
from PIL import ImageColor
fs = 16
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import libformat

kcal_mol = 0.6275

# unit of ns_CCSD and ns_pt is million
ns_CCSD, ns_pt, t_sample, t_STC, t_DLPNO, error_DLPNO = np.loadtxt("data_locality", skiprows=1, usecols=(5, 6, 7, 8, 9, 10)).T
xs = [0, 1, 2, 3]
xticks = ['1x13', '2x8', '3x5', '4x4']
get_ratio = lambda vec: vec / vec[0]


def plot(x, y, label=None, c=None, marker=None, linestyle=None, **kwargs):
    y0 = plt.ylim()[0]
    y0 = [y0] * len(x)
    plt.plot(x, y, lw=0.5, linestyle=linestyle, c='k')
    legend = plt.scatter(x, y, c=c, marker=marker, s=s, label=label, **kwargs)
    return legend


def plot_grid(ax, start_xy, axis1, axis2, n1, n2, **kwargs):
    # plot n1+1 lines parallel to axis2, n2+1 lines parallel to axis1
    for i in range(n1+1):
        start = start_xy + i*axis1
        end = start + axis2 * n2
        ax.plot([start[0], end[0]], [start[1], end[1]], **kwargs)
    for j in range(n2+1):
        start = start_xy + j*axis2
        end = start + axis1 * n1
        ax.plot([start[0], end[0]], [start[1], end[1]], **kwargs)


frameon = False
lw = 2.0
opts_PAH = dict(marker='s', linestyle='-')
opts_hBN = dict(marker='^', linestyle='-')
legend_opts = dict()

fig = plt.figure(figsize=(18, 5.6))
outer = fig.add_gridspec(1, 3, wspace=0.15)
outer.update(left=0.04, right=0.98, bottom=0.18, top=0.94)
ax1 = fig.add_subplot(outer[0])
ax2 = fig.add_subplot(outer[1])
ax3_all = outer[2].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
ax3_top = fig.add_subplot(ax3_all[0])
ax3_bot = fig.add_subplot(ax3_all[1])
s = 90
plt.sca(ax3_top)
bg = np.array(ImageColor.getcolor(config.color_1, "RGB")) / 255
bg = 1 - (1 - bg) * 0.08  # lighten the color
plt.gca().set_facecolor(bg)
c = config.color_STC
plot(xs, ns_CCSD[4:8] / ns_CCSD[0], label='CCSD, PAH', c=c, **opts_PAH)
plot(xs, ns_CCSD[0:4] / ns_CCSD[0], label='CCSD, H-hBN', c=c, **opts_hBN)
plt.text(1.5, 2.2, 'STC-CCSD', c=config.color_1, ha='center', va='center', fontweight="bold")
plt.xlim(-0.3, 3.3)
plt.ylim(0, 4.2)
plt.xticks([])
plt.yticks([0, 1, 2, 3, 4])

plt.title('Normalized $N_\\text{sample}^{\\epsilon}$')

plt.sca(ax3_bot)
bg = np.array(ImageColor.getcolor(config.color_2, "RGB")) / 255
bg = 1 - (1 - bg) * 0.08  # lighten the color
plt.gca().set_facecolor(bg)
c = config.color_STC
plot(xs, ns_pt[4:8] / ns_pt[0], label='(T), PAH', c=c, **opts_PAH)
plot(xs, ns_pt[0:4] / ns_pt[0], label='(T), H-hBN', c=c, **opts_hBN)
plt.text(1.5, 0.7, 'STC-(T)', c=config.color_2, ha='center', va='center', fontweight="bold")

plt.ylim(0, 2)
plt.xticks([0, 1, 2, 3], xticks)
plt.yticks([0, 0.5, 1, 1.5])
s = 100
plt.sca(ax1)
legend_DLPNO_PAH = plot(xs, t_DLPNO[4:8], label='DLPNO, PAH', c=config.color_DLPNO, **opts_PAH)
legend_DLPNO_hBN = plot(xs, t_DLPNO[0:4], label='DLPNO, H-hBN', c=config.color_DLPNO, **opts_hBN)
legend_STC_PAH = plot(xs, t_STC[4:8], label='STC, PAH', c=config.color_STC, **opts_PAH)
legend_STC_hBN = plot(xs, t_STC[0:4], label='STC, H-hBN', c=config.color_STC, **opts_hBN)
plt.xticks([0, 1, 2, 3], xticks)
plt.yscale('log')
fig.legend(handles=[legend_DLPNO_PAH, legend_DLPNO_hBN, legend_STC_PAH, legend_STC_hBN], frameon=frameon, loc="lower center", ncols=4, **legend_opts)
plt.title('Total computation time (min)')

plt.sca(ax2)
plot(xs, error_DLPNO[4:8] * kcal_mol, c=config.color_DLPNO, **opts_PAH)
plot(xs, error_DLPNO[0:4] * kcal_mol, c=config.color_DLPNO, **opts_hBN)
xlim = plt.xlim()
plt.plot(xlim, [0.2] * 2, linestyle='--', c=config.color_STC, lw=3)
plt.xlim(xlim)
plt.text(0.8, 0.23, 'STC target error', c="#d97706", ha='center', va='center', fontweight="bold")
plt.xticks([0, 1, 2, 3], xticks)
plt.yscale('log')
libformat.set_log_ticks(plt.gca().yaxis)
plt.yticks([0.2, 1, 5])
plt.title('Total energy error (kcal/mol)')

ax = plt.gca()
ax.annotate(
    "",
    xy=(2, 0.01),
    xycoords=("data", "axes fraction"),
    xytext=(0.60, 0.32),
    textcoords=ax.transAxes,
    arrowprops=dict(
        arrowstyle="->",
        color="0.5",
        lw=1.5,
        linestyle="--",
    ),
)

plt.ylim(0.15, 5)

# Create inset axes (size relative to parent axes)
ax_ins = inset_axes(
    ax,
    width="60%",      # adjust size here
    height="60%",
    loc="lower center",  # or 'lower left', etc.
    bbox_to_anchor=(0.06, 0, 1, 1),
    bbox_transform=plt.gca().transAxes,
)

img = plt.imread("./hBN_3x5.png")
ax_ins.imshow(img, origin="lower")
start = (-20, 25)
x_space = 58
y_space = 58
plot_grid(ax_ins, start, np.array([x_space, 0]), np.array([y_space * 0.5, y_space * np.sqrt(3) / 2]), 5, 3, c='k', lw=0.5, linestyle='--')
ax_ins.text(220, 230, '3x5 H-hBN', ha='center', va='center')
ax_ins.axis("off")  # Hide axes for the inset
for ax, lab in zip([ax1, ax2, ax3_top, ax3_bot], ['A', 'B', 'C', 'D']):
    ax.text(
        0.02, 0.95, lab,
        transform=ax.transAxes, fontsize=18,
        fontweight="bold",
        va="top",
        ha="left"
    )

plt.tight_layout()
plt.savefig("locality.png")
plt.close()
#plt.show()
