import numpy as np
import config
import matplotlib as mpl
fs = 20
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)

import matplotlib.pyplot as plt

get_mean = lambda x: np.mean(x)
get_geo_mean = lambda x: np.exp(np.mean(np.log(x)))

Nao = np.loadtxt("./stats_all", skiprows=1, usecols=2).astype(int)
order = np.argsort(Nao)
Nao = Nao[order]

molecules = np.loadtxt("./stats_all", skiprows=1, usecols=0, dtype=str)[order]
molecules = [mol.replace("-", " ") for mol in molecules]
y = np.arange(len(molecules))

err_dlpno_normal = np.abs(np.loadtxt("./stats_all", skiprows=1, usecols=3))[order]
err_dlpno_tight  = np.abs(np.loadtxt("./stats_all", skiprows=1, usecols=5))[order]
err_stc          = np.abs(np.loadtxt("./stats_all", skiprows=1, usecols=7))[order]

time_dlpno_normal = np.loadtxt("./stats_all", skiprows=1, usecols=4)[order]
time_dlpno_tight  = np.loadtxt("./stats_all", skiprows=1, usecols=6)[order]
time_stc          = np.loadtxt("./stats_all", skiprows=1, usecols=8)[order]

time_b3lyp        = np.loadtxt("./stats_all", skiprows=1, usecols=10)[order]
time_wb97m_v      = np.loadtxt("./stats_all", skiprows=1, usecols=11)[order]

err_exact = np.zeros_like(err_stc)  # zero error for exact
time_exact          = np.loadtxt("./stats_all", skiprows=1, usecols=9)[order]

color_STC = config.color_STC
color_normal = config.color_2
color_tight = config.color_DLPNO
color_exact = '#777777'

bar_h = 0.18
space = 1.2
offsets = np.array([0, 1, 2, 3]) * bar_h * space

fig, (ax_err, ax_mid, ax_time) = plt.subplots(
    1, 3,
    figsize=(12, 18),
    sharey=True,
    gridspec_kw=dict(width_ratios=[3, 2.4, 3]),
    constrained_layout=True,
)
fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)

bar_stc = ax_err.barh(y + offsets[0], err_stc,          height=bar_h, label="STC", color=color_STC)
bar_dlpno_normal = ax_err.barh(y + offsets[1], err_dlpno_normal, height=bar_h, label="DLPNO/Normal", color=color_normal)
bar_dlpno_tight = ax_err.barh(y + offsets[2], err_dlpno_tight,  height=bar_h, label="DLPNO/Tight", color=color_tight)
#bar_exact = ax_err.barh(y + offsets[3], err_exact,        height=bar_h, label="Exact", color=color_exact)

for yi, err in zip(y, err_dlpno_normal):
    if err > 3:
        ax_err.text(3.3, yi + offsets[1], f"{err:.1f}", va='center', ha='left', fontsize=fs * 0.8)

ax_err.invert_xaxis()
ax_err.set_xlabel("Energy error (kcal/mol)")
ax_err.invert_yaxis()
ax_err.set_yticks([])

bar_stc = ax_time.barh(y + offsets[0], time_stc,          height=bar_h, color=color_STC)
bar_dlpno_normal = ax_time.barh(y + offsets[1], time_dlpno_normal, height=bar_h, color=color_normal)
bar_dlpno_tight = ax_time.barh(y + offsets[2], time_dlpno_tight,  height=bar_h, color=color_tight)
bar_exact = ax_time.barh(y + offsets[3], time_exact,        height=bar_h, color=color_exact)

ax_time.set_xlabel("Computation time (min)")
ax_time.set_yticks([])

x_name = 0.05
x_nao = 1
y_shift = 0.3

text_kwargs_left = dict(ha="left", va="center", in_layout=False)
text_kwargs_right = dict(ha="right", va="center", in_layout=False)
ax_mid.text(x_name, -1 + y_shift, 'System', **text_kwargs_left, fontweight='bold')
for yi, mol in zip(y, molecules):
    ax_mid.text(x_name, yi + y_shift, mol, **text_kwargs_left)

ax_mid.text( x_nao, -1 + y_shift, '$\\boldsymbol{N}_{\\bf{basis}}$', **text_kwargs_right, fontweight='bold')
for yi, nao in zip(y, Nao):
    ax_mid.text(x_nao, yi + y_shift, str(nao), **text_kwargs_right)
ax_mid.set_xlim(0, 1)
ax_mid.set_xticks([])

ax_time.set_xscale('log')
ax_time.set_xlim(1, 12000)
ax_time.set_xticks([1, 10, 100, 1000, 10000])

ax_mid.axis('off')
ax_err.spines["top"].set_visible(False)
ax_err.spines["left"].set_visible(False)
ax_time.spines["top"].set_visible(False)
ax_time.spines["right"].set_visible(False)

ax_err.spines["right"].set_visible(False)
ax_time.spines["left"].set_visible(False)

real_ylim = (20, -2.5)

ylim = (20, -0.3)
line_target,  = ax_err.plot([0.2, 0.2], ylim, color="#d97706", linestyle='--', linewidth=2, label='STC target error')

legend_opts = dict()

legend = ax_time.legend([line_target, bar_stc, bar_dlpno_normal, bar_dlpno_tight, bar_exact], ["STC target error", "STC", "DLPNO/Normal", "DLPNO/Tight", "Exact"], ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(-0.5, 1), **legend_opts)
legend.set_in_layout(False)


ax_time.set_ylim(real_ylim)
ax_err.set_ylim(real_ylim)
ax_err.set_xlim((3, 1e-4))

for x in [1, 10, 100, 1000, 10000]:
    ax_time.plot((x, x), ylim, color="gray", lw=1, alpha=0.5, linestyle='--')


def add_serrated_edge(ax, x0, n_teeth=6, width=0.02):
    """
    Add a serrated 'axis break' at the right edge of an axis.
    Coordinates are in axes fraction.
    """
    y = np.linspace(*ylim, num=n_teeth * 2 + 1)
    x1 = np.ones_like(y) * x0
    x2 = np.ones_like(y) * x0
    x1[1::2] -= width

    ax.fill_betweenx(y, x1, x2, color='white')
    ax.plot(x1, y, color='k', lw=1.2, clip_on=False)

    ax.set_ylim(real_ylim)


add_serrated_edge(ax_err, 3.0, n_teeth=120, width=0.05)
plt.savefig("benchmarking.png")
plt.close()

print('DLPNO/Normal', get_mean(err_dlpno_normal), get_geo_mean(time_dlpno_normal))
print('DLPNO/Tight', get_mean(err_dlpno_tight), get_geo_mean(time_dlpno_tight))
print('STC', get_mean(err_stc), get_geo_mean(time_stc))
print('Exact', get_mean(err_exact), get_geo_mean(time_exact))
print('B3LYP', 'N/A', get_geo_mean(time_b3lyp))
print('wB97M-V', 'N/A', get_geo_mean(time_wb97m_v))
