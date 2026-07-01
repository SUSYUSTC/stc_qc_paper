import numpy as np
import config
import matplotlib as mpl
fs = 20
#mpl.rc('font', family='serif')
#mpl.rc('font', weight='bold')
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)

import matplotlib.pyplot as plt

# -------------------------
# data
# -------------------------
get_mean = lambda x: np.mean(x)
get_geo_mean = lambda x: np.exp(np.mean(np.log(x)))

Nao = np.loadtxt("./stats_isol24", skiprows=1, usecols=1).astype(int)
#order = np.argsort(Nao)
order = np.arange(len(Nao))
Nao = Nao[order]

molecules = np.loadtxt("./stats_isol24", skiprows=1, usecols=0, dtype=str)[order]
y = np.arange(len(molecules))

err_dlpno_tight  = np.abs(np.loadtxt("./stats_isol24", skiprows=1, usecols=2))[order]
err_stc          = np.abs(np.loadtxt("./stats_isol24", skiprows=1, usecols=3))[order]

time_dlpno_tight  = np.loadtxt("./stats_isol24", skiprows=1, usecols=4)[order]
time_stc          = np.loadtxt("./stats_isol24", skiprows=1, usecols=5)[order]

print('DLPNO/Tight', get_mean(err_dlpno_tight), get_geo_mean(time_dlpno_tight))
print('STC', get_mean(err_stc), get_geo_mean(time_stc))

color_STC = config.color_STC
color_tight = config.color_DLPNO
#color_exact = config.color_exact

# -------------------------
# layout
# -------------------------
bar_h = 0.3
space = 1.2
offsets = np.array([0, 1]) * bar_h * space

fig, (ax_err, ax_mid, ax_time) = plt.subplots(
    1, 3,
    figsize=(12, 8),
    sharey=True,
    gridspec_kw=dict(width_ratios=[3, 1, 3]),
    constrained_layout=True,
)
fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)

# -------------------------
# LEFT: error
# -------------------------
bar_stc = ax_err.barh(y + offsets[0], err_stc,          height=bar_h, label="STC", color=color_STC)
bar_dlpno_tight = ax_err.barh(y + offsets[1], err_dlpno_tight,  height=bar_h, label="DLPNO/Tight", color=color_tight)
#bar_exact = ax_err.barh(y + offsets[3], err_exact,        height=bar_h, label="Exact", color=color_exact)

ax_err.invert_xaxis()
ax_err.set_xlabel("Energy error (kcal/mol)")
ax_err.invert_yaxis()
ax_err.set_yticks([])

# -------------------------
# MIDDLE: molecule labels
# -------------------------
#ax_mid.set_xlim(0, 1)
#ax_mid.set_xticks([])
#ax_mid.set_yticks([])

bar_stc = ax_time.barh(y + offsets[0], time_stc,          height=bar_h, color=color_STC)
bar_dlpno_tight = ax_time.barh(y + offsets[1], time_dlpno_tight,  height=bar_h, color=color_tight)

ax_time.set_xlabel("Computation time (min)")
ax_time.set_yticks([])

x_name = 0.1
x_nao = 1.0
y_shift = 0.2

text_kwargs_left = dict(ha="left", va="center", in_layout=False)
text_kwargs_right = dict(ha="right", va="center", in_layout=False)
ax_mid.text(x_name, -1 + y_shift, 'ID', **text_kwargs_left, fontweight='bold')
for yi, mol in zip(y, molecules):
    ax_mid.text(x_name, yi + y_shift, mol, **text_kwargs_left)

ax_mid.text( x_nao, -1 + y_shift, '$\\boldsymbol{N}_{\\bf{basis}}$', **text_kwargs_right, fontweight='bold')
for yi, nao in zip(y, Nao):
    ax_mid.text(x_nao, yi + y_shift, str(nao), **text_kwargs_right)
ax_mid.set_xlim(0, 1)
ax_mid.set_xticks([])

#ax_err.axvline(0, color="k", lw=0.8)
#ax_time.axvline(0, color="k", lw=0.8)

#ax_err.set_xlim(3.0, 0.0)
#ax_time.set_xlim(0.0, 130)
ax_time.set_xscale('log')
ax_time.set_xlim(10, 1200)
#ax_time.set_xticks([1, 10, 100, 1000, 10000])

#ax_err.set_frame_on(False)
#ax_time.set_frame_on(False)
#ax_mid.set_frame_on(False)
#ax_nao.set_frame_on(False)

#ax_time.axis('off')
#ax_err.axis('off')
ax_mid.axis('off')
ax_err.spines["top"].set_visible(False)
ax_err.spines["left"].set_visible(False)
ax_time.spines["top"].set_visible(False)
ax_time.spines["right"].set_visible(False)

ax_err.spines["right"].set_visible(False)
ax_time.spines["left"].set_visible(False)

#ax_err.grid(True, axis='x', linestyle='--', alpha=0.4)
#ax_time.grid(True, axis='x', linestyle='--', alpha=0.6)


#ylim = ax_err.get_ylim()
real_ylim = (11, -2.5)

ylim = (11, -0.3)
line_target,  = ax_err.plot([0.2, 0.2], ylim, color="#d97706", linestyle='--', linewidth=2, label='STC target error')

#legend_opts = dict(handlelength=1.5, handletextpad=0.4, columnspacing=1)
legend_opts = dict()

#legend = ax_err.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=2)
legend = ax_time.legend([line_target, bar_stc, bar_dlpno_tight], ["STC target error", "STC", "DLPNO/Tight"], ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(-0.2, 1), **legend_opts)
#legend = fig.legend([bar_stc, bar_dlpno_normal, bar_dlpno_tight, bar_exact], ["STC", "DLPNO/Normal", "DLPNO/Tight", "Exact"], ncol=4, frameon=False, loc='upper center')
legend.set_in_layout(False)


ax_time.set_ylim(real_ylim)
ax_err.set_ylim(real_ylim)
ax_err.set_xlim((0.8, 1e-4))

#for x in [1, 10, 100, 1000, 10000]:
#for x in [200, 400, 600, 800, 1000]:
#    ax_time.plot((x, x), ylim, color="gray", lw=1, alpha=0.5, linestyle='--')


plt.savefig("isol24.pdf")
plt.savefig("isol24.png")
plt.close()
#plt.show()
