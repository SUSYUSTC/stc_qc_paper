#%config InlineBackend.figure_formats = ['svg']
import numpy as np
import matplotlib as mpl
mpl.rc('mathtext', fontset='cm')
#mpl.rc('text', usetex=True)
#mpl.rc('text.latex', preview=True)
#mpl.rc('text.latex', preamble=r"""\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}""")
#mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt
from quimb import schematic
from matplotlib.patches import Ellipse
#ax = plt.figure().add_subplot(projection='3d')


gray = (0.85, 0.85, 0.85)
presets = {
    'bond': {'linewidth': 3},
    'phys': {'linewidth': 1.5},
    'center': {
        'color': schematic.get_wong_color('blue'),
    },
    'F': {
        'color': gray,
    },
    'T': {
        #'color': (1.0, 0.8, 0.1), # dark yellow
        'color': '#d0cade',
    },
    'V': {
        #'color': (0.8, 0.1, 1.0), # purple
        'color': '#fbf8b4',
    },
}
inches_per_unit = 1
text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 22, 'color': 'k'}


def rescale(fig, ax):
    ax.set_aspect('equal', adjustable='box')
    ax.relim(); ax.autoscale_view()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx, dy = xmax - xmin, ymax - ymin

    fig_w = inches_per_unit * dx
    fig_h = inches_per_unit * dy

    fig.set_size_inches(fig_w, fig_h, forward=True)
    ax.set_position([0, 0, 1, 1])


def draw_ellipse(ax, center, width, height, color, **kwargs):
    ellipse = Ellipse(xy=center, width=width, height=height, angle=0, facecolor=color, edgecolor='k', **kwargs)
    ax.add_patch(ellipse)


r = 0.4
l = 1.2
length = 0.8
fig, ax = plt.subplots()

ax.text(-l * 1.2 * length, 0, "$E_\\text{(T)}=$", **text_kwargs)

d = schematic.Drawing(presets=presets, ax=ax)
t1 = (0, 0)
t2 = (l * length, 0)
t3 = (l * 2 * length, 0)

d.square(t1, radius=r, preset='T')
ax.text(*t1, "$\\boldsymbol{T}$", **text_kwargs)
d.square(t2, radius=r, preset='V')
ax.text(*t2, "$\\boldsymbol{V}$", **text_kwargs)
for height in [0.1, -0.1]:
    d.line((-0.5 * l * length, height), (1.5 * l * length, height), preset='bond')

rescale(fig, ax)
fig.savefig("diagram_VT.png", dpi=200)
plt.close(fig)


r = 0.4
l = 1.2
length = 0.8
fig, ax = plt.subplots()
d = schematic.Drawing(presets=presets, ax=ax)
t1 = (0, 0)
t2 = (l * length, 0)
t3 = (l * 2 * length, 0)

d.square(t1, radius=r, preset='T')
ax.text(*t1, "$\\boldsymbol{T}$", **text_kwargs)
d.square(t2, radius=r, preset='V')
ax.text(*t2, "$\\boldsymbol{V}$", **text_kwargs)
d.square(t3, radius=r, preset='T')
ax.text(*t3, "$\\boldsymbol{T}$", **text_kwargs)
for height in [0.1, -0.1]:
    d.line((-0.5 * l * length, height), (2.5 * l * length, height), preset='bond')

rescale(fig, ax)
fig.savefig("diagram_VTT.png", dpi=200)
plt.close(fig)


r = 0.4
l = 1.5
fig, ax = plt.subplots()
d = schematic.Drawing(presets=presets, ax=ax)
t_left_up = (-l * 0.6, l * 0.4)
t_right_up = (l * 0.6, l * 0.4)
t_left_down = (-l * 0.6, -l * 0.4)
t_right_down = (l * 0.6, -l * 0.4)

#ax.text(-l * 1.2, 0, "$\\mathcal_\\text{(T)}(\\boldsymbol{H},\\boldsymbol{T})=$", **text_kwargs)

d.square(t_left_up, radius=r, preset='V')
d.square(t_right_up, radius=r, preset='V')
d.square(t_left_down, radius=r, preset='T')
d.square(t_right_down, radius=r, preset='T')

ax.text(*t_left_up, "$\\boldsymbol{V}$", **text_kwargs)
ax.text(*t_right_up, "$\\boldsymbol{V}$", **text_kwargs)
ax.text(*t_left_down, "$\\boldsymbol{T}$", **text_kwargs)
ax.text(*t_right_down, "$\\boldsymbol{T}$", **text_kwargs)

draw_ellipse(ax, center=(0, 0), width=l * 0.2, height=l * 1.25, color=gray)
ax.text(0, 0, "$\\frac{1}{\\boldsymbol{\Delta}}$", **text_kwargs)
for height in [0.2, 0, -0.2]:
    d.line((-l * 0.6, l * 0.4 + height), (l * 0.6, l * 0.4 + height), preset='bond')
for height in [0.2, 0, -0.2]:
    d.line((-l * 0.6, -l * 0.4 + height), (l * 0.6, -l * 0.4 + height), preset='bond')
d.line(t_left_up, t_left_down, preset='bond')
d.line(t_right_up, t_right_down, preset='bond')

#ax.set_xlim(-l * 1.5, l)
rescale(fig, ax)
fig.savefig("diagram_pt.png", dpi=200)
plt.close(fig)
