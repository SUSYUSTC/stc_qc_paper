#%config InlineBackend.figure_formats = ['svg']
import numpy as np
import matplotlib as mpl
mpl.rc('mathtext', fontset='cm')
#rcParams["mathtext.fontset"]
import matplotlib.pyplot as plt
from quimb import schematic
#ax = plt.figure().add_subplot(projection='3d')


presets = {
    'bond': {'linewidth': 3},
    'phys': {'linewidth': 1.5},
    'center': {
        'color': '#75D2F0',
    },
    'A': {
        'color': (1.0, 0.4, 0.4), # light red
    },
    'B': {
        #'color': (0.0, 1.0, 0.0), # light green
        'color': '#b8dbb3',
    },
    'C': {
        #'color': (1.0, 0.4, 1.0), # light pink
        'color': '#fcb6a5',
    },
}
inches_per_unit = 1


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


def arrow(begin, end, c=0.3):
    a = np.array(begin) * (1-c) + np.array(end) * c
    b = np.array(begin) * (1-c-0.1) + np.array(end) * (c+0.1)
    diff = b - a
    #length = np.linalg.norm(np.array(end) - np.array(begin)) * 0.1
    length = np.linalg.norm(diff)
    plt.arrow(*a, *diff, head_width=0.15, head_length=0.1, fc='k', ec='k')


r = 0.25
text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 22, 'color': 'k'}

fig, ax = plt.subplots()

#d = schematic.Drawing(presets=presets, figsize=(4, 4), ax=ax)
d = schematic.Drawing(presets=presets, ax=ax)

d.circle((0, 0), radius=r, preset="center")
d.circle((-1, -1), radius=r, preset="center")
d.circle((1, -1), radius=r, preset="center")
ax.text(0, 0, "$|\\boldsymbol{A}|$", **text_kwargs)
ax.text(-1, -1, "$|\\boldsymbol{B}|$", **text_kwargs)
ax.text(1, -1, "$|\\boldsymbol{C}|$", **text_kwargs)

arrow((0, 0.8), (0, 0), 0.28)
arrow((0, 0), (-1, -1), 0.4)
arrow((0, 0), (1, -1), 0.4)
arrow((-1, -1), (-1, -1.8), 0.5)
arrow((1, -1), (1, -1.8), 0.5)

d.line((0, 0), (0, 0.8), preset='bond')
ax.text(0.15, 0.5, "$i$", **text_kwargs)
d.line((0, 0), (-1, -1), preset='bond')
ax.text(-0.3, -0.5, "$j$", **text_kwargs)
d.line((0, 0), (1, -1), preset='bond')
ax.text(0.3, -0.53, "$k$", **text_kwargs)
d.line((-1, -1), (-1, -1.8), preset='bond')
ax.text(-0.85, -1.4, "$l$", **text_kwargs)
d.line((1, -1), (1, -1.8), preset='bond')
ax.text(0.8, -1.4, "$m$", **text_kwargs)
#d.line((-1, -1), (-1.4, -1.8), preset='bond')
#d.line((-1, -1), (-0.6, -1.8), preset='bond')
#d.line((1, -1), (1.4, -1.8), preset='bond')
#d.line((1, -1), (0.6, -1.8), preset='bond')

rescale(fig, ax)
fig.savefig("diagram_tree.png", dpi=200)
plt.close(fig)



l = 1.5

fig, ax = plt.subplots()
d = schematic.Drawing(presets=presets, ax=ax)
height = 0.5 * np.sqrt(3) * l
up = (0, height)
left = (-0.5 * l, 0)
right = (0.5 * l, 0)

d.circle(up, radius=r, preset="center")
d.circle(left, radius=r, preset="center")
d.circle(right, radius=r, preset="center")
ax.text(*up, "$|\\boldsymbol{A}|$", **text_kwargs)
ax.text(*left, "$|\\boldsymbol{B}|$", **text_kwargs)
ax.text(*right, "$|\\boldsymbol{C}|$", **text_kwargs)

d.line(up, left, preset='bond')
d.line(up, right, preset='bond')
d.line(left, right, preset='bond')

text_height = l * 0.3
ax.text(0, text_height, "$\\tilde{p}^\\text{opt}$", **text_kwargs)
ax.text(l * 0.9, text_height, "$\\leq$", **text_kwargs)
ax.text(l * 2, text_height, "$\\tilde{p}^{\\prime}$", **text_kwargs)

shift_x = l * 2

left_up = (-0.5 * l + shift_x, height)
right_up = (0.5 * l + shift_x, height)
left_down = (-0.5 * l + shift_x, 0)
right_down = (0.5 * l + shift_x, 0)

d.circle(left_up, radius=r, preset="B")
d.circle(right_up, radius=r, preset="C")
d.circle(left_down, radius=r, preset="center")
d.circle(right_down, radius=r, preset="center")

ax.text(*left_up, "$\\boldsymbol{P}$", **text_kwargs)
ax.text(*right_up, "$\\boldsymbol{Q}$", **text_kwargs)
ax.text(*left_down, "$|\\boldsymbol{B}|$", **text_kwargs)
ax.text(*right_down, "$|\\boldsymbol{C}|$", **text_kwargs)

d.line(left_up, left_down, preset='bond')
d.line(right_up, right_down, preset='bond')
d.line(left_down, right_down, preset='bond')

rescale(fig, ax)
fig.savefig("diagram_loop.png", dpi=200)
plt.close(fig)



fig, ax = plt.subplots()
d = schematic.Drawing(presets=presets, ax=ax)
distance = 1.5
distance2 = distance + 0.8
height = 0.5 * np.sqrt(3) * l
factor = 0.7
l = 1.0
up = (0, 0)
left = (-0.5 * l * factor, -height * factor)
right = (0.5 * l * factor, -height * factor)

d.circle(up, radius=r, preset="center")
d.line(up, left, preset='bond')
d.line(up, right, preset='bond')

ax.text(0.75, 0, "$\\leq$", **text_kwargs)

ax.text(*up, "$|\\boldsymbol{A}|$", **text_kwargs)

d.circle((distance * l, 0), radius=r, preset="B")
d.circle((distance2 * l, 0), radius=r, preset="C")

d.line((distance * l, 0), (distance * l, -height * factor), preset='bond')
d.line((distance2 * l, 0), (distance2 * l, -height * factor), preset='bond')

ax.text(distance * l, 0, "$\\boldsymbol{P}$", **text_kwargs)
ax.text(distance2 * l, 0, "$\\boldsymbol{Q}$", **text_kwargs)

rescale(fig, ax)
fig.savefig("diagram_break.png", dpi=200)
plt.close(fig)
