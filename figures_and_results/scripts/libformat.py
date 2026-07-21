#!/usr/bin/env python
import numpy as np


def printf(matrix, fmt="%6.3f", length=7):
    for i in matrix:
        for j in i:
            print((fmt % j)[0:length], end='\t')
        print()


def set_log_ticks(axis):
    '''
    set_log_ticks(ax.get_xaxis())
    '''
    import matplotlib as mpl
    axis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_minor_formatter(mpl.ticker.NullFormatter())


def get_log_ticks(vs, fmt='125'):
    vmin, vmax = vs
    l = np.array(list(fmt)).astype(int)
    basemin = int(np.floor(np.log10(vmin)))
    basemax = int(np.ceil(np.log10(vmax)))
    bases = 10.0 ** np.arange(basemin, basemax)
    ls = (bases[:, None] * l).flatten()
    ls = ls[ls >= vmin]
    ls = ls[ls <= vmax]
    return ls


def get_opt_log_ticks(vs):
    vmin, vmax = vs
    for fmt in ['1', '13', '125', '1235', '12358']:
        ls = get_log_ticks(vs, fmt)
        if len(ls) >= 3:
            return ls
    return ls
