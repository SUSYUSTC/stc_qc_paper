import os

# Absolute base directories. This file lives in figures_and_results/scripts/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                 # figures_and_results
DATA_DIR = os.path.join(_ROOT, "data")         # processed data consumed by the plotting scripts
FIG_DIR = os.path.join(_ROOT, "figures")       # generated main-text figures
ASSET_DIR = os.path.join(_ROOT, "assets")      # input image assets (e.g. hBN_3x5.png)

def data(name):
    return os.path.join(DATA_DIR, name)

def fig(name):
    return os.path.join(FIG_DIR, name)

def asset(name):
    return os.path.join(ASSET_DIR, name)

import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

#color_STC = '#ff7f0e'
#color_DLPNO = '#2ca02c'
#color_exact = '#333333'

#color_1 = '#1f77b4'
#color_2 = '#b58900'


color_STC = '#f39422'
color_DLPNO = '#6fae47'
color_exact = '#333333'

color_1 = '#b16666'
color_2 = '#839fb8'
