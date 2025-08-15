
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from matplotlib.lines import Line2D


fdata_dir = 'figure_data/'
fig_dir = 'figures/'

# data import


# set general parameters 
mpl.rcParams.update({'font.size': 12, 
                     'axes.labelsize': 16, 
                     'xtick.labelsize': 14, 
                     'ytick.labelsize': 14, 
                     'axes.spines.right': False, 
                     'axes.spines.top': False})

# data stuff 

# plotting 
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# save / archive 

fig.savefig(fig_dir + 'fig1.tiff', dpi=300)