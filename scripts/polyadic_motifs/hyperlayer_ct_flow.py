from collections import Counter
from itertools import chain, combinations, accumulate
import time

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D

from scripts.functions.motif_functions import (con_binary_matrix, get_top_targets, con_bin_cos_sim, 
get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets, 
normalise_freq_cols, filter_con, map_con_targets_to_flow, get_flow_motifs, remove_incomplete_flow_motifs, 
map_con_targets_to_real_neurons, get_partner_motifs, filter_col_presyn, sort_list_by_input, remove_na, get_top_motifs_only,
get_unique_ct_target_counts)
from scripts.initialisation_scripts.get_me_started import get_me_started, get_me_labelled
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation

from scripts.functions.hyper_connectome_class import HyperCon
from scripts.functions.little_helper import get_global_top_targets
#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict, all_neurons = get_me_started()

con_rand = polyadic_edge_permutation(connector_details, rng=rng)
con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

# %% try look at the distribution of top downstream partners and randos

# think need to do this with the 'global top targets' too, otherwise it might be too high a threshold? 
# this is same for randomly permutated networks
glob_top_targets = get_global_top_targets(connector_details, only_known_targets=False, syn_threshold=3)

#%%
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']

conL['n_post'] = conL['postsynaptic_to'].apply(lambda x: len(x))
conR['n_post'] = conR['postsynaptic_to'].apply(lambda x: len(x))
con_randL['n_post'] = con_randL['postsynaptic_to'].apply(lambda x: len(x))
con_randR['n_post'] = con_randR['postsynaptic_to'].apply(lambda x: len(x))

#%% create hypercon class 
hypercon = HyperCon(filter_col='n_post')
hypercon.add_df('conL', conL)
hypercon.add_df('conR', conR)
hypercon.add_df('con_randL', con_randL)
hypercon.add_df('con_randR', con_randR)

#%% look at just layer 2 

hypercon2 = hypercon.get_all_filtered(2)

# %%
ct_names = celltype_df['name'].unique().tolist()



val_to_use = 4
results_ct = {}
for c in ct_names:
    results = hypercon.apply_multiple_functions(val_to_use, [lambda x: filter_col_presyn(x, c), lambda x: get_unique_ct_target_counts(x, remove_nan=True, ct_names=ct_names)])
    results_ct[c] = results


results_df = pd.DataFrame()
for k, res in results_ct.items():
    df_merged = None
    for key, df in res.items():
        df = df.rename(columns={'count': key})
        df_merged = pd.merge(df_merged, df, on='motif', how='outer') if df_merged is not None else df
    df_merged = df_merged.fillna(0)
    results_ct[k] = df_merged
    df_merged['motif'] = df_merged['motif'].astype(str)
    df_merged.set_index('motif', inplace=True)
    results_df = pd.concat([results_df, df_merged], axis=1, join='outer') if not results_df.empty else df_merged

results_df = results_df.fillna(0)
results_df.columns = pd.MultiIndex.from_product([results_ct.keys(), ['conL', 'conR', 'con_randL', 'con_randR']])


# %%
res_copy = results_df.copy()

#exclude RGNs cause they mess up the plot 
mask = ~results_df.columns.to_frame().apply(lambda x: x.str.contains('RGN')).any(axis=1)
results_df = results_df.loc[:, mask]

order = ['conL', 'con_randL', 'conR','con_randR']
results_df.sort_index(axis=1, level=1, inplace=True, sort_remaining=False, key=lambda idx: idx.map(lambda x: order.index(x) if x in order else len(order)))
results_df.sort_index(axis=1, level=0, inplace=True, sort_remaining=False)
results_df = results_df.div(results_df.sum(axis=0), axis=1)  # Normalize by column sums
fig, ax = plt.subplots(1, 1, figsize=(48, 30))
ax.imshow(results_df, cmap='OrRd', aspect=0.8)
ax.set_yticks(ticks=range(len(results_df.index)), labels=results_df.index)
ax.set_xticks(ticks=range(len(results_df.columns)), labels=results_df.columns, rotation=90)
#plt.tight_layout()

top10_shared_motifs = results_df.sum(axis=1).nlargest(40).index.tolist()
trd = results_df.loc[top10_shared_motifs]
wy = 8
fig, ax = plt.subplots(1, 1, figsize=(12, wy))
ax.imshow(trd, cmap='OrRd', aspect=0.9)
ax.set_yticks(ticks=range(len(trd.index)), labels=trd.index)
ax.set_xticks(ticks=range(len(trd.columns)), labels=trd.columns, rotation=90)


for xloc, cname in zip(ax.get_xticks()[::4],trd.columns.get_level_values(0)[::4]):
    xpos = xloc + 1.5
    ax.annotate(cname, xy=(xpos, 1.02), xycoords=('data', 'axes fraction'), fontsize=12, ha='center', rotation=90)


# %% replot with difference to randomised networks
results_rel = res_copy.copy()

for j in results_rel.columns.levels[0]:
    results_rel[j, 'L-diff'] = results_rel[j, 'conL'] - results_rel[j, 'con_randL']
    results_rel[j, 'R-diff'] = results_rel[j, 'conR'] - results_rel[j, 'con_randR']
results_rel.drop(['conL', 'conR', 'con_randL', 'con_randR'], level=1,axis=1, inplace=True)
print(results_rel.columns)

mask = ~results_rel.columns.to_frame().apply(lambda x: x.str.contains('RGN')).any(axis=1)
results_rel = results_rel.loc[:, mask]

# order = ['conL', 'con_randL', 'conR','con_randR']
# results_df.sort_index(axis=1, level=1, inplace=True, sort_remaining=False, key=lambda idx: idx.map(lambda x: order.index(x) if x in order else len(order)))
# results_df.sort_index(axis=1, level=0, inplace=True, sort_remaining=False)
results_rel = results_rel.div(results_rel.sum(axis=0), axis=1)  # Normalize by column sums
fig, ax = plt.subplots(1, 1, figsize=(48, 30))
ax.imshow(results_rel, cmap='coolwarm', aspect=0.8, vmin=-1, vmax=1)
ax.set_yticks(ticks=range(len(results_rel.index)), labels=results_rel.index)
ax.set_xticks(ticks=range(len(results_rel.columns)), labels=results_rel.columns, rotation=90)
#plt.tight_layout()
#%%

top10_shared_motifs = results_rel.sum(axis=1).nlargest(40).index.tolist()
trd = results_rel.loc[top10_shared_motifs]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(trd, cmap='coolwarm', aspect=0.5, vmin=-1, vmax=1)
ax.set_yticks(ticks=range(len(trd.index)), labels=trd.index)
ax.set_xticks(ticks=range(len(trd.columns)), labels=trd.columns, rotation=90)


for xloc, cname in zip(ax.get_xticks()[::2],trd.columns.get_level_values(0)[::2]):
    xpos = xloc + 0.7
    ax.annotate(cname, xy=(xpos, 1.02), xycoords=('data', 'axes fraction'), fontsize=12, ha='center', rotation=90)

# %% examine top ones for just some cell types 
ct = 'LNs'
n_plot = 40
results_rel_ct = results_rel.copy()
results_rel_ct.sort_values(by=(ct, 'L-diff'), ascending=False, inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
trd_ct = results_rel_ct.iloc[:n_plot, :]
ax.imshow(trd_ct, cmap='coolwarm', aspect=0.5, vmin=-1, vmax=1)
ax.set_yticks(ticks=range(len(trd_ct.index)), labels=trd_ct.index)
ax.set_xticks(ticks=range(len(trd_ct.columns)), labels=trd_ct.columns, rotation=90)


for xloc, cname in zip(ax.get_xticks()[::2],trd_ct.columns.get_level_values(0)[::2]):
    xpos = xloc + 0.7
    ax.annotate(cname, xy=(xpos, 1.02), xycoords=('data', 'axes fraction'), fontsize=12, ha='center', rotation=90)



# %%
