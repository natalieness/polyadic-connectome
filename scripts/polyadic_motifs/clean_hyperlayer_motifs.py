from collections import Counter
from itertools import chain, combinations, accumulate
import time
from functools import partial

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
map_con_targets_to_real_neurons, get_partner_motifs, cut_out_frags, filter_col_presyn, get_unique_ct_target_counts)
from scripts.initialisation_scripts.get_me_started import get_me_started, get_me_labelled
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
from scripts.functions.little_helper import get_global_top_targets
from scripts.functions.motif_summary_functions import (compute_motif_summary, plot_motif_bar_comparison, 
                                                       motif_summary_across_layers, compute_cosine_similarities,
                                                       extract_motif_proportions, plot_motif_summary_pure_and_mixed)

from scripts.functions.hyper_connectome_class import HyperCon

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict, all_neurons = get_me_started()

''' gonna randomize after shortening polyadic synapses to remove fragments.'''
#con_rand = polyadic_edge_permutation(connector_details, rng=rng)
#con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

# %% try look at the distribution of top downstream partners and randos

# think need to do this with the 'global top targets' too, otherwise it might be too high a threshold? 
# this is same for randomly permutated networks
glob_top_targets = get_global_top_targets(connector_details, only_known_targets=False, syn_threshold=3)

#%% shorten polyadic synapses to remove fragments before hyperlayering

con_short = cut_out_frags(connector_details, all_neurons)
con_short = get_me_labelled(con_short, skid_to_celltype, pairs, pairs_dict)
con_rand = polyadic_edge_permutation(con_short, rng=rng)
con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

#%%
# filter by hemisphere 
conL = con_short[con_short['presyn_hemi'] == 'left']
conR = con_short[con_short['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']
conL['n_post'] = conL['postsynaptic_to'].apply(lambda x: len(x))
conR['n_post'] = conR['postsynaptic_to'].apply(lambda x: len(x))
con_randL['n_post'] = con_randL['postsynaptic_to'].apply(lambda x: len(x))
con_randR['n_post'] = con_randR['postsynaptic_to'].apply(lambda x: len(x))

n_post_threshold = 0.01 * conL['n_post'].sum()
L_overview = conL['n_post'].value_counts().to_dict()
hyperlayers = [k for k, v in L_overview.items() if v > n_post_threshold]
print(f'Hyperlayers in left hemisphere: {hyperlayers}')



#%% filter unknown targets before hyperlayering 
    
hypercon = HyperCon(filter_col='n_post')
hypercon.add_df('conL_f', conL)
hypercon.add_df('conR_f', conR)
hypercon.add_df('con_randL_f', con_randL)
hypercon.add_df('con_randR_f', con_randR)

#%% flow motifs
mpl.rcParams.update({'font.size': 12, 
                     'axes.labelsize': 16, 
                     'xtick.labelsize': 14, 
                     'ytick.labelsize': 14, 
                     'axes.spines.right': False, 
                     'axes.spines.top': False})

#val = 5
map_con_targets_to_flow_partial = partial(map_con_targets_to_flow, flow_dict=flow_dict)

motif_funcs = [map_con_targets_to_flow_partial, get_flow_motifs]
motif_labels =  {'conL_f': 'Left', 'con_randL_f': 'Rand. Left', 'conR_f': 'Right', 'con_randR_f': 'Rand. Right'}
# individual layer:
# flow_motif_summary = compute_motif_summary(hypercon, val, motif_funcs, motif_labels)
# plot_motif_bar_comparison(flow_motif_summary, motif_labels, title=f"Flow motifs for {val} partners", plot_proportions=True)
# props = extract_motif_proportions(flow_motif_summary, props=None)

proportions_flow_all, cos_flow_all, summary_flow_all = motif_summary_across_layers(hypercon, motif_funcs, layer_range=range(2, 6), labels=motif_labels)

# %% top target motifs
get_partner_motifs_partial = partial(get_partner_motifs, pn_target_dict=glob_top_targets)
partner_funcs = [get_partner_motifs_partial, get_flow_motifs]


props_partner_all, cos_partner_all, summary_partner_all = motif_summary_across_layers(hypercon, partner_funcs, layer_range=range(2, 6), labels=motif_labels)

#%% apply functions based on presynaptic cell type 

ct_path = 'data/hypergraph_figs/flow_by_ct_justallneurons/'
if not os.path.exists(ct_path):
    os.makedirs(ct_path)
ct_names = celltype_df['name'].unique().tolist()
summary_flow_ct = {}
for c in ct_names:
    filter_col_partial = partial(filter_col_presyn, ct=c)
    motif_funcs_ct = [filter_col_partial, map_con_targets_to_flow_partial, get_flow_motifs]
    motif_labels =  {'conL_f': f'Left {c}', 'con_randL_f': f'Rand. Left {c}', 'conR_f': f'Right {c}', 'con_randR_f': f'Rand. Right {c}'}
    ct_path_i = f"{ct_path}{c}_"
    props_flow_ct, cos_flow_ct, summary_flow_ct[c] = motif_summary_across_layers(hypercon, motif_funcs_ct, layer_range=range(2, 6), labels=motif_labels, save_figs=ct_path_i)



#%% look at ct motifs across layers in clean hyperlayered network

val_to_use = 3
results_ct = {}
for c in ct_names:
    results = hypercon.apply_multiple_functions(val_to_use, [lambda x: filter_col_presyn(x, c), lambda x: get_unique_ct_target_counts(x, remove_nan=False, ct_names=ct_names)])
    results_ct[c] = results

results_df = pd.DataFrame()
for k, res in results_ct.items():
    df_merged = None
    for key, df in res.items():
        df = df.rename(columns={'count': key})
        df_merged = pd.merge(df_merged, df, on='motif', how='outer') if df_merged is not None else df
    df_merged = df_merged.fillna(0.0)
    results_ct[k] = df_merged
    df_merged['motif'] = df_merged['motif'].astype(str)
    df_merged.set_index('motif', inplace=True)
    results_df = pd.concat([results_df, df_merged], axis=1, join='outer') if not results_df.empty else df_merged

results_df = results_df.fillna(0)
results_df.columns = pd.MultiIndex.from_product([results_ct.keys(), ['conL', 'conR', 'con_randL', 'con_randR']])
#%% visualise ct motifs 

### all normalised counts 
res = results_df.copy()

order = ['conL', 'con_randL', 'conR','con_randR']
res.sort_index(axis=1, level=1, inplace=True, sort_remaining=False, key=lambda idx: idx.map(lambda x: order.index(x) if x in order else len(order)))
res.sort_index(axis=1, level=0, inplace=True, sort_remaining=False)
res = res.div(res.sum(axis=0), axis=1)  # Normalize by column sums

fig, ax = plt.subplots(1, 1, figsize=(48, 30))
ax.imshow(res, cmap='OrRd', aspect=0.8)
ax.set_yticks(ticks=range(len(res.index)), labels=res.index)
ax.set_xticks(ticks=range(len(res.columns)), labels=res.columns, rotation=90)

### difference to randomised network
results_rel = results_df.copy()
for j in results_rel.columns.levels[0]:
    results_rel[j, 'L-diff'] = results_rel[j, 'conL'] - results_rel[j, 'con_randL']
    results_rel[j, 'R-diff'] = results_rel[j, 'conR'] - results_rel[j, 'con_randR']
results_rel.drop(['conL', 'conR', 'con_randL', 'con_randR'], level=1,axis=1, inplace=True)
print(results_rel.columns)

norm_res = True
if norm_res:
    results_rel = results_rel.div(results_rel.sum(axis=0), axis=1)  # Normalize by column sums

top10_shared_motifs = results_rel.sum(axis=1).nlargest(40).index.tolist()
trd = results_rel.loc[top10_shared_motifs]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(trd, cmap='coolwarm', aspect=0.5, vmin=-1, vmax=1)
ax.set_yticks(ticks=range(len(trd.index)), labels=trd.index)
ax.set_xticks(ticks=range(len(trd.columns)), labels=trd.columns, rotation=90)


for xloc, cname in zip(ax.get_xticks()[::2],trd.columns.get_level_values(0)[::2]):
    xpos = xloc + 0.7
    ax.annotate(cname, xy=(xpos, 1.02), xycoords=('data', 'axes fraction'), fontsize=12, ha='center', rotation=90)

#%% plot just top of a particular celltype
ct = 'LNs'
n_plot = 40
results_rel_ct = results_rel.copy()
results_rel_ct.sort_values(by=(ct, 'L-diff'), ascending=False, inplace=True)


fig, ax = plt.subplots(1, 1, figsize=(12, 10))
trd_ct = results_rel_ct.iloc[:n_plot, :]

if norm_res:
    max_max = 1
else:
    max_max = np.max(np.abs(trd_ct.values))
ax.imshow(trd_ct, cmap='coolwarm', aspect=0.8, vmin=-max_max, vmax=max_max)
ax.set_yticks(ticks=range(len(trd_ct.index)), labels=trd_ct.index)
ax.set_xticks(ticks=range(len(trd_ct.columns)), labels=trd_ct.columns, rotation=90)


for xloc, cname in zip(ax.get_xticks()[::2],trd_ct.columns.get_level_values(0)[::2]):
    xpos = xloc + 0.7
    ax.annotate(cname, xy=(xpos, 1.02), xycoords=('data', 'axes fraction'), fontsize=12, ha='center', rotation=90)

#%% plotting just raw counts rather than some normalised version 

res_raw = results_df.copy()
order = ['conL', 'con_randL', 'conR','con_randR']
res_raw.sort_index(axis=1, level=1, inplace=True, sort_remaining=False, key=lambda idx: idx.map(lambda x: order.index(x) if x in order else len(order)))
res_raw.sort_index(axis=1, level=0, inplace=True, sort_remaining=False)

# sort by something 
ct = 'PNs'
if ct == 'all':
    res_raw = res_raw.loc[res_raw.sum(axis=1).nlargest(40).index, :]  # take top 40 motifs by sum across all conditions
else: 
    res_raw = res_raw.loc[res_raw[ct, 'conL'].nlargest(40).index, :]  

# take top 
n_plot = 40
trd_raw = res_raw.iloc[:n_plot, :]
max_max = np.max(trd_raw.values)
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

im = ax.imshow(trd_raw, cmap='OrRd', aspect=1.1, vmin=0-max_max, vmax=max_max)
ax.set_yticks(ticks=range(len(trd_raw.index)), labels=trd_raw.index)
ax.set_xticks(ticks=range(len(trd_raw.columns)), labels=trd_raw.columns, rotation=90)
cbar = plt.colorbar(im, ax=ax, fraction=0.027, pad=0.04)
cbar.set_label('# of synapses', rotation=90, labelpad=-90, fontsize=14)

for xloc, cname in zip(ax.get_xticks()[::4],trd_raw.columns.get_level_values(0)[::4]):
    xpos = xloc + 1.5
    ax.annotate(cname, xy=(xpos, 1.02), xycoords=('data', 'axes fraction'), fontsize=12, ha='center', rotation=90)

# %% check if they could just be the exact same neurons :( && remove repeats

hypercon5 = hypercon.get_all_filtered(5)
df1 = get_partner_motifs_partial(hypercon5['conR_f'])
df1_justtop = df1[df1['flow_scores'].apply(lambda x: x == df1['flow_scores'][0])]
hyperedges = df1_justtop['postsynaptic_to'].tolist()
hyperedges = [set(h) for h in hyperedges]
hyperedges_with_repeats = [h for h in hyperedges if len(h) < 5]
print(f'Number of hyperedges with repeats in left hemisphere: {len(hyperedges_with_repeats)} out of {len(hyperedges)}')

motif_labels =  {'conL_f': f'Left', 'con_randL_f': f'Rand. Left', 'conR_f': f'Right', 'con_randR_f': f'Rand. Right'}

### remove repeats 

con_ss = con_short.copy()
con_ss['postsynaptic_to'] = con_ss['postsynaptic_to'].apply(lambda x: list(set(x)))  # remove duplicates in postsynaptic partners
con_ss = con_ss.iloc[:,:5]  # keep only relevant columns
con_ss = get_me_labelled(con_ss, skid_to_celltype, pairs, pairs_dict)
con_randss = polyadic_edge_permutation(con_ss, rng=rng)
con_randss = get_me_labelled(con_randss, skid_to_celltype, pairs, pairs_dict)

conL = con_ss[con_ss['presyn_hemi'] == 'left']
conR = con_ss[con_ss['presyn_hemi'] == 'right']
con_randL = con_randss[con_randss['presyn_hemi'] == 'left']
con_randR = con_randss[con_randss['presyn_hemi'] == 'right']
conL['n_post'] = conL['postsynaptic_to'].apply(lambda x: len(x))
conR['n_post'] = conR['postsynaptic_to'].apply(lambda x: len(x))
con_randL['n_post'] = con_randL['postsynaptic_to'].apply(lambda x: len(x))
con_randR['n_post'] = con_randR['postsynaptic_to'].apply(lambda x: len(x))

n_post_threshold = 0.01 * conL['n_post'].sum()
L_overview = conL['n_post'].value_counts().to_dict()
hyperlayers = [k for k, v in L_overview.items() if v > n_post_threshold]
print(f'Hyperlayers in left hemisphere: {hyperlayers}')

hypercon_ss = HyperCon(filter_col='n_post')
hypercon_ss.add_df('conL_f', conL)
hypercon_ss.add_df('conR_f', conR)
hypercon_ss.add_df('con_randL_f', con_randL)
hypercon_ss.add_df('con_randR_f', con_randR)

###top target motifs without postsyn repeats
mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.spines.right': False, 'axes.spines.top': False})

get_partner_motifs_partial = partial(get_partner_motifs, pn_target_dict=glob_top_targets)
partner_funcs = [get_partner_motifs_partial, get_flow_motifs]


props_partner_all, cos_partner_all, summary_partner_all = motif_summary_across_layers(hypercon_ss, partner_funcs, layer_range=range(2, 6), labels=motif_labels)


#### flow motifs without postsyn repeats
#val = 5
map_con_targets_to_flow_partial = partial(map_con_targets_to_flow, flow_dict=flow_dict)

motif_funcs = [map_con_targets_to_flow_partial, get_flow_motifs]
motif_labels =  {'conL_f': 'Left', 'con_randL_f': 'Rand. Left', 'conR_f': 'Right', 'con_randR_f': 'Rand. Right'}
# individual layer:
# flow_motif_summary = compute_motif_summary(hypercon, val, motif_funcs, motif_labels)
# plot_motif_bar_comparison(flow_motif_summary, motif_labels, title=f"Flow motifs for {val} partners", plot_proportions=True)
# props = extract_motif_proportions(flow_motif_summary, props=None)

proportions_flow_all, cos_flow_all, summary_flow_all = motif_summary_across_layers(hypercon_ss, motif_funcs, layer_range=range(2, 6), labels=motif_labels)

# %%
