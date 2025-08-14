


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
map_con_targets_to_real_neurons, get_partner_motifs)
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



con_rand = polyadic_edge_permutation(connector_details, rng=rng)
con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

# %% try look at the distribution of top downstream partners and randos

# think need to do this with the 'global top targets' too, otherwise it might be too high a threshold? 
# this is same for randomly permutated networks
glob_top_targets = get_global_top_targets(connector_details, only_known_targets=False, syn_threshold=3)
#%%
# filter by hemisphere 
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']

#filter by celltype or neuron_id 
type = 'group' #individual or group
ct = 'MBONs'

conL_f, conR_f = filter_con(conL, conR, pairs_dict=pairs_dict, pairs=pairs, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)
con_randL_f, con_randR_f = filter_con(con_randL, con_randR, pairs_dict=pairs_dict, pairs=pairs, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)

# else just take the whole connector dataframe:
conL_f = conL.copy()
conR_f = conR.copy()
con_randL_f = con_randL.copy()
con_randR_f = con_randR.copy()

#%% construct hyperlayer graphs instead 

conL_f['n_post'] = conL_f['postsynaptic_to'].apply(lambda x: len(x))
conR_f['n_post'] = conR_f['postsynaptic_to'].apply(lambda x: len(x))
con_randL_f['n_post'] = con_randL_f['postsynaptic_to'].apply(lambda x: len(x))
con_randR_f['n_post'] = con_randR_f['postsynaptic_to'].apply(lambda x: len(x))

# get only layers that are not super sparse ??
# maybe only include connectors with at least 1% of all connectors in the layer?
n_post_threshold = 0.01 * conL_f['n_post'].sum()
L_overview = conL_f['n_post'].value_counts().to_dict()
hyperlayers = [k for k, v in L_overview.items() if v > n_post_threshold]
#
    
hypercon = HyperCon(filter_col='n_post')
hypercon.add_df('conL_f', conL_f)
hypercon.add_df('conR_f', conR_f)
hypercon.add_df('con_randL_f', con_randL_f)
hypercon.add_df('con_randR_f', con_randR_f)
filtered_df = hypercon.get_all_filtered(4)  # example for filtering by layer with 4 postsynaptic partners

#top_targets_hypercon = hypercon.apply_some_function_on_binarised(4, get_top_targets)

#%% example usage on a single layer
# try it 
val = 3

# use partial to supply some values to functions passed to hypercon 
map_con_targets_to_flow_partial = partial(map_con_targets_to_flow, flow_dict=flow_dict)

result_flow_motifs = hypercon.apply_multiple_functions(val, [map_con_targets_to_flow_partial, get_flow_motifs])

fm_summary = pd.DataFrame.from_dict(result_flow_motifs, orient='columns')
fm_summary.index = fm_summary.index.map(lambda x: str(x))
fm_summary = fm_summary.fillna(0)
fm_summary['within-var'] = (fm_summary['conL_f'] - fm_summary['conR_f'])/ (fm_summary['conR_f']+fm_summary['conL_f'])
fm_summary['across_L'] = (fm_summary['conL_f'] - fm_summary['con_randL_f']) / (fm_summary['conL_f'] + fm_summary['con_randL_f'])
fm_summary['across_R'] = (fm_summary['conR_f'] - fm_summary['con_randR_f']) / (fm_summary['conR_f'] + fm_summary['con_randR_f'])

within_cos_sim = cosine_similarity(np.array(fm_summary['conR_f']).reshape(1,-1), np.array(fm_summary['conL_f']).reshape(1,-1))
across_cos_sim1 = cosine_similarity(np.array(fm_summary['conL_f']).reshape(1,-1), np.array(fm_summary['con_randL_f']).reshape(1,-1))
across_cos_sim2 = cosine_similarity(np.array(fm_summary['conR_f']).reshape(1,-1), np.array(fm_summary['con_randR_f']).reshape(1,-1))
print(f"For value: {val}, Cosine similarity: L-R: {within_cos_sim}, L-randL: {across_cos_sim1}, R-randL: {across_cos_sim2}")

#%% check out celltype motifs across specific layers 
# example usage for flow 

mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.spines.right': False, 'axes.spines.top': False})

#val = 5
motif_funcs = [map_con_targets_to_flow_partial, get_flow_motifs]
motif_labels =  {'conL_f': 'Left', 'con_randL_f': 'Rand. Left', 'conR_f': 'Right', 'con_randR_f': 'Rand. Right'}
# individual layer:
# flow_motif_summary = compute_motif_summary(hypercon, val, motif_funcs, motif_labels)
# plot_motif_bar_comparison(flow_motif_summary, motif_labels, title=f"Flow motifs for {val} partners", plot_proportions=True)
# props = extract_motif_proportions(flow_motif_summary, props=None)

proportions_flow_all, cos_flow_all, summary_flow_all = motif_summary_across_layers(hypercon, motif_funcs, layer_range=range(2, 9), labels=motif_labels)


# TODO: check when removing na's would make sense - atm it doesn't make sense because it messes up the proportions by just filtering out all the mixed synapses 

# %% using partner motifs 
get_partner_motifs_partial = partial(get_partner_motifs, pn_target_dict=glob_top_targets)
partner_funcs = [get_partner_motifs_partial, get_flow_motifs]


props_partner_all, cos_partner_all, summary_partner_all = motif_summary_across_layers(hypercon, partner_funcs, layer_range=range(2, 9))

#%% try with real vs fragment 
map_con_targets_to_real_neurons_partial = partial(map_con_targets_to_real_neurons, all_neurons=all_neurons)
motif_funcs_real = [map_con_targets_to_real_neurons_partial, get_flow_motifs]
proportions_real_all, cos_real_all, summary_real_all = motif_summary_across_layers(hypercon, motif_funcs_real, layer_range=range(2, 9))

# %%
#%%

''' This whole section was used to explore the partner motifs and their distributions,
and make a bunch of plots '''

# same but delta 
df_delta = partner_summary_norm.copy()
df_delta_idx = df_delta.index.tolist()
df_delta_idx = [eval(j) for j in df_delta_idx]
idx_n = [j.count('TOP') for j in df_delta_idx]
idx_n_order = np.argsort(idx_n)
df_delta = df_delta.iloc[idx_n_order,:]
df_delta = df_delta.iloc[:,:]
df_delta['L-randL'] = df_delta['conL_f'] - df_delta['con_randL_f']
df_delta['R-randR'] = df_delta['conR_f'] - df_delta['con_randR_f']
df_delta = df_delta[['L-randL', 'R-randR']]
ax = df_delta.plot(
    kind='bar',
    stacked=False,
    figsize=(10, 6),
    color=["#037AF1", "#FA7D00"],
)
ax.set_ylabel("Delta proportion of synapses \n compared to random")
ax.legend(['Left', 'Right'], title="Hemisphere", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


#
# %%
