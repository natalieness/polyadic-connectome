from collections import Counter
from itertools import chain, combinations, accumulate
import time

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity

from scripts.functions.motif_functions import con_binary_matrix, get_top_targets, con_bin_cos_sim, get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets, filter_con
from scripts.initialisation_scripts.get_me_started import get_me_started, get_me_labelled
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict = get_me_started()

all_neurons = connector_details['presynaptic_to'].unique().tolist()

#%% get randomly permutated network to compare 

con_rand = polyadic_edge_permutation(connector_details, rng=rng)
con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

#%% 
# filter by hemisphere 
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']

#filter by celltype or neuron_id 
type = 'group' #individual or group
ct = 'sensories'

conL_f, conR_f = filter_con(conL, conR, pairs_dict=pairs_dict, pairs=pairs, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)
con_randL_f, con_randR_f = filter_con(con_randL, con_randR, pairs_dict=pairs_dict, pairs=pairs, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)


#%%
conbL_f = con_binary_matrix(conL_f, only_known_targets=True, all_neurons=all_neurons)
conbR_f = con_binary_matrix(conR_f, only_known_targets=True, all_neurons=all_neurons) 
conb_randL_f = con_binary_matrix(con_randL_f, only_known_targets=True, all_neurons=all_neurons)
conb_randR_f = con_binary_matrix(con_randR_f, only_known_targets=True, all_neurons=all_neurons)

# %%

conbL_f, top_targets_df_L = get_top_targets(conbL_f, syn_threshold=3)
conbR_f, top_targets_df_R = get_top_targets(conbR_f, syn_threshold=3)
conb_randL_f, top_targets_df_randL = get_top_targets(conb_randL_f, syn_threshold=3)
conb_randR_f, top_targets_df_randR = get_top_targets(conb_randR_f, syn_threshold=3)

#%% 
top_percentage = 0.01
L_target_df, L_target_counts= get_motifs(conbL_f, type_dict=skid_to_celltype, pairing=pairs_dict)
L_top_motifs = get_top_percentage_motifs(L_target_counts, top_percentage)
R_target_df, R_target_counts = get_motifs(conbR_f, type_dict=skid_to_celltype, pairing=pairs_dict)
R_top_motifs = get_top_percentage_motifs(R_target_counts, top_percentage)
randL_target_df, randL_target_counts = get_motifs(conb_randL_f, type_dict=skid_to_celltype, pairing=pairs_dict)
randL_top_motifs = get_top_percentage_motifs(randL_target_counts, top_percentage)
randR_target_df, randR_target_counts = get_motifs(conb_randR_f, type_dict=skid_to_celltype, pairing=pairs_dict)
randR_top_motifs = get_top_percentage_motifs(randR_target_counts, top_percentage)

print("Top motifs in left hemisphere:")
for motif, count in L_top_motifs.items():
    print(f"{motif}: {count}")
print("\nTop motifs in right hemisphere:")
for motif, count in R_top_motifs.items():
    print(f"{motif}: {count}")
print("\nTop motifs in left hemisphere (random):")
for motif, count in randL_top_motifs.items():
    print(f"{motif}: {count}")
print("\nTop motifs in right hemisphere (random):")
for motif, count in randR_top_motifs.items():
    print(f"{motif}: {count}")

#conbR_fm = get_motifs(conbR_f, type_dict=skid_to_celltype)
# %% examine if there is differences in the neurons involed in specific motifs


one_to_watch = 'PNs'
motifs_to_watchL, target_matL, pair_matL, watch_targetsL, watch_pair_idsL = get_and_plot_motif_targets(one_to_watch, L_top_motifs, L_target_df, celltype_df=celltype_df)
motifs_to_watchR, target_matR, pair_matR, watch_targetsR, watch_pair_idsR = get_and_plot_motif_targets(one_to_watch, R_top_motifs, R_target_df, celltype_df=celltype_df)
motifs_to_watch_randL, target_mat_randL, pair_mat_randL, watch_targets_randL, watch_pair_ids_randL = get_and_plot_motif_targets(one_to_watch, randL_top_motifs, randL_target_df, celltype_df=celltype_df)
motifs_to_watch_randR, target_mat_randR, pair_mat_randR, watch_targets_randR, watch_pair_ids_randR = get_and_plot_motif_targets(one_to_watch, randR_top_motifs, randR_target_df, celltype_df=celltype_df)

# %% compare motifs in real vs rand network 

def run_top_target_ct_motif_functions(conS, all_neurons, skid_to_celltype, pairs_dict, syn_threshold=3, top_percentage=0.01):
    """ 
    Run the top target motif functions for a given connector dataframe.
    """
    conbS = con_binary_matrix(conS, only_known_targets=True, all_neurons=all_neurons)
    conbS, top_targets_df_S = get_top_targets(conbS, syn_threshold=syn_threshold)
    target_df_S, target_counts_S = get_motifs(conbS, type_dict=skid_to_celltype, pairing=pairs_dict)
    top_motifs_S = get_top_percentage_motifs(target_counts_S, top_percentage)

    return target_counts_S, top_motifs_S


type='group'
ct = ['sensories']
ct_names = celltype_df['name'].unique().tolist()
for c in ct_names: 
    conL_f, conR_f = filter_con(conL, conR, pairs_dict=pairs_dict, pairs=pairs, type=type, ct=c, ct_n=5, celltype_df=celltype_df)
    con_randL_f, con_randR_f = filter_con(con_randL, con_randR, pairs_dict=pairs_dict, pairs=pairs, type=type, ct=c, ct_n=5, celltype_df=celltype_df)

    target_counts_L, top_motifs_L = run_top_target_ct_motif_functions(conL_f, all_neurons, skid_to_celltype, pairs_dict)
    target_counts_randL, top_motifs_randL = run_top_target_ct_motif_functions(con_randL_f, all_neurons, skid_to_celltype, pairs_dict)
    target_counts_R, top_motifs_R = run_top_target_ct_motif_functions(conR_f, all_neurons, skid_to_celltype, pairs_dict)
    target_counts_randR, top_motifs_randR = run_top_target_ct_motif_functions(con_randR_f, all_neurons, skid_to_celltype, pairs_dict)

    all_motifs = list(set(target_counts_L.keys()).union(set(target_counts_randL.keys())).union(set(target_counts_R.keys())).union(set(target_counts_randR.keys())))
    all_motifs = sorted(all_motifs, key=lambda x: (len(x), x))  # Sort by length and then lexicographically
    all_motifs = pd.Series(all_motifs)
    counts_df = pd.DataFrame(all_motifs, columns=['motif'])
    counts_df['L'] = counts_df['motif'].apply(lambda x: target_counts_L.get(x, 0))
    counts_df['randL'] = counts_df['motif'].apply(lambda x: target_counts_randL.get(x, 0))
    counts_df['R'] = counts_df['motif'].apply(lambda x: target_counts_R.get(x, 0))
    counts_df['randR'] = counts_df['motif'].apply(lambda x: target_counts_randR.get(x, 0))

    counts_df['L-R'] = counts_df['L'] - counts_df['R']
    counts_df['randL-randR'] = counts_df['randL'] - counts_df['randR']
    counts_df['L-randL'] = counts_df['L'] - counts_df['randL']
    counts_df['R-randR'] = counts_df['R'] - counts_df['randR']
    #counts_df['inter-vs-intra'] = abs(counts_df['L-randL'])/ (abs(counts_df['L-R'])) if counts_df['L-R'] != 0 else 0.01

    #cos_sim = cosine_similarity(counts_df[['L', 'randL']].values.reshape(1, -1))

    counts_df = counts_df.sort_values(by='L-randL', ascending=False)
    print(f"Top motifs for {c}:")
    print(counts_df.head(10))




# %%
