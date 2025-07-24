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

from scripts.polyadic_motifs.motif_functions import (con_binary_matrix, get_top_targets, con_bin_cos_sim, 
get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets, 
normalise_freq_cols, filter_con)
from scripts.initialisation_scripts.get_me_started import get_me_started
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict = get_me_started()

all_neurons = connector_details['presynaptic_to'].unique().tolist()

#%% 
# filter by hemisphere 
conA = connector_details.copy()
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']

#%% filtering and having a look at specific ones - not in a loop like below
conL_f, conR_f = filter_con(conL, conR, type='group', ct='MBONs', ct_n=5)
con_f = filter_con(conA, type='group', ct='MBONs', ct_n=5)

#
conbL_f = con_binary_matrix(conL_f, only_known_targets=True, all_neurons=all_neurons)
conbR_f = con_binary_matrix(conR_f, only_known_targets=True, all_neurons=all_neurons) 
conbA_f = con_binary_matrix(con_f, only_known_targets=True, all_neurons=all_neurons)
# 

conbL_f, top_targets_df_L = get_top_targets(conbL_f, syn_threshold=3)
conbR_f, top_targets_df_R = get_top_targets(conbR_f, syn_threshold=3)
conbA_f, top_targets_df_A = get_top_targets(conbA_f, syn_threshold=3)

# try getting flow motifs instead of celltype motifs 

flow_df_L, flow_targets_L_counts = get_simple_flow_motifs(conbL_f, conL_f, flow_dict=flow_dict, pairing=pairs_dict, flow_norm=True)
flow_df_R, flow_targets_R_counts = get_simple_flow_motifs(conbR_f, conR_f, flow_dict=flow_dict, pairing=pairs_dict, flow_norm=True)
flow_df_A, flow_targets_A_counts = get_simple_flow_motifs(conbA_f, con_f, flow_dict=flow_dict, pairing=pairs_dict, flow_norm=True)

print("Top motifs in left hemisphere:")
for motif, count in flow_targets_L_counts.items():
    print(f"{motif}: {count}")
print("\nTop motifs in right hemisphere:")
for motif, count in flow_targets_R_counts.items():
    print(f"{motif}: {count}")
print("\nTop motifs in whole brain:")
for motif, count in flow_targets_A_counts.items():
    print(f"{motif}: {count}")
#conbR_fm = get_motifs(conbR_f, type_dict=skid_to_celltype)



# %% get overview of ff / fb across celltypes 



def get_flow_for_each_celltype(conL, conR=None, celltype_df=None, flow_dict=None, pairs_dict=None):
    ffb_ct_motifs_L = pd.DataFrame(columns=['celltype', 'ff', 'fb', 'ff_fb'])
    ffb_ct_motifs_R = pd.DataFrame(columns=['celltype', 'ff', 'fb', 'ff_fb'])

    mean_flow_presyn = []
    for ct in celltype_df['name'].unique():
        if conR is None:
            conL_f, _ = filter_con(conL, type='group', ct=ct)
        else:
            conL_f, conR_f = filter_con(conL, conR, type='group', ct=ct)


        conbL_f = con_binary_matrix(conL_f, only_known_targets=True, all_neurons=all_neurons)
        conbL_f, top_targets_df_L = get_top_targets(conbL_f, syn_threshold=3)
        flow_df_L, flow_targets_L_counts = get_simple_flow_motifs(conbL_f, conL_f, flow_dict=flow_dict, pairing=pairs_dict, flow_norm=True)

        if conR is not None:
            conbR_f = con_binary_matrix(conR_f, only_known_targets=True, all_neurons=all_neurons) 
            conbR_f, top_targets_df_R = get_top_targets(conbR_f, syn_threshold=3)
            flow_df_R, flow_targets_R_counts = get_simple_flow_motifs(conbR_f, conR_f, flow_dict=flow_dict, pairing=pairs_dict, flow_norm=True)

        # get mean flow value of presynaptic neurons 
        mean_flow_presyn_L = np.mean([flow_dict.get(skid, 0) for skid in conL_f['presynaptic_to']])

        if conR is not None:
            mean_flow_presyn_R = np.mean([flow_dict.get(skid, 0) for skid in conR_f['presynaptic_to']])
            mean_flow_presyn.append(np.mean([mean_flow_presyn_L, mean_flow_presyn_R]))
        else:
            mean_flow_presyn.append(mean_flow_presyn_L)

        left_row = pd.DataFrame({
            'celltype': ct,
            'ff': flow_targets_L_counts[('FF',)],
            'fb': flow_targets_L_counts[('FB',)],
            'ff_fb': flow_targets_L_counts[('FB', 'FF')]
        }, index=[0])
        ffb_ct_motifs_L = pd.concat([ffb_ct_motifs_L, left_row], ignore_index=True)
        ffb_ct_motifs_L['whole_brain_avg_presyn_flow'] = mean_flow_presyn
        if conR is not None:
            right_row = pd.DataFrame({
                'celltype': ct,
                'ff': flow_targets_R_counts[('FF',)],
                'fb': flow_targets_R_counts[('FB',)],
                'ff_fb': flow_targets_R_counts[('FB', 'FF')]
            }, index=[0])
        
            ffb_ct_motifs_R = pd.concat([ffb_ct_motifs_R, right_row], ignore_index=True)
            ffb_ct_motifs_R['whole_brain_avg_presyn_flow'] = mean_flow_presyn

    return ffb_ct_motifs_L, ffb_ct_motifs_R

ffb_ct_motifs_L, ffb_ct_motifs_R = get_flow_for_each_celltype(conL, conR, celltype_df, flow_dict, pairs_dict)
ffb_ct_motifs_A, _ = get_flow_for_each_celltype(conA, None, celltype_df, flow_dict, pairs_dict)


#%%

ffb_ct_motifs_L = normalise_freq_cols(ffb_ct_motifs_L)
ffb_ct_motifs_R = normalise_freq_cols(ffb_ct_motifs_R)
ffb_ct_motifs_A = normalise_freq_cols(ffb_ct_motifs_A)

ffb_ct_motifs_L.sort_values(by='whole_brain_avg_presyn_flow', ascending=False, inplace=True)
ffb_ct_motifs_R.sort_values(by='whole_brain_avg_presyn_flow', ascending=False, inplace=True)
ffb_ct_motifs_A.sort_values(by='whole_brain_avg_presyn_flow', ascending=False, inplace=True)

# %%
fig, axes = plt.subplots(1, 1)

x = np.arange(len(ffb_ct_motifs_L['celltype']))
bar_width = 0.35
bar_space = 0.2

# Extract individual components
ff = ffb_ct_motifs_L['ff_norm']
ff_fb = ffb_ct_motifs_L['ff_fb_norm']
fb = ffb_ct_motifs_L['fb_norm']
ffr = ffb_ct_motifs_R['ff_norm']
ff_fbr = ffb_ct_motifs_R['ff_fb_norm']
fbr = ffb_ct_motifs_R['fb_norm']
# Plot stacked bars
axes.bar(x-bar_space, fb,  color='#58A1DB', width=bar_width)
axes.bar(x-bar_space, ff_fb, bottom=fb,  color='#A1DB58', width=bar_width)
axes.bar(x-bar_space, ff, bottom=fb + ff_fb,  color='#DB58A1', width=bar_width)
axes.bar(x+bar_space, fbr, color='#58A1DB', width=bar_width)
axes.bar(x+bar_space, ff_fbr, bottom=fbr,  color='#A1DB58', width=bar_width)
axes.bar(x+bar_space, ffr, bottom=fbr + ff_fbr,  color='#DB58A1', width=bar_width)

for xi in x:
    axes.text(xi-bar_space-(bar_width/2), 1.02, 'L')
    axes.text(xi, 1.02, 'R')

# Customize ticks and labels
axes.set_xticks(x)
axes.set_xticklabels(ffb_ct_motifs_L['celltype'], rotation=45, ha='right')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_ylabel('Normalized Proportion')
axes.legend(['FB', 'FF_FB', 'FF'], loc='upper right', frameon=True, edgecolor='white')
plt.tight_layout()

# Plot for whole brain 
fig, axes = plt.subplots(1, 1)
x = np.arange(len(ffb_ct_motifs_A['celltype']))
bar_width = 0.6
ff = ffb_ct_motifs_A['ff_norm']
ff_fb = ffb_ct_motifs_A['ff_fb_norm']
fb = ffb_ct_motifs_A['fb_norm']
# Plot stacked bars
axes.bar(x, fb,  color='#58A1DB', width=bar_width)
axes.bar(x, ff_fb, bottom=fb,  color='#A1DB58', width=bar_width)
axes.bar(x, ff, bottom=fb + ff_fb,  color='#DB58A1', width=bar_width)
axes.set_xticks(x)
axes.set_xticklabels(ffb_ct_motifs_A['celltype'], rotation=45, ha='right')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_ylabel('Normalized Proportion')
axes.legend(['FB', 'FF_FB', 'FF'], loc='upper right', frameon=True, edgecolor='white')
plt.tight_layout()


# %% Compare with random premutated polyadic network
#### TODO: Compare with rand permutated network 

con_rand = polyadic_edge_permutation(connector_details, rng=rng)
#%%
con_rand['presyn_hemi'] = con_rand['presynaptic_to'].apply(lambda x: 'right' if x in pairs['rightid'].values else 'left' if x in pairs['leftid'].values else None)
con_rand['presynaptic_celltype'] = con_rand['presynaptic_to'].apply(lambda x: skid_to_celltype.get(x, None))
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']


rand_ffb_ct_motifs_L, rand_ffb_ct_motifs_R = get_flow_for_each_celltype(con_randL, con_randR, celltype_df, flow_dict, pairs_dict)
rand_ffb_ct_motifs_L = normalise_freq_cols(rand_ffb_ct_motifs_L)
rand_ffb_ct_motifs_R = normalise_freq_cols(rand_ffb_ct_motifs_R)
rand_ffb_ct_motifs_L.sort_values(by='whole_brain_avg_presyn_flow', ascending=False, inplace=True)
rand_ffb_ct_motifs_R.sort_values(by='whole_brain_avg_presyn_flow', ascending=False, inplace=True)

#%%
fig, axes = plt.subplots(1, 1)

x = np.arange(len(rand_ffb_ct_motifs_L['celltype']))
bar_width = 0.2
bar_space = 0.1


# Extract individual components
ff = ffb_ct_motifs_L['ff_norm']
ff_fb = ffb_ct_motifs_L['ff_fb_norm']
fb = ffb_ct_motifs_L['fb_norm']
ffr = ffb_ct_motifs_R['ff_norm']
ff_fbr = ffb_ct_motifs_R['ff_fb_norm']
fbr = ffb_ct_motifs_R['fb_norm']
ff_rand = rand_ffb_ct_motifs_L['ff_norm']
ff_fb_rand = rand_ffb_ct_motifs_L['ff_fb_norm']
fb_rand = rand_ffb_ct_motifs_L['fb_norm']
ffr_rand = rand_ffb_ct_motifs_R['ff_norm']
ff_fbr_rand = rand_ffb_ct_motifs_R['ff_fb_norm']
fbr_rand = rand_ffb_ct_motifs_R['fb_norm']
# Plot stacked bars
axes.bar(x-bar_space-bar_width, fb,  color='#58A1DB', width=bar_width)
axes.bar(x-bar_space-bar_width, ff_fb, bottom=fb,  color='#A1DB58', width=bar_width)
axes.bar(x-bar_space-bar_width, ff, bottom=fb + ff_fb,  color='#DB58A1', width=bar_width)
axes.bar(x-bar_space, fbr, color='#58A1DB', width=bar_width)
axes.bar(x-bar_space, ff_fbr, bottom=fbr,  color='#A1DB58', width=bar_width)
axes.bar(x-bar_space, ffr, bottom=fbr + ff_fbr,  color='#DB58A1', width=bar_width)
axes.bar(x+bar_space, fb_rand,  color='#58A1DB', width=bar_width, alpha=0.5)
axes.bar(x+bar_space, ff_fb_rand, bottom=fb_rand,  color='#A1DB58', width=bar_width, alpha=0.5)
axes.bar(x+bar_space, ff_rand, bottom=fb_rand + ff_fb_rand,  color='#DB58A1', width=bar_width, alpha=0.5)
axes.bar(x+bar_space+bar_width, fbr_rand, color='#58A1DB', width=bar_width, alpha=0.5)
axes.bar(x+bar_space+bar_width, ff_fbr_rand, bottom=fbr_rand,  color='#A1DB58', width=bar_width, alpha=0.5)
axes.bar(x+bar_space+bar_width, ffr_rand, bottom=fbr_rand + ff_fbr_rand,  color='#DB58A1', width=bar_width, alpha=0.5)

for xi in x:
    axes.text(xi-bar_width*2, 1.02, 'Real', rotation=90)
    axes.text(xi, 1.02, 'Rand', rotation=90)


# Customize ticks and labels
axes.set_xticks(x)
axes.set_xticklabels(ffb_ct_motifs_L['celltype'], rotation=45, ha='right')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_ylabel('Normalized Proportion')
axes.legend(['FB', 'FF_FB', 'FF'], loc='upper right', frameon=True, edgecolor='white')
plt.tight_layout()

#%% 
# 
#### TODO: Examine the composition of FF/FB motifs, relatively 