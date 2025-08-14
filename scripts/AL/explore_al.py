
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.undirected_graph_functions import get_postsynaptic_co_adj, get_agg_ct_polyadj
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
from scripts.functions.group_based_stats import binarize_poly_adj, get_random_poly_adj, compare_two_sample_chi_squared, correct_pvals, plot_pvals_heatmap, plot_fold_change_heatmap, plot_significant_fold_change_heatmap, get_group_stats_from_bi_adj
from scripts.functions.motif_functions import con_binary_matrix, get_top_targets, con_bin_cos_sim, get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets, filter_con
from scripts.initialisation_scripts.get_me_started import get_me_started, get_me_labelled

rm = pymaid.CatmaidInstance(url, token, name, password)

rng = np.random.default_rng(42)  # Set a random seed for reproducibility

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% get an organise the AL annotation data 

mel_AL = pd.read_csv('input_data/Dmel_annotation-AL_04-02-2025.csv')
mel_AL['hemi'] = mel_AL['names'].apply(lambda x: 'L' if 'left' in x else 'R' if 'right' in x else 'NA')
# check if any dont have hemisphere 
mel_na = mel_AL[mel_AL['hemi'] == 'NA']
print(f'Cells with no hemisphere: {mel_na["names"]}')

broad_ct = ['ORN', 'mPN', ' PN', 'broad', 'keystone', 'picky', 'choosy', 'ventral']
mel_AL['broad_ct'] = mel_AL['names'].apply(lambda x: [ct for ct in broad_ct if ct in x] if any(ct in x for ct in broad_ct) else 'other')
mel_AL['broad_ct'] = mel_AL['broad_ct'].apply(lambda x: x[0] if isinstance(x, list) else x)
mel_other = mel_AL[mel_AL['broad_ct'] == 'other']
print(f'Cells not in broad categories: {mel_other["names"]}')

mel_AL['ct_cat'] = mel_AL['broad_ct'].apply(lambda x: 'selective_LN' if x in ['picky', 'choosy', 'ventral'] else 'global_LN' if x in ['broad','keystone'] else x)


ct_dict = {sk: ct for sk, ct in zip(mel_AL['skids'], mel_AL['ct_cat'])}

_, _, pairs, pairs_dict, _, _, _= get_me_started()

# %%
links = pymaid.get_connector_links(mel_AL['skids'].tolist(), chunk_size=50)
all_connectors = links['connector_id'].unique()
conAL = pymaid.get_connector_details(all_connectors)
conAL_rand = polyadic_edge_permutation(conAL, rng=rng)

print(f'Average number of postsynaptic partners: {conAL['postsynaptic_to'].apply(lambda x: len(x)).mean()}')

conAL['presynaptic_ct'] = conAL['presynaptic_to'].apply(lambda x: ct_dict.get(x, 'NAL'))
conAL['postsynaptic_ct'] = conAL['postsynaptic_to'].apply(lambda x: [ct_dict.get(sk, 'NAL') for sk in x])
conAL_rand['presynaptic_ct'] = conAL_rand['presynaptic_to'].apply(lambda x: ct_dict.get(x, 'NAL'))
conAL_rand['postsynaptic_ct'] = conAL_rand['postsynaptic_to'].apply(lambda x: [ct_dict.get(sk, 'NAL') for sk in x])

# %% 
def map_connector_type_region(con):
    con['from_region'] = con['presynaptic_ct'].apply(lambda x: 1 if x != 'NAL' else 0)
    con['to_region'] = con['postsynaptic_ct'].apply(lambda x: 1 if any(ct != 'NAL' for ct in x) else 0)
    con['connector_type'] = con.apply(lambda x: 'input' if x['from_region'] == 0 and x['to_region'] == 1 else
                                       'output' if x['from_region'] == 1 and x['to_region'] == 0 else
                                       'AL' if x['from_region'] == 1 and x['to_region'] == 1 else
                                       'NAL', axis=1)
    con.drop(columns=['from_region', 'to_region'], inplace=True)
    return con

conAL = map_connector_type_region(conAL)
conAL_rand = map_connector_type_region(conAL_rand)

conAL['hemi'] = conAL['presynaptic_to'].apply(lambda x: mel_AL[mel_AL['skids'] == x]['hemi'].values[0] if x in mel_AL['skids'].values else 'NA')
conAL_rand['hemi'] = conAL_rand['presynaptic_to'].apply(lambda x: mel_AL[mel_AL['skids'] == x]['hemi'].values[0] if x in mel_AL['skids'].values else 'NA')


# %%
ct = 'other'
conAL_f = conAL[conAL['presynaptic_ct'] == ct]
conAL_f_rand = conAL_rand[conAL_rand['presynaptic_ct'] == ct]
padj, ps_in_padj = get_postsynaptic_co_adj(conAL_f['postsynaptic_to'].tolist())
padj_rand, ps_in_padj_rand = get_postsynaptic_co_adj(conAL_f_rand['postsynaptic_to'].tolist())
ct_in_padj = [ct_dict.get(sk, 'NAL') for sk in ps_in_padj]
ct_in_padj_rand = [ct_dict.get(sk, 'NAL') for sk in ps_in_padj_rand]
padj_agg, unique_cts = get_agg_ct_polyadj(padj, ct_in_padj)
padj_agg_rand, unique_cts_rand = get_agg_ct_polyadj(padj_rand, ct_in_padj_rand)

# plot simple difference 
padj_diff = padj_agg - padj_agg_rand 
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(padj_diff, cmap='coolwarm', vmin=-420, vmax=420, annot=True, fmt='.0f', ax=ax)

# apply cosine similarity 
cos_sim = np.diag(cosine_similarity(padj_agg, padj_agg_rand))
for n, s in zip(padj_agg.columns, cos_sim):
    print(f'Cosine similarity for {n}: {s:.2f}')
# %% performing group based chi squared test on binarised matrices 

### CAVEAT: binarised matrix, think this is a serious issue with this type of data

def apply_stats_to_groups(poly_adj_ct1, ct_names_ct1, poly_adj_ct2, ct_names_ct2, st=0, test='chi2'):
    poly_adj_ct1 = binarize_poly_adj(poly_adj_ct1, syn_threshold=st)
    poly_adj_ct2 = binarize_poly_adj(poly_adj_ct2, syn_threshold=st)
    print(f'For celltype {ct}')
    stats_chi_ct1_vs_ct2, pvals_uncorrected_ct1_vs_ct2, group_order_ct1_vs_ct2, fold_change_ct1_vs_ct2, g_count_ct1, g_count_ct2 = compare_two_sample_chi_squared(poly_adj_ct1, ct_names_ct1, poly_adj_ct2, ct_names_ct2, test=test)
    reject_ct1_vs_ct2, pvals_corrected_ct1_vs_ct2, asidak_ct1_vs_ct2, acbonf_ct1_vs_ct2 = correct_pvals(pvals_uncorrected_ct1_vs_ct2, method='holm')            
    plot_pvals_heatmap(pvals_corrected_ct1_vs_ct2, group_order_ct1_vs_ct2)
    plot_fold_change_heatmap(fold_change_ct1_vs_ct2, group_order_ct1_vs_ct2)
    plot_significant_fold_change_heatmap(fold_change_ct1_vs_ct2, pvals_corrected_ct1_vs_ct2, group_order_ct1_vs_ct2)   

apply_stats_to_groups(padj, ct_in_padj, padj_rand, ct_in_padj_rand, st=0, test='fisher')

# %% split by hemisphere 


def split_left_right(conS):
    conS_L = conS[conS['hemi'] == 'L']
    conS_R = conS[conS['hemi'] == 'R']
    return conS_L, conS_R

ct = ' PN'
conAL_f = conAL[conAL['presynaptic_ct'] == ct]
conAL_f_rand = conAL_rand[conAL_rand['presynaptic_ct'] == ct]
conAL_L, conAL_R = split_left_right(conAL_f)
conAL_randL, conAL_randR = split_left_right(conAL_f_rand)

def run_top_target_ct_motif_functions(conS, all_neurons, skid_to_celltype, pairs_dict, syn_threshold=3, top_percentage=0.01):
    """ 
    Run the top target motif functions for a given connector dataframe.
    """
    conbS = con_binary_matrix(conS, only_known_targets=True, all_neurons=all_neurons)
    conbS, top_targets_df_S = get_top_targets(conbS, syn_threshold=syn_threshold)
    target_df_S, target_counts_S = get_motifs(conbS, type_dict=skid_to_celltype, pairing=pairs_dict)
    top_motifs_S = get_top_percentage_motifs(target_counts_S, top_percentage)

    return top_motifs_S, target_counts_S

top_motifs_L, _ = run_top_target_ct_motif_functions(conAL_L, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)
top_motifs_R, _ = run_top_target_ct_motif_functions(conAL_R, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)
top_motifs_randL, _ = run_top_target_ct_motif_functions(conAL_randL, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)
top_motifs_randR, _ = run_top_target_ct_motif_functions(conAL_randR, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)


# %%

all_counts = pd.DataFrame()

for ct in mel_AL['ct_cat'].unique(): 
    conAL_f = conAL[conAL['presynaptic_ct'] == ct]
    conAL_f_rand = conAL_rand[conAL_rand['presynaptic_ct'] == ct]
    conAL_L, conAL_R = split_left_right(conAL_f)
    conAL_randL, conAL_randR = split_left_right(conAL_f_rand)

    top_motifs_L, top_counts_L = run_top_target_ct_motif_functions(conAL_L, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)
    top_motifs_R, top_counts_R = run_top_target_ct_motif_functions(conAL_R, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)
    top_motifs_randL, top_counts_randL = run_top_target_ct_motif_functions(conAL_randL, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)
    top_motifs_randR, top_counts_randR = run_top_target_ct_motif_functions(conAL_randR, mel_AL['skids'].tolist(), ct_dict, pairs_dict, syn_threshold=3, top_percentage=0.01)

    all_motifs = list(set(top_motifs_L.keys()).union(set(top_motifs_randL.keys())).union(set(top_motifs_R.keys())).union(set(top_motifs_randR.keys())))
    all_motifs = sorted(all_motifs, key=lambda x: (len(x), x))  # Sort by length and then lexicographically
    all_motifs = pd.Series(all_motifs)
    counts_df = pd.DataFrame(all_motifs, columns=['motif'])
    counts_df['L'] = counts_df['motif'].apply(lambda x: top_counts_L.get(x, 0))
    counts_df['randL'] = counts_df['motif'].apply(lambda x: top_counts_randL.get(x, 0))
    counts_df['R'] = counts_df['motif'].apply(lambda x: top_counts_R.get(x, 0))
    counts_df['randR'] = counts_df['motif'].apply(lambda x: top_counts_randR.get(x, 0))

    counts_df['L-R'] = counts_df['L'] - counts_df['R']
    counts_df['randL-randR'] = counts_df['randL'] - counts_df['randR']
    counts_df['L-randL'] = counts_df['L'] - counts_df['randL']
    counts_df['R-randR'] = counts_df['R'] - counts_df['randR']

    counts_df['motif'] = counts_df['motif'].astype(str)
    counts_df.set_index('motif', inplace=True)
    all_counts = pd.concat([all_counts, counts_df], axis=1, join='outer') if not all_counts.empty else counts_df

    #counts_df['inter-vs-intra'] = abs(counts_df['L-randL'])/ (abs(counts_df['L-R'])) if counts_df['L-R'] != 0 else 0.01

    #cos_sim = cosine_similarity(counts_df[['L', 'randL']].values.reshape(1, -1))

    counts_df = counts_df.sort_values(by='L', ascending=False)
    print(f"Top motifs for {ct}:")
    print(counts_df.head(10))

all_counts = all_counts.fillna(0)
all_counts.columns = pd.MultiIndex.from_product([mel_AL['ct_cat'].unique(), ['L', 'randL', 'R', 'randR', 'L-R', 'randL-randR', 'L-randL', 'R-randR']])

# %%
just_L = all_counts.loc[:, (slice(None), 'L-randL')]
just_L_norm = just_L.div(just_L.sum(axis=0), axis=1)
plt.figure(figsize=(12, 8))
plt.imshow(just_L_norm, cmap='coolwarm', aspect=0.3)
plt.yticks(ticks=range(len(just_L_norm.index)), labels=just_L_norm.index)
plt.xticks(ticks=range(len(just_L_norm.columns)), labels=just_L_norm.columns, rotation=90)


# %%
sorted_by_type = all_counts.loc[:, (slice(None), ['L', 'randL', 'R', 'randR'])]
order = ['ORN', ' PN', 'mPN', 'global_LN', 'broad', 'keystone', 'selective_LN', 'picky', 'choosy', 'ventral', 'other']
sorted_by_type.sort_index(axis=1, level=0, inplace=True, sort_remaining=False, key=lambda idx: idx.map(lambda x: order.index(x) if x in order else len(order)))
sorted_by_type_norm = sorted_by_type.div(sorted_by_type.sum(axis=0), axis=1)
#sorted_by_type_norm = sorted_by_type.copy()
max_max = np.max([sorted_by_type_norm.max().max(), abs(sorted_by_type_norm.min().min())])
plt.figure(figsize=(12,8))
#plt.figure(figsize=(21, 19))
plt.imshow(sorted_by_type_norm, cmap='OrRd', aspect=0.6) #, vmin=-max_max, vmax=max_max) # 0.8
plt.yticks(ticks=range(len(sorted_by_type_norm.index)), labels=sorted_by_type_norm.index)
plt.xticks(ticks=range(len(sorted_by_type_norm.columns)), labels=sorted_by_type_norm.columns, rotation=90)


# %% just diff to rand

sorted_by_type = all_counts.loc[:, (slice(None), ['L-randL', 'R-randR'])]

order = ['ORN', ' PN', 'mPN', 'global_LN', 'broad', 'keystone', 'selective_LN', 'picky', 'choosy', 'ventral', 'other']
sorted_by_type.sort_index(axis=1, level=0, inplace=True, sort_remaining=False, key=lambda idx: idx.map(lambda x: order.index(x) if x in order else len(order)))
#sorted_by_type_norm = sorted_by_type.div(sorted_by_type.sum(axis=0), axis=1)
sorted_by_type_norm = sorted_by_type.copy()
max_max = np.max([sorted_by_type_norm.max().max(), abs(sorted_by_type_norm.min().min())])
plt.figure(figsize=(12, 12))
#plt.figure(figsize=(21, 19))
plt.imshow(sorted_by_type_norm, cmap='coolwarm', aspect=0.6, vmin=-max_max, vmax=max_max) # 1
plt.yticks(ticks=range(len(sorted_by_type_norm.index)), labels=sorted_by_type_norm.index)
plt.xticks(ticks=range(len(sorted_by_type_norm.columns)), labels=sorted_by_type_norm.columns, rotation=90)


# %% relative diff 
sorted_by_type = all_counts.loc[:, (slice(None), ['L-randL', 'R-randR', 'L-R'])]
for j in sorted_by_type.columns.levels[0]:
    sorted_by_type[(j, 'L-normdiff')] = sorted_by_type[(j, 'L-randL')] / (sorted_by_type[(j, 'L-R')] + 1e-10)
    sorted_by_type[(j, 'R-normdiff')] = sorted_by_type[(j, 'R-randR')] / (sorted_by_type[(j, 'L-R')] + 1e-10)

#sorted_by_type.drop(columns=[(j, 'L-R') for j in sorted_by_type.columns.levels[0]], inplace=True)

#sorted_by_type_norm = sorted_by_type.div(sorted_by_type.sum(axis=0), axis=1)
sorted_by_type_norm = sorted_by_type.copy()
max_max = np.max([sorted_by_type_norm.max().max(), abs(sorted_by_type_norm.min().min())])
plt.figure(figsize=(12, 12))
plt.imshow(sorted_by_type_norm, cmap='coolwarm', aspect=0.6, vmin=-max_max, vmax=max_max)
plt.yticks(ticks=range(len(sorted_by_type_norm.index)), labels=sorted_by_type_norm.index)
plt.xticks(ticks=range(len(sorted_by_type_norm.columns)), labels=sorted_by_type_norm.columns, rotation=90)