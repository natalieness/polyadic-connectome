
from itertools import chain
from collections import Counter
from itertools import combinations
import os

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from graspologic.utils import binarize
from graspologic.models import SBMEstimator

import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.undirected_graph_functions import get_postsynaptic_co_adj, get_agg_ct_polyadj
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
from scripts.functions.group_based_stats import binarize_poly_adj, get_random_poly_adj, compare_two_sample_chi_squared, correct_pvals, plot_pvals_heatmap, plot_fold_change_heatmap, plot_significant_fold_change_heatmap, get_group_stats_from_bi_adj


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


ct_dict = {sk: ct for sk, ct in zip(mel_AL['skids'], mel_AL['broad_ct'])}

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
# %%
ct = 'broad'
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
    print(poly_adj_ct1)
    stats_chi_ct1_vs_ct2, pvals_uncorrected_ct1_vs_ct2, group_order_ct1_vs_ct2, fold_change_ct1_vs_ct2, g_count_ct1, g_count_ct2 = compare_two_sample_chi_squared(poly_adj_ct1, ct_names_ct1, poly_adj_ct2, ct_names_ct2, test=test)
    reject_ct1_vs_ct2, pvals_corrected_ct1_vs_ct2, asidak_ct1_vs_ct2, acbonf_ct1_vs_ct2 = correct_pvals(pvals_uncorrected_ct1_vs_ct2, method='holm')            
    plot_pvals_heatmap(pvals_corrected_ct1_vs_ct2, group_order_ct1_vs_ct2)
    plot_fold_change_heatmap(fold_change_ct1_vs_ct2, group_order_ct1_vs_ct2)
    plot_significant_fold_change_heatmap(fold_change_ct1_vs_ct2, pvals_corrected_ct1_vs_ct2, group_order_ct1_vs_ct2)   

apply_stats_to_groups(padj, ct_in_padj, padj_rand, ct_in_padj_rand, st=0, test='fisher')

# %%
