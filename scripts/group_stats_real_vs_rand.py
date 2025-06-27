#%%

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl

from contools import Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
from scripts.functions.group_based_stats import binarize_poly_adj, get_random_poly_adj, compare_two_sample_chi_squared, correct_pvals, plot_pvals_heatmap, plot_fold_change_heatmap, plot_significant_fold_change_heatmap, get_group_stats_from_bi_adj
from scripts.functions.little_helper import get_celltype_dict, celltype_col_for_list, get_celltype_name
from scripts.functions.undirected_graph_functions import get_postsynaptic_co_adj

rm = pymaid.CatmaidInstance(url, token, name, password)

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)


#%% Testing on poly adj matrices imported from csv files and compared against degree-preserved configuration model

''' Note: using binarised adjacency matrix for polyadic connections

Functions only really make sense for this case, 
otherwise the counts of n_possible don't make sense below'''

poly_adj = pd.read_csv('data/poly_adj/adj_all_nonbi.csv').values
ct_labels = pd.read_csv('data/poly_adj/cell_group_labels.csv').values.reshape(-1)

# rather than using the binarised adjacency matrix, binarize here, so that we can use synaptic threshold if desired 

poly_adj = binarize_poly_adj(poly_adj, syn_threshold=0)

# generate random polyadic adjacency matrix to compare

r_poly_adj = get_random_poly_adj(poly_adj, rng)

# compare briefly 
print(f'Shape: poly_adj: {poly_adj.shape}, r_poly_adj: {r_poly_adj.shape}')
print(f'Sum: poly_adj: {np.sum(poly_adj)}, r_poly_adj: {np.sum(r_poly_adj)}')

#  compare group-based statistics
stats_chi, pvals_uncorrected, group_order, fold_change, g_count1, g_count2 = compare_two_sample_chi_squared(poly_adj, ct_labels, r_poly_adj, ct_labels)
reject, pvals_corrected, asidak, acbonf = correct_pvals(pvals_uncorrected, method='holm')


plot_pvals_heatmap(pvals_corrected, group_order)

plot_fold_change_heatmap(fold_change, group_order)

plot_significant_fold_change_heatmap(fold_change, pvals_corrected, group_order)


# from perspective of specific cell types 

get_group_stats_from_bi_adj('KCs')

#%% Applying to random hyperedge permutation 

#%% get cell type and synaptic data 
# get celltype data for each skid 
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()
# get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)
ct_names = celltype_df['name'].unique()

# get synaptic sites from catmaid and describe data
# select neurons to include 
wanted_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(wanted_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# get all synaptic sites associated with neurons 
links = pymaid.get_connector_links(all_neurons, chunk_size=50)

#get connector details 
all_connectors = links['connector_id'].unique()
connector_details = pymaid.get_connector_details(all_connectors)

# remove connector details without presynaptic site 
connector_details = connector_details.dropna(subset=['presynaptic_to'])

# get random permuatted polyadic synaptic connections model (including all connectors)
rand_connectors_all = polyadic_edge_permutation(connector_details, rng)

#  map skid ids in connector details to celltypes
connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))
celltype_col_for_list(connector_details, 'postsynaptic_to', skid_to_celltype=skid_to_celltype, new_col_name='postsynaptic_celltype')

# create subset of connector details with only labelled neurons
connector_details_presyn_labelled = connector_details[connector_details['presynaptic_celltype'] != 'NA']
labelled_connectors = connector_details_presyn_labelled[~connector_details_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
#remove connectors with no labelled postsynaptic celltypes
labelled_connectors = labelled_connectors[labelled_connectors['postsynaptic_celltype'].apply(lambda x: len(x) > 0)]
print(f"Number of connectors with only labelled presynaptic and postsynaptic celltypes: {len(labelled_connectors)}")

# repeat polyadic edege permuattion on filtered network - just to see the difference 
rand_connectors_labelled = polyadic_edge_permutation(labelled_connectors, rng)

# for the purpose of methods used right now, will also filter rand_connectors_all
#to only include labelled presynaptic and postsynaptic celltypes

#map to cell types
rand_connectors_all['presynaptic_celltype'] = rand_connectors_all['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))
celltype_col_for_list(rand_connectors_all, 'postsynaptic_to', skid_to_celltype=skid_to_celltype, new_col_name='postsynaptic_celltype')

#filter 
rand_connectors_all_presyn_labelled = rand_connectors_all[rand_connectors_all['presynaptic_celltype'] != 'NA']
labelled_rand_connectors_all = rand_connectors_all_presyn_labelled[~rand_connectors_all_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
#remove connectors with no labelled postsynaptic celltypes
labelled_rand_connectors_all = labelled_rand_connectors_all[labelled_rand_connectors_all['postsynaptic_celltype'].apply(lambda x: len(x) > 0)]
print(f"Number of random connectors with only labelled presynaptic and postsynaptic celltypes: {len(labelled_rand_connectors_all)}")

# %% apply stats to compare polyadic permutation model to real data 

#get polyadic pairwise incidence matrix from data
poly_adj_real, ordered_ps_in_adj_real = get_postsynaptic_co_adj(labelled_connectors['postsynaptic_to'].tolist())
poly_adj_rand1, ordered_ps_in_adj_rand1 = get_postsynaptic_co_adj(labelled_rand_connectors_all['postsynaptic_to'].tolist())
poly_adj_rand2, ordered_ps_in_adj_rand2 = get_postsynaptic_co_adj(rand_connectors_labelled['postsynaptic_to'].tolist())

poly_adjs = [poly_adj_real, poly_adj_rand1, poly_adj_rand2]
for i in range(len(poly_adjs)):
    poly_adjs[i][:] = binarize_poly_adj(poly_adjs[i], syn_threshold=0)

ct_names_real = [get_celltype_name(skid, skid_to_celltype) for skid in ordered_ps_in_adj_real]
ct_names_rand1 = [get_celltype_name(skid, skid_to_celltype) for skid in ordered_ps_in_adj_rand1]
ct_names_rand2 = [get_celltype_name(skid, skid_to_celltype) for skid in ordered_ps_in_adj_rand2]


#%% compare statistics for real vs random for whole network 

stats_chi_real_vs_rand1, pvals_uncorrected_real_vs_rand1, group_order_real_vs_rand1, fold_change_real_vs_rand1, g_count_real, g_count_rand1 = compare_two_sample_chi_squared(poly_adj_real, ct_names_real, poly_adj_rand1, ct_names_rand1)
reject_real_vs_rand1, pvals_corrected_real_vs_rand1, asidak_real_vs_rand1, acbonf_real_vs_rand1 = correct_pvals(pvals_uncorrected_real_vs_rand1, method='holm')

plot_pvals_heatmap(pvals_corrected_real_vs_rand1, group_order_real_vs_rand1)
plot_fold_change_heatmap(fold_change_real_vs_rand1, group_order_real_vs_rand1)
plot_significant_fold_change_heatmap(fold_change_real_vs_rand1, pvals_corrected_real_vs_rand1, group_order_real_vs_rand1)   


# %% compare with randomised network post filtering 
stats_chi_real_vs_rand2, pvals_uncorrected_real_vs_rand2, group_order_real_vs_rand2, fold_change_real_vs_rand2, g_count_real, g_count_rand2 = compare_two_sample_chi_squared(poly_adj_real, ct_names_real, poly_adj_rand2, ct_names_rand2)
reject_real_vs_rand2, pvals_corrected_real_vs_rand2, asidak_real_vs_rand2, acbonf_real_vs_rand2 = correct_pvals(pvals_uncorrected_real_vs_rand2, method='holm')
plot_pvals_heatmap(pvals_corrected_real_vs_rand2, group_order_real_vs_rand2)
plot_fold_change_heatmap(fold_change_real_vs_rand2, group_order_real_vs_rand2)
plot_significant_fold_change_heatmap(fold_change_real_vs_rand2, pvals_corrected_real_vs_rand2, group_order_real_vs_rand2)   

# %% filter by cell type 

rand_connectors_labelled['presynaptic_celltype'] = rand_connectors_labelled['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))


celltype = 'KCs' 
labelled_connectors_ct = labelled_connectors[labelled_connectors['presynaptic_celltype'] == celltype]
labelled_rand_connectors_ct = labelled_rand_connectors_all[labelled_rand_connectors_all['presynaptic_celltype'] == celltype]
rand_connectors_labelled_ct = rand_connectors_labelled[rand_connectors_labelled['presynaptic_celltype'] == celltype]

poly_adj_ct1, ordered_ps_in_adj_ct1 = get_postsynaptic_co_adj(labelled_connectors_ct['postsynaptic_to'].tolist())
poly_adj_ct2, ordered_ps_in_adj_ct2 = get_postsynaptic_co_adj(labelled_rand_connectors_ct['postsynaptic_to'].tolist())
poly_adj_ct3, ordered_ps_in_adj_ct3 = get_postsynaptic_co_adj(rand_connectors_labelled_ct['postsynaptic_to'].tolist())  

poly_adjs_ct = [poly_adj_ct1, poly_adj_ct2, poly_adj_ct3]
for i in range(len(poly_adjs_ct)):
    poly_adjs_ct[i][:] = binarize_poly_adj(poly_adjs_ct[i], syn_threshold=6)

ct_names_ct1 = [get_celltype_name(skid, skid_to_celltype) for skid in ordered_ps_in_adj_ct1]
ct_names_ct2 = [get_celltype_name(skid, skid_to_celltype) for skid in ordered_ps_in_adj_ct2]
ct_names_ct3 = [get_celltype_name(skid, skid_to_celltype) for skid in ordered_ps_in_adj_ct3]

def apply_stats_to_groups(poly_adj_ct1, ct_names_ct1, poly_adj_ct2, ct_names_ct2):
    stats_chi_ct1_vs_ct2, pvals_uncorrected_ct1_vs_ct2, group_order_ct1_vs_ct2, fold_change_ct1_vs_ct2, g_count_ct1, g_count_ct2 = compare_two_sample_chi_squared(poly_adj_ct1, ct_names_ct1, poly_adj_ct2, ct_names_ct2)
    reject_ct1_vs_ct2, pvals_corrected_ct1_vs_ct2, asidak_ct1_vs_ct2, acbonf_ct1_vs_ct2 = correct_pvals(pvals_uncorrected_ct1_vs_ct2, method='holm')            
    plot_pvals_heatmap(pvals_corrected_ct1_vs_ct2, group_order_ct1_vs_ct2)
    plot_fold_change_heatmap(fold_change_ct1_vs_ct2, group_order_ct1_vs_ct2)
    plot_significant_fold_change_heatmap(fold_change_ct1_vs_ct2, pvals_corrected_ct1_vs_ct2, group_order_ct1_vs_ct2)   

apply_stats_to_groups(poly_adj_ct1, ct_names_ct1, poly_adj_ct3, ct_names_ct3)

    # %%
