

from itertools import chain
from collections import Counter
from collections import defaultdict
from itertools import combinations
import os

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from contools import Celltype, Celltype_Analyzer, Promat
from graspologic.utils import binarize
from graspologic.models import SBMEstimator
import pymaid
from pymaid_creds import url, name, password, token

# local imports
from scripts.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list, get_ct_index
from scripts.undirected_graph_functions import construct_polyadic_incidence_matrix, construct_group_projection_matrix, get_skid_pair_counts, build_skid_graph
from scripts.undirected_graph_functions import get_group_pair_counts, build_group_graph, graph_normalize_weights, plot_nx_graph, centered_subgraph, plot_very_large_graph
from scripts.undirected_postoccurency_matrix_functions import compute_relative_covariance, jaccard_similarity, precompute_P_marginal, compute_PMI

rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% get and describe neuronal identity data 
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()

print("Cell types used")
n_skids = 0
for ct in celltypes:
    print(f"Name: {ct.get_name()}, Skids: {len(ct.get_skids())}, Color: {ct.get_color()}")
    n_skids += len(ct.get_skids())
print(f"Total number of skids: {n_skids}")

# get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)
ct_names = celltype_df['name'].unique()

#%% get synaptic sites from catmaid and describe data

# select neurons to include 
wanted_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(wanted_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# get all synaptic sites associated with neurons 
links = pymaid.get_connector_links(all_neurons, chunk_size=50)

# inspect connectors in links
print("Connectors in links")
n_entries, n_connectors, n_skeletons, n_nodes, n_postsynaptic, n_presynaptic = inspect_data(links, verbose=True)

#get connector details 
all_connectors = links['connector_id'].unique()
connector_details = pymaid.get_connector_details(all_connectors)

print(f"Of {len(all_connectors)} connectors in links, {len(connector_details)} have details")

# remove connector details without presynaptic site 
connector_details = connector_details.dropna(subset=['presynaptic_to'])

print(f"After removing connectors without presynaptic site, {len(connector_details)} connectors remain")

# %% # map skid ids in connector details to celltypes

connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))

celltype_col_for_list(connector_details, 'postsynaptic_to', skid_to_celltype=skid_to_celltype, new_col_name='postsynaptic_celltype')


# %% create subset of connector details with only labelled neurons
connector_details_presyn_labelled = connector_details[connector_details['presynaptic_celltype'] != 'NA']
labelled_connectors = connector_details_presyn_labelled[~connector_details_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
#remove connectors with no labelled postsynaptic celltypes
labelled_connectors = labelled_connectors[labelled_connectors['postsynaptic_celltype'].apply(lambda x: len(x) > 0)]
print(f"Number of connectors with only labelled presynaptic and postsynaptic celltypes: {len(labelled_connectors)}")
# %% get general description of labelled connectors dataset

n_presynaptic = labelled_connectors['presynaptic_to'].nunique()
print(f"Number of unique presynaptic sites: {n_presynaptic}")
unique_postsynaptic = set(chain.from_iterable(labelled_connectors['postsynaptic_to']))
n_postsynaptic = len(unique_postsynaptic)
print(f"Number of unique postsynaptic sites: {n_postsynaptic}")

for ct in celltype_df['name'].unique():
    n_presynaptic_celltype = labelled_connectors[labelled_connectors['presynaptic_celltype'] == ct]['presynaptic_to'].nunique()
    post_cts_flat = chain.from_iterable(labelled_connectors['postsynaptic_celltype'])
    counts = Counter(post_cts_flat)
    n_postsynaptic_celltype = counts[ct]
    print(f"Number of {ct} presynaptic sites: {n_presynaptic_celltype}, postsynaptic sites: {n_postsynaptic_celltype}")

# %%

#construct undirected adjacency matrix of co-occuring post-synaptic partners (independent of presynaptic partners)

def get_postsynaptic_co_adj(hyperedges):
    """
    Get the adjacency matrix of co-occuring postsynaptic partners from a list of hyperedges.
    """
    # get all unique postsynaptic partners
    all_postsynaptic = set(chain.from_iterable(hyperedges))
    # create a mapping from postsynaptic partner to index
    post_to_index = {post: i for i, post in enumerate(all_postsynaptic)}
    # create an empty adjacency matrix
    adj_matrix = np.zeros((len(all_postsynaptic), len(all_postsynaptic)))
    
    # iterate over hyperedges and fill in the adjacency matrix
    for hyperedge in hyperedges:
        indices = [post_to_index[post] for post in hyperedge]
        for i, j in combinations(indices, 2):
            adj_matrix[i, j] += 1
            adj_matrix[j, i] += 1
    
    #get ordered list of postsynaptic partner names 
    ordered_postsynaptic = [post for post, _ in sorted(post_to_index.items(), key=lambda item: item[1])]

    return adj_matrix, ordered_postsynaptic
#%% get SBM block probabilities for all labelled neurons

hyperedges = labelled_connectors['postsynaptic_to'].tolist()
ps_co_adj, ordered_ps_in_adj = get_postsynaptic_co_adj(hyperedges)
# binarize the adj matrix in preparation for SBM model 
ps_co_adj_bi = binarize(ps_co_adj)

cmap = mpl.colors.ListedColormap(['white', 'black'])
plt.imshow(ps_co_adj_bi, cmap=cmap)
plt.axis('off')

#  get cell types for each postsynaptic partner in the adjacency matrix
ps_celltype_in_adj = [skid_to_celltype[post] for post in ordered_ps_in_adj]

#  get sbm group-to-group connection probabilities 


estimator = SBMEstimator(directed=False, loops=True)
estimator.fit(ps_co_adj_bi, y=ps_celltype_in_adj)

block_probs = pd.DataFrame(estimator.block_p_, index=np.unique(ps_celltype_in_adj), columns=np.unique(ps_celltype_in_adj))
sns.heatmap(block_probs, annot=False, fmt=".1f",cmap='Blues') 
plt.title('Block probabilities for postsynaptic \nco-occurrence in polyadic synapses', fontsize=14)

# %% 
def get_sbm_block_probs_from_hyperedges(hyperedges, name='', plot=True):
    """
    Get the block probabilities for a given set of hyperedges.
    """
    # get the adjacency matrix
    adj_matrix, ordered_ps_in_adj = get_postsynaptic_co_adj(hyperedges)
    
    # binarize the adjacency matrix
    adj_matrix_bi = binarize(adj_matrix)
    if plot:
        cmap = mpl.colors.ListedColormap(['white', 'black'])
        plt.imshow(adj_matrix_bi, cmap=cmap)
        plt.axis('off')
        plt.title(f'Adjacency matrix of polyadic partners\n({name})')
        plt.show()
    
    # get the cell types for each postsynaptic partner
    ps_celltype_in_adj = [skid_to_celltype[post] for post in ordered_ps_in_adj]
    
    # fit the SBM model
    estimator = SBMEstimator(directed=False, loops=True)
    estimator.fit(adj_matrix_bi, y=ps_celltype_in_adj)
    block_probs = pd.DataFrame(estimator.block_p_, index=np.unique(ps_celltype_in_adj), columns=np.unique(ps_celltype_in_adj))
    if plot:
        sns.heatmap(block_probs, annot=False, fmt=".1f", cmap='Blues') 
        plt.title(f'Block probabilities for postsynaptic co-occurrence \n in polyadic synapses ({name})', fontsize=14)
        plt.show()
    return adj_matrix_bi, block_probs, ps_celltype_in_adj
    
hyperedges = labelled_connectors['postsynaptic_to'].tolist()
adj_all, block_probs_all, ps_celltype_in_adj_all = get_sbm_block_probs_from_hyperedges(hyperedges, name='all labelled neurons', plot=True)
#%% get top block probabilities 

def get_top_block_probs(block_probs, n=5, printing=True):
    """
    Get the top n block probabilities from a block probability matrix.
    """
    # get the upper triangle of the matrix
    upper_triangle = np.triu(block_probs.values, k=0)
    # get the indices of the top n block probabilities
    top_indices = np.unravel_index(np.argsort(upper_triangle, axis=None)[-n:], upper_triangle.shape)
    # get the top n block probabilities
    top_block_probs = [(block_probs.index[i], block_probs.columns[j], upper_triangle[i, j]) for i, j in zip(*top_indices)]
    if printing:
        print("Top block probabilities:")
        for i, (ct1, ct2, prob) in enumerate(top_block_probs[::-1]):
            print(f"{i+1}. {ct1} - {ct2}: {prob:.2f}")

    return top_block_probs
top_block_probs = get_top_block_probs(block_probs_all, n=20)


# %% Repeat but filtering by presynaptic cell type

for ct in celltype_df['name'].unique():
    print(f"Cell type: {ct}")
    # get the hyperedges for the current cell type
    hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == ct]['postsynaptic_to'].tolist()
    # get the block probabilities
    adj_ct, block_probs_ct, ps_celltype_in_adj_ct = get_sbm_block_probs_from_hyperedges(hyperedges, name=f'Postsynaptic to {ct}', plot=True)
    # get the top block probabilities
    top_block_probs_ct = get_top_block_probs(block_probs_ct, n=10)
    # save the block probabilities 
    #block_probs.to_csv(path_for_data + f'block_probs_{ct}.csv')
# %% statistically comparing sbm estimators for different subgraphs (based on presynaptic cell type)

#example: kenyon cells and mushroom body output neurons
kc_hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == 'KCs']['postsynaptic_to'].tolist()
adj_kc, block_probs_kc, kc_celltype_in_adj = get_sbm_block_probs_from_hyperedges(kc_hyperedges, name='KCs presynaptic', plot=True)

mbon_hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == 'MBONs']['postsynaptic_to'].tolist()
adj_mbon, block_probs_mbon, mbon_celltype_in_adj = get_sbm_block_probs_from_hyperedges(mbon_hyperedges, name='MBONs presynaptic', plot=True)




# %%
