''' This scripts maps connectors to the neuronal identity of their pre- and
postsynaptic sites, and then generates a directed graph of pre- to post-synaptic
connections. Using hand annotated neuronal celltypes from catmaid. 
'''
#%%
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

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

# local imports
from scripts.functions.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list
from scripts.functions.undirected_graph_functions import construct_polyadic_incidence_matrix, construct_group_projection_matrix
from scripts.functions.undirected_graph_functions import get_group_pair_counts, build_group_graph, graph_normalize_weights, plot_nx_graph, centered_subgraph

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




# %% get correlation between presynaptic and postsynaptic celltypes (independent of connector)

ct_names = celltype_df['name'].unique()

def get_ct_index(ct_name):
    return np.where(ct_names == ct_name)[0][0]

arr = np.zeros((len(ct_names), len(ct_names)))

#iterate through connectors 
for row in labelled_connectors.iterrows():
    row_presyn = row[1]['presynaptic_celltype']
    row_postsyn = row[1]['postsynaptic_celltype']
    #get index of presynaptic celltype
    presyn_index = get_ct_index(row_presyn)
    #iterate through postsynaptic celltypes
    for post_ct in row_postsyn:
        #get index of postsynaptic celltype
        post_index = get_ct_index(post_ct)
        #increment the value in the array
        arr[presyn_index, post_index] += 1


fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(arr, annot=True, fmt=".0f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names)
ax.set_xlabel('Postsynaptic Celltype')
ax.set_ylabel('Presynaptic Celltype')

#%% compute relative to total presynaptic connections per cell type 

def compute_relative(arr):
    arr_norm = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        row_sum = np.sum(arr[i, :])
        for j in range(arr.shape[1]):
            arr_norm[i, j] = arr[i, j] / row_sum if row_sum != 0 else 0
    return arr_norm

arr_norm = compute_relative(arr)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(arr_norm, annot=True, fmt=".2f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names)
ax.set_xlabel('Postsynaptic Celltype')
ax.set_ylabel('Presynaptic Celltype')



# %%
#clsutermap to sort by similarity
plt.figure()
sns.clustermap(arr, annot=True, fmt=".0f", cmap='magma', xticklabels=ct_names, yticklabels=ct_names)


# %%
