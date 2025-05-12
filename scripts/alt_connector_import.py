''' This script imports connector data from get_connector_links and filters it. 

This has been replace by the connector_details method in map_synskids.py because 
it seems like an easier way to get polyadic synapse data.'''

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
from scripts.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list
from scripts.undirected_graph_functions import construct_polyadic_incidence_matrix, construct_group_projection_matrix
from scripts.undirected_graph_functions import get_group_pair_counts, build_group_graph, graph_normalize_weights, plot_nx_graph, centered_subgraph

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


# %% Filter connectors by presynapse 

connectors = links['connector_id'].unique()
n_connector = links['connector_id'].nunique()

# Check if all connectors have at least one 'presynaptic_to' in the 'relation' column
has_presynaptic = links.groupby('connector_id')['relation'].apply(lambda x: 'presynaptic_to' in x.values)
print(f"Number of connectors with at least one 'presynaptic_to': {has_presynaptic.sum()} out of {n_connector}")

#get connectors with presynaptic site 
connector_with_presyn = has_presynaptic[has_presynaptic].index
#filter connectors by those with presynaptic sites
links_with_presyn = links[links['connector_id'].isin(connector_with_presyn)]


# %% Check if connectors with presynapse belong to any labelled neurons

#get all skids that are in celltype_df (all labelled neurons)
all_labelled_skids = celltype_df['skids'].explode().unique()

# Check if all connectors with presynapse belong to any labelled neurons
has_labelled_neuron = links_with_presyn.groupby('connector_id')['skeleton_id'].apply(lambda x: np.isin(x.values, all_labelled_skids).any())
# %%

n_labelled = has_labelled_neuron.sum()
links_with_labelled = links_with_presyn[links_with_presyn['connector_id'].isin(has_labelled_neuron[has_labelled_neuron].index)]
print(f"Number of connectors with presynapse that belong to labelled neurons: {n_labelled}")
links_with_labelled = links_with_labelled.reset_index(drop=True)
# %% get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)