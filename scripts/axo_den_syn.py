''' This script pulls connectors based on synapses labelled as axonal or dendritic to examine polyadic relationships
and how they relate to axo-dendritic identity.'''

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
import navis

from contools import Celltype, Celltype_Analyzer, Promat
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

# %% get neurons and their axo-dendritic split 
all_neus = pymaid.get_neuron(all_neurons)
all_nodes = pymaid.get_node_table(all_neurons)

# %% annotate nodes with axon/dendrite 
#iterate through all neurons
n_neus = len(all_neus)
for e, nl in enumerate(all_neus):
    #check if neuron has soma 
    if nl.soma is None:
        print(f"Neuron {nl.name} has no soma")
        continue
    #reroot at soma 
    nl.reroot(nl.soma)
    nl_split = navis.split_axon_dendrite(nl)
    #print(f"Neuron {nl.name} has {len(nl_split)} compartments")
    print(f"{e+1}/{n_neus}")
    # get axon and dendrite nodes
    for comp in nl_split:
        c_name = comp.compartment 
        c_nodes = comp.nodes 
        c_node_ids = np.array(c_nodes['node_id'])
        all_nodes.loc[all_nodes['node_id'].isin(c_node_ids),'compartment'] = c_name
# %% handle neurons with no soma 
soma_values = [neuron.soma for neuron in all_neus]
none_count = soma_values.count(None)
print(f"Number of neurons with no soma : {none_count}")

#TODO: maybe add a way to handle neurons without soma? maybe via hand annotated axo/dendrite split?
# for now just filtering them out :) 

# %% save annotated nodes to csv 
all_nodes.to_csv(path_for_data+'axo_den_labelled_nodes.csv', index=False)

#%% load and create dict of nodes

all_nodes = pd.read_csv(path_for_data+'axo_den_labelled_nodes.csv')
#get only those nodes with compartment labels 
all_nodes = all_nodes[all_nodes['compartment'].notna()]
node_to_compartment = dict(zip(all_nodes['node_id'], all_nodes['compartment']))
# %%

connector_details['presyn_ad'] = connector_details['presynaptic_to_node'].apply(lambda x: node_to_compartment.get(x, 'NA'))
connector_details.reset_index(drop=True, inplace=True)
for i in range(len(connector_details)):
    ps_nodes = connector_details['postsynaptic_to_node'].loc[i] 
    ad_list = []
    for p in ps_nodes: 
        ad_list.append(node_to_compartment.get(p, 'NA'))
    connector_details.at[i, 'postsyn_ad'] = ad_list


# %%
