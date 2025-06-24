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


#%% generate directed graph of presynaptic to postsynaptic connections

def plot_directed_graph(arr, ct_names, celltype_df, norm='pre'):

    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for presynaptic and postsynaptic celltypes
    for ct in ct_names:
        G.add_node(ct, type='celltype', color=celltype_df[celltype_df['name'] == ct]['color'].values[0])
    
    # Add edges for presynaptic to postsynaptic connections
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > 0:
                presyn_ct = ct_names[i]
                postsyn_ct = ct_names[j]
                G.add_edge(presyn_ct, postsyn_ct, weight=arr[i, j])
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    if norm == 'pre':
        # Normalize edge weights by presynaptic celltype
        for ct in ct_names:
            presyn_edges = [G[u][v]['weight'] for u, v in G.out_edges(ct)]
            if presyn_edges:
                norm_factor = sum(presyn_edges)
                for u, v in G.out_edges(ct):
                    G[u][v]['weight'] /= norm_factor if norm_factor != 0 else 1
    
    # Scale all weights by a factor for visualization
    plot_scale = 0.0006
    edges_scaled = [i * plot_scale for i in edge_weights]
    
    # remove nodes 
    #nodes_to_remove = ['CNs', 'pre-dVNCs', 'dVNCs', 'RGNs', 'LNs', 'dSEZs', 'pre-dSEZs', 'ascendings', 'LHNs', 'CNs']
    #G.remove_nodes_from(nodes_to_remove)

    # Plot directed graph
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, width=edges_scaled,
            node_color=[G.nodes[n]['color'] for n in G.nodes],
            node_size=1000, font_size=10, font_color='black', arrows=True)
    return G
    
G = plot_directed_graph(arr, ct_names, celltype_df)

#%% 

def get_in_and_Out_weights_for_neuronct(G, neuron='LHNs'):
    in_weights = [(u, G[u][v]['weight']) for u, v in G.in_edges(neuron)]
    out_weights = [(v, G[u][v]['weight']) for u, v in G.out_edges(neuron)]
    in_weights = sorted(in_weights, key=lambda x: x[1], reverse=True)
    out_weights = sorted(out_weights, key=lambda x: x[1], reverse=True)
    return in_weights, out_weights

def plot_in_out(in_weights, out_weights, neuron='LHNs'):
    GN = nx.DiGraph()
    # Add the neuron node
    GN.add_node(neuron, type='celltype', color='lightblue')
    # Add incoming edges
    for u, weight in in_weights:
        GN.add_edge(u, neuron, weight=weight)
    # Add outgoing edges
    for v, weight in out_weights:
        GN.add_edge(neuron, v, weight=weight)
    plot_scale = 30
    edge_weights = [GN[u][v]['weight'] * plot_scale for u, v in GN.edges()]
    # Plot the graph
    pos = nx.circular_layout(GN)  # positions for all nodes
    #nx.draw(GN, pos, with_labels=True, width=edge_weights,
     #       node_color=[GN.nodes[n].get('color', 'lightblue') for n in GN.nodes],
      #      node_size=1000, font_size=10, font_color='black', arrows=True)
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(GN, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(GN, pos, font_size=10)

    incoming_edges = [u  for u, _ in in_weights]
    outgoing_edges = [u for u, _ in out_weights]
    # Draw edges with curvature for both directions
    connection_style = ['arc3, rad=0.2' for _ in incoming_edges] + ['arc3, rad=-0.2' for _ in outgoing_edges]
    edges = [(u, neuron) for u in incoming_edges] + [(neuron, u) for u in outgoing_edges]
    edge_weights = [w for _, w in in_weights] + [w for _, w in out_weights]
    plot_scale = 20
    edge_weights = [w * plot_scale for w in edge_weights]

    nx.draw_networkx_edges(GN, pos, edgelist=edges, width=edge_weights,
                       connectionstyle=connection_style, arrowstyle='-|>', arrowsize=20)
    
    plt.title(f"Incoming and outgoing connections from {neuron}")
    return GN

    

LHNs_in, LHNs_out = get_in_and_Out_weights_for_neuronct(G, neuron='LHNs')
GN = plot_in_out(LHNs_in, [], neuron='LHNs')
    
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


