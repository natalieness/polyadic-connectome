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

import matplotlib as mpl
# Set matplotlib parameters to turn off top and right spines globally
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

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


#%% get all nodes without axo-dendritic split and try hand annotated 

#get all skids without annotation 
annotated_skels = all_nodes[all_nodes['compartment'].notna()]['skeleton_id'].unique()
unannotated_skels = np.setdiff1d(all_neurons, annotated_skels)
print(f"Number of unannotated skels: {len(unannotated_skels)}")


for e, nl in enumerate(all_neus):
    #check if neuron skeleton id is in unannotated skels
    skel_id = int(nl.skeleton_id)
    if skel_id in unannotated_skels:
        
        print('Unannotated skeleton id found')
        mw_axon_split = nl.tags.get('mw axon split', None)
        if mw_axon_split:
            if len(mw_axon_split) >1:
                print(f"Tag 'mw axon split' found with multiple values: {mw_axon_split}")
                continue
            else:
                print(f"Tag 'mw axon split' found: {mw_axon_split}")
                nl_axon = nl.prune_proximal_to('mw axon split', inplace=False)
                nl_dend = nl.prune_distal_to('mw axon split', inplace=False)

                #get axon and dendrite nodes 
                axon_nodes = nl_axon.nodes.node_id
                dendrite_nodes = nl_dend.nodes.node_id

                all_nodes.loc[all_nodes['node_id'].isin(axon_nodes), 'compartment'] = 'axon'
                all_nodes.loc[all_nodes['node_id'].isin(dendrite_nodes), 'compartment'] = 'dendrite'
                

        else:
            print("Tag 'mw axon split' not found")

annotated_skels = all_nodes[all_nodes['compartment'].notna()]['skeleton_id'].unique()
unannotated_skels = np.setdiff1d(all_neurons, annotated_skels)
print(f"Number of unannotated skels after hand annotations: {len(unannotated_skels)}")

#%% 
all_nodes.to_csv(path_for_data+'axo_den_labelled_nodes_withmixed_hand_anno.csv', index=False)


#%%

all_nodes = pd.read_csv(path_for_data+'axo_den_labelled_nodes_withmixed_hand_anno.csv')
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


# %% filter connectors by only those with axo-dendritic location known 

print(f"Number of connectors before filtering by axo/den: {len(connector_details)}")
print(f'Prior to filtering, average number of postsynaptic partners: {connector_details["postsynaptic_to_node"].apply(len).mean()}')

#filter 
connector_details = connector_details[connector_details['presyn_ad'] != 'NA']
connector_details = connector_details[~connector_details['postsyn_ad'].apply(lambda x: 'NA' in x)]
print(f"Number of connectors after filtering by axo/den: {len(connector_details)}")
print(f'After filtering, average number of postsynaptic partners: {connector_details["postsynaptic_to_node"].apply(len).mean()}')

# %% get connector details by presynaptic compartment 

axo_syn_dets = connector_details[connector_details['presyn_ad'] == 'axon']
den_syn_dets = connector_details[connector_details['presyn_ad'] == 'dendrite']
# Set global font sizes
mpl.rcParams['axes.titlesize'] = 20  # Title font size
mpl.rcParams['axes.labelsize'] = 16 # Axis label font size
mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick label font size
mpl.rcParams['ytick.labelsize'] = 14 
mpl.rcParams['legend.fontsize'] = 16  # Legend font size

data = [axo_syn_dets['postsyn_ad'].apply(len), den_syn_dets['postsyn_ad'].apply(len)]
names = ['Axon', 'Dendrite']
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(data[0], binwidth=1, ax=axes[0], color='blue', kde=True, kde_kws={'bw_adjust': 4})
sns.histplot(data[1], binwidth=1, ax=axes[1], color='red', kde=True, kde_kws={'bw_adjust': 4})
max_x = max([max(data[e]) for e in range(len(data))])
max_y = max([patch.get_height() for ax in axes.flatten() for patch in ax.patches])
for e, ax in enumerate(axes.flatten()):
    ax.set_title(f"{names[e]}, mean: {np.mean(data[e]):.2f}", y=1, pad=-10)
    ax.set_xlabel('Number of postsynaptic partners')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# Normalize the histograms using `stat='density'`
sns.histplot(data[0], binwidth=1, ax=ax, color='blue', kde=True, kde_kws={'bw_adjust': 4}, line_kws={'lw': 3}, stat='density')
sns.histplot(data[1], binwidth=1, ax=ax, color='red', kde=True, kde_kws={'bw_adjust': 4}, line_kws={'lw': 3}, stat='density')
max_x = max([max(data[e]) for e in range(len(data))])
max_y = 0.5
ax.set_title(f"Normalized number of axonal and dendritic postsynaptic partners", y=1, pad=25)
ax.set_xlabel('Number of postsynaptic partners')
ax.set_ylabel('Density')  
ax.set_xlim(0, max_x)
ax.set_ylim(0, max_y)
ax.legend(['Axon', 'Dendrite'], loc='upper right')
fig.tight_layout()


# %% Figure out connectivity to other compartments 

def get_postsynaptic_graph_for_any_group(hyperedges, vertex_to_group, name, node_colors=None):
    group_pair_counts = get_group_pair_counts(hyperedges, vertex_to_group)
    G = build_group_graph(group_pair_counts, vertex_to_group)
    G = graph_normalize_weights(G, factor='jaccard')
    print(f'{name} graph')
    fig_save_path = os.path.join(path_for_data, f'nt_groups/{name}_graph_jaccard.png')
    plot_nx_graph(G, plot_scale=4, save_fig=True, path=fig_save_path ,title=f'Postsynaptic partners of {name} synapses', node_colors=node_colors, node_size=1500, alpha=0.6)

get_postsynaptic_graph_for_any_group(axo_syn_dets['postsynaptic_to_node'], node_to_compartment, 'axonal ')
plt.show()
plt.figure()
get_postsynaptic_graph_for_any_group(den_syn_dets['postsynaptic_to_node'], node_to_compartment, 'dendritic')
plt.show()
# %%
