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
''' This takes a while - you can just skip and import the csv below'''
all_neus = pymaid.get_neuron(all_neurons)
all_nodes = pymaid.get_node_table(all_neurons)

#first use mw hand annotations to label axon/dendrite
for e, nl in enumerate(all_neus):
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
print(f"Number of unannotated skels after hand-annotation based split: {len(unannotated_skels)}")

# run navis axon dendrite split on neurons that are not annotated 
n_neus = len(all_neus)
for e, nl in enumerate(all_neus):
    skel_id = int(nl.skeleton_id)
    if skel_id in unannotated_skels:
        print('Unannotated skeleton id found')
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

annotated_skels = all_nodes[all_nodes['compartment'].notna()]['skeleton_id'].unique().astype(int)
unannotated_skels = np.setdiff1d(all_neurons, annotated_skels)
print(f"Number of unannotated skels after navis split: {len(unannotated_skels)}")

#  check neurons with no soma 
#soma_values = [neuron.soma for neuron in all_neus]
#none_count = soma_values.count(None)
#print(f"Number of neurons with no soma : {none_count}")

#%%  save annotated nodes to csv 
all_nodes.to_csv(path_for_data+'axo_den_labelled_nodes_new.csv', index=False)

#%% load and create dict of nodes

all_nodes = pd.read_csv(path_for_data+'axo_den_labelled_nodes_new.csv')


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
mpl.rcParams['axes.titlesize'] = 14  # Title font size
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


fig, ax = plt.subplots(1, 1)
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
    G = graph_normalize_weights(G, factor='mean')
    print(f'{name} graph')
    fig_save_path = os.path.join(path_for_data, f'nt_groups/{name}_mean.png')
    plot_nx_graph(G, plot_scale=3, save_fig=True, path=fig_save_path ,title=f'Postsynaptic partners of {name} synapses', node_colors=node_colors, node_size=800, alpha=0.8)

get_postsynaptic_graph_for_any_group(axo_syn_dets['postsynaptic_to_node'], node_to_compartment, 'axonal ')
plt.show()
plt.figure()
get_postsynaptic_graph_for_any_group(den_syn_dets['postsynaptic_to_node'], node_to_compartment, 'dendritic')
plt.show()
# %% sanity check of how many axons vs dendrites are targeted in general 

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axo_syn_dets['postsyn_ad'].explode().value_counts().plot(kind='bar', color=['C0','C1','C2'], alpha=0.8, ax=axes[0], title='Axonal presynapses')
den_syn_dets['postsyn_ad'].explode().value_counts().plot(kind='bar', color=['C0','C1','C2'], alpha=0.8, ax=axes[1], title='Dendritic presynapses')
for ax in axes.flatten():
    ax.set_xlabel('Postsynaptic compartment')
    ax.set_ylabel('No. of synaptic connections')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
# %% examine the actual makeup of hyperedges and potential flow across 

axo_hyperedges = axo_syn_dets['postsynaptic_to_node']
den_hyperedges = den_syn_dets['postsynaptic_to_node']


#%% 

def divide_connector_type_by_length(connectors_data):
    he_comps = connectors_data['postsyn_ad']
    he_comps_by_length = defaultdict(list)
    # divide pattern by length 
    for lens in range(100):
        if any(len(he) == lens for he in he_comps):
            he_comps_by_length[lens] = [he for he in he_comps if len(he) == lens]
    return he_comps_by_length

def get_pure_hyperedges(hyperedges, type_ad):
    pure_ones = [he for he in hyperedges if ((len(set(he)) == 1) and (set(he).pop() == type_ad))]
    n_pure_ones = len(pure_ones)
    return n_pure_ones

def get_hyperedge_ratio(hyperedges):
    ratios = []
    mixed_he = [he for he in hyperedges if len(set(he)) > 1]
    if len(mixed_he) > 0:
        for he in mixed_he:
            axo_count = he.count('axon')
            den_count = he.count('dendrite')
            if den_count == 0:
                print("Can't calculate ratio, no dendrites")
            else:
                ratios.append(axo_count / den_count)
    if len(ratios) == 0:
        print("No mixed hyperedges found")
        return 0, 0, []
    mean_ratio = np.mean(ratios)
    return len(mixed_he), mean_ratio, ratios



def get_partners_by_length_df(he_comps_by_length):
    partners_df = pd.DataFrame(columns=['length', 'n_just_axo', 'n_just_den', 'n_mixed', 'mean_ratio'])
    ratios_list = []
    for lens in he_comps_by_length:
        if len(he_comps_by_length[lens][0]) == 0:
            continue
        print(f"Length {lens}: {len(he_comps_by_length[lens])} axonal hyperedges")
        #get average percentage of 'pure' hyperedges 
        pure_axo = get_pure_hyperedges(he_comps_by_length[lens], 'axon')
        pure_den = get_pure_hyperedges(he_comps_by_length[lens], 'dendrite')
        print(f"Length {lens}: {pure_axo} just axonal partners, {pure_den} just dendritic partners")
        #get number of hyperedges with linkers 
        n_linker = len([he for he in he_comps_by_length[lens] if 'linker' in he])
        print(f"Length {lens}: {n_linker} hyperedges with linkers excluded")
        he_comps_by_length[lens] = [he for he in he_comps_by_length[lens] if 'linker' not in he]

        n_mix, mean_ratio, ratios = get_hyperedge_ratio(he_comps_by_length[lens])
        print(f"Length {lens}: {n_mix} mixed hyperedges, mean axo/den ratio: {mean_ratio:.2f}")
        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            'length': lens,
            'n_just_axo': pure_axo,
            'n_just_den': pure_den,
            'n_mixed': n_mix,
            'mean_ratio': mean_ratio
        }])

        # Concatenate to the existing DataFrame
        partners_df = pd.concat([partners_df, new_row], ignore_index=True)
        ratios_list.append(ratios)
    
    return partners_df,ratios_list

def plot_partners_by_length(partners_df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    df_plot = partners_df.melt(id_vars='length', value_vars=['n_just_axo', 'n_just_den', 'n_mixed'], var_name='Type', value_name='Count')
    sns.barplot(data=df_plot, x='length', y='Count', hue='Type', ax=ax, palette=['C0', 'C1', 'C2'], alpha=0.8)
    ax.set_title('Number of axonal and dendritic postsynaptic partners by hyperedge length')
    fig.tight_layout()

def plot_mean_ratio(partners_df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.barplot(data=partners_df, x='length', y='mean_ratio', ax=ax, color='gray')
    ax.set_title('Mean axo/den ratio of mixed hyperedges')
    fig.tight_layout()


def get_overview_of_hyperedge_identities(connectors_data, name):
    print('Name of group:', name)
    he_comps_by_length = divide_connector_type_by_length(connectors_data)
    partners_df, ratios_list = get_partners_by_length_df(he_comps_by_length)
    plot_partners_by_length(partners_df)
    plot_mean_ratio(partners_df)
    return partners_df , ratios_list
# divide by length
axo_partners_df, axo_ratios_list = get_overview_of_hyperedge_identities(axo_syn_dets, 'Axonal postsynaptic partners')

den_partners_df, den_ratios_list = get_overview_of_hyperedge_identities(den_syn_dets, 'Dendritic postsynaptic partners')
#%%
def ratios_list_to_df( ratios_list):
    ratios_df = pd.DataFrame(ratios_list).transpose()
    return ratios_df

ratio_list_idx = []
for i in range(1, len(axo_ratios_list)):
    ratio_list_idx.append([i]*len(axo_ratios_list[i]))

x= list(chain.from_iterable(ratio_list_idx))
y= list(chain.from_iterable(axo_ratios_list))
plt.scatter(x, y, alpha=0.5)

#%% 
axo_ratios_df = ratios_list_to_df(axo_ratios_list)

sns.stripplot(data=axo_ratios_df, size=5, jitter=1.4)

# %% what if we group hyperadges by neuron? 

def get_random_neuron(neuron_list):
    return np.random.choice(neuron_list)

def get_all_segment_connectors(axo_syn_dets, skid_id):
    axo_connectors = axo_syn_dets[(axo_syn_dets['presynaptic_to'] == skid_id) & (axo_syn_dets['presyn_ad'] == 'axon')]
    return axo_connectors

axo_neu = get_random_neuron(axo_syn_dets['presynaptic_to'])
axo_connectors = get_all_segment_connectors(axo_syn_dets, axo_neu)

random_axo_df, random_axo_ratios_list = get_overview_of_hyperedge_identities(axo_connectors, 'Postsynaptic partners of random axon')
random_axo_ratios_df = ratios_list_to_df(random_axo_ratios_list)
plt.figure()
sns.stripplot(data=random_axo_ratios_df, size=5, jitter=True)


# %%
