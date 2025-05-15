''' This script pulls connectors based on neurons labelled with neurotransmitter to examine polyadic relationships
and how they relate to neuronal identity based on neurotransmitter type.'''

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

#%% get neuron ids based on neurotransmitter 

gaba_ids = pymaid.get_skids_by_annotation('mw GABAergic')
glut_ids = pymaid.get_skids_by_annotation('mw glutamatergic')
chol_ids = pymaid.get_skids_by_annotation('mw cholinergic')
dop_ids = pymaid.get_skids_by_annotation('mw dopaminergic')
oct_ids = pymaid.get_skids_by_annotation('mw octopaminergic')

print(f'GABAergic: {len(gaba_ids)}')
print(f'Glutamatergic: {len(glut_ids)}')
print(f'Cholinergic: {len(chol_ids)}')
print(f'Dopaminergic: {len(dop_ids)}')
print(f'Octopaminergic: {len(oct_ids)}')

#create dictionary of neurotransmitter ids
nt_ids = {key: 'GABA' for key in gaba_ids}
nt_ids.update({key: 'Glut' for key in glut_ids})
nt_ids.update({key: 'Chol' for key in chol_ids})
nt_ids.update({key: 'Dop' for key in dop_ids})
nt_ids.update({key: 'Oct' for key in oct_ids})

# %% get associated synapses 

def get_synapses_by_neurotransmitter(nt_ids, name, relation=None):
    #if relation is None, get all synapses associated pre- or post-synaptically
    syns = pymaid.get_connectors(nt_ids, relation_type=relation)
    u_syns = syns['connector_id'].unique()
    syn_dets = pymaid.get_connector_details(syns)
    print(f'{name} synapses: {len(u_syns)}, with details {len(syn_dets)}')
    return syns, u_syns, syn_dets

relation='presynaptic_to'
print(f'Getting {relation} synapses')

gaba_syns, u_gaba_syns, gaba_syn_dets = get_synapses_by_neurotransmitter(gaba_ids, 'GABAergic', relation=relation)
glut_syns, u_glut_syns, glut_syn_dets = get_synapses_by_neurotransmitter(glut_ids, 'Glutamatergic', relation=relation)
chol_syns, u_chol_syns, chol_syn_dets = get_synapses_by_neurotransmitter(chol_ids, 'Cholinergic', relation=relation)
dop_syns, u_dop_syns, dop_syn_dets = get_synapses_by_neurotransmitter(dop_ids, 'Dopaminergic', relation=relation)
oct_syns, u_oct_syns, oct_syn_dets = get_synapses_by_neurotransmitter(oct_ids, 'Octopaminergic', relation=relation)

'''NOTE: not all oct and dop synapses are in the connector details table if no relation_type is provided? '''
# %% generic characterisation by neurotransmitter

def get_synapse_characteristics(syn_dets, name):
    print(f'Getting synapse characteristics for {name} with {len(syn_dets)} synapses')
    edges = list(syn_dets['postsynaptic_to'])
    edge_lengths = [len(edge) for edge in edges]
    max_len = max(edge_lengths)
    min_len = min(edge_lengths)
    mean_len = np.mean(edge_lengths)
    print(f'Polyadic partners: Max: {max_len}, Min: {min_len}, Mean: {mean_len}')
    return edge_lengths

gaba_n_post = get_synapse_characteristics(gaba_syn_dets, 'GABAergic')
glut_n_post = get_synapse_characteristics(glut_syn_dets, 'Glutamatergic')
chol_n_post = get_synapse_characteristics(chol_syn_dets, 'Cholinergic')
dop_n_post = get_synapse_characteristics(dop_syn_dets, 'Dopaminergic')
oct_n_post = get_synapse_characteristics(oct_syn_dets, 'Octopaminergic')

# %% plot histogram of number of postsynaptic polyadic partners per neurotransmitter

names = ['GABAergic', 'Glutamatergic', 'Cholinergic', 'Dopaminergic', 'Octopaminergic']
data = [gaba_n_post, glut_n_post, chol_n_post, dop_n_post, oct_n_post]

def plot_nt_n_of_postsynaptic_partners(data, names):
    fig, axes = plt.subplots(figsize=(10, 8), nrows=3, ncols=2)
    sns.histplot(data[0], binwidth=1, kde=True, ax=axes[0][0], color='C0', kde_kws={'bw_adjust': 2})
    sns.histplot(data[1], binwidth=1, kde=True, ax=axes[0][1], color='C1', kde_kws={'bw_adjust': 2})
    sns.histplot(data[2], binwidth=1, kde=True, ax=axes[1][0], color='C2', kde_kws={'bw_adjust': 2})
    sns.histplot(data[3], binwidth=1, kde=True, ax=axes[1][1], color='C3', kde_kws={'bw_adjust': 2})
    sns.histplot(data[4], binwidth=1, kde=True, ax=axes[2][0], color='C4', kde_kws={'bw_adjust': 2})
    max_x = max([max(data[e]) for e in range(len(data))])
    max_y = max([patch.get_height() for ax in axes.flatten() for patch in ax.patches])
    for e, ax in enumerate(axes.flatten()[:-1]):
        ax.set_title(f"{names[e]}, mean: {np.mean(data[e]):.2f}", y=1, pad=-10)
        ax.set_xlabel('Number of postsynaptic partners')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[2][1].axis('off')
    fig.tight_layout()
    return fig
fig = plot_nt_n_of_postsynaptic_partners(data, names)


# %% look at distribution of postsynaptic partners per neurotransmitter 

#start with GABA
gaba_ps_partners = gaba_syn_dets['postsynaptic_to']
glut_ps_partners = glut_syn_dets['postsynaptic_to']
chol_ps_partners = chol_syn_dets['postsynaptic_to']
dop_ps_partners = dop_syn_dets['postsynaptic_to']
oct_ps_partners = oct_syn_dets['postsynaptic_to']

# get general overview of the postsynaptic partner identity
def get_ps_partner_identity_flat(ps_partners, nt_ids, name):
    print(f'Getting postsynaptic partner identity for {name} with {len(ps_partners)} synapses')
    ps_partners_flat = ps_partners.explode()
    ps_nt = [nt_ids.get(skid, 'NA') for skid in ps_partners_flat]
    return ps_nt

def plot_ps_partner_identity_flat(ps_partner, nt_ids, name, ax, a1=0, a2=0, c=0):
    ps_nt = get_ps_partner_identity_flat(ps_partner, nt_ids, name)
    n_NA = (ps_nt.count('NA')/len(ps_nt))*100
    ps_nt = [x for x in ps_nt if x != 'NA']
    sns.histplot(ps_nt, ax=ax[a1][a2], kde=False, binwidth=1, color='C%i'%c)
    ax[a1][a2].annotate(f'{n_NA:.0f}% unknown', xy=(0.8, 0.98), xycoords='axes fraction', ha='center', fontsize=11)
    ax[a1][a2].set_title(f"{name}", y=1, pad=-2)
    c =+ 1
    
fig, ax = plt.subplots(nrows=3, ncols =2, figsize=(10, 8))
plot_ps_partner_identity_flat(gaba_ps_partners, nt_ids, 'GABAergic', ax, 0, 0, 0)
plot_ps_partner_identity_flat(glut_ps_partners, nt_ids, 'Glutamatergic', ax, 0, 1, 1)
plot_ps_partner_identity_flat(chol_ps_partners, nt_ids, 'Cholinergic', ax, 1, 0, 2)
plot_ps_partner_identity_flat(dop_ps_partners, nt_ids, 'Dopaminergic', ax, 1, 1, 3)
plot_ps_partner_identity_flat(oct_ps_partners, nt_ids, 'Octopaminergic', ax, 2, 0, 4)
ax[2][1].axis('off')
for e, axs in enumerate(ax.flatten()[:-1]):
    axs.set_ylabel('No. of postsynaptic partners')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

fig.tight_layout()

# %% looking at filtered polyadic postsynaptic partners by neurotransmitter

#map ps_partners to neurotransmitter
def ps_partner_to_nt(ps_partners, nt_ids):
    nt_partners = []
    for i, ps in enumerate(ps_partners):
       ps_partners = nt_partners.append([nt_ids.get(skid, 'NA') for skid in ps])
    return nt_partners

def filter_nt_partners(ps_partners, nt_ids, verbose=True):
    nt_partners = ps_partner_to_nt(ps_partners, nt_ids)
    nt_partners_filtered = []
    nt_skid_partners_filtered = []
    for i, ps in enumerate(nt_partners):
        if 'NA' not in ps:
            nt_partners_filtered.append(ps)
            nt_skid_partners_filtered.append(ps_partners[i])
    nt_partners_filtered.pop(0)
    nt_skid_partners_filtered.pop(0)
    if verbose:
        print(f'Filtering by NT identity leaves {len(nt_partners_filtered)} synapses out of {len(nt_partners)}')
    return nt_partners_filtered, nt_skid_partners_filtered


gaba_ps_nt_filtered, gaba_ps_skids_f = filter_nt_partners(gaba_ps_partners, nt_ids)
glut_ps_nt_filtered, glut_ps_skids_f = filter_nt_partners(glut_ps_partners, nt_ids)
chol_ps_nt_filtered, chol_ps_skids_f = filter_nt_partners(chol_ps_partners, nt_ids)
dop_ps_nt_filtered, dop_ps_skids_f = filter_nt_partners(dop_ps_partners, nt_ids)
oct_ps_nt_filtered, oct_ps_skids_f = filter_nt_partners(oct_ps_partners, nt_ids)


# %% 
data_ps = [gaba_ps_nt_filtered, glut_ps_nt_filtered, chol_ps_nt_filtered, dop_ps_nt_filtered, oct_ps_nt_filtered]
names = ['GABAergic', 'Glutamatergic', 'Cholinergic', 'Dopaminergic', 'Octopaminergic']

def get_nt_ps_n(data_ps):
    data_n = []
    for l in data_ps:
        l_list = []
        for ps in l:
            l_list.append(len(ps))
        data_n.append(l_list)
    return data_n

data_n = get_nt_ps_n(data_ps)
fig = plot_nt_n_of_postsynaptic_partners(data_n, names)
# %%

def get_postsynaptic_graph_for_nt_group(hyperedges, vertex_to_group, name, node_colors=None):
    group_pair_counts = get_group_pair_counts(hyperedges, vertex_to_group)
    G = build_group_graph(group_pair_counts, vertex_to_group)
    G = graph_normalize_weights(G, factor='jaccard')
    print(f'{name} graph')
    fig_save_path = os.path.join(path_for_data, f'nt_groups/{name}_graph_jaccard.png')
    plot_nx_graph(G, plot_scale=4, save_fig=True, path=fig_save_path ,title=f'Postsynaptic partners of {name} synapses', node_colors=node_colors, node_size=1500)

data_skids = [gaba_ps_skids_f, glut_ps_skids_f, chol_ps_skids_f, dop_ps_skids_f, oct_ps_skids_f]
nt_colors = {'GABA': '#49a2e0', 'Glut': '#ffb97b', 'Chol': '#81dc81', 'Dop': '#e98787', 'Oct': '#b99cd4'}
for d_skid, nam in zip(data_skids, names):
    plt.figure()
    get_postsynaptic_graph_for_nt_group(d_skid, nt_ids, nam, node_colors=nt_colors)

# %%
