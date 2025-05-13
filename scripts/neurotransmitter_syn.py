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

fig, axes = plt.subplots(figsize=(10, 8), nrows=3, ncols=2)
sns.histplot(gaba_n_post, binwidth=1, kde=True, ax=axes[0][0], color='C0', kde_kws={'bw_adjust': 2})
sns.histplot(glut_n_post, binwidth=1, kde=True, ax=axes[0][1], color='C1', kde_kws={'bw_adjust': 2})
sns.histplot(chol_n_post, binwidth=1, kde=True, ax=axes[1][0], color='C2', kde_kws={'bw_adjust': 2})
sns.histplot(dop_n_post, binwidth=1, kde=True, ax=axes[1][1], color='C3', kde_kws={'bw_adjust': 2})
sns.histplot(oct_n_post, binwidth=1, kde=True, ax=axes[2][0], color='C4', kde_kws={'bw_adjust': 2})
names = ['GABAergic', 'Glutamatergic', 'Cholinergic', 'Dopaminergic', 'Octopaminergic']
data = [gaba_n_post, glut_n_post, chol_n_post, dop_n_post, oct_n_post]
for e, ax in enumerate(axes.flatten()[:-1]):
    ax.set_title(f"{names[e]}, mean: {np.mean(data[e]):.2f}", y=1, pad=-10)
    ax.set_xlabel('Number of postsynaptic partners')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 4000)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[2][1].axis('off')

fig.tight_layout()



# %%
