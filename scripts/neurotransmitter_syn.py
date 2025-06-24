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
from scipy import stats
from scikit_posthocs import posthoc_dunn
import matplotlib as mpl

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

# local imports
from scripts.functions.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list, get_ct_index
from scripts.functions.undirected_graph_functions import construct_polyadic_incidence_matrix, construct_group_projection_matrix, get_skid_pair_counts, build_skid_graph
from scripts.functions.undirected_graph_functions import get_group_pair_counts, build_group_graph, graph_normalize_weights, plot_nx_graph, centered_subgraph, plot_very_large_graph
from scripts.functions.undirected_postoccurency_matrix_functions import compute_relative_covariance, jaccard_similarity, precompute_P_marginal, compute_PMI

rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% get neuron ids based on neurotransmitter 

#choose how to get the skids - note both can be used together
use_predictions = True #if True, use predictions, if False, use annotations
use_both = True

confidence_threshold = 0.5

if use_both:
    print('Using both predictions and annotations for neurotransmitter skids')
    # get skids for each neurotransmitter from annotations
    nt_preds = pd.read_csv('input_data/all_brain-presynapses-neuron_preds_20250610.csv')
    #filter by confidence threshold
    nt_preds = nt_preds[nt_preds['neuron_confidence'] > confidence_threshold]

    # get skids for each neurotransmitter 
    gaba_ids = nt_preds[nt_preds['neuron_label'] == 'GABA']['skeleton_id'].unique()
    glut_ids = nt_preds[nt_preds['neuron_label'] == 'Glutamate']['skeleton_id'].unique()
    chol_ids = nt_preds[nt_preds['neuron_label'] == 'Acetylcholine']['skeleton_id'].unique()
    dop_ids = nt_preds[nt_preds['neuron_label'] == 'Dopamine']['skeleton_id'].unique()
    oct_ids = nt_preds[nt_preds['neuron_label'] == 'Octopamine']['skeleton_id'].unique()

    catmaid_gaba_ids = pymaid.get_skids_by_annotation('mw GABAergic')
    catmaid_glut_ids = pymaid.get_skids_by_annotation('mw glutamatergic')
    catmaid_chol_ids = pymaid.get_skids_by_annotation('mw cholinergic')
    catmaid_dop_ids = pymaid.get_skids_by_annotation('mw dopaminergic')
    catmaid_oct_ids = pymaid.get_skids_by_annotation('mw octopaminergic')
    # combine the skids from predictions and annotations
    gaba_ids = np.unique(np.concatenate((gaba_ids, catmaid_gaba_ids)))
    glut_ids = np.unique(np.concatenate((glut_ids, catmaid_glut_ids)))
    chol_ids = np.unique(np.concatenate((chol_ids, catmaid_chol_ids)))
    dop_ids = np.unique(np.concatenate((dop_ids, catmaid_dop_ids)))
    oct_ids = np.unique(np.concatenate((oct_ids, catmaid_oct_ids)))


elif (use_predictions== True) and (use_both == False):
    print('Using predictions for neurotransmitter skids')
    nt_preds = pd.read_csv('input_data/all_brain-presynapses-neuron_preds_20250610.csv')
    #filter by confidence threshold
    nt_preds = nt_preds[nt_preds['neuron_confidence'] > confidence_threshold]

    # get skids for each neurotransmitter 
    gaba_ids = nt_preds[nt_preds['neuron_label'] == 'GABA']['skeleton_id'].unique()
    glut_ids = nt_preds[nt_preds['neuron_label'] == 'Glutamate']['skeleton_id'].unique()
    chol_ids = nt_preds[nt_preds['neuron_label'] == 'Acetylcholine']['skeleton_id'].unique()
    dop_ids = nt_preds[nt_preds['neuron_label'] == 'Dopamine']['skeleton_id'].unique()
    oct_ids = nt_preds[nt_preds['neuron_label'] == 'Octopamine']['skeleton_id'].unique()

else:
    print('Using catmaid annotations for neurotransmitter skids')
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

def plot_nt_n_of_postsynaptic_partners_kde(data, names):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for i, (d, name) in enumerate(zip(data, names)):
        sns.kdeplot(
            d, 
            ax=ax, 
            label=f"{name} (mean: {np.mean(d):.2f})", 
            color=colors[i], 
            bw_adjust=3, 
            fill=False,
            alpha=0.8, 
            common_norm=False
        )
    
    ax.set_xlabel('Number of postsynaptic partners')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of postsynaptic partners per neurotransmitter')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    fig.tight_layout()
    return fig

fig = plot_nt_n_of_postsynaptic_partners_kde(data, names)
# %% examine group differences in number of postsynaptic partners

#test for normality
for e, d in enumerate(data):
    stat1, pval1 = stats.shapiro(d)
    stats2, pval2 = stats.normaltest(d)
    if pval1 < 0.05:
        print(f'{names[e]}: Data is not normally distributed (shapiro-wilk test)')
    if pval2 < 0.05:
        print(f'{names[e]}: Data is not normally distributed (D\'Agostino and Pearson\'s test)')

# run Kruskal-Wallis test for differences in number of postsynaptic partners

stat_kw, pval_kw = stats.kruskal(gaba_n_post, glut_n_post, chol_n_post, dop_n_post, oct_n_post)
print(f'Kruskal-Wallis test: H-statistic: {stat_kw:.2f}, p-value: {pval_kw}')

data_df = pd.DataFrame()
data_df['Data'] = np.array(list(chain.from_iterable(data)))
data_df['Group'] = np.array([names[0]] * len(data[0]) + [names[1]]* len(data[1]) + [names[2]] * len(data[2]) + [names[3]] * len(data[3]) + [names[4]] * len(data[4]))

dunn_results = posthoc_dunn(data_df, val_col='Data', group_col='Group', p_adjust='bonferroni')
dunn_results.style.format("{:.4f}")

# %%
mpl.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

# Get unique group names and their positions
group_names = data_df['Group'].unique()
group_names = sorted(group_names, key=lambda x: data_df[data_df['Group'] == x]['Data'].max(), reverse=True)
group_pos = {name: i for i, name in enumerate(group_names)}

fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
colors = ['C2', 'C0', 'C1', 'C3', 'C4']
sns.boxplot(data=data_df, x='Group', y='Data', ax=ax, palette=colors,  order=list(group_pos.keys()))

# Set alpha for boxes
for patch in ax.artists:
    patch.set_alpha(0.7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('Number of postsynaptic partners')



# Set y-offset for asterisks
y_max = data_df['Data'].max()
y_offset = y_max * 0.05

hfont = {'fontname':'Arial'}
n_s = {name: 0 for name in group_names}
if use_predictions == False:
    n_s['GABAergic'] = 1  # Initialize the counter for GABAergic group
if use_both:
    n_s['Glutamatergic'] = 1
for (g1, g2) in combinations(group_names[:], 2):
    pval = dunn_results.loc[g1, g2] if g1 in dunn_results.index and g2 in dunn_results.columns else dunn_results.loc[g2, g1]
    # Choose number of asterisks based on p-value
    if pval < 0.0001:
        stars = '****'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    x1, x2 = group_pos[g1], group_pos[g2]
    # Find the max y-value for each group
    y1 = data_df.loc[data_df['Group'] == g1, 'Data'].max()
    y2 = data_df.loc[data_df['Group'] == g2, 'Data'].max()
    y = max(y1, y2) + y_offset + (3 * n_s[g1])# place annotation just above the higher box
    n_s[g1] += 1
    # Draw the annotation line
    ax.plot([x1, x1, x2, x2], [y, y + y_offset/2, y + y_offset/2, y], lw=1.0, c='k')
    # Place the asterisks
    if stars != 'ns':
        ax.text((x1 + x2) / 2, y + y_offset/2, stars, ha='center', va='bottom', color='k', fontsize=22, fontstyle='italic', fontweight='bold')
    else:
        ax.text((x1 + x2) / 2, y + y_offset/2, stars, ha='center', va='bottom', color='k', fontsize=16 )


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

# %% find the largest connectors to sort out 

chol_df = pd.DataFrame()
chol_df['connector_id'] = chol_syn_dets['connector_id']
chol_df['n partners'] = chol_n_post

chol_df.sort_values(by='n partners', ascending=False, inplace=True)

# %%


