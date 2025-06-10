''' This scripts maps connectors to the neuronal identity of their pre- and
postsynaptic sites. Using hand annotated neuronal celltypes from catmaid
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
from scripts.functions.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list, get_ct_index
from scripts.functions.undirected_graph_functions import construct_polyadic_incidence_matrix, construct_group_projection_matrix, get_skid_pair_counts, build_skid_graph
from scripts.functions.undirected_graph_functions import get_group_pair_counts, build_group_graph, graph_normalize_weights, plot_nx_graph, centered_subgraph, plot_very_large_graph
from scripts.functions.undirected_postoccurency_matrix_functions import compute_relative_covariance, jaccard_similarity, precompute_P_marginal, compute_PMI
from scripts.functions.synapse_quality_checks import count_postsyn_mutiples_of_neurons

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

#%% sort through connectors to spot potential issues: 

multi_locs = count_postsyn_mutiples_of_neurons(connector_details, mode='neuron')
multi_locs_node = count_postsyn_mutiples_of_neurons(connector_details, mode='node')

# check for same postsynaptic node co-occuring multiple times

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

# %% get co-occurency matrix between postsynaptic celltypes (independent of presynaptic partner)

post_cooccurency = np.zeros((len(ct_names), len(ct_names)))
#iterate through connectors
for row in labelled_connectors.iterrows():
    row_postsyn = row[1]['postsynaptic_celltype']
    #iterate through postsynaptic celltypes
    for e, post_ct in enumerate(row_postsyn):
        for e2, post_ct2 in enumerate(row_postsyn[e+1:]):
            #get index of postsynaptic celltype
            post_index = get_ct_index(post_ct, ct_names)
            post_index2 = get_ct_index(post_ct2, ct_names)
            #increment the value in the array
            post_cooccurency[post_index, post_index2] += 1

#adjacency matrix is constructed in a non-symmetric way, so need to sum across 
#both dimensions to get the final co-occurency matrix
#first get lower triangle of the matrix without diagonal
lower_triangle = np.tril(post_cooccurency, k=-1)
post_cooccurency = post_cooccurency + lower_triangle.T
#then just take the upper triangle with diagonal 
post_cooccurency = np.triu(post_cooccurency, k=0)

#set lower triangle to nan
lower_triangle_indices = np.tril_indices(post_cooccurency.shape[0], k=-1)

plot_post_cooccurency = post_cooccurency.copy()
plot_post_cooccurency[lower_triangle_indices] = np.nan


fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(plot_post_cooccurency, annot=True, fmt=".0f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(post_cooccurency))
#ax.set_xlabel('Celltype')
#ax.set_ylabel('Celltype') 

# compute relative covariance of postsynaptic celltypes

relative_cov = compute_relative_covariance(post_cooccurency)
fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(relative_cov, annot=True, fmt=".2f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(post_cooccurency))

#  Jaccard similarity to normalize for how frequent individual items are

post_jaccard = jaccard_similarity(post_cooccurency)

plot_jaccard = post_jaccard.copy()
# set lower triangle to nan
plot_jaccard[lower_triangle_indices] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(plot_jaccard, annot=True, fmt=".2f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(plot_jaccard))

# notelarge relative covariance of one group with another can be masked by frequent occurence of 
# only one of the groups

#  compute pointwise conditional probability

Ps_marginal = precompute_P_marginal(post_cooccurency)
pmi = compute_PMI(post_cooccurency, Ps_marginal)



# set lower triangle to nan
plot_pmi = pmi.copy()
plot_pmi[lower_triangle_indices] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(plot_pmi, annot=True, fmt=".2f", cmap='PiYG', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(plot_pmi))



# %% have a look at the data in hypergraphs 

# construct a multilevel hypergraph to represent the polyadic data structure

IM, all_skids = construct_polyadic_incidence_matrix(labelled_connectors['postsynaptic_to'])
GPM, all_celltypes = construct_group_projection_matrix(IM, all_skids, skid_to_celltype)



# %% describe based on matrices 
edge_sum = IM.sum(axis=0)
edge_min = edge_sum.min()
edge_max = edge_sum.max()
edge_mean = edge_sum.mean()

edge_zeros = np.where(edge_sum==0)[0]
print(f"Number of edges: {len(edge_sum)}")
# %% construct grup co-participation matrix

'''
A symmetric matrix 
    where each entry (i,j) represents the number of edges that connect vertices from group i to group j.
Characteristics:
Matrix form: purely numerical, best for statistical or heatmap-style analyses.

Symmetric: 

No edges or nodes — this is not a graph, but a count-based summary.

Diagonal entries 
  represent within-group interactions (e.g. how often group A interacts with itself).

Entries can be raw counts, normalized proportions, or weighted by number of vertices from each group.

Use Cases:
Heatmaps of inter-group interaction intensity

Identifying strong/weak group-pair associations

Statistical modeling or clustering on matrix

'''

# %% construct projected group graph 

# need a list of hyperedge with vertex/skid ids 
hyperedges = list(labelled_connectors['postsynaptic_to'])
# need a way to map vertex ids to group ids 
# this is skid_to_celltype 

group_pair_counts = get_group_pair_counts(hyperedges, vertex_to_group=skid_to_celltype)
G = build_group_graph(group_pair_counts, vertex_to_group=skid_to_celltype)

''' TODO: probably normalise weights based on group size or number of edges
'''

G = graph_normalize_weights(G, factor='mean')


# %% plot group graph  --- 
graph_name = 'group_graph__.png'
plot_nx_graph(G, plot_scale=1, save_fig=True, path=path_for_data+graph_name)

# %% plot group graph from perspective of one group 

FFNs_subgraph = centered_subgraph(G, 'FFNs', norm='group_participation')

# %%
path_for_subgraphs = path_for_data + 'group_graphs/'
if not os.path.exists(path_for_subgraphs):
    os.makedirs(path_for_subgraphs)
for ct in celltype_df['name'].unique():
    plt.figure()
    centered_subgraph(G, ct, norm='group_participation', plot_scale=20, save_fig=True, path=path_for_subgraphs+f'{ct}_subgraph.png')

# %% Examining group hub structure - degree 

degrees = G.degree
degrees_ordered = sorted(degrees, key=lambda x: x[1], reverse=True)
print(f"Groups by degree: {degrees_ordered}")

weighted_degrees = G.degree(weight='weight')
weighted_degrees_ordered = sorted(weighted_degrees, key=lambda x: x[1], reverse=True)
print(f"Groups by weighted degree: {weighted_degrees_ordered}")

# betweenness centrality
bw_centrality = nx.betweenness_centrality(G, weight='weight')
bw_centrality_ordered = sorted(bw_centrality.items(), key=lambda x: x[1], reverse=True)
print(f"Groups by betweenness centrality: {bw_centrality_ordered}")

# connected to other well connected groups
closeness = nx.closeness_centrality(G, distance='weight')
closeness_ordered = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
print(f"Groups by closeness centrality: {closeness_ordered}")

# %% Examining group hub structure - plotting 
# Convert to DataFrame for easier plotting
def plot_networkx_characteristics(wd, title, xlabel='Node', ylabel='Group', fsize=(3,6)):
    # convert to dataframe 
    wd = pd.DataFrame.from_dict(dict(wd), orient='index', columns=['Weighted Degree'])

    # Create a heatmap
    fig, ax = plt.subplots(figsize=fsize)
    sns.heatmap(wd, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)

    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='y', rotation=0)  # Keep y-axis labels horizontal
    return fig 

plot_networkx_characteristics(weighted_degrees_ordered, title='Weighted Degree')
plot_networkx_characteristics(bw_centrality_ordered, title='Betweenness Centrality')
plot_networkx_characteristics(closeness_ordered, title='Closeness Centrality')



# %%
from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G, weight='weight'))

#alternative community detection algorithm
#from networkx.algorithms.community import kernighan_lin_bisection
#cluster = kernighan_lin_bisection(G)

#check modularity of the communities
from networkx.algorithms.community.quality import modularity

mod_score = modularity(G, communities, weight='weight')
print("Communities:")
for i, community in enumerate(communities, start=1):
    print(f"Community {i}: {community}")

print(f"Modularity: {mod_score:.3f}")
# %% plot graph with communities
#generate color table for communities
colors = plt.cm.tab10(range(len(communities)))
# map communities to colors 
node_colors = {}
all_colors = ['#76A0EA', '#EA76D9', '#EAC076', '#76EA87']
for e, community in enumerate(communities):
    community_color = all_colors[e]
    for node in community:
        node_colors[node] = community_color

graph_name = 'group_graph_communities.png'
save_fig = False
plot_nx_graph(G, node_colors=node_colors, plot_scale=1, save_fig=save_fig, path=path_for_data+graph_name)
# %% observed vs expected weights for group pairs - not sure what to do with this yet

W = sum(d['weight'] for _, _, d in G.edges(data=True))
obs_exp_ratio = {}
expected_weights = {}
for u, v in G.edges():
    s_u = sum(G[u][nbr]['weight'] for nbr in G[u])
    s_v = sum(G[v][nbr]['weight'] for nbr in G[v])
    expected = (s_u * s_v) / (2 * W)
    observed = G[u][v]['weight']
    ratio = observed / expected if expected > 0 else 0
    expected_weights[(u, v)] = {'expected': expected, 'observed': observed, 'obs/exp': ratio}
    print (f"Edge ({u}, {v}): Expected: {expected}, Observed: {observed}, Ratio: {ratio}")

obs_exp_ratio = {edge: data['obs/exp'] for edge, data in expected_weights.items()}
obs_exp_ratio_sorted = sorted(obs_exp_ratio.items(), key=lambda x: x[1], reverse=True)
oers = dict(obs_exp_ratio_sorted[:20]) # top 20

# %% examine skid-to-skid grouping 

skid_pair_counts = get_skid_pair_counts(hyperedges, all_skids)
S = build_skid_graph(skid_pair_counts, all_skids)

# %%

plot_very_large_graph(S, node_size=1, plot_scale=0.01, save_fig=False)

#%% plot graph of nodes coloured by celltype 


'''TO DO: 
1. get node colors based on celltype
2. graph with coloured in nodes'''

node_colors = {}
#all_colors = ['#76A0EA', '#EA76D9', '#EAC076', '#76EA87'] #12 colours 
colors = plt.cm.tab20.colors[:12] 

for e, skid in enumerate(all_skids):
    #get celltype index 

    ...

    
    skid_colour = all_colors[e]
    for node in community:
        node_colors[node] = community_color


# %%

node_communities = list(greedy_modularity_communities(S, weight='weight', resolution=1.9, best_n=12))

print(f"Found {len(node_communities)} communities")
for i, community in enumerate(node_communities):
    print(f"Community {i}: {len(community)} nodes")
# %%
from networkx.algorithms.community import louvain_communities
node_communities = louvain_communities(S, weight='weight')

print(f"Found {len(node_communities)} communities with Louvain algorithm")
for i, community in enumerate(node_communities):
    print(f"Community {i}: {len(community)} nodes")
# %%
