#%%

from itertools import chain
from collections import Counter
from itertools import combinations
import os

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Arc, ConnectionPatch
import matplotlib.transforms as mtransforms

from contools import Celltype_Analyzer
from graspologic.utils import binarize
from graspologic.models import SBMEstimator
import pymaid
from pymaid_creds import url, name, password, token

# local imports
from scripts.functions.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list, get_ct_index
from scripts.functions.undirected_graph_functions import construct_polyadic_incidence_matrix, construct_group_projection_matrix, get_skid_pair_counts, build_skid_graph

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

# %%

#construct undirected adjacency matrix of co-occuring post-synaptic partners (independent of presynaptic partners)

def get_postsynaptic_co_adj(hyperedges):
    """
    Get the adjacency matrix of co-occuring postsynaptic partners from a list of hyperedges.
    """
    # get all unique postsynaptic partners
    all_postsynaptic = set(chain.from_iterable(hyperedges))
    # create a mapping from postsynaptic partner to index
    post_to_index = {post: i for i, post in enumerate(all_postsynaptic)}
    # create an empty adjacency matrix
    adj_matrix = np.zeros((len(all_postsynaptic), len(all_postsynaptic)))
    
    # iterate over hyperedges and fill in the adjacency matrix
    for hyperedge in hyperedges:
        indices = [post_to_index[post] for post in hyperedge]
        for i, j in combinations(indices, 2):
            adj_matrix[i, j] += 1
            adj_matrix[j, i] += 1
    
    #get ordered list of postsynaptic partner names 
    ordered_postsynaptic = [post for post, _ in sorted(post_to_index.items(), key=lambda item: item[1])]

    return adj_matrix, ordered_postsynaptic
#%% get SBM block probabilities for all labelled neurons

hyperedges = labelled_connectors['postsynaptic_to'].tolist()
ps_co_adj, ordered_ps_in_adj = get_postsynaptic_co_adj(hyperedges)
# binarize the adj matrix in preparation for SBM model 
ps_co_adj_bi = binarize(ps_co_adj)

cmap = mpl.colors.ListedColormap(['white', 'black'])
plt.imshow(ps_co_adj_bi, cmap=cmap)
plt.axis('off')

#  get cell types for each postsynaptic partner in the adjacency matrix
ps_celltype_in_adj = [skid_to_celltype[post] for post in ordered_ps_in_adj]

#  get sbm group-to-group connection probabilities 


estimator = SBMEstimator(directed=False, loops=True)
estimator.fit(ps_co_adj_bi, y=ps_celltype_in_adj)

block_probs = pd.DataFrame(estimator.block_p_, index=np.unique(ps_celltype_in_adj), columns=np.unique(ps_celltype_in_adj))
sns.heatmap(block_probs, annot=False, fmt=".1f",cmap='Blues') 
plt.title('Block probabilities for postsynaptic \nco-occurrence in polyadic synapses', fontsize=14)

# %% 
def get_sbm_block_probs_from_hyperedges(hyperedges, name='', plot=True):
    """
    Get the block probabilities for a given set of hyperedges.
    """
    # get the adjacency matrix
    adj_matrix, ordered_ps_in_adj = get_postsynaptic_co_adj(hyperedges)
    
    # binarize the adjacency matrix
    adj_matrix_bi = binarize(adj_matrix)
    if plot:
        cmap = mpl.colors.ListedColormap(['white', 'black'])
        plt.imshow(adj_matrix_bi, cmap=cmap)
        plt.axis('off')
        plt.title(f'Adjacency matrix of polyadic partners\n({name})')
        plt.show()
    
    # get the cell types for each postsynaptic partner
    ps_celltype_in_adj = [skid_to_celltype[post] for post in ordered_ps_in_adj]
    
    # fit the SBM model
    estimator = SBMEstimator(directed=False, loops=True)
    estimator.fit(adj_matrix_bi, y=ps_celltype_in_adj)
    block_probs = pd.DataFrame(estimator.block_p_, index=np.unique(ps_celltype_in_adj), columns=np.unique(ps_celltype_in_adj))
    if plot:
        sns.heatmap(block_probs, annot=False, fmt=".1f", cmap='Blues') 
        plt.title(f'Block probabilities for postsynaptic co-occurrence \n in polyadic synapses ({name})', fontsize=14)
        plt.show()
    return adj_matrix_bi, block_probs, ps_celltype_in_adj, adj_matrix
    
hyperedges = labelled_connectors['postsynaptic_to'].tolist()
adj_all, block_probs_all, ps_celltype_in_adj_all, adj_all_nonbi = get_sbm_block_probs_from_hyperedges(hyperedges, name='all labelled neurons', plot=True)
#%% save polyadic so adj matrices of all labelled neurons to csv for use in other scripts 

adj_all_bi_df = pd.DataFrame(adj_all)
adj_all_bi_df.to_csv(path_for_data + 'poly_adj/adj_all_bi.csv', index=False)
adj_all_nonbi_df = pd.DataFrame(adj_all_nonbi)
adj_all_nonbi_df.to_csv(path_for_data + 'poly_adj/adj_all_nonbi.csv', index=False)
cell_group_labels = pd.DataFrame(ps_celltype_in_adj_all, columns=['celltype'])
cell_group_labels.to_csv(path_for_data + 'poly_adj/cell_group_labels.csv', index=False)

#%% get top block probabilities 

def get_top_block_probs(block_probs, n=5, printing=True):
    """
    Get the top n block probabilities from a block probability matrix.
    """
    # get the upper triangle of the matrix
    upper_triangle = np.triu(block_probs.values, k=0)
    # get the indices of the top n block probabilities
    top_indices = np.unravel_index(np.argsort(upper_triangle, axis=None)[-n:], upper_triangle.shape)
    # get the top n block probabilities
    top_block_probs = [(block_probs.index[i], block_probs.columns[j], upper_triangle[i, j]) for i, j in zip(*top_indices)]
    if printing:
        print("Top block probabilities:")
        for i, (ct1, ct2, prob) in enumerate(top_block_probs[::-1]):
            print(f"{i+1}. {ct1} - {ct2}: {prob:.2f}")

    return top_block_probs
top_block_probs = get_top_block_probs(block_probs_all, n=20)


# %% Repeat but filtering by presynaptic cell type
''' fix this length issue of alignment of block probabilities with cell types '''
prect_post_poly_df = pd.DataFrame()
#get adj_matrix labels 
adj_labels_flat = []
for a in block_probs_all.index:
    for b in block_probs_all.columns:
        adj_labels_flat.append((a,b))
prect_post_poly_df['adj_labels'] = adj_labels_flat


for ct in celltype_df['name'].unique():
    print(f"Cell type: {ct}")
    # get the hyperedges for the current cell type
    hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == ct]['postsynaptic_to'].tolist()
    # get the block probabilities
    adj_ct, block_probs_ct, ps_celltype_in_adj_ct, adj_ct_nonbi = get_sbm_block_probs_from_hyperedges(hyperedges, name=f'Postsynaptic to {ct}', plot=True)
    # get the top block probabilities
    top_block_probs_ct = get_top_block_probs(block_probs_ct, n=10)

    #add to larger dataframe 
    for e, a in enumerate(block_probs_ct.index):
        for e2, b in enumerate(block_probs_ct.columns):
            idx = np.where(prect_post_poly_df['adj_labels'] == (a, b))
            print(idx)
            prect_post_poly_df.at[idx[0][0], ct] = block_probs_ct.at[a, b]

#replace NaN values with 0
prect_post_poly_df = prect_post_poly_df.fillna(0)
prect_post_poly_df = prect_post_poly_df.set_index('adj_labels')
    #prect_post_poly_df[ct] = np.array(block_probs_ct.values).flatten()
    # save the block probabilities 
    #block_probs.to_csv(path_for_data + f'block_probs_{ct}.csv')
# %% statistically comparing sbm estimators for different subgraphs (based on presynaptic cell type)

#example: kenyon cells and mushroom body output neurons
kc_hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == 'KCs']['postsynaptic_to'].tolist()
adj_kc, block_probs_kc, kc_celltype_in_adj, kc_adj_nonbi = get_sbm_block_probs_from_hyperedges(kc_hyperedges, name='KCs presynaptic', plot=True)

mbon_hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == 'MBONs']['postsynaptic_to'].tolist()
adj_mbon, block_probs_mbon, mbon_celltype_in_adj, mbon_adj_nonbi = get_sbm_block_probs_from_hyperedges(mbon_hyperedges, name='MBONs presynaptic', plot=True)




# %% try force-direct graph 

''' Network plot from the perspective of a specific pre-synaptic cell type
edge weight: co-occurrence frequency of clustering coefficient
node size: represent number of projections to that group ''' 

presynaptic_group = 'LHNs'
group_hyperedges = labelled_connectors[labelled_connectors['presynaptic_celltype'] == presynaptic_group]['postsynaptic_to'].tolist()
postsyn_celltypes = labelled_connectors[labelled_connectors['presynaptic_celltype'] == presynaptic_group]['postsynaptic_celltype'].tolist()
adj_group, block_probs_group, group_celltype_in_adj = get_sbm_block_probs_from_hyperedges(group_hyperedges, name=f'Postsynaptic to {presynaptic_group}', plot=True)

postsyn_celltypes_flat = list(chain.from_iterable(postsyn_celltypes))
postsyn_celltypes_counts = Counter(postsyn_celltypes_flat)

G = nx.from_numpy_array(block_probs_group.values)
G = nx.relabel_nodes(G, dict(zip(range(len(block_probs_group.index)), block_probs_group.index)))
#%%
#plot graph 
pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

node_size = [postsyn_celltypes_counts[ct] * 2 for ct in G.nodes()]  # scale node size by number of projections to that group
labels = {ct:f'{ct}\nn={postsyn_celltypes_counts[ct]}' for ct in G.nodes()}

#scale all weights by a factor for visualization
plot_scale = 50
edge_weights= [i*plot_scale for i in edge_weights]
''' fix from here '''
nx.draw(
    G, pos,
    labels=labels,
    with_labels=True,
    width=edge_weights,  
    node_color='lightblue', 
    node_size=node_size,
    font_size=8,
    edge_color='black', 
    alpha=0.9,
    )
plt.title(f'Network of postsynaptic partners for {presynaptic_group}', fontsize=14)
plt.savefig(path_for_data + f'network_{presynaptic_group}.png', dpi=300, bbox_inches='tight')
# %% plot graph with offset for nodes 


# Graph and layout
pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# Node sizes and labels
node_size = [postsyn_celltypes_counts[ct] * 4 for ct in G.nodes()]
labels = {ct: f'{ct}\nn={postsyn_celltypes_counts[ct]}' for ct in G.nodes()}

# Scale edge weights for visualization
plot_scale = 10
edge_weights = [w * plot_scale for w in edge_weights]

# Setup plot
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', ax=ax, alpha=0.9)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

# Convert node_size (points²) to radius in data coords
def node_size_to_radius(node_size_list, ax):
    # Convert each node size to radius in data coordinates
    fig = ax.get_figure()
    trans = ax.transData.transform
    inv = ax.transData.inverted().transform
    radii = []
    for ns in node_size_list:
        r_pts = np.sqrt(ns) / 2  # convert area to radius in points
        # Convert from points to pixels
        pixel_radius = fig.dpi_scale_trans.transform((r_pts, 0))[0]
        # Convert from pixels to data coordinates
        x0, y0 = inv((0, 0))
        x1, y1 = inv((pixel_radius, 0))
        data_radius = np.hypot(x1 - x0, y1 - y0)
        radii.append(data_radius)
    return dict(zip(G.nodes(), radii))

node_radii = node_size_to_radius(node_size, ax)

# Draw offset edges
for (u, v), width in zip(G.edges(), edge_weights):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    if dist == 0:
        continue
    offset_dx = dx / dist
    offset_dy = dy / dist
    ru = node_radii[u]
    rv = node_radii[v]
    start = (x1 + offset_dx * ru, y1 + offset_dy * ru)
    end = (x2 - offset_dx * rv, y2 - offset_dy * rv)
    line = ConnectionPatch(start, end, "data", "data", color='black', linewidth=width, alpha=0.9)
    ax.add_patch(line)


# Self-loop arc drawing using node_radii
for node in G.nodes():
    if G.has_edge(node, node):
        x, y = pos[node]
        r = node_radii[node]*3  # already calculated in your earlier code

        # Loop appearance settings
        loop_radius = r * 1.9           # how far the arc reaches from center
        vertical_offset = r * 1.4       # how far above the node center to place the arc
        arc_center = (x, y + vertical_offset)

        # Angle from 320° to 270° (skipping 270–320)
        arc = Arc(
            arc_center,
            width=2 * loop_radius,
            height=2 * loop_radius,
            angle=0,
            theta1=320,
            theta2=580,  # equivalent to 270°, wrapping around
            color='black',
            linewidth=G[node][node].get('weight', 1) * plot_scale,
            alpha=0.9
        )
        ax.add_patch(arc)



# Final plot settings
#ax.set_aspect('equal')
ax.margins(x=0.1, y=0.1)
plt.axis('off')
plt.tight_layout()
plt.show()



# %%
