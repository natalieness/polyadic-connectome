''' Functions to create and manipulate matrices and graphs representing undirected interactions and
synaptic groups, and match them to cell type categories.'''

from itertools import chain
from collections import defaultdict
from itertools import combinations
import os

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graspologic.utils import binarize
from graspologic.models import SBMEstimator


### Functions to construct a 2-level hypergraph to represent polyadic synaptic groups ###

def construct_polyadic_incidence_matrix(list_of_connectors):
    ''' Construct a polyadic incidence matrix from a list of connectors '''
    # force list 
    if not isinstance(list_of_connectors, list):
        list_of_connectors = list(list_of_connectors)
    # get all unique skids
    all_skids = tuple(set(chain.from_iterable(list_of_connectors)))
    n_skids = len(all_skids)
    n_connectors = len(list_of_connectors)

    # create incidence matrix 
    IM = np.zeros((n_skids, n_connectors))
    for e, connector in enumerate(list_of_connectors):
        for skid in connector:
            skid_idx = list(all_skids).index(skid)
            IM[skid_idx, e] += 1
    return IM, all_skids

def construct_group_projection_matrix(IM, all_skids, skid_to_celltype):

    # get all unique cell types from dictionary 
    all_celltypes = tuple(set(skid_to_celltype.values()))
    n_celltypes = len(all_celltypes)

    n_edges = IM.shape[1]

    # create group projection matrix
    GPM = np.zeros((n_celltypes, n_edges))
    for e in range(n_edges):
        # get all skids in connector
        connector_skids = np.where(IM[:, e] > 0)[0]
        # get all celltypes in connector
        connector_celltypes = [skid_to_celltype[all_skids[skid]] for skid in connector_skids]
        for c in connector_celltypes:
            ct_idx = list(all_celltypes).index(c)
            GPM[ct_idx, e] += 1
    return GPM, all_celltypes

### Functions to construct a projected graph ###

def get_group_pair_counts(hyperedges, vertex_to_group):
    # Track group pair co-occurrence counts
    group_pair_counts = defaultdict(int)
    for hedge in hyperedges:
        groups_in_edge = [vertex_to_group[v] for v in hedge]
        #unique_groups = set(groups_in_edge)

        # Count all unordered group pairs (with self-pairs)
        for g1, g2 in combinations(sorted(groups_in_edge), 2):
            group_pair_counts[(g1, g2)] += 1

        # Optionally include self-pairs (e.g., A–A if multiple A members)
        #for g in groups_in_edge:
         #   if groups_in_edge.count(g) > 1:
        #        group_pair_counts[(g, g)] += 1
    return group_pair_counts

def build_group_graph(group_pair_counts, vertex_to_group):
    G = nx.Graph()
    # Add nodes for each group
    all_groups = set(vertex_to_group.values())
    G.add_nodes_from(all_groups)

    for (g1, g2), count in group_pair_counts.items():
        G.add_edge(g1, g2, weight=count)
    return G

def graph_normalize_weights(G, factor='mean'):
    ''' 
    Normalize weights of edges in graph G based on: 
    - 'mean': Mean weight of all edges
    - 'log': Logarithm of the mean weight
    - 'jaccard': Jaccard similarity of the group/ group overlap
    '''
    if factor == 'mean':
        mean_weight = np.mean([G[u][v]['weight'] for u, v in G.edges()])
        for u, v in G.edges():
            G[u][v]['weight'] /= mean_weight
    elif factor == 'log':
        mean_weight = np.mean([G[u][v]['weight'] for u, v in G.edges()])
        for u, v in G.edges():
            G[u][v]['weight'] = np.log10(G[u][v]['weight'] / mean_weight)
    elif factor == 'jaccard':
        for u, v in G.edges():
            w = G[u][v]['weight']
            deg_u = sum(d['weight'] for _, _, d in G.edges(u, data=True))
            deg_v = sum(d['weight'] for _, _, d in G.edges(v, data=True))
            G[u][v]['weight'] = w / (deg_u + deg_v - w)
    else: 
        print("Unknown normalization factor. No normalization applied.")
    return G

### Functions to construct a skid-to-skid graph ###
##also construct a matrix that is centered on the actual neurons and contains their post-occurency 
def get_skid_pair_counts(hyperedges, all_skids):
    ''' Get pairwise co-occurency of polyadic edges for each skid to build graph
    '''
    skid_pair_counts = defaultdict(int)
    for hedge in hyperedges:
        for s1, s2 in combinations(sorted(hedge), 2):
            skid_pair_counts[(s1, s2)] += 1
    return skid_pair_counts

def build_skid_graph(skid_pair_counts, all_skids):
    G = nx.Graph()
    # Add nodes for each group
    G.add_nodes_from(all_skids)

    for (s1, s2), count in skid_pair_counts.items():
        G.add_edge(s1, s2, weight=count)
    return G

### Functions to plot projected group graphs ###

def plot_nx_graph(G, node_colors=None, plot_scale=1, save_fig=False, path='', title="Projected Group Interaction Graph", alpha=1, node_size=2000):

    #pos = nx.circular_layout(G.subgraph(G.nodes))
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    #get colors for nodes 
    if node_colors is None:
        node_colors = ['lightblue' for _ in G.nodes()]
    else:
        node_colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]
    #scale all weights by a factor for visualization
    edge_weights= [i*plot_scale for i in edge_weights]
    plt.title(title)
    nx.draw(
        G, pos,
        with_labels=True,
        width=edge_weights,  # Line thickness ~ frequency
        node_color=node_colors,
        node_size=node_size,
        font_size=8,
        edge_color='black', 
        alpha=alpha
    )
    if save_fig:
        plt.savefig(path)

def plot_very_large_graph(G, node_colors=None, node_size=1, plot_scale=0.01, save_fig=False, path=''):
    #pos = nx.circular_layout(G.subgraph(G.nodes))
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    #get colors for nodes 
    if node_colors is None:
        node_colors = ['lightblue' for _ in G.nodes()]
    else:
        node_colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]

    #scale all weights by a factor for visualization
    edge_weights= [i*plot_scale for i in edge_weights]
    nx.draw(
        G, pos,
        with_labels=False,
        width=edge_weights,  # Line thickness ~ frequency
        node_color=node_colors,
        node_size=node_size,
        font_size=10,
        edge_color='black' 
    )
    plt.title("Network Graph")
    if save_fig:
        plt.savefig(path)

def centered_subgraph(G, center_node, norm='group_participation', plot_scale=20, save_fig=False, path=''):
    '''
    Normalization options for weights:
    - 'group participation: Normalizes based on total participation of central group
    '''
    # Creat new graph of 
    H = nx.Graph() 
    
    # Add all original nodes
    H.add_nodes_from(G.nodes(data=True))

    # Add only edges connected to the specified node
    for neighbor in G.neighbors(center_node):
        edge_data = G.get_edge_data(center_node, neighbor)
        H.add_edge(center_node, neighbor, **edge_data)
    
    #plot subgraph
    edge_nodes = H.nodes - center_node
    pos = nx.circular_layout(H.subgraph(edge_nodes))
    pos[center_node] = (0, 0)  # Center node at origin
    edge_weights = [H[u][v]['weight'] for u, v in H.edges()]
    # Normalize edge weights based on the specified normalization method
    if norm == 'group_participation':
        # Normalize by the total participation of the center node
        total_participation = sum(edge_weights)
        edge_weights = [w / total_participation for w in edge_weights]
    # thicken edges based on given scale for visualization
    edge_weights= [i*plot_scale for i in edge_weights]
    nx.draw(
        H, pos,
        with_labels=True,
        width=edge_weights,  # Line thickness ~ frequency
        node_color='lightblue',
        node_size=2000,
        font_size=10,
        edge_color='black' 
    )
    plt.title(f"Subgraph centered on {center_node}")
    if save_fig:
        plt.savefig(path)
    return H


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

def map_co_adj_to_dict(adj_matrix, ordered_ps_in_adj, skid_dict, filter_adj=True):
     # get the cell types for each postsynaptic partner
    ps_type_in_adj = [skid_dict.get(post, 'NA') for post in ordered_ps_in_adj]

    if filter_adj:
        # remove 'NA' entried before fitting
        na_cells = np.where(np.array(ps_type_in_adj) == 'NA')[0]
        adj_matrix = np.delete(adj_matrix, na_cells, axis=0)
        adj_matrix = np.delete(adj_matrix, na_cells, axis=1)
        ps_type_in_adj = np.delete(ps_type_in_adj, na_cells)

    return adj_matrix, ps_type_in_adj


def get_sbm_block_probs_from_hyperedges(hyperedges, skid_to_celltype,name='', plot=True):
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
    ps_celltype_in_adj = [skid_to_celltype.get(post, 'NA') for post in ordered_ps_in_adj]

    # remove 'NA' entried before fitting
    na_cells = np.where(np.array(ps_celltype_in_adj) == 'NA')[0]
    adj_matrix = np.delete(adj_matrix, na_cells, axis=0)
    adj_matrix = np.delete(adj_matrix, na_cells, axis=1)
    adj_matrix_bi = np.delete(adj_matrix_bi, na_cells, axis=0)
    adj_matrix_bi = np.delete(adj_matrix_bi, na_cells, axis=1)
    ps_celltype_in_adj = np.delete(ps_celltype_in_adj, na_cells)
    
    # fit the SBM model
    estimator = SBMEstimator(directed=False, loops=True)
    estimator.fit(adj_matrix_bi, y=ps_celltype_in_adj)
    block_probs = pd.DataFrame(estimator.block_p_, index=np.unique(ps_celltype_in_adj), columns=np.unique(ps_celltype_in_adj))
    if plot:
        sns.heatmap(block_probs, annot=False, fmt=".1f", cmap='Blues') 
        plt.title(f'Block probabilities for postsynaptic co-occurrence \n in polyadic synapses ({name})', fontsize=14)
        plt.show()
    return adj_matrix_bi, block_probs, ps_celltype_in_adj, adj_matrix