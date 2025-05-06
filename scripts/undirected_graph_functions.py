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
        unique_groups = set(groups_in_edge)

        # Count all unordered group pairs (with self-pairs)
        for g1, g2 in combinations(sorted(unique_groups), 2):
            group_pair_counts[(g1, g2)] += 1

        # Optionally include self-pairs (e.g., A–A if multiple A members)
        for g in unique_groups:
            if groups_in_edge.count(g) > 1:
                group_pair_counts[(g, g)] += 1
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

### Functions to plot projected group graphs ###

def plot_nx_graph(G, node_colors=None, plot_scale=1, save_fig=False, path=''):

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
        with_labels=True,
        width=edge_weights,  # Line thickness ~ frequency
        node_color=node_colors,
        node_size=2000,
        font_size=10,
        edge_color='black' 
    )
    plt.title("Projected Group Interaction Graph")
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