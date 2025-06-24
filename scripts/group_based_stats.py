from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl

### Getting alternative network for comparison of polyadic connections 
### try configuration model first 
### self loops will depende on the number on whether taken into account in statistical comparison



# %%

def get_group_counts_from_poly_adj(poly_adj, ct_labels, group_order):
    '''Note this only makes sense for binarised adjacency matrix'''

    group_to_idx = {ct: i for i, ct in enumerate(group_order)}
    n_groups = len(group_order)

    group_counts = np.zeros((n_groups, n_groups), dtype=int)
    #only go along the upper triangle of the adjacency matrix since it's symmetric
    for g1 in range(poly_adj.shape[0]):
        for g2 in range(g1, poly_adj.shape[1]):
            if poly_adj[g1, g2] > 0:
                group1 = ct_labels[g1]
                group2 = ct_labels[g2]
                idx1 = group_to_idx[group1]
                idx2 = group_to_idx[group2]
                group_counts[idx1, idx2] += poly_adj[g1, g2]
                if idx1 != idx2:
                    group_counts[idx2, idx1] += poly_adj[g1, g2]
    return group_counts

def get_n_per_group(ct_labels, group_order):
    group_counts = Counter(ct_labels)
    #order in the same way as group_order
    group_counts = {ct: group_counts[ct] for ct in group_order}
    label_counts = list(group_counts.values())
    return label_counts

def get_n_possible_matrix(ct_labels, group_order):
    label_counts = get_n_per_group(ct_labels, group_order)
    # note this matrix works for binarised adjacency matrix for between group connections
    n_pos_mat = np.outer(label_counts, label_counts)
    # adjusting the diagonal to look at within group connections 
    # allow self-loops between nodes of the same group 
    intra_group_n_pos = [(j * (j + 1)) / 2 for j in label_counts]
    np.fill_diagonal(n_pos_mat, intra_group_n_pos)
    return n_pos_mat, group_order


def compare_two_sample_chi_squared(poly_adj1, ct_labels1, poly_adj2, ct_labels2):
    """
    Compare two polyadic adjacency matrices using a chi-squared test.
    Assumes both matrices are binarised and have the same cell group labels.
    """
    group_order = np.unique(ct_labels1)
    group_counts1 = get_group_counts_from_poly_adj(poly_adj1, ct_labels1, group_order)
    group_counts2 = get_group_counts_from_poly_adj(poly_adj2, ct_labels2, group_order)

    n_pos_mat1, group_order1 = get_n_possible_matrix(ct_labels1, group_order)
    n_pos_mat2, group_order2 = get_n_possible_matrix(ct_labels2, group_order)
    assert np.array_equal(group_order1, group_order2), "Group orders must match for both matrices"

    stats_chi = np.zeros((len(group_order), len(group_order)), dtype=float)
    pvals_uncorrected = np.ones((len(group_order), len(group_order)), dtype=float)

    for e, g1 in enumerate(group_order):
        for f, g2 in enumerate(group_order[e:]):
            n_obs1 = group_counts1[e, f + e]
            n_obs2 = group_counts2[e, f + e]
            n_pos1 = n_pos_mat1[e, f + e]
            n_pos2 = n_pos_mat2[e, f + e]
            if n_obs1 + n_obs2 == 0:
                continue
            cont_table = np.array([[n_obs1, n_pos1 - n_obs1], 
                                   [n_obs2, n_pos2 - n_obs2]])
            chi2, p_val, _, _ = stats.chi2_contingency(cont_table)
            stats_chi[e, f + e] = chi2
            pvals_uncorrected[e, f + e] = p_val

    #also get fold change between models 
    fold_change = np.zeros((len(group_order), len(group_order)), dtype=float)
    fold_change[:] = np.nan
    for e, g1 in enumerate(group_order):
        for f, g2 in enumerate(group_order[e:]):
            n_obs1 = group_counts1[e, f + e]
            n_obs2 = group_counts2[e, f + e]
            #using log2(n_obs1 / n_obs2) to get fold change 
            pseudocount = 1 #e-6  # to avoid division by zero
            fold_change[e, f+e] = np.log2((n_obs1 + pseudocount) / (n_obs2 + pseudocount))

            #if n_obs1 + n_obs2 == 0:
                #fold_change[e, f + e] = 1 #np.nan
            #else:
                #fold_change[e, f + e] = n_obs1 / n_obs2 if n_obs2 > 0 else n_obs1 #think this makes sense, but could be better way?
    # this doesnt make sense because a negative change will just result in 0 
    return stats_chi, pvals_uncorrected, group_order, fold_change, group_counts1, group_counts2


def get_random_poly_adj(poly_adj, rng):

    n_nodes = poly_adj.shape[0]
    # Create a degree sequence from the polyadic adjacency matrix
    degree_sequence = np.sum(poly_adj, axis=1)
    degree_sequence = degree_sequence.astype(int) 
    # Create a random graph with the same degree sequence
    random_graph = nx.configuration_model(degree_sequence, create_using=nx.Graph(), seed=rng)
    # remove any parallel edges 
    random_graph = nx.Graph(random_graph)  # Convert to a simple graph
    random_adj = nx.to_numpy_array(random_graph, nodelist=range(n_nodes))
    
    return random_adj

def correct_pvals(pvals_uncorrected, method='holm'):
    pvals_shape = pvals_uncorrected.shape
    pvals_flat = pvals_uncorrected.flatten()
    reject, pvals_corrected, asidak, acbonf = multipletests(pvals_flat, method=method, alpha=0.05, is_sorted=False, returnsorted=False)
    pvals_corrected = pvals_corrected.reshape(pvals_shape)
    reject = reject.reshape(pvals_shape)
    return reject, pvals_corrected, asidak, acbonf
            
### Plotting functions ###
def get_pval_cbar():
    cmap_sig = (mpl.colors.ListedColormap(['gray', 'darkgray', 'lightgray']).with_extremes(over='white', under='black'))
    bounds = [0.0001, 0.001, 0.01, 0.05]
    norm = mpl.colors.BoundaryNorm(bounds, cmap_sig.N)
    return cmap_sig, norm

def plot_pvals_heatmap(pvals_corrected, group_order, title='Corrected p-values for comparison to degree-matched config model'):
    cmap_sig, norm = get_pval_cbar()
    fig, ax = plt.subplots()
    img = ax.imshow(pvals_corrected, cmap=cmap_sig, norm=norm)
    ax.set_xticks(ticks=np.arange(len(group_order)), labels=group_order, rotation=90)
    ax.set_yticks(ticks=np.arange(len(group_order)), labels=group_order)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, orientation='vertical', label='Corrected p-value', extend='both')

def plot_fold_change_heatmap(fold_change, group_order):
    """Plot a heatmap of fold change values."""
    # Center colormap on 0

    fc_max = np.nanmax(np.abs(fold_change))

    fig, ax = plt.subplots()
    img = ax.imshow(fold_change, cmap='coolwarm', vmin=-fc_max, vmax=fc_max)
    ax.set_xticks(ticks=np.arange(len(group_order)), labels=group_order, rotation=90)
    ax.set_yticks(ticks=np.arange(len(group_order)), labels=group_order)
    ax.set_title('log2 Fold change between polyadic connections and config model')
    fig.colorbar(img, ax=ax, orientation='vertical', label='Fold change')

def plot_significant_fold_change_heatmap(fold_change, pvals_corrected, group_order):
    """Plot a heatmap of fold change values for significant p-values."""
    significant_mask = pvals_corrected < 0.05
    fold_change_sig = np.where(significant_mask, fold_change, np.nan)
    # Center colormap on 0
    fc_max = np.nanmax(np.abs(fold_change))
    fig, ax = plt.subplots()
    img = ax.imshow(fold_change_sig, cmap='coolwarm', vmin=-fc_max, vmax=fc_max )
    ax.set_xticks(ticks=np.arange(len(group_order)), labels=group_order, rotation=90)
    ax.set_yticks(ticks=np.arange(len(group_order)), labels=group_order)
    ax.set_title('log2 fold change between polyadic connections and config model \n for significant changes only')
    fig.colorbar(img, ax=ax, orientation='vertical', label='Log2 fold change')

#%% Temporary to test the code

''' Note: using binarised adjacency matrix for polyadic connections

Functions only really make sense for this case, 
otherwise the counts of n_possible don't make sense below'''

poly_adj = pd.read_csv('data/poly_adj/adj_all_nonbi.csv').values
ct_labels = pd.read_csv('data/poly_adj/cell_group_labels.csv').values.reshape(-1)

# rather than using the binarised adjacency matrix, binarize here, so that we can use synaptic threshold if desired 
def binarize_poly_adj(poly_adj, syn_threshold=0):
    ''' Binarize the polyadic adjacency matrix based on a synaptic threshold '''
    binarized_adj = np.where(poly_adj > syn_threshold, 1, 0)
    return binarized_adj

poly_adj = binarize_poly_adj(poly_adj, syn_threshold=0)

#%% generate random polyadic adjacency matrix to compare
seed = 40
#generate numpy random instance
rng = np.random.default_rng(seed=seed)
r_poly_adj = get_random_poly_adj(poly_adj, rng)

# compare briefly 
print(f'Shape: poly_adj: {poly_adj.shape}, r_poly_adj: {r_poly_adj.shape}')
print(f'Sum: poly_adj: {np.sum(poly_adj)}, r_poly_adj: {np.sum(r_poly_adj)}')

#  compare group-based statistics
stats_chi, pvals_uncorrected, group_order, fold_change, g_count1, g_count2 = compare_two_sample_chi_squared(poly_adj, ct_labels, r_poly_adj, ct_labels)
reject, pvals_corrected, asidak, acbonf = correct_pvals(pvals_uncorrected, method='holm')


plot_pvals_heatmap(pvals_corrected, group_order)

plot_fold_change_heatmap(fold_change, group_order)

plot_significant_fold_change_heatmap(fold_change, pvals_corrected, group_order)


#%% from perspective of specific cell types 
def get_group_stats_from_bi_adj(celltype):
    ct_adj = pd.read_csv(f'data/poly_adj/adj_{celltype}_nonbi.csv').values
    ct_labels = pd.read_csv(f'data/poly_adj/{celltype}_group_labels.csv').values.reshape(-1)

    #binarise
    ct_adj = binarize_poly_adj(ct_adj, syn_threshold=0)

    ''' this adj matrix is not even - which violates the adj because each edge needs to be connected to two nodes 
    This is because of the binarization that's needed for the current methodology, so will need to introduce a random edge to make it even
    Not ideal so need to think about this more'''
    #check if the adjacency matrix is even
    if np.sum(ct_adj) % 2 != 0:
        print(f'Adjacency matrix for {celltype} is not even, adding a self-loop to make it even')
        ct_adj[0,0] += 1  # add a self-loop to make it even

    r_ct_adj = get_random_poly_adj(ct_adj, rng)
    if np.sum(r_ct_adj) % 2 != 0:
        #remove self loop
        ct_adj[0,0] -= 1  # remove the self-loop to restore original adjacency matrix

    stats_chi_ct, pvals_uncorrected_ct, group_order_ct, fold_change_ct, g_count1_ct, g_count2_ct = compare_two_sample_chi_squared(ct_adj, ct_labels, r_ct_adj, ct_labels)
    reject_ct, pvals_corrected_ct, asidak_ct, acbonf_ct = correct_pvals(pvals_uncorrected_ct, method='holm')

    plot_pvals_heatmap(pvals_corrected_ct, group_order_ct, title=f'Corrected p-values for comparison to degree-matched config model ({celltype})')
    plot_fold_change_heatmap(fold_change_ct, group_order_ct)
    plot_significant_fold_change_heatmap(fold_change_ct, pvals_corrected_ct, group_order_ct)


get_group_stats_from_bi_adj('KCs')
#
#  %%


