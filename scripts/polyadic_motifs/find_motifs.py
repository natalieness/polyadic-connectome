''' Find polyadic motifs across neurons
Idea1: direct matchign of connectors
Idea2: dim reduction and clustering of connectors
'''

from collections import Counter
from itertools import chain, combinations, accumulate
import time

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
import prince
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jaccard
import umap
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh  # or use np.linalg.eigh for dense
from sklearn import cluster, mixture
from sklearn.cluster import KMeans

from contools import Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.little_helper import get_celltype_dict, celltype_col_for_list, get_celltype_name, celltype_col_for_nestedlist, get_pairs_dict

rm = pymaid.CatmaidInstance(url, token, name, password)

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

#%% get bilateral pairs 

pairs = pd.read_csv('data/pairs-2022-02-14.csv')
right_ns = pairs['rightid'].unique()
left_ns = pairs['leftid'].unique()

pairs_dict = get_pairs_dict(pairs)

#%% get cell type and synaptic data 
# get celltype data for each skid 
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()
# get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)
ct_names = celltype_df['name'].unique()

# get synaptic sites from catmaid and describe data
# select neurons to include 
wanted_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(wanted_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# get all synaptic sites associated with neurons 
links = pymaid.get_connector_links(all_neurons, chunk_size=50)

#get connector details 
all_connectors = links['connector_id'].unique()
connector_details = pymaid.get_connector_details(all_connectors)

# remove connector details without presynaptic site 
connector_details = connector_details.dropna(subset=['presynaptic_to'])

#  map skid ids in connector details to celltypes
connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))
celltype_col_for_list(connector_details, 'postsynaptic_to', skid_to_celltype=skid_to_celltype, new_col_name='postsynaptic_celltype')

# assign hemisphere of presynaptic neuron 

connector_details['presyn_hemi'] = connector_details['presynaptic_to'].apply(lambda x: 'right' if x in right_ns else 'left' if x in left_ns else 'NA')


#%% Some functions for constructing connector binary matrices and direct comparison

### Constructing vector binary matrices for connectors ###
def con_binary_matrix(con, only_known_targets=False, all_neurons=all_neurons):
    all_ps_flat = np.unique(list(chain.from_iterable(con['postsynaptic_to'].values)))
    if only_known_targets:
        all_ps_flat = np.intersect1d(all_ps_flat, all_neurons)

    n_targets = len(all_ps_flat)
    n_cons = con.shape[0]

    con_bin = np.zeros((n_cons, n_targets), dtype=int)
    for s in range(n_cons):
        c_id = con['connector_id'].values[s]
        ps = con['postsynaptic_to'].values[s]
        for p in ps:
            if p not in all_ps_flat:
                continue
            p_idx = np.where(all_ps_flat == p)[0][0]
            con_bin[s, p_idx] = 1
    con_bin = pd.DataFrame(con_bin, columns=all_ps_flat, index=con['connector_id'])
    return con_bin

### Functions to sort and match connector binary matrices ###

def get_and_sort_by_match(con_bin1, con_bin2,  pairs_dict, verbose=True):
    '''
    Sort connector binary matrices by matching pairs.'''
    # get postsynaptic partners from binary matrices
    all_ps_flat1 = list(con_bin1.columns.values)
    all_ps_flat2 = list(con_bin2.columns.values)
    

    # check for direct matches of connectors 
    direct_matches = list(set(all_ps_flat1) & set(all_ps_flat2))
    if verbose:
        print(f"Out of {len(all_ps_flat1)} and {len(all_ps_flat2)} postsynaptic partners.")
        print(f"Direct matches: {len(direct_matches)}")
    
    # check for bilateral pair matches 
    pair_id1 = [pairs_dict.get(x, None) for x in all_ps_flat1]
    pair_id2 = [pairs_dict.get(x, None) for x in all_ps_flat2]
    bilateral_matches = list(set(pair_id1) & set(pair_id2))
    if verbose:
        print(f"Bilateral matches: {len(bilateral_matches)}")
    bil_match1 = list(np.where(np.isin(pair_id1, bilateral_matches))[0])
    bil_match2 = list(np.where(np.isin(pair_id2, bilateral_matches))[0])

    bil_match_neu1 = [all_ps_flat1[i] for i in bil_match1]
    bil_match_neu2 = [all_ps_flat2[i] for i in bil_match2]
    #remove direct matches 
    bil_match_neu1 = [x for x in bil_match_neu1 if x not in direct_matches]
    bil_match_neu2 = [x for x in bil_match_neu2 if x not in direct_matches]
    if verbose:
        print(f"Bilateral matches without direct matches: {len(bil_match_neu1)}")
    # sort 
    all_ps = list(np.concatenate([all_ps_flat1, all_ps_flat2]))
    non_matched = [x for x in all_ps if x not in direct_matches and x not in bil_match_neu1 and x not in bil_match_neu2]
    sorted_ps = direct_matches + bil_match_neu1 + bil_match_neu2 + non_matched
    sorted_ps1 = [x for x in sorted_ps if x in con_bin1.columns]
    sorted_ps2 = [x for x in sorted_ps if x in con_bin2.columns]

    if verbose:
        print(f"Sorted postsynaptic partners: {len(sorted_ps)}")
    
    # sort columns 
    con_bin1_sorted = con_bin1[sorted_ps1]
    con_bin2_sorted = con_bin2[sorted_ps2]

    n_matches = len(direct_matches) + len(bil_match_neu1) 

    return con_bin1_sorted, con_bin2_sorted, sorted_ps, n_matches

def first_nonzero_index(row):
    nonzero = np.nonzero(row)[0]
    return nonzero[0] if nonzero.size > 0 else np.inf

def sort_mat_rows_by_occ(mat):
    '''
    Sort rows of a binary matrix by the number of non-zero entries in descending order.
    '''
    first_indices = np.apply_along_axis(first_nonzero_index, 1, mat.values)
    sorted_indices = np.argsort(first_indices)
    return mat.iloc[sorted_indices]

### Apply cosine similarity to connector binary matrices ###

def con_bin_cos_sim(con_bin1, con_bin2, n_match):
    '''
    Calculate cosine similarity between two connector binary matrices.
    '''
    cos_sim_arr = np.zeros((con_bin1.shape[0], con_bin2.shape[0]))
    for r in range(con_bin1.shape[0]):
        a = con_bin1.iloc[r, :n_match].values.reshape(1, -1)
        for r2 in range(con_bin2.shape[0]):
            b = con_bin2.iloc[r2, :n_match].values.reshape(1, -1)
            cos_sim = cosine_similarity(a, b)[0][0]
            cos_sim_arr[r, r2] = cos_sim
    return cos_sim_arr

# try on only connectors with top downstream partners 
def get_top_targets(conb, syn_threshold=3):
    '''
    Get top targets based on a threshold of synaptic connections.
    Filter connectors to only contain those that contain strong connections.
    '''
    conb_filtered = conb.loc[:, conb.sum(axis=0) >= syn_threshold]
    top_targets = conb_filtered.columns
    top_counts = conb_filtered.sum(axis=0)
    top_targets_df = pd.DataFrame({'target': top_targets, 'count': top_counts})
    top_targets_df = top_targets_df.sort_values(by='count', ascending=False)
    # filter out connectors that do not have any top targets 
    conb_filtered = conb_filtered.loc[conb_filtered.sum(axis=1) > 0]
    return conb_filtered, top_targets_df

# %% start with a single neuron 
pair_pick = 240
presyn_neu = pairs['leftid'][pair_pick]
presyn_neuR = pairs['rightid'][pair_pick]
print(f"Cell type {skid_to_celltype.get(presyn_neu, 'NA')}")
con = connector_details[connector_details['presynaptic_to'] == presyn_neu]
conR = connector_details[connector_details['presynaptic_to'] == presyn_neuR]

# try with specific celltype instead 
con = connector_details[connector_details['presynaptic_celltype'] == 'LNs']
conR = connector_details[connector_details['presynaptic_celltype'] == 'LNs']

all_ps_flat = np.unique(list(chain.from_iterable(con['postsynaptic_to'].values)))
n_targets = len(all_ps_flat)
n_cons = con.shape[0]
print(f"Presynaptic neuron {presyn_neu} has {n_targets} postsynaptic targets and {n_cons} connectors.")

con_bin = con_binary_matrix(con, only_known_targets=True, all_neurons=all_neurons)
con_binR = con_binary_matrix(conR, only_known_targets=True, all_neurons=all_neurons)

#%% just get top targets combos based on cell types and pairs

# map postsynaptic partners to pair ids
con['presyn pair_id'] = con['presynaptic_to'].apply(lambda x: pairs_dict.get(x, None))
conR['presyn pair_id'] = conR['presynaptic_to'].apply(lambda x: pairs_dict.get(x, None))
# get postsynaptic pairs for connectors
con['postsyn pair_id'] = con['postsynaptic_to'].apply(lambda x: [pairs_dict.get(v, None) for v in x])
conR['postsyn pair_id'] = conR['postsynaptic_to'].apply(lambda x: [pairs_dict.get(v, None) for v in x])

# get top postsynaptic partners based on threshold of synaptic connections


# %% sort by bilateral matches and fitting rows 

conbL_filtered = get_top_targets(con_bin, syn_threshold=3)[0]
conbR_filtered = get_top_targets(con_binR, syn_threshold=3)[0]

con_bin_sorted, con_binR_sorted, sorted_ps, n_matches = get_and_sort_by_match(conbL_filtered, conbR_filtered, pairs_dict)
con_bin_verysorted = sort_mat_rows_by_occ(con_bin_sorted)
con_binR_verysorted = sort_mat_rows_by_occ(con_binR_sorted)

# %% apply cosine sim directly to binary matrices of bilarteral pairs

cos_sim_arr = con_bin_cos_sim(con_bin_sorted, con_binR_sorted, n_matches)   

# %% check if cos sim results are just high for single neurons targeted
all_scores = np.unique(cos_sim_arr)[::-1]  # sort scores in descending order
for score in all_scores:
    u, v = np.where(cos_sim_arr == score)
    sums1 = []
    sums2 = []
    for u_, v_ in zip(u, v):
        s1 = con_bin_sorted.iloc[u_, :n_matches].sum()
        s2 = con_binR_sorted.iloc[v_, :n_matches].sum()
        sums1.append(s1)
        sums2.append(s2)
    print(f"Score: {score}, with average postsynaptic partners {np.mean(sums1)} and {np.mean(sums2)}")
    print(sums1)
    print(sums2)

### Problem: how do we score similarity with different number of postsynaptic partners??

# %% try single neuron motif finding using low dim embedding and clustering 

### Linear methods ###
# Correspondence Analysis (CA) won't work because it can't deal with the sparsity of the data, i.e. all zero rows and columns. 
# Principal Component Analysis (PCA) is not suitable for binary data, as it assumes continuous data.

### Just plotting or visualisation function used for multiple methods ###
def get_color_scheme(con_bin_sorted_nonzero, skid_to_celltype, mode='n_partners', cmap_= 'Greys'):
    '''
    Get color scheme for the connectors based on the number of postsynaptic partners or a specific cell type.
    '''
    cmap = mpl.colormaps[cmap_]
    
    if mode == 'n_partners':
        n_partners = con_bin_sorted_nonzero.sum(axis=1)
        norm = mpl.colors.Normalize(vmin=n_partners.min()-1, vmax=n_partners.max())
        colors = [cmap(norm(x)) for x in n_partners]
        u_vals = np.unique(n_partners)
        color_name = f'# postsynaptic partners'
        vals = n_partners
    else:
        cellname = mode
        LHNs = [c for c in con_bin_sorted_nonzero.columns if cellname == skid_to_celltype.get(c, '')]
        n_lhn_connections = con_bin_sorted_nonzero[LHNs].sum(axis=1)
        norm = mpl.colors.Normalize(vmin=n_lhn_connections.min()-1, vmax=n_lhn_connections.max())
        colors = [cmap(norm(x)) for x in n_lhn_connections]
        u_vals = np.unique(n_lhn_connections)
        color_name = f'# postsynaptic connections to {cellname}'
        vals = n_lhn_connections
    vals = list(vals)
    return colors, u_vals, color_name, vals

### PCA or ASE embedding ###
def get_connector_similarity(con_bin_sorted_nonzero, con_binR_sorted_nonzero):
    '''
    Get connector similarity scores for PCA or ASE embedding.
    '''

    X = con_bin_sorted_nonzero.values
    X2 = con_binR_sorted_nonzero.values

    # Compute similarity matrix (1 - Jaccard distance)
    similarity = 1 - pairwise_distances(X, metric='jaccard')
    similarity2 = 1 - pairwise_distances(X2, metric='jaccard')

    # Optional: sparsify (threshold) to create adjacency
    adj_matrix =  similarity #(similarity > 0.3).astype(int)  # tune threshold
    adj_matrix2 = similarity2 #(similarity2 > 0.3).astype(int)  # tune threshold
    return adj_matrix, adj_matrix2

def get_pca_embedding(adj_mat, n_components=6):
    pca = PCA(n_components=6)
    embedding = pca.fit_transform(adj_mat)
    return embedding

def get_ase_embedding(adj_mat, n_eigenvectors=6):
    # Symmetrize in case it's not perfectly symmetric
    adj_mat = (adj_mat + adj_mat.T) / 2

    # Compute top k eigenvectors (largest magnitude)
    eigvals, eigvecs = eigsh(adj_mat, k=n_eigenvectors)

    # ASE embedding: use the eigenvectors as coordinates
    embedding = eigvecs
    return embedding

### UMAP embedding ###
def umap_embed_connectors(CB):
    X = CB.copy()

    X = X.reset_index(drop=True)

    # Instantiate and fit UMAP
    reducer = umap.UMAP(metric='jaccard', random_state=42)  # Jaccard is good for binary data
    embedding = reducer.fit_transform(X)

    # Create a DataFrame for visualization
    embedding_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    embedding_df["connector_id"] = CB.index
    return embedding_df



def plot_umap_embedding(embedding_df, CB, skid_dict, mode='n_partners', cmap_='coolwarm'):
    _,_, color_name, acc_ns = get_color_scheme(CB, skid_dict, mode=mode, cmap_=cmap_)
    # Plot the 2D embedding
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=embedding_df, x="UMAP_1", y="UMAP_2", hue=acc_ns, palette='coolwarm')
    plt.title("UMAP projection of binary feature matrix")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True)
    plt.legend(title=color_name)

# %% get target cell types and con bin sorted nonzero 
target_ct = [skid_to_celltype.get(x, 'NA') for x in con_bin_sorted.columns]
print(Counter(target_ct))

# %% Get connector similarity scores for PCA or ASE embedding
con_bin_sorted_nonzero = con_bin_sorted.loc[con_bin_sorted.sum(axis=1) > 1]
con_binR_sorted_nonzero = con_binR_sorted.loc[con_binR_sorted.sum(axis=1) > 1]



#%% compute PCA embedding 

adj_matrix, adj_matrix2 = get_connector_similarity(con_bin_sorted_nonzero, con_binR_sorted_nonzero)
embedding = get_pca_embedding(adj_matrix)

# for cbar 
colors, u_vals, color_name, acc_ns = get_color_scheme(con_bin_sorted_nonzero, skid_to_celltype, mode='n_partners', cmap_='Greys')

# Plot PCA embedding
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, dim in zip([0,1,2,4,5,6], combinations(range(6), 2)):
    axes[i].scatter(embedding[:, dim[0]], embedding[:, dim[1]], c=colors)
    #axes[i].set_title(f"PCA Embedding (Dimensions {dim[0]+1} and {dim[1]+1})")
    axes[i].set_xlabel(f"PCA {dim[0]+1}", fontsize=16)
    axes[i].set_ylabel(f"PCA {dim[1]+1}", fontsize=16)
    axes[i].axis("equal")
    axes[i].set_xlim([-2, 2.4])
    axes[i].set_ylim([-2, 2.4])
    #axes[i].set_xticks([])
    #axes[i].set_yticks([])
    axes[i].grid(False)

axes[3].set_xlim(0, 1)
axes[3].set_ylim(0, len(u_vals))

for i, (val, color) in enumerate(zip(u_vals[::-1], u_colors)):
    axes[3].add_patch(plt.Rectangle((0, i), 0.2, 1, color=color))
    axes[3].text(0.25, i + 0.5, str(val), va='center', fontsize=18)

axes[3].set_title(f"{color_name}", fontsize=20)

axes[3].axis('off')
axes[7].axis('off')

#%% compute ASE embedding
n_eigenvectors = 6  # Number of eigenvectors to use
embedding = get_ase_embedding(adj_matrix, n_eigenvectors=n_eigenvectors)

for dim in combinations(range(n_eigenvectors), 2):
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, dim[0]], embedding[:, dim[1]])
    plt.title(f"ASE Embedding (Dimensions {dim[0]+1} and {dim[1]+1})")
    plt.xlabel(f"ASE {dim[0]+1}")
    plt.ylabel(f"ASE {dim[1]+1}")
    plt.axis("equal")
    plt.grid(True)


# check out one of the ASE dimensions 

group1 = np.argwhere(embedding[:, 5] > 0.08).flatten()
group2 = np.argwhere(embedding[:, 5] < 0.08).flatten()

fig, axes = plt.subplots(1, 2)
axes[0].imshow(con_bin_sorted_nonzero.iloc[group1, :n_matches].values, aspect='auto')
axes[0].set_title('Group 1')
axes[1].imshow(con_bin_sorted_nonzero.iloc[group2, :n_matches].values, aspect='auto')
axes[1].set_title('Group 2')
# 
g1_ids = con_bin_sorted_nonzero.iloc[group1, :n_matches].sum(axis=0)
g2_ids = con_bin_sorted_nonzero.iloc[group2, :n_matches].sum(axis=0)

print("group1")
for n in g1_ids.index:
    print(f"{n}: {skid_to_celltype.get(n, 'NA')}")
print("group2")
for n in g2_ids.index:
    print(f"{n}: {skid_to_celltype.get(n, 'NA')}")

# %% UMAP embedding 

# on whole connector binary matrix

mode='n_partners'
L_embed = umap_embed_connectors(con_bin_sorted_nonzero)
plot_umap_embedding(L_embed, con_bin_sorted_nonzero, skid_to_celltype, mode=mode)

R_embed = umap_embed_connectors(con_binR_sorted_nonzero)
plot_umap_embedding(R_embed, con_binR_sorted_nonzero, skid_to_celltype, mode=mode)

#%%

con_top_L, top_targets_L= get_top_targets(con_bin_sorted_nonzero, syn_threshold=3)
con_top_R, top_targets_R = get_top_targets(con_binR_sorted_nonzero, syn_threshold=3)

con_top_L = con_top_L.loc[con_top_L.sum(axis=1) > 1]
con_top_R = con_top_R.loc[con_top_R.sum(axis=1) > 1]

# UMAP embedding on top targets
L_embed1 = umap_embed_connectors(con_top_L)
R_embed1 = umap_embed_connectors(con_top_R)
plot_umap_embedding(L_embed1, con_top_L, skid_to_celltype, mode=mode)
plot_umap_embedding(R_embed1, con_top_R, skid_to_celltype, mode=mode)
L_embed1.dropna(inplace=True)
R_embed1.dropna(inplace=True)

# %%try clustering 
n_components = 7
eps=0.3
min_samples = 17

def get_umap_clusters(L_embed1, R_embed1, type='GMM', n_components=2, eps=eps, min_samples=min_samples):
    if type == 'GMM':
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(L_embed1[['UMAP_1', 'UMAP_2']])
        L_embed1['cluster'] = gmm.predict(L_embed1[['UMAP_1', 'UMAP_2']])
        gmmR = mixture.GaussianMixture(n_components=n_components)
        gmmR.fit(R_embed1[['UMAP_1', 'UMAP_2']])
        R_embed1['cluster'] = gmmR.predict(R_embed1[['UMAP_1', 'UMAP_2']])
    elif type == 'DBSCAN':
        dbs = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(L_embed1[['UMAP_1', 'UMAP_2']])
        L_embed1['cluster'] = dbs.labels_
        dbsR = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(R_embed1[['UMAP_1', 'UMAP_2']])
        R_embed1['cluster'] = dbsR.labels_
    elif type == 'KMeans':
        kmeans = KMeans(n_clusters=n_components)
        kmeans.fit(L_embed1[['UMAP_1', 'UMAP_2']])
        L_embed1['cluster'] = kmeans.labels_
        kmeansR = KMeans(n_clusters=n_components)
        kmeansR.fit(R_embed1[['UMAP_1', 'UMAP_2']])
        R_embed1['cluster'] = kmeansR.labels_

    return L_embed1, R_embed1

    
L_embed1, R_embed1 = get_umap_clusters(L_embed1, R_embed1, type='KMeans', n_components=n_components)

# plot clusters
def plot_gmm_cluster_umap(embedding_df, title="UMAP projection with GMM clustering"):
    plt.figure()
    sns.scatterplot(data=embedding_df, x="UMAP_1", y="UMAP_2", hue="cluster", palette='Set1')
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")

plot_gmm_cluster_umap(L_embed1, title="Left hemisphere UMAP with GMM clustering")
plot_gmm_cluster_umap(R_embed1, title="Right hemisphere UMAP with GMM clustering")


for c in np.unique(L_embed1['cluster']):
    c_ids_L = L_embed1[L_embed1['cluster'] == c]['connector_id'].values
    c_ids_R = R_embed1[R_embed1['cluster'] == c]['connector_id'].values
    top_L_c = con_top_L.loc[c_ids_L,:].sum(axis=0).to_frame()
    top_L_c.columns = ['count']
    top_R_c = con_top_R.loc[c_ids_R,:].sum(axis=0).to_frame()
    top_R_c.columns = ['count']
    top_L_c['celltype'] = top_L_c.index.map(skid_to_celltype)
    top_R_c['celltype'] = top_R_c.index.map(skid_to_celltype)
    top_L_c['pair_id'] = [pairs_dict.get(x, 'NA') for x in top_L_c.index]
    top_R_c['pair_id'] = [pairs_dict.get(x, 'NA') for x in top_R_c.index]
    print(f"Cluster {c} in left hemisphere:")
    print(top_L_c.sort_values(by='count', ascending=False).head(10))
    print(f"Cluster {c} in right hemisphere:")
    print(top_R_c.sort_values(by='count', ascending=False).head(10))

#%%
def construct_polyadic_motif_counts(con_top_, cluster_ids, skid_to_celltype, pairs_dict):
    #cluster_celltypes =  [skid_to_celltype.get(x, 'NA') for x in cluster_ids]
    #cluster_pairs = [pairs_dict.get(x, 'NA') for x in cluster_ids]
    ct_combos = []

    con_top_ = con_top_.loc[cluster_ids, :]  # filter connectors by cluster ids

    for connector in con_top_.index:
        interm = con_top_.loc[connector, :]
        interm = interm[interm > 0].index  # keep only non-zero counts
        interm_ct = [skid_to_celltype.get(x, 'NA') for x in interm]
        interm_unique_ct = np.unique(interm_ct)
        ct_combos.append(interm_unique_ct)
        counter = Counter(tuple(sorted(arr)) for arr in ct_combos)

    return ct_combos, counter

d, d_= construct_polyadic_motif_counts(con_top_R, c_ids_R, skid_to_celltype, pairs_dict)
d_

for c in np.unique(L_embed1['cluster']):
    c_ids_L = L_embed1[L_embed1['cluster'] == c]['connector_id'].values
    c_ids_R = R_embed1[R_embed1['cluster'] == c]['connector_id'].values
    l, l_ = construct_polyadic_motif_counts(con_top_L, c_ids_L, skid_to_celltype, pairs_dict)
    r, r_ = construct_polyadic_motif_counts(con_top_R, c_ids_R, skid_to_celltype, pairs_dict)
    print(f"Cluster {c} in left hemisphere:")
    print(l_)
    print(f"Cluster {c} in right hemisphere:")
    print(r_)



#%%% extract groups (manually) just do have a quick look 

### rearrangin of connectors for vis ###
def get_connector_overview(X1c, conb, skid_to_celltype):
    x = conb.loc[X1c].sum(axis=0).to_frame()
    #x = x[x[0] > 1].sort_values(by=0, ascending=False)
    x['celltype'] = x.index.map(skid_to_celltype)
    return x 

def sort_by_ct(conb, skid_to_celltype):
    neuronids = conb.columns
    cts = [skid_to_celltype.get(x, 'NA') for x in neuronids]
    sorted_idx = np.argsort(cts)
    sorted_cts = [ct for ct in np.array(cts)[sorted_idx]]
    n_cts_sorted = list(Counter(sorted_cts).values())
    
    conb_ct_sorted = conb.iloc[:, sorted_idx]
    return conb_ct_sorted, sorted_cts, n_cts_sorted

def get_ticks(n_cts_sorted):
    acc = [0] + list(accumulate(n_cts_sorted))
    ticks = [(x/2)+acc[e] for e, x in enumerate(n_cts_sorted)]
    return ticks

L1 = np.argwhere(L_embed['UMAP_1'] < 5).flatten()
L2 = np.argwhere(L_embed['UMAP_1'] > 5).flatten()
R1 = np.argwhere(R_embed['UMAP_1'] < 7).flatten()
R2 = np.argwhere(R_embed['UMAP_1'] > 7).flatten()

L1c = L_embed[L_embed['UMAP_1'] < 5]['connector_id'].values
L2c = L_embed[L_embed['UMAP_1'] > 5]['connector_id'].values
R1c = R_embed[R_embed['UMAP_1'] < 7]['connector_id'].values
R2c = R_embed[R_embed['UMAP_1'] > 7]['connector_id'].values

l1 = get_connector_overview(L1c, con_bin_sorted_nonzero, skid_to_celltype)
l2 = get_connector_overview(L2c, con_bin_sorted_nonzero, skid_to_celltype)
r1 = get_connector_overview(R1c, con_binR_sorted_nonzero, skid_to_celltype)
r2 = get_connector_overview(R2c, con_binR_sorted_nonzero, skid_to_celltype)

mergedl = pd.merge(l1, l2, left_index=True, right_index=True, how='outer', suffixes=('_L1', '_L2'))
mergedr = pd.merge(r1, r2, left_index=True, right_index=True, how='outer', suffixes=('_R1', '_R2'))
merged = pd.merge(mergedl, mergedr, left_index=True, right_index=True, how='outer')
merged = merged.fillna(0)

thresh = 3.0
merged.sort_values(by='0_L1', ascending=False, inplace=True)
top_l1 = merged[merged['0_L1'] > thresh].index
top_l1_ct = [skid_to_celltype.get(x, 'NA') for x in top_l1]
merged.sort_values(by='0_L2', ascending=False, inplace=True)
top_l2 = merged[merged['0_L2'] > thresh].index
top_l2_ct = [skid_to_celltype.get(x, 'NA') for x in top_l2]

merged.sort_values(by='0_R1', ascending=False, inplace=True)
top_r1 = merged[merged['0_R1'] > thresh].index
top_r1_ct = [skid_to_celltype.get(x, 'NA') for x in top_r1]
merged.sort_values(by='0_R2', ascending=False, inplace=True)
top_r2 = merged[merged['0_R2'] > thresh].index
top_r2_ct = [skid_to_celltype.get(x, 'NA') for x in top_r2]


fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.flatten()
for i, l, r in zip(range(2), [L1, L2], [R1, R2]):
    left_conb = con_bin_sorted_nonzero.iloc[l, :n_matches]
    right_conb = con_binR_sorted_nonzero.iloc[r, :n_matches]
    # drop non-zeroes to make easier to visualize
    left_conb = left_conb.loc[left_conb.sum(axis=1) > 0]
    right_conb = right_conb.loc[right_conb.sum(axis=1) > 0]

    left_conb, L_cts, L_ct_counts = sort_by_ct(left_conb, skid_to_celltype)
    right_conb, R_cts, R_ct_counts = sort_by_ct(right_conb, skid_to_celltype)

    axes[i].imshow(left_conb.values, aspect='auto')  
    axes[i+2].imshow(right_conb.values, aspect='auto')

    # setting tick labels 
    axes[i].set_xticks(get_ticks(L_ct_counts))
    axes[i+2].set_xticks(get_ticks(R_ct_counts))
    axes[i].set_xticklabels(np.unique(L_cts), rotation=90)
    axes[i+2].set_xticklabels(np.unique(R_cts), rotation=90)

    axes[i].set_title(f'Group {i+1} Left')
    axes[i+2].set_title(f'Group {i+1} Right')
    axes[i+2].set_xlabel('Postsynaptic partner')

axes[0].set_ylabel('Connectors')
axes[2].set_ylabel('Connectors')

fig.tight_layout()
#%%

# sort by cell type for visualisation? 

# global sort by cell type only 

L_celltypes = [skid_to_celltype.get(x, 'NA') for x in con_bin_sorted_nonzero.columns]
R_celltypes = [skid_to_celltype.get(x, 'NA') for x in con_binR_sorted_nonzero.columns]
L_celltypes.sort()
R_celltypes.sort()
L_ct_counts = list(Counter(L_celltypes).values())
R_ct_counts = list(Counter(R_celltypes).values())

cons_ct_sorted = con_bin_sorted_nonzero[L_celltypes]
consR_ct_sorted = con_binR_sorted_nonzero[R_celltypes]

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
for i, g in enumerate([gR1, gR2, gR3, gR4]):
    axes[i].imshow(con_binR_sorted_nonzero.iloc[g, :n_matches].values, aspect='auto')
    axes[i].set_title(f'Group {i+1}')

# %%
