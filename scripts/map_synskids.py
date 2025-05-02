''' This scripts maps connectors to the neuronal identity of their pre- and
postsynaptic sites. Using hand annotated neuronal celltypes from catmaid
'''
#%%
from itertools import chain
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token
from scripts.little_helper import inspect_data

rm = pymaid.CatmaidInstance(url, token, name, password)

#%% get and describe neuronal identity data 
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()

print("Cell types used")
n_skids = 0
for ct in celltypes:
    print(f"Name: {ct.get_name()}, Skids: {len(ct.get_skids())}, Color: {ct.get_color()}")
    n_skids += len(ct.get_skids())
print(f"Total number of skids: {n_skids}")

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


# %% Filter connectors by presynapse 

connectors = links['connector_id'].unique()
n_connector = links['connector_id'].nunique()

# Check if all connectors have at least one 'presynaptic_to' in the 'relation' column
has_presynaptic = links.groupby('connector_id')['relation'].apply(lambda x: 'presynaptic_to' in x.values)
print(f"Number of connectors with at least one 'presynaptic_to': {has_presynaptic.sum()} out of {n_connector}")

#get connectors with presynaptic site 
connector_with_presyn = has_presynaptic[has_presynaptic].index
#filter connectors by those with presynaptic sites
links_with_presyn = links[links['connector_id'].isin(connector_with_presyn)]


# %% Check if connectors with presynapse belong to any labelled neurons

#get all skids that are in celltype_df (all labelled neurons)
all_labelled_skids = celltype_df['skids'].explode().unique()

# Check if all connectors with presynapse belong to any labelled neurons
has_labelled_neuron = links_with_presyn.groupby('connector_id')['skeleton_id'].apply(lambda x: np.isin(x.values, all_labelled_skids).any())
# %%

n_labelled = has_labelled_neuron.sum()
links_with_labelled = links_with_presyn[links_with_presyn['connector_id'].isin(has_labelled_neuron[has_labelled_neuron].index)]
print(f"Number of connectors with presynapse that belong to labelled neurons: {n_labelled}")
links_with_labelled = links_with_labelled.reset_index(drop=True)
# %% easily map skids to celltypes

skid_to_celltype = {
    skid: row['name']
    for _, row in celltype_df.iterrows()
    for skid in row['skids']
}
def get_celltype_name(skid):
    return skid_to_celltype.get(skid, "NA")  # Returns "NA" if skid is not found


# %% to do set this up so you can use connector_details to easily check identity at each intersection

# remove connector details without presynaptic site 
connector_details = connector_details.dropna(subset=['presynaptic_to'])



# %% # map skid ids in connector details to celltypes

connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(get_celltype_name)

def celltype_col_for_list(connector_df, col_name, new_col_name='postsynaptic_celltype'):
    df_series = connector_df[col_name]
    new_df_series = []
    for l in df_series:
        #each element is a list of skids 
        new_l = []
        for skid in l:
            new_l.append(get_celltype_name(skid))
        new_df_series.append(new_l)
    connector_df[new_col_name] = new_df_series

celltype_col_for_list(connector_details, 'postsynaptic_to', new_col_name='postsynaptic_celltype')


# %% create subset of connector details with only labelled neurons
connector_details_presyn_labelled = connector_details[connector_details['presynaptic_celltype'] != 'NA']
labelled_connectors = connector_details_presyn_labelled[~connector_details_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
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




# %% get correlation between presynaptic and postsynaptic celltypes (independent of connector)

ct_names = celltype_df['name'].unique()

def get_ct_index(ct_name):
    return np.where(ct_names == ct_name)[0][0]

arr = np.zeros((len(ct_names), len(ct_names)))

#iterate through connectors 
for row in labelled_connectors.iterrows():
    row_presyn = row[1]['presynaptic_celltype']
    row_postsyn = row[1]['postsynaptic_celltype']
    #get index of presynaptic celltype
    presyn_index = get_ct_index(row_presyn)
    #iterate through postsynaptic celltypes
    for post_ct in row_postsyn:
        #get index of postsynaptic celltype
        post_index = get_ct_index(post_ct)
        #increment the value in the array
        arr[presyn_index, post_index] += 1


fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(arr, annot=True, fmt=".0f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names)
ax.set_xlabel('Postsynaptic Celltype')
ax.set_ylabel('Presynaptic Celltype')

#%% compute relative to total presynaptic connections per cell type 

def compute_relative(arr):
    arr_norm = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        row_sum = np.sum(arr[i, :])
        for j in range(arr.shape[1]):
            arr_norm[i, j] = arr[i, j] / row_sum if row_sum != 0 else 0
    return arr_norm

arr_norm = compute_relative(arr)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(arr_norm, annot=True, fmt=".2f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names)
ax.set_xlabel('Postsynaptic Celltype')
ax.set_ylabel('Presynaptic Celltype')



# %%
#clsutermap to sort by similarity
plt.figure()
sns.clustermap(arr, annot=True, fmt=".0f", cmap='magma', xticklabels=ct_names, yticklabels=ct_names)


# %% get co-occurency matrix between postsynaptic celltypes (independent of presynaptic partner)

post_cooccurency = np.zeros((len(ct_names), len(ct_names)))
#iterate through connectors
for row in labelled_connectors.iterrows():
    row_postsyn = row[1]['postsynaptic_celltype']
    #iterate through postsynaptic celltypes
    for e, post_ct in enumerate(row_postsyn):
        for e2, post_ct2 in enumerate(row_postsyn[e+1:]):
            #get index of postsynaptic celltype
            post_index = get_ct_index(post_ct)
            post_index2 = get_ct_index(post_ct2)
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
sns.heatmap(plot_post_cooccurency, annot=True, fmt=".0f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(post_cooccurency))
ax.set_xlabel('Celltype')
ax.set_ylabel('Celltype') 

#%% compute relative covariance of postsynaptic celltypes

def compute_relative_covariance(post_cooccurency):
    #mirror matrix to be able to compute covariance relative to each row 
    upper_triangle = np.triu(post_cooccurency, k=1)
    mirrored = post_cooccurency + upper_triangle.T
    row_sums = np.sum(mirrored, axis=1)
    cov = np.zeros(post_cooccurency.shape)
    for i in range(mirrored.shape[0]):
        for j in range(mirrored.shape[1]):
            cov[i, j] = mirrored[i, j] / row_sums[i] 

    return cov

relative_cov = compute_relative_covariance(post_cooccurency)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(relative_cov, annot=True, fmt=".2f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(post_cooccurency))






# %% Jaccard similarity to normalize for how frequent individual items are

def jaccard_similarity(arr):
    ''' Calculate jaccard similarity based on a co-occurency matrix '''
    jaccard_sim = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            intersection = arr[i, j]
            print(f"Intersection: {intersection}")
            union = np.sum(arr[i, :]) + np.sum(arr[:, j]) - intersection
            print(f"Union: {union}")
            jaccard_sim[i, j] = intersection / union if union != 0 else 0
    return jaccard_sim

post_jaccard = jaccard_similarity(post_cooccurency)

plot_jaccard = post_jaccard.copy()
# set lower triangle to nan
plot_jaccard[lower_triangle_indices] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(plot_jaccard, annot=True, fmt=".2f", cmap='magma', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(plot_jaccard))

# notelarge relative covariance of one group with another can be masked by frequent occurence of 
# only one of the groups


# %% compute pointwise conditional probability

# compute marginal probability of each cell type co-occuring with itself or other 
def get_P_marginal(arr, ct_index):
    num = np.sum(arr[ct_index, :]) #sum of all postsynaptic celltypes pairs involving this cell type
    denom = np.sum(arr) #sum of all postsynaptic celltype pairs 
    return num / denom

def precompute_P_marginal(arr):
    Ps_marginal = []
    for i in range(arr.shape[0]):
        Ps_marginal.append(get_P_marginal(arr, i))
    return Ps_marginal

def get_P_ij(arr, ct_index_i, ct_index_j):
    num = arr[ct_index_i, ct_index_j] #sum of all postsynaptic celltypes pairs involving this cell type
    denom = np.sum(arr) #sum of all postsynaptic celltype pairs 
    return num / denom

def compute_PMI(arr, Ps_marginal):
    pmi = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(i, arr.shape[1]):
            Pij = get_P_ij(arr, i, j)
            pmi[i, j] = np.log2(Pij / (Ps_marginal[i] * Ps_marginal[j]))
            if Pij == 0:
                pmi[i, j] = np.nan
    return pmi


Ps_marginal = precompute_P_marginal(post_cooccurency)
pmi = compute_PMI(post_cooccurency, Ps_marginal)

# set lower triangle to nan
plot_pmi = pmi.copy()
plot_pmi[lower_triangle_indices] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(plot_pmi, annot=True, fmt=".2f", cmap='PiYG', cbar=True, xticklabels=ct_names, yticklabels=ct_names, mask=np.isnan(plot_pmi))




# %% probability of occuring alone 


## DOUBLE CHECK THIS CONCEPTUALLY

def get_P_alone(df_series, ct_name):
    n_ct = 0
    n_sets = 0 
    for l in df_series:
        n_sets += 1
        if ct_name in l:
            if len(l) ==1:
                n_ct += 1
    return n_ct / n_sets

def get_P_alone_all(df_series, ct_names):
    alone = []
    for ct in ct_names.unique():
        alone.append(get_P_alone(df_series, ct))
    return alone

ct_P_alone = get_P_alone_all(labelled_connectors['postsynaptic_celltype'], celltype_df['name'])


# %%
