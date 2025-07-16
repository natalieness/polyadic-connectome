''' Find polyadic motifs across neurons
Idea1: direct matchign of connectors
Idea2: dim reduction and clustering of connectors
'''

from collections import Counter
from itertools import chain, combinations
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


# %% start with a single neuron 
presyn_neu = pairs['leftid'][30]
presyn_neuR = pairs['rightid'][30]
con = connector_details[connector_details['presynaptic_to'] == presyn_neu]
conR = connector_details[connector_details['presynaptic_to'] == presyn_neuR]

all_ps_flat = np.unique(list(chain.from_iterable(con['postsynaptic_to'].values)))
n_targets = len(all_ps_flat)
n_cons = con.shape[0]
print(f"Presynaptic neuron {presyn_neu} has {n_targets} postsynaptic targets and {n_cons} connectors.")

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

con_bin = con_binary_matrix(con, only_known_targets=True, all_neurons=all_neurons)
con_binR = con_binary_matrix(conR, only_known_targets=True, all_neurons=all_neurons)


# %%


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

con_bin_sorted, con_binR_sorted, sorted_ps, n_matches = get_and_sort_by_match(con_bin, con_binR, pairs_dict)
con_bin_verysorted = sort_mat_rows_by_occ(con_bin_sorted)
con_binR_verysorted = sort_mat_rows_by_occ(con_binR_sorted)

# %%

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

cos_sim_arr = con_bin_cos_sim(con_bin_sorted, con_binR_sorted, n_matches)


    

# %% check if this is just single neurons targeted 

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

# %% try single neuron motif finding using spectral embedding and clustering 


