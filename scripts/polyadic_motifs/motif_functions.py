from itertools import chain
from collections import Counter

import pandas as pd
import numpy as np


### Constructing vector binary matrices for connectors ###
def con_binary_matrix(con, only_known_targets=False, all_neurons=None):
    if all_neurons is None:
        raise ValueError("Please provide a list of all neurons to filter targets.")
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