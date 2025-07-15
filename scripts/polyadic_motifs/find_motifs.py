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

from contools import Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.little_helper import get_celltype_dict, celltype_col_for_list, get_celltype_name, celltype_col_for_nestedlist

rm = pymaid.CatmaidInstance(url, token, name, password)

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

#%% get bilateral pairs 

pairs = pd.read_csv('data/pairs-2022-02-14.csv')
right_ns = pairs['rightid'].unique()
left_ns = pairs['leftid'].unique()

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
    return con_bin, all_ps_flat

con_bin, all_ps_flat = con_binary_matrix(con, only_known_targets=True, all_neurons=all_neurons)
con_binR, all_ps_flatR = con_binary_matrix(conR, only_known_targets=True, all_neurons=all_neurons)


# %%
