''' This scripts maps connectors to the neuronal identity of their pre- and
postsynaptic sites. Using hand annotated neuronal celltypes from catmaid
'''
#%%
from itertools import chain
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    #get index of pr

