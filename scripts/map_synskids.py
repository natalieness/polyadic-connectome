''' This scripts maps connectors to the neuronal identity of their pre- and
postsynaptic sites. Using hand annotated neuronal celltypes from catmaid
'''
#%%
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


# %% Filter connectors by presynapse and check if they belong to annotated neurons

connectors = links['connector_id'].unique()

# Check if all connectors have at least one 'presynaptic_to' in the 'relation' column
has_presynaptic = links.groupby('connector_id')['relation'].apply(lambda x: 'presynaptic_to' in x.values)
print(f"Number of connectors with at least one 'presynaptic_to': {has_presynaptic.sum()} out of {n_connector}")

#get connectors with presynaptic site 
connector_with_presyn = has_presynaptic[has_presynaptic].index
#filter connectors by those with presynaptic sites
links_with_presyn = links[links['connector_id'].isin(connector_with_presyn)]

