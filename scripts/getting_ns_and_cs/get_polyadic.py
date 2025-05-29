#%%

import numpy as np
import pandas as pd
import contools
import matplotlib.pyplot as plt
import pymaid 
import os
from scripts.functions.little_helper import inspect_data

#get parent directory path
current_file = __file__  # Replace with your file path if not in a script
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)

#%% get connectors from catmaid

from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

# select neurons to include 
wanted_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(wanted_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# get all synaptic sites associated with neurons 
links = pymaid.get_connector_links(all_neurons, chunk_size=50)

#get all connectors 
all_links = pymaid.get_connectors(all_neurons)

#%% inspect how to get all synaptic sites including neuronal fragments 

#include incomplete and partially differentiated neurons to compare numbers 
links_wB = pymaid.get_connector_links(wanted_neurons, chunk_size=50)
n_entries_wB, n_connectors_wB, n_skeletons_wB, n_nodes_wB, n_postsynaptic_wB, n_presynaptic_wB = inspect_data(links_wB, verbose=True)

# %% inspect shape of data
n_entries, n_connectors, n_skeletons, n_nodes, n_postsynaptic, n_presynaptic = inspect_data(links, verbose=True)

data_numbers = pd.DataFrame()
data_numbers['Object'] = ['entries', 'connectors', 'skeletons', 'nodes', 'postsynaptic sites', 'presynaptic sites'] 
data_numbers['All connections to and from brain, input and access. neurons; except very incomplete, motor or partially diff.'] = [n_entries, n_connectors, n_skeletons, n_nodes, n_postsynaptic, n_presynaptic]
data_numbers['All connections to and from brain, input and access. neurons'] = [n_entries_wB, n_connectors_wB, n_skeletons_wB, n_nodes_wB, n_postsynaptic_wB, n_presynaptic_wB]
data_numbers.to_excel(parent_dir+'/data/data_numbers.xlsx', index=False)

n_pre_relations = links.groupby('relation').value_counts()

#%% get connectors associated with connectors used here 
all_connectors = links['connector_id'].unique()
connector_details = pymaid.get_connector_details(all_connectors)
#INFO  : Data for 218070 of 221015 unique connector IDs retrieved (pymaid) 



# %% figure out what exactly a catmaid connector is

#get everything associated with 1 connector ID
connector_id = links['connector_id'].unique()[0]
connector = links[links['connector_id'] == connector_id]
connector 

#a connector in catmaid appears to be a synaptic site - with one pre- and multiple post-synaptic partners

# %% get connectors with presynaptic sites

connectors = links['connector_id'].unique()

# Check if all connectors have at least one 'presynaptic_to' in the 'relation' column
has_presynaptic = links.groupby('connector_id')['relation'].apply(lambda x: 'presynaptic_to' in x.values)
n_with_presynaptic = has_presynaptic.sum()
print(f"Number of connectors with at least one 'presynaptic_to': {n_with_presynaptic} out of {n_connector}")

#get connectors with presynaptic site 
connector_with_presyn = has_presynaptic[has_presynaptic].index
#filter connectors by those with presynaptic sites
links_with_presyn = links[links['connector_id'].isin(connector_with_presyn)]

# %% find out the average number of post-synaptic partners per connector

mean_post_all = links.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).mean()
print(f"Mean number of postsynaptic partners per connector (all): {mean_post_all}")

mean_post_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).mean()
print(f"Mean number of postsynaptic partners per connector (filtered by with presynaptic site): {mean_post_filtered}")

max_post_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).max()
print(f"Max number of postsynaptic partners per connector (filtered): {max_post_filtered}")

min_post_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).min()
print(f"Min number of postsynaptic partners per connector (filtered): {min_post_filtered}")
#%%
#look at distribution of number of postsynaptic partners per connector
postperconnect_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum())

# Plot histogram of the number of postsynaptic partners per connector
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(postperconnect_filtered, bins=range(postperconnect_filtered.min(), postperconnect_filtered.max() + 2))

# Annotate each bin with its count
for count, bin_edge in zip(counts, bins):
    if count > 0:  
        count_str = str(int(count))
        if count > 1000:
            count_str = f"{int(count / 1000)}k"
        ax.text(bin_edge + 0.5, count, count_str, ha='center', va='bottom', rotation=90, fontsize=8)

ax.set_xlabel('Number of Postsynaptic Partners')
ax.set_ylabel('Number of Connectors')
ax.set_title('Distribution of Postsynaptic Partners per Connector')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.show()

#%% Identification or grouping of connection partners 

# %%
