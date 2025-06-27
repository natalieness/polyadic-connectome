#%%

from itertools import chain
from collections import Counter
from itertools import combinations
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


from contools import Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

# local imports
from scripts.functions.little_helper import inspect_data, get_celltype_dict, get_celltype_name, celltype_col_for_list
from scripts.functions.signal_flow import signal_flow

rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% get and describe neuronal identity data 

''' note: leaving this here in case we want to use it later, but first looking at flow independent of celltype'''
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()

print("Cell types used")
n_skids = 0
for ct in celltypes:
    print(f"Name: {ct.get_name()}, Skids: {len(ct.get_skids())}, Color: {ct.get_color()}")
    n_skids += len(ct.get_skids())
print(f"Total number of skids: {n_skids}")

# get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)
ct_names = celltype_df['name'].unique()

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

# remove connector details without presynaptic site 
connector_details = connector_details.dropna(subset=['presynaptic_to'])

print(f"After removing connectors without presynaptic site, {len(connector_details)} connectors remain")


# %% # map skid ids in connector details to celltypes

#connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))
#celltype_col_for_list(connector_details, 'postsynaptic_to', skid_to_celltype=skid_to_celltype, new_col_name='postsynaptic_celltype')
# create subset of connector details with only labelled neurons
#connector_details_presyn_labelled = connector_details[connector_details['presynaptic_celltype'] != 'NA']
#labelled_connectors = connector_details_presyn_labelled[~connector_details_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
#remove connectors with no labelled postsynaptic celltypes
#labelled_connectors = labelled_connectors[labelled_connectors['postsynaptic_celltype'].apply(lambda x: len(x) > 0)]
#print(f"Number of connectors with only labelled presynaptic and postsynaptic celltypes: {len(labelled_connectors)}")
# %% order skids using signal flow metric 

#construct directed adjacency matrix
directed_adj = pymaid.adjacency_matrix(all_neurons, targets=all_neurons)
adj_arr = directed_adj.to_numpy()

# apply signal flow metric
# this involves a step to remove self-loops, so does not need to be done here
z = signal_flow(adj_arr)

# map z metric to skids 
z_dict = dict(zip(all_neurons, z))

# %% examine whether z metric / signal flow position is related to number of polyadic synaptic partners 

# in connector details, count number of postsynaptic partners at each connector 

connector_details['n_post'] = connector_details['postsynaptic_to'].apply(lambda x: len(x))
#filter out connectors with no postsynaptic partners
connector_details = connector_details[connector_details['n_post'] > 0]

mean_ns = []
std_ns = []
for skid in z_dict.keys():
    mean_n_post = connector_details[connector_details['presynaptic_to'] == skid]['n_post'].mean()
    std_n_post = connector_details[connector_details['presynaptic_to'] == skid]['n_post'].std()
    mean_ns.append(mean_n_post)
    std_ns.append(std_n_post)

z_based_df = pd.DataFrame()
z_based_df['skid'] = list(z_dict.keys())
z_based_df['z'] = list(z_dict.values())
z_based_df['mean_n_post'] = mean_ns
z_based_df['std_n_post'] = std_ns
#%%
# bin z metric 
bin_int = 0.5 
z_min = np.floor(z_based_df['z'].min())
z_max = np.ceil(z_based_df['z'].max())

z_bins = np.arange(z_min, z_max+bin_int, bin_int)
z_based_df['z_bin'] = pd.cut(z_based_df['z'], bins=z_bins, include_lowest=True)


# plot z metric again mean number of postsynaptic partners

g = sns.catplot(x='z_bin', y='mean_n_post', data=z_based_df, kind='strip', height=5, aspect=1.5)
ax = g.ax
ax.set_xticks(z_bins[1:-1])
sns.pointplot(x='z_bin', y='mean_n_post', data=z_based_df, estimator=np.mean, ax=ax, color='black', join=False, markers='_', markersize=3, errwidth=5, capsize=0.1, zorder=3)

# %% scatterplot with rolling mean
mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 15, 'axes.titlesize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 14})
fig, ax = plt.subplots()
sns.scatterplot(x='z', y='mean_n_post', data=z_based_df, ax=ax, color='C0', s=3, alpha=0.7)
# rolling mean 
plot_z_df = z_based_df[['z', 'mean_n_post']].copy()
plot_z_df.sort_values(by='z', inplace=True)
round_by = 1
bins = plot_z_df.groupby(plot_z_df.z.round(round_by)).mean()

ax.plot(bins.index, bins.mean_n_post, color='C1', linewidth=2, label='Rolling mean')
ax.set_ylabel('Mean number of postsynaptic partners')
ax.set_xlabel('Signal flow metric (z)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
fig.tight_layout()


# %%
