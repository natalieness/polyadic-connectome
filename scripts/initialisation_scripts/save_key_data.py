# save all neuron connector of pymaid to avoid having to call again 
import pandas as pd
import numpy as np

import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)

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

connector_details.to_csv('init_data/connector_details2025.csv', index=False)


#%% get signal flow data 

ann = pymaid.get_annotation_list()
ann = ann['name'].to_list()

ann_lvl7 = [a for a in ann if ('level-7_clusterID' in a) and ('signal-flow' in a)]

flow_neurons = pd.DataFrame(columns=['skid', 'flow_score'])
for a in ann_lvl7:
    val = a[-6:]
    try:
        val = float(val) # this should only work for negative values
    except ValueError:
        val = val[1:] # remove underscore 
        val = float(val)

    lvl_skids = pymaid.get_skids_by_annotation(a)
    flow_neurons = pd.concat([flow_neurons, pd.DataFrame({'skid': lvl_skids, 'flow_score': val})], ignore_index=True)

flow_neurons.to_csv('init_data/flow_neurons.csv', index=False)
