from collections import Counter
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl

from scripts.functions.random_polyadic_networks import polyadic_edge_permutation

import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)
#%% import and filter data from catmaid 
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

#%% script to test functions - generate random polyadic network 

rng = np.random.default_rng(42)  # Set a random seed for reproducibility
rand_connectors = polyadic_edge_permutation(connector_details, rng)
