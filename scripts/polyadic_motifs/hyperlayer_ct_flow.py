from collections import Counter
from itertools import chain, combinations, accumulate
import time

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D

from scripts.polyadic_motifs.motif_functions import (con_binary_matrix, get_top_targets, con_bin_cos_sim, 
get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets, 
normalise_freq_cols, filter_con, map_con_targets_to_flow, get_flow_motifs, remove_incomplete_flow_motifs, map_con_targets_to_real_neurons, get_partner_motifs)
from scripts.initialisation_scripts.get_me_started import get_me_started, get_me_labelled
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict = get_me_started()

all_neurons = connector_details['presynaptic_to'].unique().tolist()

con_rand = polyadic_edge_permutation(connector_details, rng=rng)
con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

# %% try look at the distribution of top downstream partners and randos

# think need to do this with the 'global top targets' too, otherwise it might be too high a threshold? 
# this is same for randomly permutated networks
glob_top_targets = {}
for pn in all_neurons:
    conb = con_binary_matrix(connector_details[connector_details['presynaptic_to'] == pn],
                            only_known_targets=False, all_neurons=all_neurons)
    _, top_targets_df = get_top_targets(conb, syn_threshold=3)
    glob_top_targets[pn] = list(top_targets_df['target'])
#%%
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']

conL['n_post'] = conL['postsynaptic_to'].apply(lambda x: len(x))
conR['n_post'] = conR['postsynaptic_to'].apply(lambda x: len(x))
con_randL['n_post'] = con_randL['postsynaptic_to'].apply(lambda x: len(x))
con_randR['n_post'] = con_randR['postsynaptic_to'].apply(lambda x: len(x))

#%%
class HyperCon:
    # to manage and filter multiple layers of connector dataframes
    def __init__(self, filter_col: str = 'n_post'):
        self.df = {} #store original connector dataframe by name
        self.filter_col = filter_col

    def add_df(self, name: str, data: pd.DataFrame):
        self.df[name] = data
    
    def filter_by_layer(self, name: str, value: int) -> pd.DataFrame:
        if name not in self.df:
            raise ValueError(f"Dataframe {name} not found.")
        data = self.df[name]
        return data[data[self.filter_col] == value].reset_index(drop=True)
    
    def get_all_filtered(self, value: int) -> dict:
        """ Returns a dictionary of filtered dataframes for all stored dataframes """
        return {name: self.filter_by_layer(name, value) for name in self.df}
    
    def apply_some_function_on_binarised(self, value: int, func) -> dict:
        """ Returns a dictionary of binary matrices for all stored dataframes """
        results = {}
        for name, data in self.get_all_filtered(value).items():
            bin_data = con_binary_matrix(data, only_known_targets=True, all_neurons=all_neurons)
            results[name] = func(bin_data)
        return results
        
    def apply_some_function(self, value: int, func) -> dict:
        results = {}
        for name, data in self.get_all_filtered(value).items():
            results[name] = func(data)
        return results
    
    def apply_multiple_functions(self, value: int, func_list: list) -> dict:
        results = {}
        for name, data in self.get_all_filtered(value).items():
            result = data
            for func in func_list:
                result = func(result)
            results[name] = result
        return results

    
hypercon = HyperCon(filter_col='n_post')
hypercon.add_df('conL', conL)
hypercon.add_df('conR', conR)
hypercon.add_df('con_randL', con_randL)
hypercon.add_df('con_randR', con_randR)

#%% look at just layer 2 

hypercon2 = hypercon.get_all_filtered(2)

# %%
ct_names = celltype_df['name'].unique().tolist()

def filter_col_presyn(con, ct):
    return con[con['presynaptic_celltype'] == ct]

def sort_list_by_input(unordered_list, order):
    cat = pd.Categorical(unordered_list, categories=order, ordered=True)
    return pd.Series(unordered_list).sort_values(key=lambda x: cat).tolist()

def remove_na(target_df):
    target_df = target_df[target_df['motif'].apply(lambda x: 'NA' not in x)]
    return target_df

def get_top_motifs_only(target_df, top_percentage=0.01):
    all_syns = target_df['count'].sum()
    thresh = all_syns * top_percentage
    target_df = target_df[target_df['count'] >= thresh]
    return target_df

def get_unique_ct_target_counts(con, remove_nan=True, top_percentage=0.01):
    ps_targets = con['postsynaptic_celltype'].tolist()
    ps_targets = [['NA' if k is None else k for k in sublist] for sublist in ps_targets]
    ps_targets_sorted = [tuple(sort_list_by_input(y, ct_names)) for y in ps_targets] 
    unique_ct_targets = np.unique(ps_targets_sorted).tolist()
    ct_target_counts = Counter(ps_targets_sorted)
    target_df = pd.DataFrame(ct_target_counts.items(), columns=['motif','count'])
    if remove_nan:
        target_df = remove_na(target_df)
    if top_percentage is not None:
        target_df = get_top_motifs_only(target_df, top_percentage=top_percentage)
    target_df = target_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    return target_df

results_ct = {}
for c in ct_names:
    results = hypercon.apply_multiple_functions(2, [lambda x: filter_col_presyn(x, c), get_unique_ct_target_counts])
    results_ct[c] = results


for k, res in results_ct.items():
    df_merged = None
    for key, df in res.items():
        df = df.rename(columns={'count': key})
        if df_merged is None:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, on='motif', how='outer')
    df_merged = df_merged.fillna(0)
    results_ct[k] = df_merged



# %%
