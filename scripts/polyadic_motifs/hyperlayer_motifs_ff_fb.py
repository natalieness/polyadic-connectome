


from collections import Counter
from itertools import chain, combinations, accumulate
import time

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity

from scripts.polyadic_motifs.motif_functions import (con_binary_matrix, get_top_targets, con_bin_cos_sim, 
get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets, 
normalise_freq_cols, filter_con)
from scripts.initialisation_scripts.get_me_started import get_me_started, get_me_labelled
from scripts.functions.random_polyadic_networks import polyadic_edge_permutation

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict = get_me_started()

all_neurons = connector_details['presynaptic_to'].unique().tolist()

con_rand = polyadic_edge_permutation(connector_details, rng=rng)
con_rand = get_me_labelled(con_rand, skid_to_celltype, pairs, pairs_dict)

#%%
# filter by hemisphere 
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']

#filter by celltype or neuron_id 
type = 'group' #individual or group
ct = 'LNs'

conL_f, conR_f = filter_con(conL, conR, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)
con_randL_f, con_randR_f = filter_con(con_randL, con_randR, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)

#%% 

#%%
conbL_f = con_binary_matrix(conL_f, only_known_targets=True, all_neurons=all_neurons)
conbR_f = con_binary_matrix(conR_f, only_known_targets=True, all_neurons=all_neurons) 
conb_randL_f = con_binary_matrix(con_randL_f, only_known_targets=True, all_neurons=all_neurons)
conb_randR_f = con_binary_matrix(con_randR_f, only_known_targets=True, all_neurons=all_neurons)

# %%

conbL_f, top_targets_df_L = get_top_targets(conbL_f, syn_threshold=3)
conbR_f, top_targets_df_R = get_top_targets(conbR_f, syn_threshold=3)
conb_randL_f, top_targets_df_randL = get_top_targets(conb_randL_f, syn_threshold=3)
conb_randR_f, top_targets_df_randR = get_top_targets(conb_randR_f, syn_threshold=3)

#%% construct hyperlayer graphs instead 

conL_f['n_post'] = conL_f['postsynaptic_to'].apply(lambda x: len(x))
conR_f['n_post'] = conR_f['postsynaptic_to'].apply(lambda x: len(x))
con_randL_f['n_post'] = con_randL_f['postsynaptic_to'].apply(lambda x: len(x))
con_randR_f['n_post'] = con_randR_f['postsynaptic_to'].apply(lambda x: len(x))

# get only layers that are not super sparse ??
# maybe only include connectors with at least 1% of all connectors in the layer?
n_post_threshold = 0.01 * conL_f['n_post'].sum()
L_overview = conL_f['n_post'].value_counts().to_dict()
hyperlayers = [k for k, v in L_overview.items() if v > n_post_threshold]
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
hypercon.add_df('conL_f', conL_f)
hypercon.add_df('conR_f', conR_f)
hypercon.add_df('con_randL_f', con_randL_f)
hypercon.add_df('con_randR_f', con_randR_f)
filtered_df = hypercon.get_all_filtered(4)  # example for filtering by layer with 4 postsynaptic partners

top_targets_hypercon = hypercon.apply_some_function_on_binarised(4, get_top_targets)

#%%

def map_con_targets_to_flow(con, flow_dict=flow_dict):
    flow_scores_series = pd.Series(index=con.index, dtype=object)
    for c in con.index:
        c_id = con.loc[c].connector_id
        postsynaptic_to = con.loc[c, 'postsynaptic_to']
        own_flow = flow_dict.get(con.loc[c, 'presynaptic_to'], None)
        if own_flow is not None:
            flow_scores = [flow_dict.get(skid, 'NA') for skid in postsynaptic_to]
            flow_scores = [(score-own_flow if isinstance(score, (int, float)) else 'NA') 
                           for score in flow_scores]
            flow_scores = [('FB' if score > 0 else 'FF') if isinstance(score, (int, float)) else 'NA' 
                           for score in flow_scores]
            flow_scores_series[c] = flow_scores
    con['flow_scores'] = flow_scores_series
    return con

def get_flow_motifs(con):
    # make sure orientation is always in order 
    flow_scores_series = con['flow_scores']
    flow_scores_series = flow_scores_series.apply(lambda x: tuple(sorted(x)) if isinstance(x, (list, tuple)) else x)
    flow_motifs = flow_scores_series.value_counts().to_dict()
    return flow_motifs

#%%
# try it 
val = 8
result_flow_motifs = hypercon.apply_multiple_functions(val, [map_con_targets_to_flow, get_flow_motifs])

fm_summary = pd.DataFrame.from_dict(result_flow_motifs, orient='columns')
fm_summary.index = fm_summary.index.map(lambda x: str(x))
fm_summary = fm_summary.fillna(0)

within_cos_sim = cosine_similarity(np.array(fm_summary['conR_f']).reshape(1,-1), np.array(fm_summary['conL_f']).reshape(1,-1))
across_cos_sim1 = cosine_similarity(np.array(fm_summary['conL_f']).reshape(1,-1), np.array(fm_summary['con_randL_f']).reshape(1,-1))
across_cos_sim2 = cosine_similarity(np.array(fm_summary['conR_f']).reshape(1,-1), np.array(fm_summary['con_randR_f']).reshape(1,-1))
print(f"For value: {val}, Cosine similarity: L-R: {within_cos_sim}, L-randL: {across_cos_sim1}, R-randL: {across_cos_sim2}")



# %%
