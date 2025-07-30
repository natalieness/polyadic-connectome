


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

#%%
# filter by hemisphere 
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']
con_randL = con_rand[con_rand['presyn_hemi'] == 'left']
con_randR = con_rand[con_rand['presyn_hemi'] == 'right']

#filter by celltype or neuron_id 
type = 'group' #individual or group
ct = 'MBONs'

conL_f, conR_f = filter_con(conL, conR, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)
con_randL_f, con_randR_f = filter_con(con_randL, con_randR, type=type, ct=ct, ct_n=5, celltype_df=celltype_df)

# else just take the whole connector dataframe:
# conL_f = conL.copy()
# conR_f = conR.copy()
# con_randL_f = con_randL.copy()
# con_randR_f = con_randR.copy()

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
#
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

#top_targets_hypercon = hypercon.apply_some_function_on_binarised(4, get_top_targets)

#%%
# try it 
val = 3
result_flow_motifs = hypercon.apply_multiple_functions(val, [map_con_targets_to_flow, get_flow_motifs])

fm_summary = pd.DataFrame.from_dict(result_flow_motifs, orient='columns')
fm_summary.index = fm_summary.index.map(lambda x: str(x))
fm_summary = fm_summary.fillna(0)
fm_summary['within-var'] = (fm_summary['conL_f'] - fm_summary['conR_f'])/ (fm_summary['conR_f']+fm_summary['conL_f'])
fm_summary['across_L'] = (fm_summary['conL_f'] - fm_summary['con_randL_f']) / (fm_summary['conL_f'] + fm_summary['con_randL_f'])
fm_summary['across_R'] = (fm_summary['conR_f'] - fm_summary['con_randR_f']) / (fm_summary['conR_f'] + fm_summary['con_randR_f'])

within_cos_sim = cosine_similarity(np.array(fm_summary['conR_f']).reshape(1,-1), np.array(fm_summary['conL_f']).reshape(1,-1))
across_cos_sim1 = cosine_similarity(np.array(fm_summary['conL_f']).reshape(1,-1), np.array(fm_summary['con_randL_f']).reshape(1,-1))
across_cos_sim2 = cosine_similarity(np.array(fm_summary['conR_f']).reshape(1,-1), np.array(fm_summary['con_randR_f']).reshape(1,-1))
print(f"For value: {val}, Cosine similarity: L-R: {within_cos_sim}, L-randL: {across_cos_sim1}, R-randL: {across_cos_sim2}")

#%% iterate through all layers and get flow motifs to compare distribution

def sort_and_bar_plot_motifs(partner_summary, title=None):
    idxes = partner_summary.index.tolist()
    idxes = [eval(j) for j in idxes]
    loc_true = np.where(['NA' not in j for j in idxes])[0]
    partner_summary_f = partner_summary.iloc[loc_true,:]
    print(loc_true)
    idxes = [j for e, j in enumerate(idxes) if e in loc_true]
    n_ff = [j.count('FF') for j in idxes]
    n_ff_order = np.argsort(n_ff)
    partner_summary_f = partner_summary_f.iloc[n_ff_order,:]

    # now plot
    #partner_summary_norm = partner_summary_f / partner_summary_f.sum(axis=0).values
    fig, ax = plt.subplots(figsize=(10, 6))
    xvals = np.arange(partner_summary_f.shape[0])
    bw = 0.15
    colrs = ["#037AF1","#89BBEE", "#FA7D00", "#ECBA87"]
    for i, col in enumerate(['conL_f', 'con_randL_f', 'conR_f', 'con_randR_f']):
        ax.bar(xvals + i * bw, partner_summary_f[col], width=bw, label=col, alpha=0.7, color=colrs[i])
    ax.set_xticks(xvals + 1.5 * bw)
    ax.set_xticklabels(partner_summary_f.index, rotation=45, ha='right')
    ax.set_ylabel('Number of synapses')
    ax.legend(['Left', 'Rand. Left', 'Right', 'Rand. Right'], loc='upper left')


cos_sims = pd.DataFrame(index= ['within', 'widthin_rand', 'across_L', 'across_R'])
proportions = pd.DataFrame(index= ['L', 'R', 'randL', 'randR'])

for n in range(2,9):
    val = n 
    #note i'm just removing motifs with fragments or unknown neurons here - maybe not the best way to solve this
    the_motifs = hypercon.apply_multiple_functions(val, [map_con_targets_to_flow, get_flow_motifs])
    partner_summary = pd.DataFrame.from_dict(the_motifs, orient='columns')
    partner_summary.index = partner_summary.index.map(lambda x: str(x))
    partner_summary = partner_summary.fillna(0)
    sort_and_bar_plot_motifs(partner_summary, title=f"Flow motifs for {val} partners")

    sums = partner_summary.sum(axis=0).values
    top_str = str(tuple((['FF'])*val))
    low_str = str(tuple((['FB'])*val))
    if top_str in partner_summary.index:
        just_tops = partner_summary.loc[str(tuple((['FF'])*val))].values / sums
    else:
        just_tops = np.zeros_like(sums)
    if low_str in partner_summary.index:
        just_lows = partner_summary.loc[str(tuple((['FB'])*val))].values / sums
    else:
        just_lows = np.zeros_like(sums)
    all_motifs = 1 - just_tops - just_lows

    within_partner_cos_sim = cosine_similarity(np.array(partner_summary['conR_f']).reshape(1,-1), np.array(partner_summary['conL_f']).reshape(1,-1))
    within_partner_cos_sim2 = cosine_similarity(np.array(partner_summary['con_randL_f']).reshape(1,-1), np.array(partner_summary['con_randR_f']).reshape(1,-1))
    across_partner_cos_sim1 = cosine_similarity(np.array(partner_summary['conL_f']).reshape(1,-1), np.array(partner_summary['con_randL_f']).reshape(1,-1))
    across_partner_cos_sim2 = cosine_similarity(np.array(partner_summary['conR_f']).reshape(1,-1), np.array(partner_summary['con_randR_f']).reshape(1,-1))
    cos_sims[f'val_{n}'] = [within_partner_cos_sim[0][0], within_partner_cos_sim2[0][0],
                            across_partner_cos_sim1[0][0], across_partner_cos_sim2[0][0]]
    proportions[f'{n}-ff'] = just_tops
    proportions[f'{n}-fb'] = just_lows
    proportions[f'{n}-mixed'] = all_motifs


#
proportions = proportions.T
#proportions['L-randL'] = proportions['L'] - proportions['randL']
#proportions['R-randR'] = proportions['R'] - proportions['randR']

new_index = proportions.index.str.extract(r'(?P<num>\d+)-(?P<class>\w+)')
proportions.index = pd.MultiIndex.from_frame(new_index)
proportions['class'] = proportions.index.get_level_values('class')
proportions['num'] = proportions.index.get_level_values('num').astype(int)

proportions = proportions.reset_index(drop=True)
prop_plot = proportions.groupby('class').agg(list)
propPlot = prop_plot.drop(columns=['num'])
propPlot = propPlot.loc[['ff','mixed','fb']]
#propPlot = propPlot.explode('L').explode('R').explode('randL').explode('randR')
#propPlot = propPlot.melt(value_vars=['L','R','randL','randR'], ignore_index=False)
#propPlot.sort_values(by=['variable'], key=lambda x: x.map({'L': 0, 'randL': 1, 'R': 2, 'randR': 3}), inplace=True)

mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.spines.right': False, 'axes.spines.top': False})

fig, ax = plt.subplots(figsize=(7, 6))
x_offset = 0.15
x_vals = np.arange(propPlot.shape[0])
x_flat = x_vals
x_vals = [ [x]*len(propPlot['L'][0]) for x in x_vals]
x_vals = list(chain.from_iterable(x_vals))
colrs = ["#037AF1","#89BBEE", "#FA7D00", "#ECBA87"]
ms_min = 12
ms_max = 90
markersizes = np.linspace(ms_min, ms_max, 7)
markersizes = list(chain.from_iterable([markersizes]*propPlot.shape[0]))

size_legend = [
    Line2D([0], [0], marker='o', color='none', label='2', 
           markerfacecolor='gray', markersize=np.sqrt(ms_min)),  # sqrt to match scatter sizing
    Line2D([0], [0], marker='o', color='none', label='8', 
           markerfacecolor='gray', markersize=np.sqrt(ms_max)),
]


for i, (col, lbl) in enumerate(zip(['L','randL','R','randR'], ['Left', 'Rand. Left', 'Right', 'Rand. Right'])):
    ax.scatter(np.array(x_vals) + i * x_offset, propPlot[col].explode(), label=lbl, alpha=0.5, color=colrs[i], s=markersizes)
    mean_vals = propPlot
    ax.errorbar(np.array(x_flat) + i * x_offset, [np.mean(y) for y in list(propPlot[col])], fmt='_', color=colrs[i], markersize=13, mew=3, capsize=5, elinewidth=0)

ax.set_xticks(x_flat + 1.5 * x_offset)
ax.set_xticklabels(propPlot.index)
ax.set_ylabel('Proportion of postsynaptic partners')

data_legend = ax.legend(title='Presynaptic neurons', loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.add_artist(data_legend) 
ax.legend(handles=size_legend, title='No. of partners', loc='upper right', bbox_to_anchor=(1.3, 0.73))

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


#%%

''' This whole section was used to explore the partner motifs and their distributions,
and make a bunch of plots '''


val = 4
result_partner_motifs = hypercon.apply_multiple_functions(val, [get_partner_motifs, get_flow_motifs])
partner_summary = pd.DataFrame.from_dict(result_partner_motifs, orient='columns')
partner_summary.index = partner_summary.index.map(lambda x: str(x))

within_partner_cos_sim = cosine_similarity(np.array(partner_summary['conR_f']).reshape(1,-1), np.array(partner_summary['conL_f']).reshape(1,-1))
within_partner_cos_sim2 = cosine_similarity(np.array(partner_summary['con_randL_f']).reshape(1,-1), np.array(partner_summary['con_randR_f']).reshape(1,-1))
across_partner_cos_sim1 = cosine_similarity(np.array(partner_summary['conL_f']).reshape(1,-1), np.array(partner_summary['con_randL_f']).reshape(1,-1))
across_partner_cos_sim2 = cosine_similarity(np.array(partner_summary['conR_f']).reshape(1,-1), np.array(partner_summary['con_randR_f']).reshape(1,-1))
print(f"For value: {val}, Cosine similarity: L-R: {within_partner_cos_sim}, L-randL: {across_partner_cos_sim1}, R-randL: {across_partner_cos_sim2}")
partner_summary

partner_summary_norm = partner_summary / partner_summary.sum(axis=0).values
fig, ax = plt.subplots(figsize=(10, 6))
xvals = np.arange(partner_summary.shape[0])
bw = 0.15
colrs = ["#037AF1","#89BBEE", "#FA7D00", "#ECBA87"]
for i, col in enumerate(['conL_f', 'con_randL_f', 'conR_f', 'con_randR_f']):
    ax.bar(xvals + i * bw, partner_summary_norm[col], width=bw, label=col, alpha=0.7, color=colrs[i])
ax.set_xticks(xvals + 1.5 * bw)
ax.set_xticklabels(partner_summary.index, rotation=45, ha='right')

#
# Transpose the DataFrame to switch index and columns
df_transposed = partner_summary_norm #.iloc[::-1] #.T
df_transposed = df_transposed[['conL_f', 'con_randL_f', 'conR_f', 'con_randR_f']]

# Plot the transposed DataFrame as a stacked bar chart
ax = df_transposed.plot(
    kind='bar',
    stacked=False,
    figsize=(10, 6),
    color=["#037AF1", "#89BBEE", "#FA7D00", "#ECBA87"],
)

ax.set_title("Stacked Bar Plot of MultiIndex Rows (Stacked) Across Columns")
ax.set_xlabel("Motif")
ax.set_ylabel("Proportion of synapses")
ax.legend(title="Index", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# same but delta 
df_delta = partner_summary_norm.copy()
df_delta_idx = df_delta.index.tolist()
df_delta_idx = [eval(j) for j in df_delta_idx]
idx_n = [j.count('TOP') for j in df_delta_idx]
idx_n_order = np.argsort(idx_n)
df_delta = df_delta.iloc[idx_n_order,:]
df_delta = df_delta.iloc[:,:]
df_delta['L-randL'] = df_delta['conL_f'] - df_delta['con_randL_f']
df_delta['R-randR'] = df_delta['conR_f'] - df_delta['con_randR_f']
df_delta = df_delta[['L-randL', 'R-randR']]
ax = df_delta.plot(
    kind='bar',
    stacked=False,
    figsize=(10, 6),
    color=["#037AF1", "#FA7D00"],
)
ax.set_ylabel("Delta proportion of synapses \n compared to random")
ax.legend(['Left', 'Right'], title="Hemisphere", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


#
cos_sims = pd.DataFrame(index= ['within', 'widthin_rand', 'across_L', 'across_R'])
proportions = pd.DataFrame(index= ['L', 'R', 'randL', 'randR'])
for n in range(2,9):
    val = n 
    result_partner_motifs = hypercon.apply_multiple_functions(val, [get_partner_motifs, get_flow_motifs])
    partner_summary = pd.DataFrame.from_dict(result_partner_motifs, orient='columns')
    partner_summary.index = partner_summary.index.map(lambda x: str(x))
    partner_summary = partner_summary.fillna(0)

    sums = partner_summary.sum(axis=0).values
    just_tops = partner_summary.loc[str(tuple((['TOP'])*val))].values / sums
    just_lows = partner_summary.loc[str(tuple((['LOW'])*val))].values / sums
    all_motifs = 1 - just_tops - just_lows

    within_partner_cos_sim = cosine_similarity(np.array(partner_summary['conR_f']).reshape(1,-1), np.array(partner_summary['conL_f']).reshape(1,-1))
    within_partner_cos_sim2 = cosine_similarity(np.array(partner_summary['con_randL_f']).reshape(1,-1), np.array(partner_summary['con_randR_f']).reshape(1,-1))
    across_partner_cos_sim1 = cosine_similarity(np.array(partner_summary['conL_f']).reshape(1,-1), np.array(partner_summary['con_randL_f']).reshape(1,-1))
    across_partner_cos_sim2 = cosine_similarity(np.array(partner_summary['conR_f']).reshape(1,-1), np.array(partner_summary['con_randR_f']).reshape(1,-1))
    cos_sims[f'val_{n}'] = [within_partner_cos_sim[0][0], within_partner_cos_sim2[0][0],
                            across_partner_cos_sim1[0][0], across_partner_cos_sim2[0][0]]
    proportions[f'{n}-tops'] = just_tops
    proportions[f'{n}-lows'] = just_lows
    proportions[f'{n}-mixed'] = all_motifs


#%
proportions = proportions.T
#proportions['L-randL'] = proportions['L'] - proportions['randL']
#proportions['R-randR'] = proportions['R'] - proportions['randR']

new_index = proportions.index.str.extract(r'(?P<num>\d+)-(?P<class>\w+)')
proportions.index = pd.MultiIndex.from_frame(new_index)
proportions['class'] = proportions.index.get_level_values('class')
proportions['num'] = proportions.index.get_level_values('num').astype(int)

proportions = proportions.reset_index(drop=True)
#proportions = proportions.unstack('class')
#proportions = proportions.swaplevel(axis=1).sort_index(axis=1)
proportions


#

prop_plot = proportions.groupby('class').agg(list)
propPlot = prop_plot.drop(columns=['num'])
#propPlot = propPlot.explode('L').explode('R').explode('randL').explode('randR')
#propPlot = propPlot.melt(value_vars=['L','R','randL','randR'], ignore_index=False)
#propPlot.sort_values(by=['variable'], key=lambda x: x.map({'L': 0, 'randL': 1, 'R': 2, 'randR': 3}), inplace=True)

mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.spines.right': False, 'axes.spines.top': False})

fig, ax = plt.subplots(figsize=(7, 6))
x_offset = 0.15
x_vals = np.arange(propPlot.shape[0])
x_flat = x_vals
x_vals = [ [x]*len(propPlot['L'][0]) for x in x_vals]
x_vals = list(chain.from_iterable(x_vals))
colrs = ["#037AF1","#89BBEE", "#FA7D00", "#ECBA87"]
ms_min = 12
ms_max = 90
markersizes = np.linspace(ms_min, ms_max, 7)
markersizes = list(chain.from_iterable([markersizes]*propPlot.shape[0]))

size_legend = [
    Line2D([0], [0], marker='o', color='none', label='2', 
           markerfacecolor='gray', markersize=np.sqrt(ms_min)),  # sqrt to match scatter sizing
    Line2D([0], [0], marker='o', color='none', label='8', 
           markerfacecolor='gray', markersize=np.sqrt(ms_max)),
]


for i, (col, lbl) in enumerate(zip(['L','randL','R','randR'], ['Left', 'Rand. Left', 'Right', 'Rand. Right'])):
    ax.scatter(np.array(x_vals) + i * x_offset, propPlot[col].explode(), label=lbl, alpha=0.5, color=colrs[i], s=markersizes)
    mean_vals = propPlot
    ax.errorbar(np.array(x_flat) + i * x_offset, [np.mean(y) for y in list(propPlot[col])], fmt='_', color=colrs[i], markersize=13, mew=3, capsize=5, elinewidth=0)

ax.set_xticks(x_flat + 1.5 * x_offset)
ax.set_xticklabels(['Low only', 'Mixed targets', 'Top only'])
ax.set_ylabel('Proportion of postsynaptic partners')

data_legend = ax.legend(title='Presynaptic neurons', loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.add_artist(data_legend) 
ax.legend(handles=size_legend, title='No. of partners', loc='upper right', bbox_to_anchor=(1.3, 0.73))


# %%
