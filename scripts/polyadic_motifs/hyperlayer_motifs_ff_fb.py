


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
conL_f = conL.copy()
conR_f = conR.copy()
con_randL_f = con_randL.copy()
con_randR_f = con_randR.copy()

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

#%% example usage on a single layer
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

#%% check out celltype motifs across specific layers 

val = 3 



# %% attempt to delete all the code above in lieu for this nicer general version 

def compute_motif_summary(hypercon, val, motif_funcs, motif_labels=None):
    """
    Applies list of functions to hypercon object filtered by layer 'val'
    Parameters:
    hypercon: HyperCon object containing connector dataframes
    val: int, the layer
    motif_funcs: list of functions to apply to the filtered connector dataframes
    motif_labels: list of str (optional) to override column names in result dataframe  
    """
    result = hypercon.apply_multiple_functions(val, motif_funcs)
    df = pd.DataFrame.from_dict(result, orient='columns')
    df.index = df.index.map(lambda x: str(x))
    df = df.fillna(0)
    # sort the df columns to group by hemisphere 
    df = df[['conL_f', 'con_randL_f', 'conR_f', 'con_randR_f']]
    if motif_labels: 
        df.columns = df.columns.map(lambda x: motif_labels[x] if x in motif_labels else x)
    
    return df

def plot_motif_bar_comparison(summary_df, colors=None, title=None, plot_proportions=False, remove_na = False):
    """
    Create grouped bar plot comparing motif counts or proportions across conditions.
    
    Parameters:
        summary_df: pd.DataFrame - motif summary
        labels: list of str - legend labels for each column
        colors: list of str - colors per condition
        title: str - optional plot title
        plot_proportions: bool - whether to plot proportions
    """
    if plot_proportions:
        summary_df = summary_df.div(summary_df.sum(axis=0), axis=1)

    labels = summary_df.columns.tolist()

    # sort the motifs to make it easier to compare 
    # get all the different options in the motifs 
    idxes = [eval(j) for j in summary_df.index.tolist()]
    if remove_na:
        idxes_real = [i for i, tup in enumerate(idxes) if 'NA' not in tup]
        summary_df = summary_df.iloc[idxes_real,:]
        idxes = [j for i, j in enumerate(idxes) if i in idxes_real]
        
    motif_elements = np.unique(list(chain.from_iterable(idxes)))
    # sort the elements if any fo the following are in there 
    favourites = ['FF', 'REAL', 'TOP']
    el = [j for j in favourites if j in motif_elements]
    if len(el) > 0:
        motif_elements = np.append(el, motif_elements[~np.isin(motif_elements, el)])

    # now sort the index of the summary_df by the motif elements
    n_first = [j.count(motif_elements[0]) for j in idxes]
    n_first_order = np.argsort(n_first)
    summary_df = summary_df.iloc[n_first_order[::-1],:]

    fig, ax = plt.subplots(figsize=(10, 6))
    xvals = np.arange(summary_df.shape[0])
    bw = 0.15
    if summary_df.shape[1] == 4:
        colrs = ["#037AF1","#89BBEE", "#FA7D00", "#ECBA87"]
    else:
        colrs = colors if colors else sns.color_palette("tab10", n_colors=summary_df.shape[1])

    for i, col in enumerate(summary_df.columns):
        ax.bar(xvals + i * bw, summary_df[col], width=bw, label=labels[i], alpha=0.7, color=colrs[i])

    ax.set_xticks(xvals + (len(summary_df.columns)-1)/2 * bw)
    ax.set_xticklabels(summary_df.index, rotation=45, ha='right')
    ax.set_ylabel('Proportion of synapses' if plot_proportions else 'Motif count')
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()


def compute_cosine_similarities(df):
    """
    Compute cosine similarities between specified column pairs.

    Parameters:
        df: pd.DataFrame - motif summary table

    Returns:
        dict: {pair: cosine similarity}
    """
    results = {}
    for (a, b) in combinations(df.columns, 2):
        sim = cosine_similarity(df[a].values.reshape(1, -1), df[b].values.reshape(1, -1))[0][0]
        results[f'{a}-{b}'] = sim
    return results

def extract_motif_proportions(summary_df, props=None):
    """
    Extract specific motif proportions (e.g., TOP/LOW/mixed) across datasets.

    Parameters:
        summary_df: pd.DataFrame

    Returns:
        dict: proportions by type and condition
    """
    sums = summary_df.sum(axis=0).values
    idxes = [eval(j) for j in summary_df.index.tolist()]
    elements = np.unique(list(chain.from_iterable(idxes)))
    elements = [j for j in elements if j != 'NA'] # remove nas from elements to examine
    if len(elements) > 2:
        raise ValueError("Expected only two elements in motifs, e.g. 'TOP' and 'LOW'.")
    
    inferred_val = len(idxes[0]) # number of elements in each motif 
    top_str = str(tuple(([elements[0]])*inferred_val))
    low_str = str(tuple(([elements[1]])*inferred_val))

    if top_str in summary_df.index:
        just_tops = summary_df.loc[top_str].values / sums
    else:
        just_tops = np.zeros_like(sums)
    if low_str in summary_df.index:
        just_lows = summary_df.loc[low_str].values / sums
    else:
        just_lows = np.zeros_like(sums)
    all_motifs = 1 - just_tops - just_lows

    if props is None:
        props = pd.DataFrame(index=summary_df.columns)
    props[f'{inferred_val}-{elements[0]}'] = just_tops
    props[f'{inferred_val}-mixed'] = all_motifs
    props[f'{inferred_val}-{elements[1]}'] = just_lows
    
    return props

def plot_motif_summary_pure_and_mixed(proportions):

    # reformat proportions for plotting 
    proportions['class'] = proportions.index.get_level_values('class')
    proportions['num'] = proportions.index.get_level_values('num').astype(int)

    proportions = proportions.reset_index(drop=True)
    prop_plot = proportions.groupby('class').agg(list)
    propPlot = prop_plot.drop(columns=['num'])
    # ensure mixed type is in the middle 
    loc_mid = np.where(propPlot.index.unique() == 'mixed')[0][0]
    other_locs = [f for f in range(propPlot.shape[0]) if f != loc_mid]
    propPlot = propPlot.iloc[[other_locs[0], loc_mid, other_locs[1]], :]

    fig, ax = plt.subplots(figsize=(7, 6))
    x_offset = 0.15
    x_vals = np.arange(propPlot.shape[0])
    x_flat = x_vals
    x_vals = [ [x]*len(propPlot.iloc[0,0]) for x in x_vals]
    x_vals = list(chain.from_iterable(x_vals))
    if propPlot.shape[1] == 4:
        colrs = ["#037AF1","#89BBEE", "#FA7D00", "#ECBA87"]
    else:
        colrs = sns.color_palette("tab10", n_colors=propPlot.shape[1])
    ms_min = 12
    ms_max = 90
    markersizes = np.linspace(ms_min, ms_max, len(propPlot.iloc[0,0]))
    markersizes = list(chain.from_iterable([markersizes]*propPlot.shape[0]))

    size_legend = [
        Line2D([0], [0], marker='o', color='none', label='2', 
            markerfacecolor='gray', markersize=np.sqrt(ms_min)),  # sqrt to match scatter sizing
        Line2D([0], [0], marker='o', color='none', label=f'{len(propPlot.iloc[0,0])+1}', 
            markerfacecolor='gray', markersize=np.sqrt(ms_max)),
    ]


    for i, col in enumerate(propPlot.columns):
        ax.scatter(np.array(x_vals) + i * x_offset, propPlot[col].explode(), label=col, alpha=0.5, color=colrs[i], s=markersizes)
        ax.errorbar(np.array(x_flat) + i * x_offset, [np.mean(y) for y in list(propPlot[col])], fmt='_', color=colrs[i], markersize=13, mew=3, capsize=5, elinewidth=0)

    ax.set_xticks(x_flat + 1.5 * x_offset)
    ax.set_xticklabels(propPlot.index)
    ax.set_ylabel('Proportion of postsynaptic partners')

    data_legend = ax.legend(title='Presynaptic neurons', loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.add_artist(data_legend) 
    ax.legend(handles=size_legend, title='No. of partners', loc='upper right', bbox_to_anchor=(1.3, 0.73))


def motif_summary_across_layers(hypercon, motif_funcs, layer_range=range(2, 9), labels=None, plot_individual=True, plot_summary=True):
    """
    Run analysis across multiple layers and return summary tables.

    Returns:
        proportions_df: pd.DataFrame
        cosine_df: pd.DataFrame
    """
    props=None
    cos_sims = {}
    summary_all = {}

    for val in layer_range:
        summary = compute_motif_summary(hypercon, val, motif_funcs, motif_labels=labels)

        props = extract_motif_proportions(summary, props=props)

        cos = compute_cosine_similarities(summary)
        cos_sims[f'{val}'] = cos

        summary_all[val] = summary

        if plot_individual: # plot each layer's summary bar graph of motifs 
            plot_motif_bar_comparison(summary, title=f"Flow motifs for {val} partners", plot_proportions=False, remove_na=True)

    proportions = props.T
    new_index = proportions.index.str.extract(r'(?P<num>\d+)-(?P<class>\w+)')
    proportions.index = pd.MultiIndex.from_frame(new_index)
    if plot_summary:
        plot_motif_summary_pure_and_mixed(proportions)

    cos_df = pd.DataFrame.from_dict(cos_sims, orient='index')

    return proportions, cos_df, summary_all



# example usage for flow 

mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.spines.right': False, 'axes.spines.top': False})

#val = 5
motif_funcs = [map_con_targets_to_flow, get_flow_motifs]
motif_labels =  {'conL_f': 'Left', 'con_randL_f': 'Rand. Left', 'conR_f': 'Right', 'con_randR_f': 'Rand. Right'}
# individual layer:
# flow_motif_summary = compute_motif_summary(hypercon, val, motif_funcs, motif_labels)
# plot_motif_bar_comparison(flow_motif_summary, motif_labels, title=f"Flow motifs for {val} partners", plot_proportions=True)
# props = extract_motif_proportions(flow_motif_summary, props=None)

proportions_flow_all, cos_flow_all, summary_flow_all = motif_summary_across_layers(hypercon, motif_funcs, layer_range=range(2, 9), labels=motif_labels)


# TODO: check when removing na's would make sense - atm it doesn't make sense because it messes up the proportions by just filtering out all the mixed synapses 

# %% using partner motifs 
partner_funcs = [get_partner_motifs, get_flow_motifs]

props_partner_all, cos_partner_all, summary_partner_all = motif_summary_across_layers(hypercon, partner_funcs, layer_range=range(2, 9))

# %%
#%%

''' This whole section was used to explore the partner motifs and their distributions,
and make a bunch of plots '''

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