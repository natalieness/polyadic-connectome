
from itertools import chain, combinations


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D


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
    return fig 


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
    return fig


def motif_summary_across_layers(hypercon, motif_funcs, layer_range=range(2, 9), labels=None, plot_individual=True, plot_summary=True, save_figs=None):
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
            bar_fig = plot_motif_bar_comparison(summary, title=f"Flow motifs for {val} partners", plot_proportions=False, remove_na=True)
            if save_figs is not None:
                bar_fig.savefig(f"{save_figs}motif_summary_{val}_ps.png", bbox_inches='tight')

    proportions = props.T
    new_index = proportions.index.str.extract(r'(?P<num>\d+)-(?P<class>\w+)')
    proportions.index = pd.MultiIndex.from_frame(new_index)
    if plot_summary:
        summary_fig = plot_motif_summary_pure_and_mixed(proportions)
        if save_figs is not None:
            summary_fig.savefig(f"{save_figs}motif_summary_pure_and_mixed.png", bbox_inches='tight')

    cos_df = pd.DataFrame.from_dict(cos_sims, orient='index')

    return proportions, cos_df, summary_all