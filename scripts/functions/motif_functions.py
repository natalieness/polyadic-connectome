from itertools import chain
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


### Constructing vector binary matrices for connectors ###
def con_binary_matrix(con, only_known_targets=False, all_neurons=None):
    if all_neurons is None:
        raise ValueError("Please provide a list of all neurons to filter targets.")
    all_ps_flat = np.unique(list(chain.from_iterable(con['postsynaptic_to'].values)))
    if only_known_targets:
        all_ps_flat = np.intersect1d(all_ps_flat, all_neurons)

    n_targets = len(all_ps_flat)
    n_cons = con.shape[0]

    con_bin = np.zeros((n_cons, n_targets), dtype=int)
    for s in range(n_cons):
        c_id = con['connector_id'].values[s]
        ps = con['postsynaptic_to'].values[s]
        for p in ps:
            if p not in all_ps_flat:
                continue
            p_idx = np.where(all_ps_flat == p)[0][0]
            con_bin[s, p_idx] = 1
    con_bin = pd.DataFrame(con_bin, columns=all_ps_flat, index=con['connector_id'])
    return con_bin

def filter_con(conLeft, conRight=None, pairs_dict=None, pairs=None, type='individual', ct='LNs', ct_n=5, celltype_df=None):
    """ Filter connectors either by an individual presynaptic neuron 
    or a group of presynaptic neurons of a specific celltype.
    """
    conR_f = 0

    if type =='group':
        conL_f = conLeft[conLeft['presynaptic_celltype'] == ct]
        if conRight is not None:
            conR_f = conRight[conRight['presynaptic_celltype'] == ct]
        print(f'Filtered by all presynaptic neurons of celltype {ct}.')
    elif type == 'individual':
        # filter by specific neuron 
        presyn = celltype_df[celltype_df['name'] == ct]['skids'].values[0][ct_n]
        print(f'Celltype {ct} presynaptic neuron: {presyn}')
        if pairs_dict or pairs is None:
            raise ValueError("Please provide a pairs_dict and pairs to map neuron IDs to pairs.")
        pair_no = pairs_dict[presyn]
        presyn_neuL = pairs['leftid'].loc[pair_no]
        if conRight is not None:
            presyn_neuR = pairs['rightid'].loc[pair_no]
        conL_f = conLeft[conLeft['presynaptic_to'] == presyn_neuL]
        if conRight is not None:
            conR_f = conRight[conRight['presynaptic_to'] == presyn_neuR]

    return conL_f, conR_f

# similar but different simpler filtering function
def filter_col_presyn(con, ct):
    try:
        con = con[con['presynaptic_celltype'] == ct]
    except KeyError:
        print(f"Column 'presynaptic_celltype' not found in the connector dataframe.")
        return con
    if con.empty:
        print(f"No data for presynaptic celltype: {ct}")
        return con
    return con

# remove fragments 
def cut_out_frags(con, all_neurons):
    con = con.iloc[:,:5]
    con = con[con['presynaptic_to'].isin(all_neurons)]
    for c in con.index:
        con.at[c, 'postsynaptic_to'] = [p for p in con.at[c, 'postsynaptic_to'] if p in all_neurons]
    return con

# try on only connectors with top downstream partners 
def get_top_targets(conb, syn_threshold=3):
    '''
    Get top targets based on a threshold of synaptic connections.
    Filter connectors to only contain those that contain strong connections.
    '''
    conb_filtered = conb.loc[:, conb.sum(axis=0) >= syn_threshold]
    top_targets = conb_filtered.columns
    top_counts = conb_filtered.sum(axis=0)
    top_targets_df = pd.DataFrame({'target': top_targets, 'count': top_counts})
    top_targets_df = top_targets_df.sort_values(by='count', ascending=False)
    # filter out connectors that do not have any top targets 
    conb_filtered = conb_filtered.loc[conb_filtered.sum(axis=1) > 0]
    return conb_filtered, top_targets_df

def con_bin_cos_sim(con_bin1, con_bin2, n_match):
    '''
    Calculate cosine similarity between two connector binary matrices.
    '''
    cos_sim_arr = np.zeros((con_bin1.shape[0], con_bin2.shape[0]))
    for r in range(con_bin1.shape[0]):
        a = con_bin1.iloc[r, :n_match].values.reshape(1, -1)
        for r2 in range(con_bin2.shape[0]):
            b = con_bin2.iloc[r2, :n_match].values.reshape(1, -1)
            cos_sim = cosine_similarity(a, b)[0][0]
            cos_sim_arr[r, r2] = cos_sim
    return cos_sim_arr


### Functions for finding simple cell type motifs ###

def get_motifs(conb, type_dict=None, pairing=None):
    if type_dict is None:
        raise ValueError("Please provide a type_dict to map neuron IDs to types.")
    if pairing is None:
        raise ValueError("Please provide a pairing dictionary to map neuron IDs to their partners.")

    target_df = pd.DataFrame(columns=['connector', 'target', 'type', 'n_targets', 'target_pairs'])
    for connector in conb.index:
        # Get the postsynaptic targets for the connector
        neuron_idx = np.where(conb.loc[connector] > 0)[0]
        targets = conb.columns[neuron_idx].tolist()
        ct_targets = [type_dict.get(t, None) for t in targets]
        ct_targets = [t for t in ct_targets if t is not None]
        u_ct_targets = list(np.unique(ct_targets))
        target_pairs = [pairing.get(t, None) for t in targets]
        # Create a DataFrame for the targets
        if (len(targets) > 1) & (len(u_ct_targets) > 0): # don't include connectors with only one known target
            target_df = pd.concat([target_df, pd.DataFrame({
                'connector': connector,
                'target': [targets],
                'type': [tuple(u_ct_targets)],
                'n_targets': len(targets),
                'target_pairs': [target_pairs]
            })], ignore_index=True)
    # get unique target motifs 
    target_counts = Counter(target_df['type'])
    return target_df, target_counts
    
def get_top_percentage_motifs(target_counts, top_percentage=0.01):
    total_count = sum(target_counts.values())
    top_count = int(total_count * top_percentage)
    top_motifs = {k: v for k, v in target_counts.items() if v >= top_count}
    #order motifs by count
    top_motifs = dict(sorted(top_motifs.items(), key=lambda item: item[1], reverse=True))
    return top_motifs

### Function for getting feedforward and feedback motifs ###

def get_flow_motifs(conb, con, flow_dict=None, pairing=None, flow_norm=True):
    """ 
    If flow norm true, normalise postsynaptic target flow score by the presynaptic neuron flow score."""
    if flow_dict is None:
        raise ValueError("Please provide a flow_dict to map neuron IDs to flow scores.")
    if pairing is None:
        raise ValueError("Please provide a pairing dictionary to map neuron IDs to their partners.")

    target_df = pd.DataFrame(columns=['connector', 'target', 'flow', 'n_targets', 'target_pairs'])
    for connector in conb.index:
        # Get the postsynaptic targets for the connector
        neuron_idx = np.where(conb.loc[connector] > 0)[0]
        targets = conb.columns[neuron_idx].tolist()
        # get the flow score for presynaptic neuron and targets 
        presyn_neuron = con[con['connector_id'] == connector]['presynaptic_to'].values[0]
        presyn_flow = flow_dict.get(presyn_neuron, None)

        flow_targets = [flow_dict.get(t, None) for t in targets]
        flow_targets = [t for t in flow_targets if t is not None]
        if flow_norm:
            #check if presyn_flow is not None to avoid division by zero
            if presyn_flow is None:
                print(f'skipping connector {connector} due to missing presynaptic flow score')
                continue
            else:
                flow_targets = [f - presyn_flow for f in flow_targets]
        
        u_flow_targets = list(np.unique(flow_targets))
        target_pairs = [pairing.get(t, None) for t in targets]
        # Create a DataFrame for the targets
        if (len(targets) > 1) & (len(u_flow_targets) > 0): # don't include connectors with only one known target
            target_df = pd.concat([target_df, pd.DataFrame({
                'connector': connector,
                'target': [targets],
                'flow': [tuple(u_flow_targets)],
                'n_targets': len(targets),
                'target_pairs': [target_pairs]
            })], ignore_index=True)
    # get unique target motifs 
    target_counts = Counter(target_df['flow'])
    return target_df, target_counts

def get_simple_flow_motifs(conb, con, flow_dict=None, pairing=None, flow_norm=True):
    """ 
    If flow norm true, normalise postsynaptic target flow score by the presynaptic neuron flow score."""
    if flow_dict is None:
        raise ValueError("Please provide a flow_dict to map neuron IDs to flow scores.")
    if pairing is None:
        raise ValueError("Please provide a pairing dictionary to map neuron IDs to their partners.")

    target_df = pd.DataFrame(columns=['connector', 'target', 'flow', 'n_targets', 'target_pairs'])
    for connector in conb.index:
        # Get the postsynaptic targets for the connector
        neuron_idx = np.where(conb.loc[connector] > 0)[0]
        targets = conb.columns[neuron_idx].tolist()
        # get the flow score for presynaptic neuron and targets 
        presyn_neuron = con[con['connector_id'] == connector]['presynaptic_to'].values[0]
        presyn_flow = flow_dict.get(presyn_neuron, None)

        flow_targets = [flow_dict.get(t, None) for t in targets]
        flow_targets = [t for t in flow_targets if t is not None]
        if flow_norm:
            #check if presyn_flow is not None to avoid division by zero
            if presyn_flow is None:
                print(f'skipping connector {connector} due to missing presynaptic flow score')
                continue
            else:
                flow_targets = [f - presyn_flow for f in flow_targets]

        # convert flow_targets to FF and FB 
        flow_targets = ['FB' if f > 0 else 'FF' for f in flow_targets]
        
        u_flow_targets = list(np.unique(flow_targets))
        target_pairs = [pairing.get(t, None) for t in targets]
        # Create a DataFrame for the targets
        if (len(targets) > 1) & (len(u_flow_targets) > 0): # don't include connectors with only one known target
            target_df = pd.concat([target_df, pd.DataFrame({
                'connector': connector,
                'target': [targets],
                'flow': [tuple(u_flow_targets)],
                'n_targets': len(targets),
                'target_pairs': [target_pairs]
            })], ignore_index=True)
    # get unique target motifs 
    target_counts = Counter(target_df['flow'])
    return target_df, target_counts

### Still flow but simplified for multilayer class motifs ###

def map_con_targets_to_flow(con, flow_dict=None):
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

def remove_incomplete_flow_motifs(flow_motifs):
    """ Remove motifs that are not complete, i.e. do not have both FF and FB """
    complete_motifs = {k: v for k, v in flow_motifs.items() if 'NA' not in k}
    return complete_motifs

def map_con_targets_to_real_neurons(con, all_neurons=None):
    reality_check_series = pd.Series(index=con.index, dtype=object)
    for c in con.index:
        postsynaptic_to = con.loc[c, 'postsynaptic_to']
        reality_check = ['REAL' if skid in all_neurons else 'FRAG' for skid in postsynaptic_to]
        reality_check_series[c] = reality_check
    con['flow_scores'] = reality_check_series # again need to change this in future but leaving to keep it working with other functions for now
    return con

### Get motifs of top targets vs low targets ###

def get_partner_motifs(con, syn_threshold=3, use_global=True, pn_target_dict=None):

    # get top targets based on synapse count threshold for each presynaptic neuron
    presyn_neurons = con['presynaptic_to'].unique()
    if use_global:
        if pn_target_dict is None:
            raise ValueError("Please provide a pn_target_dict to use global top targets.")
    else:
        pn_target_dict = {}
        for pn in presyn_neurons:
            con_p = con[con['presynaptic_to'] == pn]
            # binarise to apply the top target function - should probably just rewrite this
            conb_p = con_binary_matrix(con_p, only_known_targets=False, all_neurons=all_neurons)
            if conb_p.empty:
                continue
            # get top targets for this presynaptic neuron
            _, top_targets_df = get_top_targets(conb_p, syn_threshold=syn_threshold)
            pn_target_dict[pn] = list(top_targets_df['target'])

    # get motif counts with top targets vs random targets
    motif_series = pd.Series(index=con.index, dtype=object)
    for c in con.index:
        # get top targets expected for this presynaptic neuron
        presyn_neuron = con.loc[c, 'presynaptic_to']
        if presyn_neuron not in pn_target_dict:
            raise ValueError(f"Presynaptic neuron {presyn_neuron} not found in top target dictionary.")
        pn_targets = pn_target_dict[presyn_neuron]
        # get postsynaptic partners for this connector
        postsynaptic_to = con.loc[c, 'postsynaptic_to']
        # cmap targets to top targets or low targets 
        target_motif = ['TOP' if skid in pn_targets else 'LOW' for skid in postsynaptic_to]
        motif_series[c] = target_motif
    con['flow_scores'] = motif_series # just not renaming cause I hard coded the other function - will fix later
    return con

### Get and plot top motifs ### 
def generate_target_heatmap(watch_targets, watch_pair_ids, motifs_to_watch):
    all_targets = np.unique(list(chain.from_iterable(watch_targets)))
    all_pair_ids = np.unique(list(chain.from_iterable(watch_pair_ids)))

    target_matrix = np.zeros((len(watch_targets), len(all_targets)))
    pair_matrix = np.zeros((len(watch_targets), len(all_pair_ids)))

    for i, (targs, pars) in enumerate(zip(watch_targets, watch_pair_ids)):
        for t in targs:
            target_matrix[i, np.where(all_targets == t)[0][0]] += 1
        for p in pars:
            pair_matrix[i, np.where(all_pair_ids == p)[0][0]] += 1
    target_matrix = pd.DataFrame(target_matrix, columns=all_targets, index=motifs_to_watch)
    pair_matrix = pd.DataFrame(pair_matrix, columns=all_pair_ids, index=motifs_to_watch)
    cols_sorted = target_matrix.columns.sort_values()
    target_matrix = target_matrix[cols_sorted]
    cols_sorted = pair_matrix.columns.sort_values()
    pair_matrix = pair_matrix[cols_sorted]

    return target_matrix, pair_matrix

def normalise_rows(df):
    return df.div(df.sum(axis=1), axis=0)

def get_and_plot_motif_targets(one_to_watch, top_motifs, target_df, celltype_df=None):
    # this is obv not very systematic, just trying to explore 
    # this was to explore synapses from LNs

    motifs_to_watch = [k for k, v in top_motifs.items() if one_to_watch in k]
    motif_len = [len(m) for m in motifs_to_watch]
    motif_len_sort = np.argsort(motif_len)
    motifs_to_watch = [motifs_to_watch[i] for i in motif_len_sort]

    watch_targets = []
    watch_pair_ids = []
    for motif in motifs_to_watch:
        w_ids = celltype_df[celltype_df['name'] == one_to_watch]['skids'].explode().tolist()
        w_targs = target_df[target_df['type'] == motif]['target'].explode().tolist()
        w_pair_ids = target_df[target_df['type'] == motif]['target_pairs'].explode().tolist()

        w_locs = np.where(np.isin(w_targs, w_ids))[0]
        watch_targets.append(np.array(w_targs)[w_locs].tolist())
        watch_pair_ids.append(np.array(w_pair_ids)[w_locs].tolist())

    target_mat, pair_mat = generate_target_heatmap(watch_targets, watch_pair_ids, motifs_to_watch)
    target_mat_norm = normalise_rows(target_mat)
    pair_mat_norm = normalise_rows(pair_mat)

    fig, axes = plt.subplots(1,2, figsize=(11, 4))
    sns.heatmap(target_mat_norm, cmap='viridis', ax=axes[0])
    sns.heatmap(pair_mat_norm, cmap='viridis', ax=axes[1])
    fig.tight_layout()

    return motifs_to_watch, target_mat, pair_mat, watch_targets, watch_pair_ids

def normalise_freq_cols(df):
    """ Normalise frequency columns in a DataFrame by the sum of the row. """
    freq_cols = ['ff', 'fb', 'ff_fb']
    for col in freq_cols:
        df[f'{col}_norm'] = df[col] / (df[freq_cols[0]] + df[freq_cols[1]] + df[freq_cols[2]] + 1e-10)  # Adding a small value to avoid division by zero
    return df

### from hyperlayer ct flow 
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

def get_unique_ct_target_counts(con, remove_nan=False, top_percentage=0.01, ct_names=None):
    if ct_names is None:
        raise ValueError("Please provide a list of cell type names to sort the targets.")
    ps_targets = con['postsynaptic_celltype'].tolist()
    ps_targets = [['NA' if k is None else k for k in sublist] for sublist in ps_targets]
    ps_targets_sorted = [tuple(sort_list_by_input(y, ct_names)) for y in ps_targets] 
    unique_ct_targets = np.unique(ps_targets_sorted).tolist()
    ct_target_counts = Counter(ps_targets_sorted)
    target_df = pd.DataFrame(ct_target_counts.items(), columns=['motif','count'])
    if target_df.empty:
        print("No motifs found for the given connector data.")
        return target_df
    if remove_nan:
        target_df = remove_na(target_df)
    if top_percentage is not None:
        target_df = get_top_motifs_only(target_df, top_percentage=top_percentage)
    target_df = target_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    return target_df