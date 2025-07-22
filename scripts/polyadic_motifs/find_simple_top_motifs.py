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

from scripts.polyadic_motifs.motif_functions import con_binary_matrix, get_top_targets, con_bin_cos_sim
from scripts.initialisation_scripts.get_me_started import get_me_started

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df = get_me_started()

all_neurons = connector_details['presynaptic_to'].unique().tolist()


#%% 
# filter by hemisphere 
conL = connector_details[connector_details['presyn_hemi'] == 'left']
conR = connector_details[connector_details['presyn_hemi'] == 'right']

#filter by celltype or neuron_id 
ct = 'LNs'
conL_f = conL[conL['presynaptic_celltype'] == ct]
conR_f = conR[conR['presynaptic_celltype'] == ct]

#try this with mbins
# filter by specific neuron 
presyn = celltype_df[celltype_df['name'] == ct]['skids'].values[0][5]
print(presyn)
pair_no = pairs_dict[presyn] # get the bilateral id of the neuron

presyn_neuL = pairs['leftid'].loc[pair_no]
presyn_neuR = pairs['rightid'].loc[pair_no]

conL_f = conL_f[conL_f['presynaptic_to'] == presyn_neuL]
conR_f = conR_f[conR_f['presynaptic_to'] == presyn_neuR]

#%%
conbL_f = con_binary_matrix(conL_f, only_known_targets=True, all_neurons=all_neurons)
conbR_f = con_binary_matrix(conR_f, only_known_targets=True, all_neurons=all_neurons) 

# %%

conbL_f, top_targets_df_L = get_top_targets(conbL_f, syn_threshold=3)
conbR_f, top_targets_df_R = get_top_targets(conbR_f, syn_threshold=3)


# %%
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

#%% 
top_percentage = 0.01
L_target_df, L_target_counts= get_motifs(conbL_f, type_dict=skid_to_celltype, pairing=pairs_dict)
L_top_motifs = get_top_percentage_motifs(L_target_counts, top_percentage=0.01)
R_target_df, R_target_counts = get_motifs(conbR_f, type_dict=skid_to_celltype, pairing=pairs_dict)
R_top_motifs = get_top_percentage_motifs(R_target_counts, top_percentage=0.01)

print("Top motifs in left hemisphere:")
for motif, count in L_top_motifs.items():
    print(f"{motif}: {count}")
print("\nTop motifs in right hemisphere:")
for motif, count in R_top_motifs.items():
    print(f"{motif}: {count}")
#conbR_fm = get_motifs(conbR_f, type_dict=skid_to_celltype)
# %% examine if there is differences in the neurons involed in specific motifs 
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

def get_and_plot_motif_targets(one_to_watch, top_motifs, target_df):
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


one_to_watch = 'PNs'
motifs_to_watchL, target_matL, pair_matL, watch_targetsL, watch_pair_idsL = get_and_plot_motif_targets(one_to_watch, L_top_motifs, L_target_df)
motifs_to_watchR, target_matR, pair_matR, watch_targetsR, watch_pair_idsR = get_and_plot_motif_targets(one_to_watch, R_top_motifs, R_target_df)





# %%
