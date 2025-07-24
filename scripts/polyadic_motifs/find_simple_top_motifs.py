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

from scripts.polyadic_motifs.motif_functions import con_binary_matrix, get_top_targets, con_bin_cos_sim, get_simple_flow_motifs, get_motifs, get_top_percentage_motifs, get_and_plot_motif_targets
from scripts.initialisation_scripts.get_me_started import get_me_started

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df, flow_dict = get_me_started()

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


one_to_watch = 'PNs'
motifs_to_watchL, target_matL, pair_matL, watch_targetsL, watch_pair_idsL = get_and_plot_motif_targets(one_to_watch, L_top_motifs, L_target_df)
motifs_to_watchR, target_matR, pair_matR, watch_targetsR, watch_pair_idsR = get_and_plot_motif_targets(one_to_watch, R_top_motifs, R_target_df)





# %%
