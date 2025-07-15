#%%

from collections import Counter
from itertools import chain, combinations
import time

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib as mpl

from contools import Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
from scripts.functions.group_based_stats import binarize_poly_adj, get_random_poly_adj, compare_two_sample_chi_squared, correct_pvals, plot_pvals_heatmap, plot_fold_change_heatmap, plot_significant_fold_change_heatmap, get_group_stats_from_bi_adj
from scripts.functions.little_helper import get_celltype_dict, celltype_col_for_list, get_celltype_name, celltype_col_for_nestedlist
from scripts.functions.undirected_graph_functions import get_postsynaptic_co_adj, map_co_adj_to_dict

rm = pymaid.CatmaidInstance(url, token, name, password)

#generate numpy random instance
seed = 40
rng = np.random.default_rng(seed=seed)

#%% get bilateral pairs 

pairs = pd.read_csv('data/pairs-2022-02-14.csv')
right_ns = pairs['rightid'].unique()
left_ns = pairs['leftid'].unique()

#%% get cell type and synaptic data 
# get celltype data for each skid 
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()
# get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)
ct_names = celltype_df['name'].unique()

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

#  map skid ids in connector details to celltypes
connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(lambda x: get_celltype_name(x, skid_to_celltype=skid_to_celltype))
celltype_col_for_list(connector_details, 'postsynaptic_to', skid_to_celltype=skid_to_celltype, new_col_name='postsynaptic_celltype')

# assign hemisphere of presynaptic neuron 

connector_details['presyn_hemi'] = connector_details['presynaptic_to'].apply(lambda x: 'right' if x in right_ns else 'left' if x in left_ns else 'NA')


#%% assess bilateral symmetry of synaptic sites

#get number of neurons in each hemisphere
print(connector_details['presyn_hemi'].value_counts())

connector_details['n_postsyn_partners'] = connector_details['postsynaptic_to'].apply(lambda x: len(x) if isinstance(x, list) else 0)

left_origin = connector_details[connector_details['presyn_hemi'] == 'left']
right_origin = connector_details[connector_details['presyn_hemi'] == 'right']

#get average number of synapses per neuron in each hemisphere
left_avg_synapses = left_origin.groupby('presynaptic_to').size().mean()
right_avg_synapses = right_origin.groupby('presynaptic_to').size().mean()
print(f"Average number of synapses per neuron in left hemisphere: {left_avg_synapses}")
print(f"Average number of synapses per neuron in right hemisphere: {right_avg_synapses}")

# get average number of postsynaptic partners per presynaptic neuron in each hemisphere
left_avg_partners = left_origin.groupby('presynaptic_to')['n_postsyn_partners'].mean().mean()
right_avg_partners = right_origin.groupby('presynaptic_to')['n_postsyn_partners'].mean().mean()
print(f"Average number of postsynaptic partners per presynaptic neuron in left hemisphere: {left_avg_partners}")
print(f"Average number of postsynaptic partners per presynaptic neuron in right hemisphere: {right_avg_partners}")

# get difference in number of postsynaptic partners per synapse between bilateral pairs of neurons 

pairs['left_n_partners'] = pairs['leftid'].apply(lambda x: left_origin[left_origin['presynaptic_to'] == x]['n_postsyn_partners'].mean() if x in left_origin['presynaptic_to'].values else np.nan)
pairs['right_n_partners'] = pairs['rightid'].apply(lambda x: right_origin[right_origin['presynaptic_to'] == x]['n_postsyn_partners'].mean() if x in right_origin['presynaptic_to'].values else np.nan)
pairs['n_partner_diff'] = pairs.apply(
    lambda row: np.nan if row['left_n_partners'] is np.nan or row['right_n_partners'] is np.nan
    else row['left_n_partners'] - row['right_n_partners'],
    axis=1
)
print(f'Average difference in number of postsynaptic partners between pairs: {pairs["n_partner_diff"].dropna().mean()}')

#%%
# try random reshuffle and comparison
def get_random_pairs(pairs, rng):
    pairs_permutated = pairs.iloc[:, :2].copy()
    pairs_permutated['rightid'] = rng.permutation(pairs_permutated['rightid'].values)
    return pairs_permutated

def get_avg_n_partners_bw_pairs(pairs_permutated, left_origin, right_origin):
    pairs_permutated['left_n_partners'] = pairs_permutated['leftid'].apply(lambda x: left_origin[left_origin['presynaptic_to'] == x]['n_postsyn_partners'].mean() if x in left_origin['presynaptic_to'].values else np.nan)
    pairs_permutated['right_n_partners'] = pairs_permutated['rightid'].apply(lambda x: right_origin[right_origin['presynaptic_to'] == x]['n_postsyn_partners'].mean() if x in right_origin['presynaptic_to'].values else np.nan)
    pairs_permutated['n_partner_diff'] = pairs_permutated.apply(
        lambda row: np.nan if row['left_n_partners'] is np.nan or row['right_n_partners'] is np.nan
        else row['left_n_partners'] - row['right_n_partners'],
        axis=1
    )   
    print(f'Average difference in number of postsynaptic partners between permutated pairs: {pairs_permutated["n_partner_diff"].dropna().mean()}')
    return pairs_permutated['n_partner_diff'].dropna().mean()

def run_partner_permutation(pairs, rng, left_origin, right_origin, n_permutations=20):
    random_diffs = []
    for i in range(n_permutations):
        pairs_permutated = get_random_pairs(pairs, rng)
        avg_diff = get_avg_n_partners_bw_pairs(pairs_permutated, left_origin, right_origin)
        random_diffs.append(avg_diff)
    return random_diffs

random_diffs_20 = run_partner_permutation(pairs, rng, left_origin, right_origin, n_permutations=20)

#%% examine specific postsynaptic partners of bilateral pairs

def get_known_ps_partners(connector_df):
    #u_presyn = connector_df['presynaptic_to'].unique()
    #group by 'presynaptic_to' and aggregate multiple into lists
    ps = connector_df.groupby('presynaptic_to')['presynaptic_to'].first().to_list()
    ps_ct = connector_df.groupby('presynaptic_to')['presynaptic_celltype'].first().to_list()
    ps_postsyn = connector_df.groupby('presynaptic_to')['postsynaptic_to'].apply(list).to_list()
    ps_postsyn_ct = connector_df.groupby('presynaptic_to')['postsynaptic_celltype'].apply(list).to_list()

    grouped_ps = pd.DataFrame()
    grouped_ps['presynaptic_to'] = ps
    grouped_ps['presynaptic_celltype'] = ps_ct
    grouped_ps['postsynaptic_to'] = ps_postsyn
    grouped_ps['postsynaptic_celltype'] = ps_postsyn_ct



    #grouped_ps = connector_df.groupby('presynaptic_to')['postsynaptic_to'].apply(list).to_frame()
    #grouped_ps['presynaptic_to'] = grouped_ps.index
    grouped_ps = grouped_ps.reset_index(drop=True)
    grouped_ps['postsynaptic_flat'] = grouped_ps['postsynaptic_to'].apply(lambda x: list(chain.from_iterable(x)))
    grouped_ps['postsynaptic_unique'] = grouped_ps['postsynaptic_flat'].apply(lambda x: list(set(x)))
    
    return grouped_ps

grouped_ps_left = get_known_ps_partners(left_origin)
grouped_ps_right = get_known_ps_partners(right_origin)

#%% direct comparison

pairs_dict = {}
for index, row in pairs.iterrows():
    # check if duplicate 
    if row['leftid'] in pairs_dict.keys() or row['rightid'] in pairs_dict.keys():
        print(f"Duplicate pair found: {row['leftid']} - {row['rightid']}")
        if row['leftid'] in pairs_dict.keys():
            existing_idx = pairs_dict[row['leftid']]
            pairs_dict[row['rightid']] = existing_idx
            continue
        elif row['rightid'] in pairs_dict.keys():
            existing_idx = pairs_dict[row['rightid']]
            pairs_dict[row['leftid']] = existing_idx
            continue
        else:
            print('something is wrong with the pairs_dict')

    else:
        pairs_dict[row['leftid']] = index
        pairs_dict[row['rightid']] = index

grouped_ps_left['presyn_pair'] = grouped_ps_left['presynaptic_to'].apply(lambda x: pairs_dict[x] if x in pairs_dict else np.nan)
grouped_ps_right['presyn_pair'] = grouped_ps_right['presynaptic_to'].apply(lambda x: pairs_dict[x] if x in pairs_dict else np.nan)

grouped_ps_left['postsyn_pair'] = grouped_ps_left['postsynaptic_unique'].apply(lambda x: [pairs_dict[ps] for ps in x if ps in pairs_dict])
grouped_ps_right['postsyn_pair'] = grouped_ps_right['postsynaptic_unique'].apply(lambda x: [pairs_dict[ps] for ps in x if ps in pairs_dict])

#%% compare 
remove_fragments = True # whether to remove fragments from the postsynaptic partners
min_occ = 6
# get unique postsynaptic partners in common for each pair 
overlaps = []
diffs = []
overlaps_pairs = []
diffs_pairs = []

for i in range(pairs.shape[0]):
    left_pre = pairs['leftid'][i]
    right_pre = pairs['rightid'][i]
    if left_pre not in grouped_ps_left['presynaptic_to'].values or right_pre not in grouped_ps_right['presynaptic_to'].values:
        overlaps.append([])
        diffs.append([])
        overlaps_pairs.append([])
        diffs_pairs.append([])
        continue
    left_ps = grouped_ps_left[grouped_ps_left['presynaptic_to'] == left_pre]['postsynaptic_unique'].to_list()[0]
    right_ps = grouped_ps_right[grouped_ps_right['presynaptic_to'] == right_pre]['postsynaptic_unique'].to_list()[0]
    if remove_fragments: 
        left_ps = [ps for ps in left_ps if ps in all_neurons]
        right_ps = [ps for ps in right_ps if ps in all_neurons]
    
    # only include postsynaptic partners that occur at least min_occ times in the whole dataset
    left_ps = [ps for ps in left_ps if len(grouped_ps_left[grouped_ps_left['postsynaptic_flat'].apply(lambda x: ps in x)]) >= min_occ]
    right_ps = [ps for ps in right_ps if len(grouped_ps_right[grouped_ps_right['postsynaptic_flat'].apply(lambda x: ps in x)]) >= min_occ]

    overlaps.append(list(set(left_ps) & set(right_ps)))
    diffs.append(list(set(left_ps) ^ set(right_ps)))

    left_p = grouped_ps_left[grouped_ps_left['presynaptic_to'] == left_pre]['postsyn_pair'].to_list()[0]
    right_p = grouped_ps_right[grouped_ps_right['presynaptic_to'] == right_pre]['postsyn_pair'].to_list()[0]

    overlaps_pairs.append(list(set(left_p) & set(right_p)))
    diffs_pairs.append(list(set(left_p) ^ set(right_p)))

pairs['overlap'] = overlaps
pairs['diff'] = diffs
pairs['overlap_pairs'] = overlaps_pairs
pairs['diff_pairs'] = diffs_pairs

pairs['n_overlap'] = pairs['overlap'].apply(lambda x: len(x) if isinstance(x, list) else 0)
pairs['n_diff'] = pairs['diff'].apply(lambda x: len(x) if isinstance(x, list) else 0)
pairs['n_overlap_pairs'] = pairs['overlap_pairs'].apply(lambda x: len(x) if isinstance(x, list) else 0)
pairs['n_diff_pairs'] = pairs['diff_pairs'].apply(lambda x: len(x) if isinstance(x, list) else 0) 

pairs['fraction_overlap'] = pairs['n_overlap'] / (pairs['n_overlap'] + pairs['n_diff'])
pairs['fraction_overlap_pairs'] = pairs['n_overlap_pairs'] / (pairs['n_overlap_pairs'] + pairs['n_diff_pairs'])

print(f"Exclude fragments: {remove_fragments}, min occurrence of partners: {min_occ}")
print(f"Average fraction of postsynaptic partners in common: {pairs['fraction_overlap'].mean()}")
print(f"Average fraction of postsynaptic partners in common (pairs): {pairs['fraction_overlap_pairs'].mean()}")


#%% cos sim across pairs 


def evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_dict, unique_cells, rng, presyn_restrict=None, randomize=False, verbose=False):
    #restrict the pairs to compare if desired - otherwise will compare across all pairs 
    start_time = time.time()
    if presyn_restrict is not None:
        grouped_ps_left = grouped_ps_left[grouped_ps_left['presynaptic_to'].isin(presyn_restrict)].reset_index(drop=True)
        grouped_ps_right = grouped_ps_right[grouped_ps_right['presynaptic_to'].isin(presyn_restrict)].reset_index(drop=True)  
        #also filter pairs so that all possible neurons are included, and not so many nan's for all other pairs
        pairs = pairs[pairs['leftid'].isin(presyn_restrict)]
        pairs = pairs.reset_index(drop=True)

    cos_sim_flat = []
    cos_sim_hyper_flat = []
    for c in range(len(pairs)):
        L1 = grouped_ps_left[grouped_ps_left['presynaptic_to']== pairs['leftid'][c]]
        if randomize:
            choice_right_neurons = grouped_ps_right['presynaptic_to'].unique()
            c2 = rng.choice(choice_right_neurons)
            #c2 = rng.integers(0, len(pairs))
            R1 = grouped_ps_right[grouped_ps_right['presynaptic_to']== c2] #pairs['rightid'][c2]]
        else:
            R1 = grouped_ps_right[grouped_ps_right['presynaptic_to']== pairs['rightid'][c]]
        if L1.empty or R1.empty:
            if verbose:
                print(f"Skipping pair {c} as one of the neurons does not have any postsynaptic partners.")
            cos_sim_flat.append(np.nan)
            continue

        # TODO make sure this is also compatible with pairs, not just celltypes. added this line to not rely on celltype
        # check that both neurons project to some known postsynaptic partners
        # map dict to L1 and R1 

        celltype_col_for_nestedlist(L1, 'postsynaptic_to', skid_to_celltype=skid_dict, new_col_name='mapped')
        celltype_col_for_nestedlist(R1, 'postsynaptic_to', skid_to_celltype=skid_dict, new_col_name='mapped')

        allL = list(chain.from_iterable(L1['mapped'].to_list()[0])) #note worked with L1['postsynaptic_celltype'] inherited from grouped_ps_left
        allR = list(chain.from_iterable(R1['mapped'].to_list()[0]))
        #print(allL)
        #print(skid_to_celltype.values())
        L_ = any(v in allL for v in skid_dict.values())
        R_ = any(v in allR for v in skid_dict.values())
        if not (L_ and R_):
            if verbose:
                print(f"Skipping pair {c} as one of the neurons does not have any known postsynaptic partners.")
            cos_sim_flat.append(np.nan)
            cos_sim_hyper_flat.append(np.nan)
            continue
        

        L1_adj, L1_adj_neurons = get_postsynaptic_co_adj(L1['postsynaptic_to'].to_list()[0])
        R1_adj, R1_adj_neurons = get_postsynaptic_co_adj(R1['postsynaptic_to'].to_list()[0])

        L1_adj, L1_ct_in_adj = map_co_adj_to_dict(L1_adj, L1_adj_neurons, skid_dict, filter_adj=True)
        R1_adj, R1_ct_in_adj = map_co_adj_to_dict(R1_adj, R1_adj_neurons, skid_dict, filter_adj=True)

        L1_ct_in_adj = [int(v) for v in L1_ct_in_adj]
        R1_ct_in_adj = [int(v) for v in R1_ct_in_adj]

        count_df = get_counts(L1_ct_in_adj, R1_ct_in_adj)

        cosine_sim = get_cosine_similarity(count_df)
        

        cos_sim_flat.append(cosine_sim[0][0])

        # to the same but with hyperedge co-occurrence 
        L1_agg, L1_unique_cts = get_agg_ct_polyadj(L1_adj, L1_ct_in_adj, unique_cts=unique_cells)
        R1_agg, R1_unique_cts = get_agg_ct_polyadj(R1_adj, R1_ct_in_adj, unique_cts=unique_cells)

        L1_agg_flat = L1_agg.values.flatten()
        R1_agg_flat = R1_agg.values.flatten()
        hyper_cosine_sim = cosine_similarity(L1_agg_flat.reshape(1, -1), R1_agg_flat.reshape(1, -1))
        #print(f'Pair {c}: hyper_cos_sim {hyper_cosine_sim}, just val {hyper_cosine_sim[0][0]}')

        cos_sim_hyper_flat.append(hyper_cosine_sim[0][0])

    print(f"Average cosine similarity of postsynaptic cell types between bilateral pairs: {np.nanmean(cos_sim_flat)}")
    print(f"Average hyperedge cosine similarity of postsynaptic cell types between bilateral pairs: {np.nanmean(cos_sim_hyper_flat)}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return cos_sim_flat, cos_sim_hyper_flat

#%%
# all cells - divide postsynaptic partners by cell type 
#cos_sim_flat, cos_sim_hyper_flat = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cells=ct_names, rng=rng, presyn_restrict=None, randomize=False)
#cos_sim_flat_random, cos_sim_hyper_flat_random = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cells=ct_names, rng=rng, presyn_restrict=None, randomize=True)

# all cells - divide postsynaptic partners by specific neuron or bilateral partner 
# maybe need to find some way to avoid KCs not being paired, not affecting it? so should not weigh non-paired cells 

# use pairs_dict to map to pairs id rather than cell type 
unique_pairs = np.unique([k for k in pairs_dict.values()])
cos_sim_flat_pairs, cos_sim_hyper_flat_pairs = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, pairs_dict, unique_pairs, rng, presyn_restrict=None, randomize=False)
cos_sim_flat_pairs_rand, cos_sim_hyper_flat_pairs_rand = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, pairs_dict, unique_pairs, rng, presyn_restrict=None, randomize=True)

#%% for each presynaptic cell type 
cos_sim_presyn = []
cos_sim_hyper_presyn = []
cos_sim_presyn_rand = []
cos_sim_hyper_presyn_rand = []
for presyn_ct in ct_names:
    presyn_restrict = celltype_df[celltype_df['name'] == presyn_ct]['skids'].values[0]

    cos_sim_flat_presyn, cos_sim_hyper_flat_presyn = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cells=ct_names, rng=rng, presyn_restrict=presyn_restrict, randomize=False)
    cos_sim_flat_presyn_rand, cos_sim_hyper_flat_presyn_rand = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cells=ct_names, rng=rng, presyn_restrict=presyn_restrict, randomize=True)
    cos_sim_presyn.append(cos_sim_flat_presyn)
    cos_sim_hyper_presyn.append(cos_sim_hyper_flat_presyn)
    cos_sim_presyn_rand.append(cos_sim_flat_presyn_rand)
    cos_sim_hyper_presyn_rand.append(cos_sim_hyper_flat_presyn_rand)

#%% again for specific neuron id, not just cell type


cos_sim_presyn_neuronid = []
cos_sim_presyn_hyper_neuronid = []
cos_sim_presyn_neuronid_rand = []
cos_sim_presyn_hyper_neuronid_rand = []

for presyn_ct in ct_names:
    presyn_restrict = celltype_df[celltype_df['name'] == presyn_ct]['skids'].values[0]

    cos_sim_flat_presyn_neuronid, cos_sim_hyper_flat_presyn_neuronid = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, pairs_dict, unique_pairs, rng=rng, presyn_restrict=presyn_restrict, randomize=False)
    cos_sim_flat_presyn_neuronid_rand, cos_sim_hyper_flat_presyn_neuronid_rand = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, pairs_dict, unique_pairs, rng=rng, presyn_restrict=presyn_restrict, randomize=True)
    cos_sim_presyn_neuronid.append(cos_sim_flat_presyn_neuronid)
    cos_sim_presyn_hyper_neuronid.append(cos_sim_hyper_flat_presyn_neuronid)
    cos_sim_presyn_neuronid_rand.append(cos_sim_flat_presyn_neuronid_rand)
    cos_sim_presyn_hyper_neuronid_rand.append(cos_sim_hyper_flat_presyn_neuronid_rand)




#%% plot cosine similarity
mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,

})

def sem(data):
    return np.nanstd(data) / np.sqrt(len(data))

def plot_summary_cos_sim(data, data_groups, data_names, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))

    # get number of groups 
    num_groups = len(np.unique(data_groups))

    if num_groups == 2:
        x_width = 0.4
        xw = x_width + 0.01
        x_locs = [1 - xw/2, 1 + xw/2, 2 - xw/2, 2 + xw/2]
        colors = ['#1e20e1', '#6667ea', 'orangered', 'coral']
    elif num_groups == 3:
        x_width = 0.3
        xw = x_width + 0.01
        x_locs = [1 - xw, 1, 1 + xw, 2.3 - xw, 2.3, 2.3 + xw]
        colors = ['#1e20e1', '#6667ea', '#9999f1', 'orangered', 'coral', 'darkorange']

    means = [np.nanmean(v) for v in data]
    clean_data = [[v for v in sublist if not np.isnan(v)] for sublist in data]
    sems = [sem(v) for v in clean_data]
    ns = [len(v) for v in clean_data]
    ax.bar(x_locs,
           means,
           yerr=sems, 
           width=x_width,
           capsize=5, color=colors, alpha=0.7)
    for i in range(len(means)):
        ax.text(x_locs[i], means[i] + sems[i] + 0.02, f'n={ns[i]}', ha='center', va='bottom', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x_locs)
    ax.set_xticklabels(data_names, rotation=45, ha='right')
    ax.set_ylabel('Cosine Similarity')
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0,1)

data = [cos_sim_flat, cos_sim_flat_random, cos_sim_flat_pairs, cos_sim_hyper_flat, cos_sim_hyper_flat_random, cos_sim_hyper_flat_pairs]
data_names = ['Directed (celltype)', 'Directed (rand ct)', 'Directed (neuron id)', 'Polyadic (celltype)', 'Polyadic (rand ct)', 'Polyadic (neuron id)']
data_groups = [0,1,2,0,1,2]  # 0 for pairs, 1 for polyadic pairs

plot_summary_cos_sim(data, data_groups, data_names)

data = [cos_sim_flat, cos_sim_flat_random, cos_sim_hyper_flat, cos_sim_hyper_flat_random]
data_names = ['Directed (celltype)', 'Directed (rand ct)','Polyadic (celltype)', 'Polyadic (rand ct)']
data_groups = [0,1,0,1]  # 0 for pairs, 1 for polyadic pairs

plot_summary_cos_sim(data, data_groups, data_names)

data = [cos_sim_flat_pairs, cos_sim_flat_pairs_rand, cos_sim_hyper_flat_pairs, cos_sim_hyper_flat_pairs_rand]
data_names = ['Directed (neuronid)', 'Directed (rand neuronid)', 'Polyadic (neuronid)', 'Polyadic (rand neuronid)']
data_groups = [0,1,0,1]  # 0 for pairs, 1 for polyadic pairs

plot_summary_cos_sim(data, data_groups, data_names)

#now for pairs data specifically for LNs 
data = [cos_sim_presyn_neuronid, cos_sim_presyn_neuronid_rand, cos_sim_presyn_hyper_neuronid, cos_sim_presyn_hyper_neuronid_rand]
data = [v[0] for v in data]
data_names = ['Directed (neuron id)', 'Directed (rand)', 'Polyadic (neuron id)', 'Polyadic (rand)']
data_groups = [0,1,0,1] 
plot_summary_cos_sim(data, data_groups, data_names, title='Cosine Similarity of postsynaptic partners for LNs')

data = [cos_sim_presyn[4], cos_sim_presyn_rand[4], cos_sim_hyper_presyn[4], cos_sim_hyper_presyn_rand[4]]
#data = [v[0] for v in data]
data_names = ['Directed (celltype)', 'Directed (rand)', 'Polyadic (celltype)', 'Polyadic (rand)']
data_groups = [0,1,0,1] 
plot_summary_cos_sim(data, data_groups, data_names, title='Cosine Similarity of postsynaptic partners for LNs')



#%% plot cosine similarity per presynaptic cell type


def plot_cos_sim_bar_plot(cos_sim_list, cos_sim_rand_list, title):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    #remove nans before calculating means and sems
    cos_sim_list = [[v for v in cos_sim if not np.isnan(v)] for cos_sim in cos_sim_list]
    cos_sim_rand_list = [[v for v in cos_sim if not np.isnan(v)] for cos_sim in cos_sim_rand_list]
    means = [np.nanmean(cos_sim) for cos_sim in cos_sim_list]
    rand_means = [np.nanmean(cos_sim) for cos_sim in cos_sim_rand_list]

    sems = [sem(cos_sim) for cos_sim in cos_sim_list]
    rand_sems = [np.nanstd(cos_sim)/np.sqrt(len(cos_sim)) for cos_sim in cos_sim_rand_list]

    x = np.arange(len(ct_names))  # the label locations
    width = 0.35  # the width of the bars
    colors = ['C%i' % i for i in range(len(ct_names))]

    ax.bar(x - width/2, means, width, label='Bilateral pairs', yerr=sems, capsize=5, color=colors, alpha=1.)
    ax.bar(x + width/2, rand_means, width, label='Random pairs', yerr=rand_sems, capsize=5, color=colors, alpha=0.5)

    ax.set_ylabel('Cosine Similarity across bilateral pairs')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ct_names, rotation=45, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.titlesize': 22

})

plot_cos_sim_bar_plot(cos_sim_presyn, cos_sim_presyn_rand, 'Cosine Similarity of postsynaptic partners per Presynaptic Cell Type')
plot_cos_sim_bar_plot(cos_sim_hyper_presyn, cos_sim_hyper_presyn_rand, 'Cosine Similarity of polyadic postsynaptic partners per Presynaptic Cell Type')

plot_cos_sim_bar_plot(cos_sim_presyn_neuronid, cos_sim_presyn_neuronid_rand, 'General partners - neuron id')
plot_cos_sim_bar_plot(cos_sim_presyn_hyper_neuronid, cos_sim_presyn_hyper_neuronid_rand, 'Polyadic partners - neuron id')

#%% try counter and cosine sim

def get_counts(ct_list1, ct_list2):
    """
    Get the counts of each cell type in both lists.
    """
    ct1_counts = Counter(ct_list1)
    ct2_counts = Counter(ct_list2) 
    
    all_cts = set(ct1_counts.keys()).union(set(ct2_counts.keys()))
    count_df = pd.DataFrame(all_cts, columns=['cell_type'])
    count_df['ct1'] = 0
    count_df['ct2'] = 0

    for e, ct in enumerate(all_cts):
        count_df.at[e, 'ct1'] = ct1_counts.get(ct, 0)
        count_df.at[e, 'ct2'] = ct2_counts.get(ct, 0)

    return count_df

def get_cosine_similarity(count_df):
    ct1 = count_df['ct1'].values.reshape(1, -1)
    ct2 = count_df['ct2'].values.reshape(1, -1)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(ct1, ct2)
    return cosine_sim

def get_agg_ct_polyadj(adj, ct_in_adj, unique_cts=None):
    #s = time.time()
    if unique_cts is None:
        unique_cts = np.unique(ct_in_adj)
    # Map cell types to integer labels
    ctype_to_int = {ctype: i for i, ctype in enumerate(unique_cts)}
    ct_labels = np.vectorize(ctype_to_int.get)(ct_in_adj)

    n_types = len(unique_cts)
    n_cells = len(ct_labels)

    # Build assignment matrix
    assign_matrix = np.zeros((n_types, n_cells))
    assign_matrix[ct_labels, np.arange(n_cells)] = 1

    # Perform fast aggregation using matrix multiplication
    agg_matrix = assign_matrix @ adj @ assign_matrix.T

    # Convert to DataFrame
    agg_df = pd.DataFrame(agg_matrix, index=unique_cts, columns=unique_cts)
    #e = time.time()
    #print(f"Time taken for aggregation: {e - s:.2f} seconds")
    return agg_df, unique_cts

#%%
# create subset of connector details with only labelled neurons
connector_details_presyn_labelled = connector_details[connector_details['presynaptic_celltype'] != 'NA']
labelled_connectors = connector_details_presyn_labelled[~connector_details_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
#remove connectors with no labelled postsynaptic celltypes
labelled_connectors = labelled_connectors[labelled_connectors['postsynaptic_celltype'].apply(lambda x: len(x) > 0)]
print(f"Number of connectors with only labelled presynaptic and postsynaptic celltypes: {len(labelled_connectors)}")
