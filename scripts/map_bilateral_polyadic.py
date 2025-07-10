#%%

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

from contools import Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

from scripts.functions.random_polyadic_networks import polyadic_edge_permutation
from scripts.functions.group_based_stats import binarize_poly_adj, get_random_poly_adj, compare_two_sample_chi_squared, correct_pvals, plot_pvals_heatmap, plot_fold_change_heatmap, plot_significant_fold_change_heatmap, get_group_stats_from_bi_adj
from scripts.functions.little_helper import get_celltype_dict, celltype_col_for_list, get_celltype_name
from scripts.functions.undirected_graph_functions import get_postsynaptic_co_adj, get_sbm_block_probs_from_hyperedges, map_co_adj_to_dict

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


#%% try compare cell type targets - one neuron to try 


def evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cts, rng, presyn_restrict=None, randomize=False, verbose=False):
    #restrict the pairs to compare if desired - otherwise will compare across all pairs 
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

        # check that both neurons project to some known postsynaptic partners
        allL = list(chain.from_iterable(L1['postsynaptic_celltype'].to_list()[0]))
        allR = list(chain.from_iterable(R1['postsynaptic_celltype'].to_list()[0]))
        L_ = any(v in allL for v in skid_to_celltype.values())
        R_ = any(v in allR for v in skid_to_celltype.values())
        if not (L_ and R_):
            if verbose:
                print(f"Skipping pair {c} as one of the neurons does not have any known postsynaptic partners.")
            cos_sim_flat.append(np.nan)
            cos_sim_hyper_flat.append(np.nan)
            continue
        
        # TODO: replace this with my poly adj function & the NA filtering of cell types so that we avoid the necessity of using cell types, to make more flexible
        L1_adj_bi, L1_block_probs, L1_ct_in_adj, L1_adj = get_sbm_block_probs_from_hyperedges(L1['postsynaptic_to'].to_list()[0], skid_to_celltype, 'Left', plot=False)
        R1_adj_bi, R1_block_probs, R1_ct_in_adj, R1_adj = get_sbm_block_probs_from_hyperedges(R1['postsynaptic_to'].to_list()[0], skid_to_celltype, 'Right', plot=False)

        count_df = get_counts(L1_ct_in_adj, R1_ct_in_adj)
        cosine_sim = get_cosine_similarity(count_df)
        cos_sim_flat.append(cosine_sim[0][0])

        # to the same but with hyperedge co-occurrence 

        L1_agg, L1_unique_cts = get_agg_ct_polyadj(L1_adj, L1_ct_in_adj, unique_cts=unique_cts)
        R1_agg, R1_unique_cts = get_agg_ct_polyadj(R1_adj, R1_ct_in_adj, unique_cts=unique_cts)
        L1_agg_flat = L1_agg.values.flatten()
        R1_agg_flat = R1_agg.values.flatten()
        hyper_cosine_sim = cosine_similarity(L1_agg_flat.reshape(1, -1), R1_agg_flat.reshape(1, -1))

        cos_sim_hyper_flat.append(hyper_cosine_sim[0][0])

    print(f"Average cosine similarity of postsynaptic cell types between bilateral pairs: {np.nanmean(cos_sim_flat)}")
    print(f"Average hyperedge cosine similarity of postsynaptic cell types between bilateral pairs: {np.nanmean(cos_sim_hyper_flat)}")
    return cos_sim_flat, cos_sim_hyper_flat


#cos_sim_flat, cos_sim_hyper_flat = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cts=ct_names, rng=rng, presyn_restrict=None, randomize=False)
#cos_sim_flat_random, cos_sim_hyper_flat_random = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cts=ct_names, rng=rng, presyn_restrict=None, randomize=True)

cos_sim_presyn = []
cos_sim_hyper_presyn = []
cos_sim_presyn_rand = []
cos_sim_hyper_presyn_rand = []
for presyn_ct in ct_names:
    presyn_restrict = celltype_df[celltype_df['name'] == presyn_ct]['skids'].values[0]

    cos_sim_flat_presyn, cos_sim_hyper_flat_presyn = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cts=ct_names, rng=rng, presyn_restrict=presyn_restrict, randomize=False)
    cos_sim_flat_presyn_rand, cos_sim_hyper_flat_presyn_rand = evaluate_cossim_across_pairs(pairs, grouped_ps_left, grouped_ps_right, skid_to_celltype, unique_cts=ct_names, rng=rng, presyn_restrict=presyn_restrict, randomize=True)
    cos_sim_presyn.append(cos_sim_flat_presyn)
    cos_sim_hyper_presyn.append(cos_sim_hyper_flat_presyn)
    cos_sim_presyn_rand.append(cos_sim_flat_presyn_rand)
    cos_sim_hyper_presyn_rand.append(cos_sim_hyper_flat_presyn_rand)


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

fig, ax = plt.subplots(1, 1)
means = [np.nanmean(cos_sim_flat), np.nanmean(cos_sim_flat_random), np.nanmean(cos_sim_hyper_flat), np.nanmean(cos_sim_hyper_flat_random)]
#remove nans before calculating sem here 
clean_data = [[v for v in sublist if not np.isnan(v)] for sublist in [cos_sim_flat, cos_sim_flat_random, cos_sim_hyper_flat, cos_sim_hyper_flat_random]]
sems = [sem(v) for v in clean_data]
ns = [len(v) for v in clean_data]

ax.bar(['Partners', 'Partners (rand)', 'Polyadic partners', 'Polyadic partners (rand)'], 
       means,
       yerr=sems, 
       capsize=5, color=['blue', 'deepskyblue', 'orangered', 'coral'])
for i in range(len(means)):
    ax.text(i, means[i] + sems[i] + 0.02, f'n={ns[i]}', ha='center', va='bottom', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticklabels(['Partners', 'Partners (rand)', 'Polyadic partners', 'Polyadic partners (rand)'], rotation=45, ha='right')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Cosine Similarity between bilateral and permutated pairs')
ax.set_ylim(0,1)

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

    ax.set_ylabel('Cosine Similarity across pairs')
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
    if unique_cts is None:
        unique_cts = np.unique(ct_in_adj)
    type_to_indices = {ctype: np.where(np.array(ct_in_adj) == ctype)[0] for ctype in unique_cts}
    agg_matrix = np.zeros((len(unique_cts), len(unique_cts)))

    for i, type_i in enumerate(unique_cts):
        for j, type_j in enumerate(unique_cts):
            idx_i = type_to_indices[type_i]
            idx_j = type_to_indices[type_j]
            submatrix = adj[np.ix_(idx_i, idx_j)]
            agg_matrix[i, j] = submatrix.sum()

    agg_df = pd.DataFrame(agg_matrix, index=unique_cts, columns=unique_cts)
    return agg_df, unique_cts

#%%
# create subset of connector details with only labelled neurons
connector_details_presyn_labelled = connector_details[connector_details['presynaptic_celltype'] != 'NA']
labelled_connectors = connector_details_presyn_labelled[~connector_details_presyn_labelled['postsynaptic_celltype'].apply(lambda x: 'NA' in x)]
#remove connectors with no labelled postsynaptic celltypes
labelled_connectors = labelled_connectors[labelled_connectors['postsynaptic_celltype'].apply(lambda x: len(x) > 0)]
print(f"Number of connectors with only labelled presynaptic and postsynaptic celltypes: {len(labelled_connectors)}")
