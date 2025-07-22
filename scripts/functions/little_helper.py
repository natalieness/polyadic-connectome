'''Little helper functions '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymaid
import networkx as nx

### just printing a bunch of info about the data ###

def inspect_data(links_df, verbose=True):
    n_entries = links_df.shape[0]
    n_connectors = links_df['connector_id'].nunique()
    n_skeletons = links_df['skeleton_id'].nunique()
    n_nodes = links_df['node_id'].nunique()
    n_postsynaptic = links_df[links_df['relation'] == 'postsynaptic_to'].shape[0]
    n_presynaptic = links_df[links_df['relation'] == 'presynaptic_to'].shape[0]
    if verbose:
        print(f"Number of entries: {n_entries}")
        print(f"Number of connectors: {n_connectors}")
        print(f"Number of skeletons: {n_skeletons}")
        print(f"Number of nodes: {n_nodes}")
        print(f"Number of postsynaptic sites: {n_postsynaptic}")
        print(f"Number of presynaptic sites: {n_presynaptic}")
    return n_entries, n_connectors, n_skeletons, n_nodes, n_postsynaptic, n_presynaptic

### Mapping skids to cell types ###

def get_celltype_dict(celltype_df):
    skid_to_celltype = {
        skid: row['name']
        for _, row in celltype_df.iterrows()
        for skid in row['skids']
    }
    return skid_to_celltype

def get_celltype_name(skid, skid_to_celltype):
    return skid_to_celltype.get(skid, "NA")  # Returns "NA" if skid is not found

def get_ct_index(ct_name, ct_names):
    return np.where(ct_names == ct_name)[0][0]

### Mapping skids to cell types in specific structures ###

def celltype_col_for_list(connector_df, col_name, skid_to_celltype, new_col_name='postsynaptic_celltype'):
    df_series = connector_df[col_name] 

    new_df_series = []
    for l in df_series:
        #each element is a list of skids 
        new_l = []
        for skid in l:
            new_l.append(get_celltype_name(skid, skid_to_celltype))
        new_df_series.append(new_l)
    connector_df[new_col_name] = new_df_series

def celltype_col_for_nestedlist(connector_df, col_name, skid_to_celltype, new_col_name='postsynaptic_celltype'):
    df_series = connector_df[col_name].values[0] # changed this for pairs_dict, if it breaks for celltype, check this line
    #print(f"df series: {df_series}")
    new_df_series = []
    for l in df_series:
        #each element is a list of skids 
        #print(f'l in df_series: {l}')
        new_l = []
        for skid in l:
            #  print(f'skid: {skid}')
            new_l.append(get_celltype_name(skid, skid_to_celltype))
        new_df_series.append(new_l)
    new_df_series = [new_df_series]
    connector_df[new_col_name] = new_df_series


def get_pairs_dict(pairs):
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

    return pairs_dict