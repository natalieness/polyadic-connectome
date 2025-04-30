import pandas as pd
import numpy as np

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