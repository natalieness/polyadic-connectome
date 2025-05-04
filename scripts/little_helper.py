'''Little helper functions '''


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

def get_celltype_name(skid, skid_to_celltype=skid_to_celltype):
    return skid_to_celltype.get(skid, "NA")  # Returns "NA" if skid is not found

### Mapping skids to cell types in specific structures ###

def celltype_col_for_list(connector_df, col_name, new_col_name='postsynaptic_celltype'):
    df_series = connector_df[col_name]
    new_df_series = []
    for l in df_series:
        #each element is a list of skids 
        new_l = []
        for skid in l:
            new_l.append(get_celltype_name(skid))
        new_df_series.append(new_l)
    connector_df[new_col_name] = new_df_series