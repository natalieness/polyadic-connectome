import pandas as pd
import numpy as np
import ast

from scripts.initialisation_scripts.create_neuron_class import get_neuron_class
from scripts.functions.little_helper import get_celltype_dict, get_pairs_dict


def get_me_started():
    '''This is a little shortcut for me to get data that I want 
    for most of my scripts.'''

    # Load connectors 
    connector_details = pd.read_csv('init_data/connector_details2025.csv')
    all_presyn = connector_details['presynaptic_to'].unique()

    connector_details['postsynaptic_to'] = connector_details['postsynaptic_to'].apply(ast.literal_eval)
    connector_details['postsynaptic_to_node'] = connector_details['postsynaptic_to_node'].apply(ast.literal_eval)
    
    #get cell types
    celltype_df = pd.read_csv('init_data/celltype_df.csv')
    celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)
    skid_to_celltype = get_celltype_dict(celltype_df)

    pairs = pd.read_csv('init_data/pairs-2022-02-14.csv')
    pairs_dict = get_pairs_dict(pairs)

    # Load the neuron class 
    neuron_objects = get_neuron_class(
        all_presyn, skid_to_celltype, pairs_dict
    )

    #  map skid ids in connector details to celltypes
    connector_details['presynaptic_celltype'] = connector_details['presynaptic_to'].apply(lambda x: skid_to_celltype.get(x, None))
    connector_details['postsynaptic_celltype'] = connector_details['postsynaptic_to'].apply(
        lambda x: [skid_to_celltype.get(v, None) for v in x])

     # assign hemisphere of presynaptic neuron 
    left_ns = pairs['leftid'].unique()
    right_ns = pairs['rightid'].unique()
    connector_details['presyn_hemi'] = connector_details['presynaptic_to'].apply(lambda x: 'right' if x in right_ns else 'left' if x in left_ns else None)

    connector_details['presynaptic_pair'] = connector_details['presynaptic_to'].apply(lambda x: pairs_dict.get(x, None))
    connector_details['postsynaptic_pair'] = connector_details['postsynaptic_to'].apply(lambda x: [pairs_dict.get(v, None) for v in x])

    return connector_details, skid_to_celltype, pairs, pairs_dict, neuron_objects, celltype_df

