''' Quality checks for synapse data to spot potential systematic issues with annotations of polyadic synapses.'''

from collections import Counter

import numpy as np
import pandas as pd


def count_postsyn_mutiples_of_neurons(connector_dets, mode='neuron'):
    ''' Check if for any synaptic sites, there are multiples of the 
    same postsynaptic target neuron or node. 
    Multiple nodes is definitely a mistake, multiple neurons may be 
    biologically relevant, but coould also indicate annotation issues. 
    
    Inputs: 
        connector_dets: pandas dataframe based on pymaid.get_connector_details()
        mode: 'neuron' or 'node', determines whether to check for multiple neurons 
        or nodes. 
        
    Returns:
        multi_locs: list of indices in connector_dets where multiple 
        postsynaptic targets are found for the same presynaptic site.
    '''
    if mode == 'neuron':
        # get postsynaptic sites as a list of lists
        postsyn_to = connector_dets['postsynaptic_to']
        
    elif mode == 'node':
        postsyn_to = connector_dets['postsynaptic_to_node']
    else:
        raise ValueError("Mode must be 'neuron' or 'node'")
    
    postsyn_to_id = connector_dets['connector_id']
    multi_locs = []
    for e, ps in zip(postsyn_to_id, postsyn_to): #should all be lists
        if isinstance(ps, list):
            counts = Counter(ps)
            for skid, count in counts.items():
                if count > 1:
                    multi_locs.append(e)
    print(f"Number of presynaptic sites with multiple of the same postsynaptic {mode}: {len(multi_locs)} out of {len(postsyn_to)}")
    print(f" {len(multi_locs)/len(connector_dets)*100:.2f}%")
    return multi_locs

def count_multiples_of_postsyn_nodes_across_connectors(connector_dets):
    ''' Check if for any synaptic sites, there are multiple postsynaptic 
    nodes across different connectors. This could be an indication of mistakes
    but also should frequently occur across larger neurons, as a node in catmaid can 
    cover a large area with multiple synaptic sites.
    
    Inputs: 
        connector_dets: pandas dataframe based on pymaid.get_connector_details()
        
    Returns:
        multi_locs: dict of postsynaptic nodes with number of times they occur.
    '''
    postsyn_to_node = connector_dets['postsynaptic_to_node']
    postsyn_to_node_flat = [item for sublist in postsyn_to_node for item in sublist]
    counts = Counter(postsyn_to_node_flat)
    return counts

