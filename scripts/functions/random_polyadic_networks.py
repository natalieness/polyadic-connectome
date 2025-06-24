
from itertools import chain

import numpy as np
import pandas as pd


# functions to generate a random polyadic network

def polyadic_edge_permutation(connectors, rng=None):
    ''' Expecting a dataframe similar to connector_details from catmaid '''
    if rng is None:
        rng = np.random.default_rng(42)
    #initialise new connectors dataframe
    new_connectors = pd.DataFrame(columns=['connector_id', 'presynaptic_to', 'postsynaptic_to'])
    # get all unique presynaptic neurons 
    u_presyn = connectors['presynaptic_to'].unique()

    #iterate over unique presynaptic neurons
    for ps in u_presyn:
        # get all connectors with this presynaptic neuron
        ps_connectors = connectors[connectors['presynaptic_to'] == ps]
        # get all postsynaptic partners 
        ps_partners = list(chain.from_iterable(ps_connectors['postsynaptic_to']))
        # randomly permutate partners so they can be drawn on later 
        rng.shuffle(ps_partners)
        # iterate over each connector, and randomly select new postsynaptic partners 
        u_connector = ps_connectors['connector_id'].unique()
        for c in u_connector:
            # get the number of postsynaptic partners for this connector
            n_partners = len(ps_connectors[ps_connectors['connector_id'] == c]['postsynaptic_to'].values[0])
            # randomly draw new partners from all partners
            new_partners = ps_partners[:n_partners]
            ps_partners = ps_partners[n_partners:]  # remove drawn partners from the list
            # create a new connector with the same presynaptic neuron and new postsynaptic
            new_row = pd.DataFrame({'connector_id': c,
                                     'presynaptic_to': ps,
                                     'postsynaptic_to': [new_partners]})
            new_connectors = pd.concat([new_connectors, new_row], ignore_index=True)
        # check that all ps_partners have been used 
        if len(ps_partners) >0:
            print('something went wrong, not all partners used')
    return new_connectors
    

