
''' download directed adj matrix of neurons, their celltype metadata and neurotransmitter predictions from catmaid.'''

import numpy as np
import pandas as pd


from contools import  Celltype_Analyzer
import pymaid
from pymaid_creds import url, name, password, token

# local imports
from scripts.functions.little_helper import get_celltype_dict
rm = pymaid.CatmaidInstance(url, token, name, password)



# select neurons to include 
wanted_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(wanted_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)


celltype_df,celltypes = Celltype_Analyzer.default_celltypes()

print("Cell types used")
n_skids = 0
for ct in celltypes:
    print(f"Name: {ct.get_name()}, Skids: {len(ct.get_skids())}, Color: {ct.get_color()}")
    n_skids += len(ct.get_skids())
print(f"Total number of skids: {n_skids}")

# get dictionary to map skids to celltypes 
skid_to_celltype = get_celltype_dict(celltype_df)

adj = pymaid.adjacency_matrix(all_neurons, all_neurons)
adj.to_csv('data/for_collab/adj.csv')

neus = pymaid.get_neurons(all_neurons)

neuron_df = pd.DataFrame()
neuron_df['skeleton_id'] = neus.skeleton_id
neuron_df['name'] = neus.name
neuron_df['skeleton_id'] = neuron_df['skeleton_id'].apply(lambda x: int(x))


neuron_df['celltype'] = neuron_df['skeleton_id'].map(skid_to_celltype)

#%% polyadic celltypes 

links = pymaid.get_connector_links(all_neurons, chunk_size=50)
all_connectors = links['connector_id'].unique()
connector_details = pymaid.get_connector_details(all_connectors)

connector_details = connector_details.iloc[:,:3]

#%% 
connector_details.columns = ['connector_id', 'presynaptic_id', 'postsynaptic_id']
connector_details.to_csv('data/for_collab/polyadic_connectors.csv', index=False)



#%%
neuron_nt_preds = pd.read_csv('input_data/all_brain-presynapses-neuron_preds_20250610.csv')
#remove low predictions 
#neuron_nt_preds = neuron_nt_preds[neuron_nt_preds['neuron_confidence'] > 0.5]
#%% make neuron overview 

neuron_df = neuron_df.merge(
    neuron_nt_preds[['skeleton_id', 'neuron_label', 'neuron_confidence']],
    on='skeleton_id',
    how='left'
)
# rename columns
neuron_df = neuron_df.rename(columns={
    'neuron_label': 'predicted_neurotransmitter',
    'neuron_confidence': 'neurotransmitter_confidence'
})

#%%
neuron_df.to_csv('data/for_collab/neuron_details_with_nt.csv', index=False)

#%% get neurotransmitter data from catmaid
nt_neus = pymaid.get_annotated('mw predicted NT')

pred_gaba = pymaid.get_skids_by_annotation('mw predicted-gaba 2023-09-13')
pred_glut = pymaid.get_skids_by_annotation('mw predicted-glut 2023-09-13')
pred_ach = pymaid.get_skids_by_annotation('mw predicted-ach 2023-09-13')


neuron_df['predicted_neurotransmitter'] = np.nan
neuron_df.loc[neuron_df['skeleton_id'].isin(pred_gaba), 'predicted_neurotransmitter'] = 'GABA'
neuron_df.loc[neuron_df['skeleton_id'].isin(pred_glut), 'predicted_neurotransmitter'] = 'Glutamate'
neuron_df.loc[neuron_df['skeleton_id'].isin(pred_ach), 'predicted_neurotransmitter'] = 'Acetylcholine'
