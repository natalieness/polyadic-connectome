''' code snippets to get neuronal identity from larval d melanogaster connectome from pymaid 

with little overview of numbers in each category
'''
#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

#access catmaid instance
rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  # Replace with your file path if not in a script
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)

#%% to get hand annotated cell types from fig s4
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()

#note this comes from https://github.com/mwinding/connectome_tools/blob/main/contools/celltype.py#L454

#%% inspect cell type data from fig s4

#inspect whats in the celltypes class
for ct in celltypes:
    print(f"Name: {ct.get_name()}, Skids: {len(ct.get_skids())}, Color: {ct.get_color()}")

#inspect whats in the celltype_df - probably the same? 
n_skids_total = 0
for row in celltype_df.iterrows():
    index, data = row
    print(f"Name: {data['name']}, Skids: {len(data['skids'])}, Color: {data['color']}")
    n_skids_total += len(data['skids']) 

print(f"Total number of skids: {n_skids_total}")

## these objects are the same, just different ways of accessing the data

#%%
#to get cluster hierarchy at any particular level (in this case lvel 7) from fig 3

cluster_lvl = 7 
clusters, cluster_names = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_lvl}', split=True)
# note The code for get_skids_from_meta_annotation() is here:
# https://github.com/mwinding/connectome_tools/blob/main/contools/celltype.py#L438

#%% inspect cluster data from fig 3

print(f"Number of clusters: {len(clusters)}")
print(f"Number of named clusters in cluster_names: {len(cluster_names)}")

all_skids = [*sum(clusters, [])]
all_unique_skids = list(set(all_skids))

print(f"Total number of skids: {len(all_skids)}")
print(f"Number of unique skids: {len(all_unique_skids)}")
## each nested list in clusters corresponds to a cluster, with name in associated cluster_names



#%%
#to get sensory neurons 2nd 3rd or 4th order neurons based on modality from fig 4


order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']

sens = [Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {celltype}') for celltype in order]
order2_ct = [Celltype(f'2nd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 2nd_order')) for celltype in order]
order3_ct = [Celltype(f'3rd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 3rd_order')) for celltype in order]
order4_ct = [Celltype(f'4th-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 4th_order')) for celltype in order]

print(f"Sensory neurons, and 2nd to 4th order neurons from each sensory modality. Each order has neurons for  {len(order)} modalities")

# %%
