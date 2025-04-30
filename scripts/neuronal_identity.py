''' code snippets to get neuronal identity from larval d melanogaster connectome from pymaid 
'''

#to get hand annotated cell types from fig s4 

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)
celltype_df,celltypes = Celltype_Analyzer.default_celltypes()

#note this comes from https://github.com/mwinding/connectome_tools/blob/main/contools/celltype.py#L454

#%%
#to get cluster hierarchy at any particular level (in this case lvel 7) from fig 3

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)

cluster_lvl = 7 
clusters, cluster_names = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_lvl}', split=True)
# note The code for get_skids_from_meta_annotation() is here:
# https://github.com/mwinding/connectome_tools/blob/main/contools/celltype.py#L438

#%%
#to get sensory neurons 2nd 3rd or 4th order neurons based on modality from fig 4

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)

order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']

sens = [Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {celltype}') for celltype in order]
order2_ct = [Celltype(f'2nd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 2nd_order')) for celltype in order]
order3_ct = [Celltype(f'3rd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 3rd_order')) for celltype in order]
order4_ct = [Celltype(f'4th-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 4th_order')) for celltype in order]


