import numpy as np
import pandas as pd
import contools
import matplotlib.pyplot as plt
import pymaid 
import pickle
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Image

#%% decide whether to save data pdf or not 
save_pdf = True

if save_pdf:
    #set up file
    folder = os.getcwd()
    filename = os.path.join(folder, 'polyadic_connectors_overview.pdf')
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4



    def add_text(c, text, x=50, y=800):
        c.setFont("Helvetica", 12)
        c.drawString(x, y, text)
        y -= 20
        return y

    def add_figure(c, fig, fig_name, x=50, y=400, width_scale=0.5):
        y -= 400
        fig_path = os.path.join(folder, fig_name)
        fig.savefig(fig_path)
        c.drawImage(fig_path, x, y, width=width*width_scale, preserveAspectRatio=True)
        os.remove(fig_path)  # Optional: clean up temp file
        y -= 20
        return y
else:
    c=None

#text positions
x=50
y=800

def double_print(text, save_pdf=False, c=None, x=50, y=800):
    print(text)
    if save_pdf:
        y = add_text(c, text, x, y)
    return y


#%% get connectors from catmaid

from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

# select neurons to include 
all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# get all synaptic sites associated with neurons 
links = pymaid.get_connector_links(all_neurons, chunk_size=50)

#get all connectors 
all_links = pymaid.get_connectors(all_neurons)

# %% inspect shape of data

y = double_print(f"links df has {links.shape[0]} entries ", save_pdf, c, x, y)

n_connector = links['connector_id'].nunique()
y = double_print(f"Number of connectors: {n_connector}", save_pdf, c, x, y)

n_skeleton = links['skeleton_id'].nunique()

y = double_print(f"Skeletons: {n_skeleton}", save_pdf, c, x, y)
n_nodes = links['node_id'].nunique()

y = double_print(f"Number of nodes: {n_nodes}", save_pdf, c, x, y)

n_pre_relations = links.groupby('relation').value_counts()
# %% figure out what exactly a catmaid connector is

#get everything associated with 1 connector ID
connector_id = links['connector_id'].unique()[0]
connector = links[links['connector_id'] == connector_id]
connector 

#a connector in catmaid appears to be a synaptic site - with one pre- and multiple post-synaptic partners

# %% get connectors with presynaptic sites

connectors = links['connector_id'].unique()

# Check if all connectors have at least one 'presynaptic_to' in the 'relation' column
has_presynaptic = links.groupby('connector_id')['relation'].apply(lambda x: 'presynaptic_to' in x.values)
n_with_presynaptic = has_presynaptic.sum()
y = double_print(f"Number of connectors with at least one 'presynaptic_to': {n_with_presynaptic} out of {n_connector}", save_pdf, c, x, y)

#get connectors with presynaptic site 
connector_with_presyn = has_presynaptic[has_presynaptic].index
#filter connectors by those with presynaptic sites
links_with_presyn = links[links['connector_id'].isin(connector_with_presyn)]

# %% find out the average number of post-synaptic partners per connector

mean_post_all = links.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).mean()
y = double_print(f"Mean number of postsynaptic partners per connector (all): {mean_post_all}", save_pdf, c, x, y)

mean_post_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).mean()
y = double_print(f"Mean number of postsynaptic partners per connector (filtered by with presynaptic site): {mean_post_filtered}", save_pdf, c, x, y)

max_post_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).max()
y = double_print(f"Max number of postsynaptic partners per connector (filtered): {max_post_filtered}", save_pdf, c, x, y)

min_post_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum()).min()
y = double_print(f"Min number of postsynaptic partners per connector (filtered): {min_post_filtered}", save_pdf, c, x, y)
#%%
#look at distribution of number of postsynaptic partners per connector
postperconnect_filtered = links_with_presyn.groupby('connector_id')['relation'].apply(lambda x: (x == 'postsynaptic_to').sum())

# Plot histogram of the number of postsynaptic partners per connector
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(postperconnect_filtered, bins=range(postperconnect_filtered.min(), postperconnect_filtered.max() + 2))

# Annotate each bin with its count
for count, bin_edge in zip(counts, bins):
    if count > 0:  
        count_str = str(int(count))
        if count > 1000:
            count_str = f"{int(count / 1000)}k"
        ax.text(bin_edge + 0.5, count, count_str, ha='center', va='bottom', rotation=90, fontsize=8)

ax.set_xlabel('Number of Postsynaptic Partners')
ax.set_ylabel('Number of Connectors')
ax.set_title('Distribution of Postsynaptic Partners per Connector')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#add figure to pdf
y = add_figure(c, fig, 'postsynaptic_partners_distribution.png', x=x, y=y, width_scale=0.8)
#then show 
plt.show()
# %%

''' Check what i can learn about neuronal identities from the connector data
'''



#%% close pdf 
c.save()
print(f"PDF saved at {filename}.")

# %%
