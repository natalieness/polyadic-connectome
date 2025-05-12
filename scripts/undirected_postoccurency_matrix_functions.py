''' Functions to examine the undirected post co-occurency matrix 
of polyadic synaptic sites.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
from collections import defaultdict
import os


#compute relative covariance of postsynaptic celltypes
def compute_relative_covariance(post_cooccurency):
    #mirror matrix to be able to compute covariance relative to each row 
    upper_triangle = np.triu(post_cooccurency, k=1)
    mirrored = post_cooccurency + upper_triangle.T
    row_sums = np.sum(mirrored, axis=1)
    cov = np.zeros(post_cooccurency.shape)
    for i in range(mirrored.shape[0]):
        for j in range(mirrored.shape[1]):
            cov[i, j] = mirrored[i, j] / row_sums[i] 

    return cov


def jaccard_similarity(arr):
    ''' Calculate jaccard similarity based on a co-occurency matrix '''
    jaccard_sim = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            intersection = arr[i, j]
            print(f"Intersection: {intersection}")
            union = np.sum(arr[i, :]) + np.sum(arr[:, j]) - intersection
            print(f"Union: {union}")
            jaccard_sim[i, j] = intersection / union if union != 0 else 0
    return jaccard_sim

# compute marginal probability of each cell type co-occuring with itself or other 
def get_P_marginal(arr, ct_index):
    num = np.sum(arr[ct_index, :]) #sum of all postsynaptic celltypes pairs involving this cell type
    denom = np.sum(arr) #sum of all postsynaptic celltype pairs 
    return num / denom

def precompute_P_marginal(arr):
    Ps_marginal = []
    for i in range(arr.shape[0]):
        Ps_marginal.append(get_P_marginal(arr, i))
    return Ps_marginal

def get_P_ij(arr, ct_index_i, ct_index_j):
    num = arr[ct_index_i, ct_index_j] #sum of all postsynaptic celltypes pairs involving this cell type
    denom = np.sum(arr) #sum of all postsynaptic celltype pairs 
    return num / denom

def compute_PMI(arr, Ps_marginal):
    pmi = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(i, arr.shape[1]):
            Pij = get_P_ij(arr, i, j)
            pmi[i, j] = np.log2(Pij / (Ps_marginal[i] * Ps_marginal[j]))
            if Pij == 0:
                pmi[i, j] = np.nan
    return pmi

#  probability of occuring alone 


## DOUBLE CHECK THIS CONCEPTUALLY

def get_P_alone(df_series, ct_name):
    n_ct = 0
    n_sets = 0 
    for l in df_series:
        n_sets += 1
        if ct_name in l:
            if len(l) ==1:
                n_ct += 1
    return n_ct / n_sets

def get_P_alone_all(df_series, ct_names):
    alone = []
    for ct in ct_names.unique():
        alone.append(get_P_alone(df_series, ct))
    return alone

ct_P_alone = get_P_alone_all(labelled_connectors['postsynaptic_celltype'], celltype_df['name'])

