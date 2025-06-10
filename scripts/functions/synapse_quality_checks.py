


# check for number of same post-synaptic neuron co-occuring with presynaptic neuron 
def count_postsyn_mutiples_of_neurons(connector_dets, mode='neuron'):
    if mode == 'neuron':
        # get postsynaptic sites as a list of lists
        postsyn_to = connector_dets['postsynaptic_to']
    elif mode == 'node':
        postsyn_to = connector_dets['postsynaptic_to_node']
    else:
        raise ValueError("Mode must be 'neuron' or 'node'")
    multi_locs = []
    for e, ps in enumerate(postsyn_to): #should all be lists
        if isinstance(ps, list):
            counts = Counter(ps)
            for skid, count in counts.items():
                if count > 1:
                    multi_locs.append(e)
    print(f"Number of presynaptic sites with multiple of the same postsynaptic {mode}: {len(multi_locs)} out of {len(postsyn_to)}")
    print(f" {len(multi_locs)/len(connector_dets)*100:.2f}%")
    return multi_locs