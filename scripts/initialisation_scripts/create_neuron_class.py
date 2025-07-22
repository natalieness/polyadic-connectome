
class Neuron: 
    def __init__(self, skid, celltype=None, pair_id=None):
        self.skid = skid
        self.celltype = celltype
        self.pair_id = pair_id
    def get_bilateral_id(self, neuron_dict):
        """ 
        Returns a list of skids with the same pair_id, excluding self
        """
        if self.pair_id is None:
            return []
        return [
            skid for skid, neuron in neuron_dict.items()
            if neuron.pair_id == self.pair_id and skid != self.skid
        ]
    
def get_neuron_class(skids, skid_to_celltype, pairs_dict):
    neuron_objects = {}
    for skid in skids:
        celltypes = skid_to_celltype.get(skid, None)
        pair_id = pairs_dict.get(skid, None)
        neuron = Neuron(skid, celltype=celltypes, pair_id=pair_id)
        neuron_objects[skid] = neuron
    return neuron_objects

