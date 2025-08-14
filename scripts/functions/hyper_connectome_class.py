
from functools import partial

import pandas as pd
import numpy as np

from scripts.functions.motif_functions import con_binary_matrix


class HyperCon:
    # to manage and filter multiple layers of connector dataframes
    def __init__(self, filter_col: str = 'n_post'):
        self.df = {} #store original connector dataframe by name
        self.filter_col = filter_col

    def add_df(self, name: str, data: pd.DataFrame):
        self.df[name] = data
    
    def filter_by_layer(self, name: str, value: int) -> pd.DataFrame:
        if name not in self.df:
            raise ValueError(f"Dataframe {name} not found.")
        data = self.df[name]
        return data[data[self.filter_col] == value].reset_index(drop=True)
    
    def get_all_filtered(self, value: int) -> dict:
        """ Returns a dictionary of filtered dataframes for all stored dataframes """
        return {name: self.filter_by_layer(name, value) for name in self.df}
    
    def apply_some_function_on_binarised(self, value: int, func) -> dict:
        """ Returns a dictionary of binary matrices for all stored dataframes """
        results = {}
        for name, data in self.get_all_filtered(value).items():
            bin_data = con_binary_matrix(data, only_known_targets=True, all_neurons=None)
            results[name] = func(bin_data)
        return results
        
    def apply_some_function(self, value: int, func) -> dict:
        results = {}
        for name, data in self.get_all_filtered(value).items():
            results[name] = func(data)
        return results
    
    def apply_multiple_functions(self, value: int, func_list: list) -> dict:
        results = {}
        for name, data in self.get_all_filtered(value).items():
            result = data
            for func in func_list:
                result = func(result)
            results[name] = result
        return results