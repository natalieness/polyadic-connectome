import ast

import pandas as pd 
import numpy as np 
#from typing import Optional 

def prep_motif_df_for_analysis(df_results: pd.DataFrame, norm: bool = True) -> pd.DataFrame:
    ''' Takes multi-index and layered columns dataframe of motif frequencies and formats it into 
     a long version for easier downstream analysis.'''
    df_copy = df_results.copy()
    ct_groups = df_results.columns.get_level_values(0).unique().to_list() 

    # normalise frequency counts 
    if norm:
        df_copy = df_copy.div(df_copy.sum(axis=0), axis=1)  # Normalize by column sums
        print("Normalised motif frequencies within each group.")
    else:
        print("Motif frequencies were not normalised.")

    # convert multi-index to column
    df_copy['syn_motif'] = [ast.literal_eval(x) for x in df_results.index]
    df_copy.reset_index(drop=True, inplace=True)  

    #remove multilayered columns 
    df_copy.columns = df_copy.columns.map('_'.join) 
    df_copy.rename(columns={'syn_motif_': 'motif'}, inplace=True)

    # convert to long format 
    df_long = pd.melt(df_copy, id_vars=['motif'], var_name='group', value_name='p')
    df_long['celltype'] = df_long['group'].apply(lambda x: x.split('_')[0])
    df_long['hemi'] = df_long['group'].apply(lambda x: 'L' if 'L' in x else 'R')
    df_long['type'] = df_long['group'].apply(lambda x: 'rand' if 'rand' in x else 'real')
    df_long.drop(columns=['group'], inplace=True)

    #df_wide = df_long.pivot(index=["motif", "celltype", "hemi"], columns="type", values="p").reset_index()
    df_out = df_long[df_long['type'] == 'real'].reset_index(drop=True)
    df_rand = df_long[df_long['type'] == 'rand'].reset_index(drop=True)
    df_out.rename(columns={'p': 'p_real'}, inplace=True)
    p_rand = df_rand['p'].to_list()
    df_out['p_rand'] = p_rand
    df_out['delta'] = df_out['p_real'] - df_out['p_rand']
    df_out['dir'] = np.sign(df_out['delta'])

    df_out.drop(columns=['type'], inplace=True)
    df_out = df_out[['motif', 'celltype', 'hemi', 'p_real', 'p_rand', 'delta', 'dir']]

    return df_out

def summarise_by_cellgroup_motif(df_out: pd.DataFrame) -> pd.DataFrame:

    keep_cols = ["p_real", "p_rand", "delta"]

    wide = df_out.pivot_table(index=["celltype", "motif"],
                             columns="hemi",
                             values=keep_cols)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]  # ('delta','left')->'delta_left'
    wide = wide.reset_index()

    def _sign(x):
        if pd.isna(x): return 0
        return 1 if x > 0 else (-1 if x < 0 else 0)

    wide["sign_L"]  = wide["delta_L"].apply(_sign)
    wide["sign_R"] = wide["delta_R"].apply(_sign)
    wide["concordant"] = (wide["sign_L"] == wide["sign_R"]) & (wide["sign_L"] != 0)

    wide["mean_abs_delta"] = (wide["delta_L"].abs() + wide["delta_R"].abs()) / 2.0
    wide["min_abs_delta"]  = wide[["delta_L", "delta_R"]].abs().min(axis=1)

    # Inter-sample (real-vs-rand) vs inter-hemisphere (left-vs-right) variability
    wide["abs_LR_delta"] = (wide["p_real_L"] - wide["p_real_R"]).abs()

    return wide


def filter_consistent_differences(
    wide: pd.DataFrame,
    min_abs_delta: float = 0.01,
    variability_margin: float = 1.0,
    require_concordance: bool = True,
) -> pd.DataFrame:
    """
    Keep motifs that:
      - are concordant across hemispheres (same sign of delta),
      - have effect size >= min_abs_delta in BOTH hemispheres (via min_abs_delta),
      - have mean_abs_delta >= variability_margin * abs_LR_delta.
    """
    crit = pd.Series(True, index=wide.index)

    if require_concordance:
        crit &= wide["concordant"].fillna(False)

    crit &= (wide["min_abs_delta"] >= min_abs_delta)
    crit &= (wide["mean_abs_delta"] >= variability_margin * wide["abs_LR_delta"])

    result = wide.loc[crit].copy()
    return result.sort_values(
        ["celltype", "sign_L", "mean_abs_delta"],
        ascending=[True, True, False]
    )
