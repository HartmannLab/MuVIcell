
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import muon as mu
from plotnine import *

def plot_variance_explained(mdata_or_model, max_factors=None, by_view=True):
    if hasattr(mdata_or_model, "mdata"):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
    
    var_exp = mdata.uns["muvi_variance_explained"]
    plot_data = []
    for view_name, view_var in var_exp.items():
        for factor_idx, variance in enumerate(view_var):
            if max_factors is not None and factor_idx >= max_factors:
                break
            plot_data.append({
                "factor": f"Factor_{factor_idx}",
                "view": view_name,
                "variance_explained": variance,
                "factor_idx": factor_idx
            })
    
    df = pd.DataFrame(plot_data)
    p = (ggplot(df, aes(x="factor", y="variance_explained", fill="view")) +
         geom_bar(stat="identity", position="dodge") +
         theme_minimal())
    return p

def plot_factor_scores(mdata_or_model, factors=(0, 1), color_by=None, size=1.0):
    if hasattr(mdata_or_model, "mdata"):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    factor_scores = mdata.obsm["X_muvi"]
    plot_data = pd.DataFrame({
        f"Factor_{factors[0]}": factor_scores[:, factors[0]],
        f"Factor_{factors[1]}": factor_scores[:, factors[1]]
    })
    
    if color_by and color_by in mdata.obs.columns:
        plot_data[color_by] = mdata.obs[color_by].values
        p = (ggplot(plot_data, aes(x=f"Factor_{factors[0]}", y=f"Factor_{factors[1]}", color=color_by)) +
             geom_point(size=size, alpha=0.7) + theme_minimal())
    else:
        p = (ggplot(plot_data, aes(x=f"Factor_{factors[0]}", y=f"Factor_{factors[1]}")) +
             geom_point(size=size, alpha=0.7, color="steelblue") + theme_minimal())
    return p

def plot_factor_loadings(mdata_or_model, view, factors=None, top_genes=5):
    if hasattr(mdata_or_model, "mdata"):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    loadings = mdata.mod[view].varm["muvi_loadings"]
    gene_names = mdata.mod[view].var_names
    
    if factors is None:
        factors = [0, 1, 2]
    
    plot_data = []
    for factor_idx in factors:
        if factor_idx < loadings.shape[1]:
            factor_loadings = loadings[:, factor_idx]
            top_indices = np.argsort(np.abs(factor_loadings))[-top_genes:]
            for idx in top_indices:
                plot_data.append({
                    "gene": gene_names[idx],
                    "loading": factor_loadings[idx],
                    "factor": f"Factor_{factor_idx}"
                })
    
    df = pd.DataFrame(plot_data)
    p = (ggplot(df, aes(x="gene", y="loading", fill="factor")) +
         geom_bar(stat="identity", position="dodge") +
         theme_minimal() + theme(axis_text_x=element_text(rotation=45)))
    return p

def plot_factor_associations(mdata_or_model, associations_df):
    p = (ggplot(associations_df, aes(x="factor", y="metadata", fill="p_value")) +
         geom_tile() + scale_fill_gradient(low="red", high="blue") + theme_minimal())
    return p

def plot_factor_comparison(mdata_or_model, factors, group_by, plot_type="boxplot"):
    if hasattr(mdata_or_model, "mdata"):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    if group_by not in mdata.obs.columns:
        raise ValueError(f"Column {group_by} not found in mdata.obs")
        
    factor_scores = mdata.obsm["X_muvi"]
    plot_data = []
    
    for factor_idx in factors:
        if factor_idx < factor_scores.shape[1]:
            for i, group_val in enumerate(mdata.obs[group_by]):
                plot_data.append({
                    "factor": f"Factor_{factor_idx}",
                    "score": factor_scores[i, factor_idx],
                    group_by: group_val
                })
    
    df = pd.DataFrame(plot_data)
    p = (ggplot(df, aes(x="factor", y="score", fill=group_by)) +
         geom_boxplot() + theme_minimal())
    return p
