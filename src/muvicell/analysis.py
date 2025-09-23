"""
Analysis utilities for MuVIcell factor interpretation.

This module provides functions for analyzing and interpreting MuVI factors,
including pathway enrichment analysis and factor characterization.
"""

import muon as mu
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
import scipy.stats as stats
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import warnings


def identify_factor_associations(
    model,
    metadata_columns: Optional[List[str]] = None,
    categorical_test: str = 'kruskal',
    continuous_test: str = 'pearson'
) -> pd.DataFrame:
    """
    Identify associations between factors and sample metadata.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    metadata_columns : List[str], optional
        Specific metadata columns to test. If None, tests all columns
    categorical_test : str, default 'kruskal'
        Statistical test for categorical variables ('kruskal', 'anova')
    continuous_test : str, default 'pearson'
        Correlation method for continuous variables ('pearson', 'spearman')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with factor-metadata associations and p-values
        
    Examples
    --------
    >>> associations = identify_factor_associations(model)
    >>> significant = associations[associations['p_value'] < 0.05]
    """
    assert hasattr(model, 'get_factor_scores'), "Model must have get_factor_scores method"
    assert hasattr(model, 'mdata_original'), "Model must have mdata_original attribute"
    
    factor_scores = model.get_factor_scores()
    obs_df = model.mdata_original.obs
    
    if metadata_columns is None:
        metadata_columns = list(obs_df.columns)
    
    results = []
    n_factors = factor_scores.shape[1]
    
    for factor_idx in range(n_factors):
        factor_values = factor_scores[:, factor_idx]
        
        for col in metadata_columns:
            if col not in obs_df.columns:
                continue
            
            metadata_values = obs_df[col]
            
            # Skip if too many missing values
            valid_mask = ~(pd.isna(metadata_values) | pd.isna(factor_values))
            if valid_mask.sum() < 10:
                continue
            
            factor_valid = factor_values[valid_mask]
            metadata_valid = metadata_values[valid_mask]
            
            # Determine if metadata is categorical or continuous
            if pd.api.types.is_categorical_dtype(metadata_valid) or \
               pd.api.types.is_object_dtype(metadata_valid):
                # Categorical variable
                try:
                    groups = [factor_valid[metadata_valid == group] 
                             for group in metadata_valid.unique() 
                             if len(factor_valid[metadata_valid == group]) > 0]
                    
                    if len(groups) < 2:
                        continue
                    
                    if categorical_test == 'kruskal':
                        statistic, p_value = stats.kruskal(*groups)
                        test_name = 'Kruskal-Wallis'
                    elif categorical_test == 'anova':
                        statistic, p_value = stats.f_oneway(*groups)
                        test_name = 'ANOVA'
                    else:
                        continue
                        
                except Exception:
                    continue
                    
            else:
                # Continuous variable
                try:
                    if continuous_test == 'pearson':
                        statistic, p_value = stats.pearsonr(factor_valid, metadata_valid)
                        test_name = 'Pearson correlation'
                    elif continuous_test == 'spearman':
                        statistic, p_value = stats.spearmanr(factor_valid, metadata_valid)
                        test_name = 'Spearman correlation'
                    else:
                        continue
                except Exception:
                    continue
            
            results.append({
                'factor': f'Factor_{factor_idx}',
                'metadata': col,
                'test': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'n_valid': valid_mask.sum()
            })
    
    results_df = pd.DataFrame(results)
    
    # Add multiple testing correction
    if len(results_df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_adj'] = p_adj
    
    return results_df


def cluster_cells_by_factors(
    model,
    factors_to_use: Optional[List[int]] = None,
    n_clusters: int = 3,
    method: str = 'kmeans'
) -> np.ndarray:
    """
    Cluster samples based on factor scores.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    factors_to_use : List[int], optional
        Specific factors to use for clustering. If None, uses all factors
    n_clusters : int, default 3
        Number of clusters
    method : str, default 'kmeans'
        Clustering method ('kmeans', 'hierarchical')
        
    Returns
    -------
    np.ndarray
        Cluster assignments for each sample
        
    Examples
    --------
    >>> clusters = cluster_cells_by_factors(model, n_clusters=4)
    >>> print(f"Cluster distribution: {np.bincount(clusters)}")
    """
    assert hasattr(model, 'get_factor_scores'), "Model must have get_factor_scores method"
    
    factor_scores = model.get_factor_scores()
    
    if factors_to_use is not None:
        factor_scores = factor_scores[:, factors_to_use]
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = clusterer.fit_predict(factor_scores)
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clusterer.fit_predict(factor_scores)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return clusters