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


def characterize_factors(
    mdata_or_model,
    top_genes_per_factor: int = 50,
    loading_threshold: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Characterize factors by identifying top contributing genes per view.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
    top_genes_per_factor : int, default 50
        Number of top genes to extract per factor
    loading_threshold : float, default 0.1
        Minimum absolute loading threshold
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with top genes per view and factor
        
    Examples
    --------
    >>> factor_genes = characterize_factors(model, top_genes_per_factor=25)
    >>> print(factor_genes['view1'].head())
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    factor_characterization = {}
    
    for view_name in mdata.mod.keys():
        # Try to get loadings from different possible keys
        loadings_key = None
        possible_keys = ['muvi_loadings', 'loadings', 'factor_loadings']
        
        for key in possible_keys:
            if key in mdata.mod[view_name].varm:
                loadings_key = key
                break
        
        if loadings_key is None:
            continue
            
        loadings = mdata.mod[view_name].varm[loadings_key]
        gene_names = mdata.mod[view_name].var_names
        n_factors = loadings.shape[1]
        
        view_results = []
        
        for factor_idx in range(n_factors):
            factor_loadings = loadings[:, factor_idx]
            
            # Get genes above threshold
            significant_mask = np.abs(factor_loadings) >= loading_threshold
            if not np.any(significant_mask):
                continue
            
            # Sort by absolute loading value
            sorted_indices = np.argsort(np.abs(factor_loadings[significant_mask]))[::-1]
            top_indices = np.where(significant_mask)[0][sorted_indices[:top_genes_per_factor]]
            
            for idx in top_indices:
                view_results.append({
                    'factor': factor_idx,
                    'gene': gene_names[idx],
                    'loading': factor_loadings[idx],
                    'abs_loading': np.abs(factor_loadings[idx])
                })
        
        factor_characterization[view_name] = pd.DataFrame(view_results)
    
    return factor_characterization


def calculate_factor_correlations(mdata_or_model) -> pd.DataFrame:
    """
    Calculate correlations between factors.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
        
    Returns
    -------
    pd.DataFrame
        Factor correlation matrix
        
    Examples
    --------
    >>> factor_corr = calculate_factor_correlations(model)
    >>> print(factor_corr)
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    factor_scores = mdata.obsm['X_muvi']
    n_factors = factor_scores.shape[1]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(factor_scores.T)
    
    # Create DataFrame with factor names
    factor_names = [f'Factor_{i}' for i in range(n_factors)]
    corr_df = pd.DataFrame(
        corr_matrix,
        index=factor_names,
        columns=factor_names
    )
    
    return corr_df


def identify_factor_associations(
    mdata_or_model,
    metadata_columns: Optional[List[str]] = None,
    categorical_test: str = 'kruskal',
    continuous_test: str = 'pearson'
) -> pd.DataFrame:
    """
    Identify associations between factors and cell metadata.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
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
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    factor_scores = mdata.obsm['X_muvi']
    n_factors = factor_scores.shape[1]
    
    if metadata_columns is None:
        metadata_columns = list(mdata.obs.columns)
    
    results = []
    
    for factor_idx in range(n_factors):
        factor_values = factor_scores[:, factor_idx]
        
        for col in metadata_columns:
            if col not in mdata.obs.columns:
                continue
            
            metadata_values = mdata.obs[col]
            
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
        _, results_df['p_value_corrected'], _, _ = multipletests(
            results_df['p_value'], method='fdr_bh'
        )
    
    return results_df


def cluster_cells_by_factors(
    mdata_or_model,
    n_clusters: int = 5,
    factors_to_use: Optional[List[int]] = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster cells based on factor scores.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
    n_clusters : int, default 5
        Number of clusters
    factors_to_use : List[int], optional
        Specific factors to use for clustering. If None, uses all factors
    random_state : int, default 42
        Random state for reproducibility
        
    Returns
    -------
    np.ndarray
        Cluster labels for each cell
        
    Examples
    --------
    >>> clusters = cluster_cells_by_factors(model, n_clusters=8)
    >>> model.mdata.obs['factor_clusters'] = clusters
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    factor_scores = mdata.obsm['X_muvi']
    
    if factors_to_use is not None:
        factor_scores = factor_scores[:, factors_to_use]
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(factor_scores)
    
    return cluster_labels


def calculate_factor_distances(
    mdata_or_model,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Calculate pairwise distances between cells in factor space.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
    metric : str, default 'euclidean'
        Distance metric ('euclidean', 'cosine', 'manhattan')
        
    Returns
    -------
    np.ndarray
        Pairwise distance matrix
        
    Examples
    --------
    >>> distances = calculate_factor_distances(model, metric='cosine')
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    factor_scores = mdata.obsm['X_muvi']
    distance_matrix = pairwise_distances(factor_scores, metric=metric)
    
    return distance_matrix


def summarize_factor_activity(
    mdata_or_model,
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Summarize factor activity across cell groups.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    group_by : str, optional
        Column in mdata.obs to group cells by
        
    Returns
    -------
    pd.DataFrame
        Summary of factor activity per group
        
    Examples
    --------
    >>> summary = summarize_factor_activity(mdata_muvi, group_by='cell_type')
    >>> print(summary)
    """
    factor_scores = mdata.obsm['X_muvi']
    n_factors = factor_scores.shape[1]
    factor_names = [f'Factor_{i}' for i in range(n_factors)]
    
    if group_by is None:
        # Overall summary
        summary_data = {
            'factor': factor_names,
            'mean_activity': np.mean(factor_scores, axis=0),
            'std_activity': np.std(factor_scores, axis=0),
            'min_activity': np.min(factor_scores, axis=0),
            'max_activity': np.max(factor_scores, axis=0)
        }
    else:
        # Group-wise summary
        if group_by not in mdata.obs.columns:
            raise ValueError(f"Column '{group_by}' not found in mdata.obs")
        
        groups = mdata.obs[group_by].unique()
        summary_rows = []
        
        for group in groups:
            if pd.isna(group):
                continue
            group_mask = mdata.obs[group_by] == group
            group_scores = factor_scores[group_mask, :]
            
            for factor_idx, factor_name in enumerate(factor_names):
                summary_rows.append({
                    'group': group,
                    'factor': factor_name,
                    'mean_activity': np.mean(group_scores[:, factor_idx]),
                    'std_activity': np.std(group_scores[:, factor_idx]),
                    'n_cells': group_mask.sum()
                })
        
        summary_data = pd.DataFrame(summary_rows)
        return summary_data
    
    return pd.DataFrame(summary_data)