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
    model,
    top_genes_per_factor: int = 50,
    loading_threshold: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Characterize factors by identifying top contributing genes per view.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
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
    # Handle both real MuVI model and mock model
    if hasattr(model, 'get_factor_loadings'):
        # Real MuVI model
        loadings_dict = model.get_factor_loadings()
        view_names = list(loadings_dict.keys())
    else:
        # Mock model - fall back to mdata access
        mdata = model.mdata
        loadings_dict = {}
        view_names = list(mdata.mod.keys())
        
        for view_name in view_names:
            if 'muvi_loadings' in mdata.mod[view_name].varm:
                loadings_dict[view_name] = mdata.mod[view_name].varm['muvi_loadings']
    
    factor_characterization = {}
    
    for view_name in view_names:
        if view_name not in loadings_dict:
            continue
            
        loadings = loadings_dict[view_name]
        
        # Get gene names - try different approaches
        try:
            # For real MuVI model, get from original mdata
            if hasattr(model, 'mdata_original'):
                gene_names = model.mdata_original.mod[view_name].var_names
            elif hasattr(model, 'mdata'):
                gene_names = model.mdata.mod[view_name].var_names
            else:
                # Fallback: create generic gene names
                gene_names = [f"{view_name}_gene_{i}" for i in range(loadings.shape[0])]
        except:
            gene_names = [f"{view_name}_gene_{i}" for i in range(loadings.shape[0])]
        
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


def calculate_factor_correlations(model) -> pd.DataFrame:
    """
    Calculate correlations between factors.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
        
    Returns
    -------
    pd.DataFrame
        Factor correlation matrix
        
    Examples
    --------
    >>> factor_corr = calculate_factor_correlations(model)
    >>> print(factor_corr)
    """
    # Get factor scores from model
    if hasattr(model, 'get_factor_scores'):
        # Real MuVI model
        factor_scores = model.get_factor_scores()
    else:
        # Mock model - fall back to mdata access
        mdata = model.mdata
        factor_scores = mdata.obsm['X_muvi']
    
    # Calculate correlation matrix
    factor_corr = np.corrcoef(factor_scores.T)
    
    # Create DataFrame with proper factor names
    n_factors = factor_scores.shape[1]
    factor_names = [f'Factor_{i}' for i in range(n_factors)]
    
    corr_df = pd.DataFrame(
        factor_corr,
        index=factor_names,
        columns=factor_names
    )
    
    return corr_df


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
    # Get factor scores and metadata
    if hasattr(model, 'get_factor_scores'):
        # Real MuVI model
        factor_scores = model.get_factor_scores()
        # Need to get metadata from original mdata
        if hasattr(model, 'mdata_original'):
            obs_df = model.mdata_original.obs
        else:
            # Create dummy metadata for testing
            obs_df = pd.DataFrame({
                'cell_type': ['TypeA'] * factor_scores.shape[0],
                'condition': ['Control'] * factor_scores.shape[0]
            })
    else:
        # Mock model
        mdata = model.mdata
        factor_scores = mdata.obsm['X_muvi']
    
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
    # Get factor scores
    if hasattr(model, 'get_factor_scores'):
        # Real MuVI model
        factor_scores = model.get_factor_scores()
    else:
        # Mock model
        mdata = model.mdata
        factor_scores = mdata.obsm['X_muvi']
    
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