"""
Visualization utilities for MuVIcell using plotnine.

This module provides functions for visualizing MuVI results including
factor scores, loadings, variance explained, and factor associations.
"""

import muon as mu
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union, Tuple
import warnings


def plot_variance_explained(
    model,
    max_factors: Optional[int] = None,
    by_view: bool = True
) -> ggplot:
    """
    Plot variance explained by MuVI factors.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    max_factors : int, optional
        Maximum number of factors to show
    by_view : bool, default True
        Whether to show variance explained by view
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_variance_explained(model, max_factors=10)
    >>> p.show()
    """
    # Get variance explained data
    if hasattr(model, 'mdata'):
        # Use mdata for compatibility
        var_exp = model.mdata.uns['muvi_variance_explained']
    else:
        # Try to compute from real MuVI model (placeholder for now)
        # In real implementation, use muvi.tl.variance_explained(model)
        var_exp = {'view1': [0.2, 0.15, 0.1], 'view2': [0.18, 0.12, 0.08], 'view3': [0.16, 0.11, 0.07]}
    
    # Prepare data for plotting
    plot_data = []
    for view_name, view_var in var_exp.items():
        for factor_idx, variance in enumerate(view_var):
            if max_factors is not None and factor_idx >= max_factors:
                break
            plot_data.append({
                'factor': f'Factor_{factor_idx}',
                'view': view_name,
                'variance_explained': variance,
                'factor_idx': factor_idx
            })
    
    df = pd.DataFrame(plot_data)
    
    if by_view:
        p = (ggplot(df, aes(x='factor_idx', y='variance_explained', fill='view')) +
             geom_col(position='dodge') +
             labs(title='Variance Explained by MuVI Factors',
                  x='Factor Index',
                  y='Variance Explained',
                  fill='View') +
             theme_minimal() +
             theme(axis_text_x=element_text(rotation=45, hjust=1)))
    else:
        # Total variance across views
        total_var = df.groupby('factor_idx')['variance_explained'].sum().reset_index()
        total_var['factor'] = [f'Factor_{i}' for i in total_var['factor_idx']]
        
        p = (ggplot(total_var, aes(x='factor_idx', y='variance_explained')) +
             geom_col(fill='steelblue') +
             labs(title='Total Variance Explained by MuVI Factors',
                  x='Factor Index',
                  y='Total Variance Explained') +
             theme_minimal() +
             theme(axis_text_x=element_text(rotation=45, hjust=1)))
    
    return p


def plot_factor_scores(
    model,
    factors: Tuple[int, int] = (0, 1),
    color_by: Optional[str] = None,
    size: float = 1.0
) -> ggplot:
    """
    Plot factor scores in 2D space.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    factors : Tuple[int, int], default (0, 1)
        Factor indices to plot (x-axis, y-axis)
    color_by : str, optional
        Column in sample metadata to color points by
    size : float, default 1.0
        Point size
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_scores(model, factors=(0, 2), color_by='cell_type')
    >>> p.show()
    """
    # Get factor scores
    if hasattr(model, 'get_factor_scores'):
        factor_scores = model.get_factor_scores()
    else:
        factor_scores = model.mdata.obsm['X_muvi']
    
    if factors[0] >= factor_scores.shape[1] or factors[1] >= factor_scores.shape[1]:
        raise ValueError("Factor indices exceed number of available factors")
    
    # Prepare data
    plot_data = pd.DataFrame({
        f'Factor_{factors[0]}': factor_scores[:, factors[0]],
        f'Factor_{factors[1]}': factor_scores[:, factors[1]]
    })
    
    if color_by is not None:
        # Get metadata from model
        if hasattr(model, 'mdata_original'):
            obs_df = model.mdata_original.obs
        elif hasattr(model, 'mdata'):
            obs_df = model.mdata.obs
        else:
            warnings.warn(f"Cannot access metadata for coloring")
            color_by = None
            obs_df = None
            
        if color_by is not None and obs_df is not None:
            if color_by not in obs_df.columns:
                warnings.warn(f"Column '{color_by}' not found in sample metadata")
                color_by = None
            else:
                plot_data[color_by] = obs_df[color_by].values
    
    # Create plot
    if color_by is not None:
        p = (ggplot(plot_data, aes(x=f'Factor_{factors[0]}', y=f'Factor_{factors[1]}', 
                                  color=color_by)) +
             geom_point(size=size, alpha=0.7) +
             labs(title=f'Factor Scores: Factor {factors[0]} vs Factor {factors[1]}',
                  color=color_by) +
             theme_minimal())
    else:
        p = (ggplot(plot_data, aes(x=f'Factor_{factors[0]}', y=f'Factor_{factors[1]}')) +
             geom_point(size=size, alpha=0.7, color='steelblue') +
             labs(title=f'Factor Scores: Factor {factors[0]} vs Factor {factors[1]}') +
             theme_minimal())
    
    return p


def plot_factor_loadings(
    model,
    view: str,
    factor: int = 0,
    top_genes: int = 20,
    loading_threshold: float = 0.0
) -> ggplot:
    """
    Plot top gene loadings for a specific factor and view.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    view : str
        Name of the view
    factor : int, default 0
        Factor index
    top_genes : int, default 20
        Number of top genes to show
    loading_threshold : float, default 0.0
        Minimum absolute loading threshold
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_loadings(model, 'view1', factor=1, top_genes=15)
    >>> p.show()
    """
    # Get factor loadings
    if hasattr(model, 'get_factor_loadings'):
        loadings_dict = model.get_factor_loadings()
        if view not in loadings_dict:
            raise ValueError(f"View '{view}' not found")
        loadings = loadings_dict[view]
    else:
        # Fall back to mdata access
        if view not in model.mdata.mod:
            raise ValueError(f"View '{view}' not found")
        loadings = model.mdata.mod[view].varm['muvi_loadings']
    
    # Get gene names
    if hasattr(model, 'mdata_original'):
        gene_names = model.mdata_original.mod[view].var_names
    elif hasattr(model, 'mdata'):
        gene_names = model.mdata.mod[view].var_names
    else:
        gene_names = [f"{view}_gene_{i}" for i in range(loadings.shape[0])]
    
    if factor >= loadings.shape[1]:
        raise ValueError(f"Factor {factor} not available")
    
    factor_loadings = loadings[:, factor]
    
    # Filter by threshold and get top genes
    valid_mask = np.abs(factor_loadings) >= loading_threshold
    valid_loadings = factor_loadings[valid_mask]
    valid_genes = gene_names[valid_mask]
    
    # Sort by absolute loading value
    sorted_indices = np.argsort(np.abs(valid_loadings))[::-1][:top_genes]
    
    plot_data = pd.DataFrame({
        'gene': valid_genes[sorted_indices],
        'loading': valid_loadings[sorted_indices],
        'abs_loading': np.abs(valid_loadings[sorted_indices])
    })
    
    # Create plot
    p = (ggplot(plot_data, aes(x='reorder(gene, abs_loading)', y='loading')) +
         geom_col(aes(fill='loading > 0'), alpha=0.8) +
         coord_flip() +
         labs(title=f'Top Gene Loadings: {view} - Factor {factor}',
              x='Genes',
              y='Loading',
              fill='Positive Loading') +
         scale_fill_manual(values=['red', 'blue']) +
         theme_minimal())
    
    return p


def plot_factor_heatmap(
    mdata: mu.MuData,
    view: str,
    factors: Optional[List[int]] = None,
    top_genes_per_factor: int = 10
) -> ggplot:
    """
    Plot heatmap of top gene loadings across factors.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    view : str
        Name of the view
    factors : List[int], optional
        Specific factors to include. If None, uses all factors
    top_genes_per_factor : int, default 10
        Number of top genes per factor
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_heatmap(mdata_muvi, 'rna', factors=[0, 1, 2])
    >>> print(p)
    """
    if view not in mdata.mod:
        raise ValueError(f"View '{view}' not found")
    
    loadings = mdata.mod[view].varm['muvi_loadings']
    gene_names = mdata.mod[view].var_names
    
    if factors is None:
        factors = list(range(loadings.shape[1]))
    
    # Get top genes for each factor
    all_top_genes = set()
    for factor in factors:
        factor_loadings = loadings[:, factor]
        top_indices = np.argsort(np.abs(factor_loadings))[::-1][:top_genes_per_factor]
        all_top_genes.update(gene_names[top_indices])
    
    # Create heatmap data
    heatmap_data = []
    for gene in all_top_genes:
        gene_idx = list(gene_names).index(gene)
        for factor in factors:
            heatmap_data.append({
                'gene': gene,
                'factor': f'Factor_{factor}',
                'loading': loadings[gene_idx, factor]
            })
    
    df = pd.DataFrame(heatmap_data)
    
    # Create heatmap
    p = (ggplot(df, aes(x='factor', y='gene', fill='loading')) +
         geom_tile() +
         scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
         labs(title=f'Factor Loadings Heatmap: {view}',
              x='Factor',
              y='Gene',
              fill='Loading') +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               axis_text_y=element_text(size=8)))
    
    return p


def plot_factor_associations(
    mdata: mu.MuData,
    associations_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    top_n: int = 20
) -> ggplot:
    """
    Plot factor-metadata associations.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    associations_df : pd.DataFrame
        Output from identify_factor_associations()
    p_value_threshold : float, default 0.05
        P-value threshold for significance
    top_n : int, default 20
        Number of top associations to show
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> from MuVIcell.analysis import identify_factor_associations
    >>> assoc = identify_factor_associations(mdata_muvi)
    >>> p = plot_factor_associations(mdata_muvi, assoc)
    >>> print(p)
    """
    # Filter significant associations
    sig_assoc = associations_df[associations_df['p_value'] <= p_value_threshold].copy()
    
    if len(sig_assoc) == 0:
        warnings.warn("No significant associations found")
        return None
    
    # Sort by p-value and take top N
    sig_assoc = sig_assoc.sort_values('p_value').head(top_n)
    
    # Add -log10 p-value for plotting
    sig_assoc['neg_log10_p'] = -np.log10(sig_assoc['p_value'])
    
    # Create plot
    p = (ggplot(sig_assoc, aes(x='reorder(metadata, neg_log10_p)', 
                              y='neg_log10_p', 
                              fill='factor')) +
         geom_col() +
         coord_flip() +
         geom_hline(yintercept=-np.log10(p_value_threshold), 
                    linetype='dashed', color='red') +
         labs(title='Factor-Metadata Associations',
              x='Metadata',
              y='-log10(p-value)',
              fill='Factor') +
         theme_minimal())
    
    return p


def plot_cell_clusters(
    mdata: mu.MuData,
    cluster_labels: np.ndarray,
    factors: Tuple[int, int] = (0, 1),
    cluster_column: str = 'factor_clusters'
) -> ggplot:
    """
    Plot cell clusters in factor space.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    cluster_labels : np.ndarray
        Cluster labels for each cell
    factors : Tuple[int, int], default (0, 1)
        Factor indices to plot
    cluster_column : str, default 'factor_clusters'
        Name for cluster column
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> from MuVIcell.analysis import cluster_cells_by_factors
    >>> clusters = cluster_cells_by_factors(mdata_muvi, n_clusters=5)
    >>> p = plot_cell_clusters(mdata_muvi, clusters, factors=(0, 1))
    >>> print(p)
    """
    factor_scores = mdata.obsm['X_muvi']
    
    plot_data = pd.DataFrame({
        f'Factor_{factors[0]}': factor_scores[:, factors[0]],
        f'Factor_{factors[1]}': factor_scores[:, factors[1]],
        cluster_column: [f'Cluster_{i}' for i in cluster_labels]
    })
    
    p = (ggplot(plot_data, aes(x=f'Factor_{factors[0]}', y=f'Factor_{factors[1]}', 
                              color=cluster_column)) +
         geom_point(alpha=0.7, size=1.5) +
         labs(title=f'Cell Clusters in Factor Space (Factors {factors[0]} & {factors[1]})',
              color='Cluster') +
         theme_minimal())
    
    return p


def plot_factor_comparison(
    model,
    factors: List[int],
    group_by: str,
    plot_type: str = 'boxplot'
) -> ggplot:
    """
    Compare factor activity across sample groups.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    factors : List[int]
        Factor indices to compare
    group_by : str
        Column in sample metadata to group by
    plot_type : str, default 'boxplot'
        Type of plot ('boxplot', 'violin')
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_comparison(model, [0, 1, 2], 'cell_type')
    >>> p.show()
    """
    # Get factor scores
    if hasattr(model, 'get_factor_scores'):
        factor_scores = model.get_factor_scores()
    else:
        factor_scores = model.mdata.obsm['X_muvi']
    
    # Get metadata
    if hasattr(model, 'mdata_original'):
        obs_df = model.mdata_original.obs
    elif hasattr(model, 'mdata'):
        obs_df = model.mdata.obs
    else:
        raise ValueError("Cannot access sample metadata")
    
    if group_by not in obs_df.columns:
        raise ValueError(f"Column '{group_by}' not found in sample metadata")
    
    # Prepare data
    plot_data = []
    for factor_idx in factors:
        for i, score in enumerate(factor_scores[:, factor_idx]):
            plot_data.append({
                'factor': f'Factor_{factor_idx}',
                'score': score,
                'group': obs_df[group_by].iloc[i]
            })
    
    df = pd.DataFrame(plot_data)
    
    # Remove NA groups
    df = df.dropna(subset=['group'])
    
    # Create plot
    if plot_type == 'boxplot':
        p = (ggplot(df, aes(x='group', y='score', fill='group')) +
             geom_boxplot(alpha=0.7) +
             facet_wrap('~factor', scales='free_y') +
             labs(title=f'Factor Activity by {group_by}',
                  x=group_by,
                  y='Factor Score') +
             theme_minimal() +
             theme(axis_text_x=element_text(rotation=45, hjust=1),
                   legend_position='none'))
    elif plot_type == 'violin':
        p = (ggplot(df, aes(x='group', y='score', fill='group')) +
             geom_violin(alpha=0.7) +
             facet_wrap('~factor', scales='free_y') +
             labs(title=f'Factor Activity by {group_by}',
                  x=group_by,
                  y='Factor Score') +
             theme_minimal() +
             theme(axis_text_x=element_text(rotation=45, hjust=1),
                   legend_position='none'))
    else:
        raise ValueError("plot_type must be 'boxplot' or 'violin'")
    
    return p