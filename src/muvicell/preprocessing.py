"""
Preprocessing utilities for MuVIcell.

This module provides functions for preprocessing multi-view data
before MuVI analysis, including normalization and filtering.
"""

import muon as mu
import scanpy as sc
import numpy as np
from typing import Optional, Dict, List, Union
import warnings


def normalize_views(
    mdata: mu.MuData,
    view_configs: Optional[Dict[str, Dict]] = None,
    log_transform: bool = True,
    scale: bool = True
) -> mu.MuData:
    """
    Normalize data in each view of the muon object.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with multiple views
    view_configs : Dict[str, Dict], optional
        View-specific normalization configurations
    log_transform : bool, default True
        Whether to apply log transformation
    scale : bool, default True  
        Whether to scale data to unit variance
        
    Returns
    -------
    mu.MuData
        Normalized muon data object
        
    Examples
    --------
    >>> mdata_norm = normalize_views(mdata)
    >>> # With custom configurations
    >>> configs = {'rna': {'log_transform': True}, 'protein': {'log_transform': False}}
    >>> mdata_norm = normalize_views(mdata, view_configs=configs)
    """
    mdata_norm = mdata.copy()
    
    for view_name, view_data in mdata_norm.mod.items():
        # Get view-specific config or use defaults
        if view_configs and view_name in view_configs:
            config = view_configs[view_name]
            view_log = config.get('log_transform', log_transform)
            view_scale = config.get('scale', scale)
        else:
            view_log = log_transform
            view_scale = scale
        
        # Store raw data
        view_data.raw = view_data
        
        # Normalize total counts per cell
        sc.pp.normalize_total(view_data, target_sum=1e4)
        
        # Log transform if specified
        if view_log:
            sc.pp.log1p(view_data)
        
        # Scale to unit variance if specified
        if view_scale:
            sc.pp.scale(view_data, max_value=10)
    
    return mdata_norm


def filter_views(
    mdata: mu.MuData,
    min_cells_per_gene: int = 3,
    min_genes_per_cell: int = 200,
    max_genes_per_cell: Optional[int] = None,
    view_specific_filters: Optional[Dict[str, Dict]] = None
) -> mu.MuData:
    """
    Filter cells and genes in each view.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object
    min_cells_per_gene : int, default 3
        Minimum number of cells expressing a gene
    min_genes_per_cell : int, default 200
        Minimum number of genes per cell
    max_genes_per_cell : int, optional
        Maximum number of genes per cell
    view_specific_filters : Dict[str, Dict], optional
        View-specific filtering parameters
        
    Returns
    -------
    mu.MuData
        Filtered muon data object
        
    Examples
    --------
    >>> mdata_filt = filter_views(mdata, min_cells_per_gene=5)
    """
    mdata_filt = mdata.copy()
    
    for view_name, view_data in mdata_filt.mod.items():
        # Get view-specific filters or use defaults
        if view_specific_filters and view_name in view_specific_filters:
            filters = view_specific_filters[view_name]
            min_cells = filters.get('min_cells_per_gene', min_cells_per_gene)
            min_genes = filters.get('min_genes_per_cell', min_genes_per_cell)
            max_genes = filters.get('max_genes_per_cell', max_genes_per_cell)
        else:
            min_cells = min_cells_per_gene
            min_genes = min_genes_per_cell
            max_genes = max_genes_per_cell
        
        # Filter genes
        sc.pp.filter_genes(view_data, min_cells=min_cells)
        
        # Filter cells
        sc.pp.filter_cells(view_data, min_genes=min_genes)
        if max_genes is not None:
            sc.pp.filter_cells(view_data, max_genes=max_genes)
    
    return mdata_filt


def find_highly_variable_genes(
    mdata: mu.MuData,
    n_top_genes: int = 2000,
    view_specific_n_genes: Optional[Dict[str, int]] = None
) -> mu.MuData:
    """
    Find highly variable genes in each view.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object
    n_top_genes : int, default 2000
        Number of highly variable genes to select
    view_specific_n_genes : Dict[str, int], optional
        View-specific number of genes to select
        
    Returns
    -------
    mu.MuData
        Muon data object with highly variable genes identified
        
    Examples
    --------
    >>> mdata_hvg = find_highly_variable_genes(mdata, n_top_genes=1500)
    """
    mdata_hvg = mdata.copy()
    
    for view_name, view_data in mdata_hvg.mod.items():
        # Get view-specific number of genes or use default
        if view_specific_n_genes and view_name in view_specific_n_genes:
            n_genes = view_specific_n_genes[view_name]
        else:
            n_genes = n_top_genes
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(
            view_data,
            n_top_genes=n_genes,
            subset=False  # Don't subset yet, just mark
        )
    
    return mdata_hvg


def subset_to_hvg(mdata: mu.MuData) -> mu.MuData:
    """
    Subset each view to highly variable genes.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with highly variable genes identified
        
    Returns
    -------
    mu.MuData
        Muon data object subsetted to highly variable genes
        
    Examples
    --------
    >>> mdata_hvg = find_highly_variable_genes(mdata)
    >>> mdata_subset = subset_to_hvg(mdata_hvg)
    """
    mdata_subset = mdata.copy()
    
    for view_name, view_data in mdata_subset.mod.items():
        if 'highly_variable' in view_data.var.columns:
            # Subset to highly variable genes
            view_data = view_data[:, view_data.var.highly_variable].copy()
            mdata_subset.mod[view_name] = view_data
        else:
            warnings.warn(
                f"No highly variable genes found in view '{view_name}'. "
                "Run find_highly_variable_genes() first."
            )
    
    return mdata_subset


def preprocess_for_muvi(
    mdata: mu.MuData,
    filter_cells: bool = True,
    filter_genes: bool = True,
    normalize: bool = True,
    find_hvg: bool = True,
    subset_hvg: bool = True,
    **kwargs
) -> mu.MuData:
    """
    Complete preprocessing pipeline for MuVI analysis.
    
    Parameters
    ----------
    mdata : mu.MuData
        Raw muon data object
    filter_cells : bool, default True
        Whether to filter cells
    filter_genes : bool, default True
        Whether to filter genes
    normalize : bool, default True
        Whether to normalize data
    find_hvg : bool, default True
        Whether to find highly variable genes
    subset_hvg : bool, default True
        Whether to subset to highly variable genes
    **kwargs
        Additional arguments passed to individual preprocessing functions
        
    Returns
    -------
    mu.MuData
        Preprocessed muon data object ready for MuVI analysis
        
    Examples
    --------
    >>> mdata_processed = preprocess_for_muvi(mdata_raw)
    """
    mdata_processed = mdata.copy()
    
    if filter_cells or filter_genes:
        mdata_processed = filter_views(
            mdata_processed,
            **{k: v for k, v in kwargs.items() if k in [
                'min_cells_per_gene', 'min_genes_per_cell', 'max_genes_per_cell',
                'view_specific_filters'
            ]}
        )
    
    if normalize:
        mdata_processed = normalize_views(
            mdata_processed,
            **{k: v for k, v in kwargs.items() if k in [
                'view_configs', 'log_transform', 'scale'
            ]}
        )
    
    if find_hvg:
        mdata_processed = find_highly_variable_genes(
            mdata_processed,
            **{k: v for k, v in kwargs.items() if k in [
                'n_top_genes', 'view_specific_n_genes'
            ]}
        )
    
    if subset_hvg:
        mdata_processed = subset_to_hvg(mdata_processed)
    
    return mdata_processed