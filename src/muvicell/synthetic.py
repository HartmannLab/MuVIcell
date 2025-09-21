"""
Synthetic data generation utilities for MuVIcell.

This module provides functions for generating synthetic multi-view data
in muon format for testing and demonstration purposes.
"""

import muon as mu
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings


def generate_synthetic_data(
    n_cells: int = 200,
    view_configs: Optional[Dict[str, Dict]] = None,
    random_state: int = 42
) -> mu.MuData:
    """
    Generate synthetic multi-view data in muon format.
    
    Parameters
    ----------
    n_cells : int, default 200
        Number of cells to generate
    view_configs : Dict[str, Dict], optional
        Configuration for each view. If None, creates default 3 views
        with 5, 10, and 15 features respectively
    random_state : int, default 42
        Random state for reproducibility
        
    Returns
    -------
    mu.MuData
        Synthetic muon data object with multiple views
        
    Examples
    --------
    >>> # Generate default synthetic data
    >>> mdata = generate_synthetic_data(n_cells=300)
    >>> 
    >>> # Generate custom synthetic data
    >>> configs = {
    ...     'rna': {'n_vars': 100, 'sparsity': 0.7},
    ...     'protein': {'n_vars': 50, 'sparsity': 0.3}
    ... }
    >>> mdata = generate_synthetic_data(n_cells=200, view_configs=configs)
    """
    np.random.seed(random_state)
    
    # Default configuration if none provided
    if view_configs is None:
        view_configs = {
            'view1': {'n_vars': 5, 'sparsity': 0.3},
            'view2': {'n_vars': 10, 'sparsity': 0.4}, 
            'view3': {'n_vars': 15, 'sparsity': 0.5}
        }
    
    # Generate shared cell metadata
    cell_types = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=n_cells, p=[0.4, 0.35, 0.25])
    conditions = np.random.choice(['Control', 'Treatment'], size=n_cells, p=[0.6, 0.4])
    
    obs_df = pd.DataFrame({
        'cell_id': [f'Cell_{i}' for i in range(n_cells)],
        'cell_type': cell_types,
        'condition': conditions,
        'batch': np.random.choice(['Batch1', 'Batch2'], size=n_cells),
        'total_counts': np.random.lognormal(mean=8, sigma=0.5, size=n_cells)
    })
    obs_df.index = obs_df['cell_id']
    
    # Generate views
    view_adatas = {}
    
    for view_name, config in view_configs.items():
        n_vars = config.get('n_vars', 10)
        sparsity = config.get('sparsity', 0.4)
        
        # Generate synthetic expression data
        view_data = _generate_view_data(
            n_cells=n_cells,
            n_vars=n_vars,
            cell_types=cell_types,
            conditions=conditions,
            sparsity=sparsity,
            random_state=random_state + hash(view_name) % 1000
        )
        
        # Create var dataframe
        var_df = pd.DataFrame({
            'gene_id': [f'{view_name}_gene_{i}' for i in range(n_vars)],
            'gene_name': [f'{view_name}_Gene{i}' for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], size=n_vars, p=[0.3, 0.7])
        })
        var_df.index = var_df['gene_id']
        
        # Create AnnData object
        adata = ad.AnnData(
            X=view_data,
            obs=obs_df.copy(),
            var=var_df
        )
        
        view_adatas[view_name] = adata
    
    # Create MuData object
    mdata = mu.MuData(view_adatas)
    
    return mdata


def _generate_view_data(
    n_cells: int,
    n_vars: int,
    cell_types: np.ndarray,
    conditions: np.ndarray,
    sparsity: float = 0.4,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic expression data for a single view with realistic structure.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_vars : int
        Number of variables/genes
    cell_types : np.ndarray
        Cell type labels
    conditions : np.ndarray
        Condition labels
    sparsity : float, default 0.4
        Fraction of zeros in the data
    random_state : int, default 42
        Random state for reproducibility
        
    Returns
    -------
    np.ndarray
        Synthetic expression matrix (cells x genes)
    """
    np.random.seed(random_state)
    
    # Create base expression levels
    base_expression = np.random.lognormal(mean=2, sigma=1, size=(n_cells, n_vars))
    
    # Add cell type effects
    unique_cell_types = np.unique(cell_types)
    for i, cell_type in enumerate(unique_cell_types):
        cell_mask = cell_types == cell_type
        
        # Some genes are differentially expressed in this cell type
        de_genes = np.random.choice(n_vars, size=max(1, n_vars // 4), replace=False)
        effect_size = np.random.normal(0, 0.5, size=len(de_genes))
        
        for j, gene_idx in enumerate(de_genes):
            base_expression[cell_mask, gene_idx] *= np.exp(effect_size[j])
    
    # Add condition effects
    unique_conditions = np.unique(conditions)
    for i, condition in enumerate(unique_conditions):
        if condition == 'Control':
            continue  # No effect for control
        
        condition_mask = conditions == condition
        
        # Some genes are differentially expressed in this condition
        de_genes = np.random.choice(n_vars, size=max(1, n_vars // 6), replace=False)
        effect_size = np.random.normal(0, 0.3, size=len(de_genes))
        
        for j, gene_idx in enumerate(de_genes):
            base_expression[condition_mask, gene_idx] *= np.exp(effect_size[j])
    
    # Add noise
    noise = np.random.normal(0, 0.1, size=(n_cells, n_vars))
    expression_data = base_expression * np.exp(noise)
    
    # Introduce sparsity
    dropout_mask = np.random.binomial(1, sparsity, size=(n_cells, n_vars))
    expression_data[dropout_mask.astype(bool)] = 0
    
    # Ensure non-negative values
    expression_data = np.maximum(expression_data, 0)
    
    return expression_data


def add_latent_structure(
    mdata: mu.MuData,
    n_latent_factors: int = 5,
    factor_variance: List[float] = None
) -> mu.MuData:
    """
    Add realistic latent factor structure to synthetic data.
    
    Parameters
    ----------
    mdata : mu.MuData
        Synthetic muon data object
    n_latent_factors : int, default 5
        Number of latent factors to simulate
    factor_variance : List[float], optional
        Variance explained by each factor. If None, uses decreasing variance
        
    Returns
    -------
    mu.MuData
        Modified muon data object with latent structure
        
    Examples
    --------
    >>> mdata = generate_synthetic_data()
    >>> mdata_structured = add_realistic_structure(mdata, n_latent_factors=3)
    """
    if factor_variance is None:
        # Decreasing variance explained
        factor_variance = [0.3, 0.2, 0.15, 0.1, 0.05][:n_latent_factors]
    
    if len(factor_variance) != n_latent_factors:
        raise ValueError("Length of factor_variance must match n_latent_factors")
    
    n_cells = mdata.n_obs
    
    # Generate latent factors (cell loadings)
    latent_factors = np.random.normal(0, 1, size=(n_cells, n_latent_factors))
    
    # Add factor structure to each view
    mdata_structured = mdata.copy()
    
    for view_name, view_data in mdata_structured.mod.items():
        n_vars = view_data.n_vars
        
        # Generate factor loadings (gene weights)
        factor_loadings = np.random.normal(0, 1, size=(n_vars, n_latent_factors))
        
        # Some genes are not associated with any factor
        active_genes = np.random.choice(
            n_vars, 
            size=int(n_vars * 0.7),  # 70% of genes are active
            replace=False
        )
        factor_mask = np.zeros((n_vars, n_latent_factors), dtype=bool)
        factor_mask[active_genes, :] = True
        factor_loadings[~factor_mask] = 0
        
        # Apply factor variance scaling
        for factor_idx, variance in enumerate(factor_variance):
            factor_loadings[:, factor_idx] *= np.sqrt(variance)
        
        # Generate structured expression
        structured_expression = latent_factors @ factor_loadings.T
        
        # Add to existing expression (scaled down to avoid overwhelming signal)
        current_expression = view_data.X.toarray() if hasattr(view_data.X, 'toarray') else view_data.X
        combined_expression = current_expression + 0.5 * structured_expression
        
        # Ensure non-negative
        combined_expression = np.maximum(combined_expression, 0)
        
        # Update the view data
        mdata_structured.mod[view_name].X = combined_expression
    
    # Store true latent factors for evaluation
    mdata_structured.obsm['true_factors'] = latent_factors
    mdata_structured.uns['true_factor_variance'] = factor_variance
    
    # Add shared metadata columns for easier access
    # Extract from the first view (without prefixes)
    first_view = list(mdata_structured.mod.keys())[0]
    first_view_obs = mdata_structured.mod[first_view].obs
    
    # Add shared metadata columns to the main obs
    if 'cell_type' in first_view_obs.columns:
        mdata_structured.obs['cell_type'] = first_view_obs['cell_type'].values
    if 'condition' in first_view_obs.columns:
        mdata_structured.obs['condition'] = first_view_obs['condition'].values
    if 'batch' in first_view_obs.columns:
        mdata_structured.obs['batch'] = first_view_obs['batch'].values
    
    return mdata_structured


def generate_batch_effects(
    mdata: mu.MuData,
    batch_column: str = 'batch',
    effect_strength: float = 0.2
) -> mu.MuData:
    """
    Add batch effects to synthetic data.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object
    batch_column : str, default 'batch'
        Column name containing batch information
    effect_strength : float, default 0.2
        Strength of batch effects
        
    Returns
    -------
    mu.MuData
        Muon data object with batch effects
        
    Examples
    --------
    >>> mdata = generate_synthetic_data()
    >>> mdata_batch = generate_batch_effects(mdata, effect_strength=0.3)
    """
    if batch_column not in mdata.obs.columns:
        warnings.warn(f"Column '{batch_column}' not found. No batch effects added.")
        return mdata
    
    mdata_batch = mdata.copy()
    batches = mdata.obs[batch_column].unique()
    
    for view_name, view_data in mdata_batch.mod.items():
        current_expression = view_data.X.toarray() if hasattr(view_data.X, 'toarray') else view_data.X
        
        for batch in batches:
            batch_mask = mdata.obs[batch_column] == batch
            
            # Generate batch-specific effects
            batch_effects = np.random.normal(
                0, effect_strength, 
                size=(batch_mask.sum(), view_data.n_vars)
            )
            
            # Apply multiplicative batch effects
            current_expression[batch_mask, :] *= np.exp(batch_effects)
        
        # Ensure non-negative
        current_expression = np.maximum(current_expression, 0)
        mdata_batch.mod[view_name].X = current_expression
    
    return mdata_batch


def simulate_missing_data(
    mdata: mu.MuData,
    missing_rate: float = 0.1,
    view_specific_rates: Optional[Dict[str, float]] = None
) -> mu.MuData:
    """
    Simulate missing data in views.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object
    missing_rate : float, default 0.1
        Global missing data rate
    view_specific_rates : Dict[str, float], optional
        View-specific missing rates
        
    Returns
    -------
    mu.MuData
        Muon data object with missing data
        
    Examples
    --------
    >>> mdata = generate_synthetic_data()
    >>> mdata_missing = simulate_missing_data(mdata, missing_rate=0.15)
    """
    mdata_missing = mdata.copy()
    
    for view_name, view_data in mdata_missing.mod.items():
        # Get view-specific missing rate
        if view_specific_rates and view_name in view_specific_rates:
            rate = view_specific_rates[view_name]
        else:
            rate = missing_rate
        
        current_expression = view_data.X.toarray() if hasattr(view_data.X, 'toarray') else view_data.X
        
        # Generate missing data mask
        missing_mask = np.random.binomial(1, rate, size=current_expression.shape)
        
        # Set missing values to 0
        current_expression[missing_mask.astype(bool)] = 0
        
        mdata_missing.mod[view_name].X = current_expression
    
    return mdata_missing