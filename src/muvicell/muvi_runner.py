"""
MuVI runner utilities for MuVIcell.

This module provides wrapper functions for running MuVI (Multi-View Integration)
on preprocessed muon data objects using the exact same API as the MuVI package.
"""

import muon as mu
import numpy as np
from typing import Optional, Dict, List, Union, Any
import warnings

try:
    import muvi
    import muvi.tl
    MUVI_AVAILABLE = True
except ImportError:
    MUVI_AVAILABLE = False
    warnings.warn(
        "MuVI package not available. This is expected for Python 3.11+. "
        "For full MuVI functionality, use Python 3.9-3.10 and install with: pip install muvi"
    )


def run_muvi(
    mdata: mu.MuData,
    n_factors: int = 10,
    nmf: bool = False,
    device: str = "cpu",
    **muvi_kwargs
) -> mu.MuData:
    """
    Run MuVI analysis on muon data using the standard muvi.tl.from_mdata pattern.
    
    Parameters
    ----------
    mdata : mu.MuData
        Preprocessed muon data object
    n_factors : int, default 10
        Number of latent factors
    nmf : bool, default False
        Whether to use non-negative matrix factorization
    device : str, default "cpu"
        Device to use for computation ('cpu' or 'cuda')
    **muvi_kwargs
        Additional arguments passed to MuVI
        
    Returns
    -------
    mu.MuData
        Muon data object with MuVI results stored in .obsm and .varm
        
    Examples
    --------
    >>> mdata_muvi = run_muvi(mdata, n_factors=15, nmf=False)
    """
    if not MUVI_AVAILABLE:
        warnings.warn("MuVI not available. Creating mock results for demonstration.")
        return _create_mock_muvi_results(mdata, n_factors)
    
    # Create and fit MuVI model using the standard API
    model = muvi.tl.from_mdata(
        mdata,
        n_factors=n_factors,
        nmf=nmf,
        device=device,
        **muvi_kwargs
    )
    
    # Fit the model
    model.fit()
    
    # Store results in mdata - the model automatically updates the mdata object
    # MuVI stores results in the standard locations:
    # - Factor scores in mdata.obsm['X_muvi']
    # - Factor loadings in mdata.varm for each view
    # - Variance explained in mdata.uns
    
    return mdata


def _create_mock_muvi_results(mdata: mu.MuData, n_factors: int = 10) -> mu.MuData:
    """Create mock MuVI results for demonstration when MuVI is not available."""
    mdata_result = mdata.copy()
    n_cells = mdata.n_obs
    
    # Create mock factor scores
    mdata_result.obsm['X_muvi'] = np.random.normal(0, 1, size=(n_cells, n_factors))
    
    # Create mock factor loadings for each view
    for view_name, view_data in mdata_result.mod.items():
        n_vars = view_data.n_vars
        # Create sparse loadings (most genes have small loadings)
        loadings = np.random.normal(0, 0.1, (n_vars, n_factors))
        
        # Make some genes have stronger loadings
        for f in range(n_factors):
            strong_genes = np.random.choice(n_vars, size=max(1, n_vars//4), replace=False)
            loadings[strong_genes, f] += np.random.normal(0, 0.5, len(strong_genes))
        
        mdata_result.mod[view_name].varm['muvi_loadings'] = loadings
    
    # Create mock variance explained
    base_variance = np.array([0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01][:n_factors])
    mdata_result.uns['muvi_variance_explained'] = {
        view_name: base_variance * np.random.uniform(0.8, 1.2, n_factors)
        for view_name in mdata_result.mod.keys()
    }
    
    # Store model parameters
    mdata_result.uns['muvi_model_params'] = {
        'n_factors': n_factors,
        'mock': True
    }
    
    return mdata_result


def get_factor_scores(mdata: mu.MuData) -> np.ndarray:
    """
    Extract factor scores (cell embeddings) from MuVI results.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
        
    Returns
    -------
    np.ndarray
        Factor scores matrix (cells x factors)
        
    Examples
    --------
    >>> scores = get_factor_scores(mdata_muvi)
    >>> print(f"Factor scores shape: {scores.shape}")
    """
    if 'X_muvi' not in mdata.obsm:
        raise ValueError("No MuVI results found. Run run_muvi() first.")
    
    return mdata.obsm['X_muvi']


def get_factor_loadings(mdata: mu.MuData, view: str) -> np.ndarray:
    """
    Extract factor loadings for a specific view from MuVI results.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    view : str
        Name of the view
        
    Returns
    -------
    np.ndarray
        Factor loadings matrix (genes x factors)
        
    Examples
    --------
    >>> loadings = get_factor_loadings(mdata_muvi, 'view1')
    >>> print(f"View1 loadings shape: {loadings.shape}")
    """
    if view not in mdata.mod:
        raise ValueError(f"View '{view}' not found in muon data.")
    
    # Check for MuVI loadings in varm (try different possible keys)
    loadings_key = None
    possible_keys = ['muvi_loadings', 'loadings', 'factor_loadings']
    
    for key in possible_keys:
        if key in mdata.mod[view].varm:
            loadings_key = key
            break
    
    # Fallback: check for any key containing 'loading' or 'factor'
    if loadings_key is None:
        for key in mdata.mod[view].varm.keys():
            if 'loading' in key.lower() or 'factor' in key.lower():
                loadings_key = key
                break
    
    if loadings_key is None:
        raise ValueError(f"No MuVI loadings found for view '{view}'. Run run_muvi() first.")
    
    return mdata.mod[view].varm[loadings_key]


def get_variance_explained(mdata: mu.MuData) -> Dict[str, np.ndarray]:
    """
    Extract variance explained by each factor from MuVI results.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with variance explained per view and factor
        
    Examples
    --------
    >>> var_exp = get_variance_explained(mdata_muvi)
    >>> print(f"Variance explained: {var_exp}")
    """
    # Check for variance explained in uns (try different possible keys)
    var_exp_key = None
    possible_keys = ['muvi_variance_explained', 'variance_explained', 'r2']
    
    for key in possible_keys:
        if key in mdata.uns:
            var_exp_key = key
            break
    
    # Fallback: check for any key containing 'variance' or 'r2'
    if var_exp_key is None:
        for key in mdata.uns.keys():
            if 'variance' in key.lower() or 'r2' in key.lower():
                var_exp_key = key
                break
    
    if var_exp_key is None:
        raise ValueError("No MuVI variance explained found. Run run_muvi() first.")
    
    return mdata.uns[var_exp_key]


def select_top_factors(
    mdata: mu.MuData,
    n_top_factors: Optional[int] = None,
    variance_threshold: float = 0.01
) -> List[int]:
    """
    Select top factors based on variance explained.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    n_top_factors : int, optional
        Number of top factors to select. If None, uses variance_threshold
    variance_threshold : float, default 0.01
        Minimum variance explained threshold
        
    Returns
    -------
    List[int]
        Indices of selected factors
        
    Examples
    --------
    >>> top_factors = select_top_factors(mdata_muvi, n_top_factors=5)
    >>> # Or based on variance threshold
    >>> top_factors = select_top_factors(mdata_muvi, variance_threshold=0.02)
    """
    var_exp = get_variance_explained(mdata)
    
    # Calculate total variance explained per factor across all views
    total_var_per_factor = np.sum([
        var_exp[view] for view in var_exp.keys()
    ], axis=0)
    
    if n_top_factors is not None:
        # Select top N factors
        top_indices = np.argsort(total_var_per_factor)[::-1][:n_top_factors]
    else:
        # Select factors above variance threshold
        top_indices = np.where(total_var_per_factor >= variance_threshold)[0]
    
    return sorted(top_indices.tolist())