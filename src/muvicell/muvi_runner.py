"""
MuVI runner utilities for MuVIcell.

This module provides wrapper functions for running MuVI (Multi-View Integration)
on preprocessed muon data objects.
"""

import muon as mu
import numpy as np
from typing import Optional, Dict, List, Union, Any
import warnings

try:
    import muvi
    MUVI_AVAILABLE = True
except ImportError:
    MUVI_AVAILABLE = False
    warnings.warn(
        "MuVI package not available. To use MuVI functionality, install it with:\n"
        "pip install muvi\n"
        "Note: MuVI requires Python <3.11. If you're using Python 3.11+, "
        "the mock implementation will be used for demonstration purposes."
    )


def setup_muvi_model(
    mdata: mu.MuData,
    n_factors: int = 10,
    likelihood_per_view: Optional[Dict[str, str]] = None,
    sparsity_inducing: bool = True,
    **model_kwargs
):
    """
    Set up MuVI model for multi-view integration.
    
    Parameters
    ----------
    mdata : mu.MuData
        Preprocessed muon data object
    n_factors : int, default 10
        Number of latent factors
    likelihood_per_view : Dict[str, str], optional
        Likelihood for each view. If None, uses 'normal' for all views
    sparsity_inducing : bool, default True
        Whether to use sparsity-inducing priors
    **model_kwargs
        Additional arguments passed to MuVI model
        
    Returns
    -------
    muvi.MuVI or dict
        Configured MuVI model (or dict if muvi not available)
        
    Examples
    --------
    >>> model = setup_muvi_model(mdata, n_factors=15)
    >>> # With custom likelihoods
    >>> likelihoods = {'rna': 'normal', 'protein': 'normal'}
    >>> model = setup_muvi_model(mdata, likelihood_per_view=likelihoods)
    """
    if not MUVI_AVAILABLE:
        warnings.warn("MuVI not available. Returning mock configuration.")
        return {
            'n_factors': n_factors,
            'likelihood_per_view': likelihood_per_view or {view: 'normal' for view in mdata.mod.keys()},
            'sparsity_inducing': sparsity_inducing,
            **model_kwargs
        }
    
    # Set default likelihoods if not provided
    if likelihood_per_view is None:
        likelihood_per_view = {view: 'normal' for view in mdata.mod.keys()}
    
    # Validate that all views have specified likelihoods
    for view in mdata.mod.keys():
        if view not in likelihood_per_view:
            likelihood_per_view[view] = 'normal'
            warnings.warn(f"No likelihood specified for view '{view}', using 'normal'")
    
    # Create MuVI model
    model = muvi.MuVI(
        n_factors=n_factors,
        likelihood=likelihood_per_view,
        sparsity_inducing=sparsity_inducing,
        **model_kwargs
    )
    
    return model


def run_muvi(
    mdata: mu.MuData,
    n_factors: int = 10,
    n_iterations: int = 1000,
    likelihood_per_view: Optional[Dict[str, str]] = None,
    convergence_tolerance: float = 1e-3,
    verbose: bool = True,
    **muvi_kwargs
) -> mu.MuData:
    """
    Run MuVI analysis on muon data.
    
    Parameters
    ----------
    mdata : mu.MuData
        Preprocessed muon data object
    n_factors : int, default 10
        Number of latent factors
    n_iterations : int, default 1000
        Maximum number of iterations
    likelihood_per_view : Dict[str, str], optional
        Likelihood for each view
    convergence_tolerance : float, default 1e-3
        Convergence tolerance for training
    verbose : bool, default True
        Whether to show training progress
    **muvi_kwargs
        Additional arguments passed to MuVI
        
    Returns
    -------
    mu.MuData
        Muon data object with MuVI results stored in .obsm and .varm
        
    Examples
    --------
    >>> mdata_muvi = run_muvi(mdata, n_factors=15, n_iterations=1500)
    """
    # Set up model
    model = setup_muvi_model(
        mdata, 
        n_factors=n_factors,
        likelihood_per_view=likelihood_per_view,
        **{k: v for k, v in muvi_kwargs.items() if k != 'convergence_tolerance'}
    )
    
    # Check if we have a real MuVI model or mock
    if isinstance(model, dict):
        # Mock implementation
        if verbose:
            print("Using mock MuVI implementation for demonstration.")
        return _create_mock_muvi_results(mdata, n_factors)
    
    # Fit model
    model.fit(
        mdata,
        n_iterations=n_iterations,
        convergence_tolerance=convergence_tolerance,
        verbose=verbose
    )
    
    # Store results in mdata
    mdata_result = mdata.copy()
    
    # Add factor scores (cell embeddings)
    mdata_result.obsm['X_muvi'] = model.get_factor_scores()
    
    # Add factor loadings for each view
    for view_name in mdata.mod.keys():
        loadings = model.get_factor_loadings(view=view_name)
        mdata_result.mod[view_name].varm['muvi_loadings'] = loadings
    
    # Add factor variance explained
    mdata_result.uns['muvi_variance_explained'] = model.get_variance_explained()
    
    # Store model parameters
    mdata_result.uns['muvi_model_params'] = {
        'n_factors': n_factors,
        'n_iterations': n_iterations,
        'likelihood_per_view': likelihood_per_view,
        'convergence_tolerance': convergence_tolerance
    }
    
    return mdata_result


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
    >>> loadings = get_factor_loadings(mdata_muvi, 'rna')
    >>> print(f"RNA loadings shape: {loadings.shape}")
    """
    if view not in mdata.mod:
        raise ValueError(f"View '{view}' not found in muon data.")
    
    if 'muvi_loadings' not in mdata.mod[view].varm:
        raise ValueError(f"No MuVI loadings found for view '{view}'. Run run_muvi() first.")
    
    return mdata.mod[view].varm['muvi_loadings']


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
    >>> print(f"Total variance explained: {var_exp}")
    """
    if 'muvi_variance_explained' not in mdata.uns:
        raise ValueError("No MuVI variance explained found. Run run_muvi() first.")
    
    return mdata.uns['muvi_variance_explained']


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