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
    n_factors: int = 3,
    nmf: bool = False,
    device: str = "cpu",
    **muvi_kwargs
):
    """
    Run MuVI analysis on muon data using the standard muvi.tl.from_mdata pattern.
    
    Parameters
    ----------
    mdata : mu.MuData
        Preprocessed muon data object
    n_factors : int, default 3
        Number of latent factors (should match number of true factors in synthetic data)
    nmf : bool, default False
        Whether to use non-negative matrix factorization
    device : str, default "cpu"
        Device to use for computation ('cpu' or 'cuda')
    **muvi_kwargs
        Additional arguments passed to MuVI
        
    Returns
    -------
    model
        Trained MuVI model object (or mock model for demonstration)
        
    Examples
    --------
    >>> model = run_muvi(mdata, n_factors=3, nmf=False)
    >>> model.fit()
    """
    if not MUVI_AVAILABLE:
        warnings.warn("MuVI not available. Creating mock results for demonstration.")
        return _create_mock_muvi_model(mdata, n_factors)
    
    # Create MuVI model using the standard API
    model = muvi.tl.from_mdata(
        mdata,
        n_factors=n_factors,
        nmf=nmf,
        device=device,
        **muvi_kwargs
    )
    
    # Store reference to original mdata for access to metadata and gene names
    model.mdata_original = mdata
    
    return model


def _create_mock_muvi_model(mdata: mu.MuData, n_factors: int = 3):
    """Create mock MuVI model for demonstration when MuVI is not available."""
    
    class MockMuVIModel:
        def __init__(self, mdata, n_factors):
            self.mdata = mdata
            self.mdata_original = mdata
            self.n_factors = n_factors
            self.fitted = False
            
        def fit(self, **kwargs):
            """Fit the mock model and create realistic results."""
            n_samples = self.mdata.n_obs
            
            # Create mock factor scores with realistic structure
            np.random.seed(42)
            self.factor_scores = np.random.normal(0, 1, size=(n_samples, self.n_factors))
            
            # Create mock factor loadings for each view  
            self.factor_loadings = {}
            for view_name, view_data in self.mdata.mod.items():
                n_vars = view_data.n_vars
                
                # Create realistic sparse loadings
                loadings = np.random.normal(0, 0.05, (n_vars, self.n_factors))
                
                # Make each factor strongly load on different sets of genes
                for f in range(self.n_factors):
                    # Each factor gets 20-40% of genes with strong loadings
                    n_strong = max(1, int(n_vars * np.random.uniform(0.2, 0.4)))
                    strong_genes = np.random.choice(n_vars, size=n_strong, replace=False)
                    loadings[strong_genes, f] += np.random.normal(0, 0.3, len(strong_genes))
                
                self.factor_loadings[view_name] = loadings
                
                # Store in mdata for compatibility
                self.mdata.mod[view_name].varm['muvi_loadings'] = loadings
            
            # Store factor scores in mdata for compatibility
            self.mdata.obsm['X_muvi'] = self.factor_scores
            
            # Create realistic variance explained
            base_variance = np.array([0.25, 0.15, 0.10][:self.n_factors])
            self.mdata.uns['muvi_variance_explained'] = {
                view_name: base_variance * np.random.uniform(0.9, 1.1, self.n_factors)
                for view_name in self.mdata.mod.keys()
            }
            
            self.fitted = True
            return self
            
        def get_factor_scores(self):
            """Get factor scores - mimics real MuVI API."""
            if not self.fitted:
                raise ValueError("Model must be fitted first")
            return self.factor_scores
            
        def get_factor_loadings(self):
            """Get factor loadings - mimics real MuVI API."""
            if not self.fitted:
                raise ValueError("Model must be fitted first")
            return self.factor_loadings
            
    return MockMuVIModel(mdata, n_factors)


def get_factor_scores(mdata_or_model) -> np.ndarray:
    """
    Extract factor scores (cell embeddings) from MuVI results.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
        
    Returns
    -------
    np.ndarray
        Factor scores matrix (cells x factors)
        
    Examples
    --------
    >>> scores = get_factor_scores(model.mdata)
    >>> print(f"Factor scores shape: {scores.shape}")
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
    if 'X_muvi' not in mdata.obsm:
        raise ValueError("No MuVI results found. Run model.fit() first.")
    
    return mdata.obsm['X_muvi']


def get_factor_loadings(mdata_or_model, view: str) -> np.ndarray:
    """
    Extract factor loadings for a specific view from MuVI results.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
    view : str
        Name of the view
        
    Returns
    -------
    np.ndarray
        Factor loadings matrix (genes x factors)
        
    Examples
    --------
    >>> loadings = get_factor_loadings(model.mdata, 'view1')
    >>> print(f"View1 loadings shape: {loadings.shape}")
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
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
        raise ValueError(f"No MuVI loadings found for view '{view}'. Run model.fit() first.")
    
    return mdata.mod[view].varm[loadings_key]


def get_variance_explained(mdata_or_model) -> Dict[str, np.ndarray]:
    """
    Extract variance explained by each factor from MuVI results.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with variance explained per view and factor
        
    Examples
    --------
    >>> var_exp = get_variance_explained(model.mdata)
    >>> print(f"Variance explained: {var_exp}")
    """
    # Handle both mdata and model objects
    if hasattr(mdata_or_model, 'mdata'):
        mdata = mdata_or_model.mdata
    else:
        mdata = mdata_or_model
        
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
        raise ValueError("No MuVI variance explained found. Run model.fit() first.")
    
    return mdata.uns[var_exp_key]


def select_top_factors(
    mdata_or_model,
    n_top_factors: Optional[int] = None,
    variance_threshold: float = 0.01
) -> List[int]:
    """
    Select top factors based on variance explained.
    
    Parameters
    ----------
    mdata_or_model : mu.MuData or MuVI model
        Muon data object with MuVI results or fitted MuVI model
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
    >>> top_factors = select_top_factors(model.mdata, n_top_factors=5)
    >>> # Or based on variance threshold
    >>> top_factors = select_top_factors(model.mdata, variance_threshold=0.02)
    """
    var_exp = get_variance_explained(mdata_or_model)
    
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