"""
Synthetic data generation utilities for MuVIcell.

This module provides functions for generating synthetic multi-view data
in muon format for testing and demonstration purposes.

All randomness is controlled by numpy.random.default_rng with a
user-provided random_state.
"""

import muon as mu
import anndata as ad
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings


def generate_synthetic_data(
    n_samples: int = 200,
    view_configs: Optional[Dict[str, Dict]] = None,
    random_state: int = 42
) -> mu.MuData:
    """
    Generate reproducible synthetic multi-view data in muon format.
    
    Parameters
    ----------
    n_samples : int, default 200
        Number of samples to generate (each row = a sample, not individual cells).
    view_configs : Dict[str, Dict], optional
        Configuration for each view. If None, creates default 3 views
        with 5, 10, and 15 features respectively.
    random_state : int, default 42
        Random state for reproducibility.
        
    Returns
    -------
    mu.MuData
        Synthetic muon data object with multiple views.
    """
    rng = np.random.default_rng(random_state)
    
    # Default config
    if view_configs is None:
        view_configs = {
            'Type1': {'n_vars': 5, 'sparsity': 0.3},
            'Type2': {'n_vars': 10, 'sparsity': 0.4}, 
            'Type3': {'n_vars': 15, 'sparsity': 0.5}
        }
    
    # Global obs
    obs_df = pd.DataFrame({
        'sample_id': [f'Sample_{i}' for i in range(n_samples)],
        'batch': rng.choice(['Batch1', 'Batch2'], size=n_samples),
        'total_counts': rng.lognormal(mean=8, sigma=0.5, size=n_samples)
    }, index=[f'Sample_{i}' for i in range(n_samples)])
    
    view_adatas = {}
    for idx, (view_name, config) in enumerate(view_configs.items()):
        n_vars = config.get('n_vars', 10)
        sparsity = config.get('sparsity', 0.4)
        
        # Use a deterministic per-view RNG
        view_rng = np.random.default_rng(random_state + idx)
        
        view_data = _generate_view_data(
            n_samples=n_samples,
            n_vars=n_vars,
            sparsity=sparsity,
            rng=view_rng
        )
        
        var_df = pd.DataFrame({
            'gene_id': [f'ft_{i}' for i in range(n_vars)],
            'gene_name': [f'{view_name}_Gene{i}' for i in range(n_vars)],
            'highly_variable': rng.choice([True, False], size=n_vars, p=[0.3, 0.7])
        }).set_index("gene_id")
        
        adata = ad.AnnData(X=view_data, obs=obs_df.copy(), var=var_df)
        view_adatas[view_name] = adata
    
    return mu.MuData(view_adatas)


# -------------------------------------------------------------------------
# Single-view generator
# -------------------------------------------------------------------------

def _generate_view_data(
    n_samples: int,
    n_vars: int,
    sparsity: float = 0.4,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate synthetic expression data for a single view.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    base_expression = rng.lognormal(mean=2, sigma=1, size=(n_samples, n_vars))
    noise = rng.normal(0, 0.1, size=(n_samples, n_vars))
    expression_data = base_expression * np.exp(noise)
    
    dropout_mask = rng.binomial(1, sparsity, size=(n_samples, n_vars))
    expression_data[dropout_mask.astype(bool)] = 0
    
    return np.maximum(expression_data, 0)



def add_latent_structure(
    mdata: mu.MuData,
    n_latent_factors: int = 5,
    factor_variance: Optional[List[float]] = None,
    structure_strength: float = 1.0,
    baseline_strength: float = 1.0,
    random_state: int = 42
) -> mu.MuData:
    """
    Add realistic latent factor structure to synthetic data.
    """
    rng = np.random.default_rng(random_state)
    
    if factor_variance is None:
        factor_variance = [0.3, 0.2, 0.15, 0.1, 0.05][:n_latent_factors]
    if len(factor_variance) != n_latent_factors:
        raise ValueError("Length of factor_variance must match n_latent_factors")
    
    n_samples = mdata.n_obs
    latent_factors = rng.normal(0, 1, size=(n_samples, n_latent_factors))
    
    mdata_structured = mdata.copy()
    true_factor_loadings = {}
    
    for vidx, (view_name, view_data) in enumerate(mdata_structured.mod.items()):
        n_vars = view_data.n_vars
        factor_loadings = rng.normal(0, 1, size=(n_vars, n_latent_factors))
        
        active_genes = rng.choice(n_vars, size=int(n_vars * 0.7), replace=False)
        mask = np.zeros((n_vars, n_latent_factors), dtype=bool)
        mask[active_genes, :] = True
        factor_loadings[~mask] = 0
        
        for fidx, var in enumerate(factor_variance):
            factor_loadings[:, fidx] *= np.sqrt(var)
        
        true_factor_loadings[view_name] = factor_loadings.copy()
        
        structured_expression = latent_factors @ factor_loadings.T
        current_expression = view_data.X.toarray() if hasattr(view_data.X, 'toarray') else view_data.X
        baseline_scale = np.mean(current_expression, axis=0, keepdims=True)
        
        structured_scaled = structured_expression * baseline_scale
        combined = baseline_strength * current_expression + structure_strength * structured_scaled
        combined = np.maximum(combined, 0)
        
        mdata_structured.mod[view_name].X = combined
        mdata_structured.mod[view_name].varm['true_factor_loadings'] = factor_loadings
    
    mdata_structured.obsm['true_factors'] = latent_factors
    mdata_structured.uns['true_factor_variance'] = factor_variance
    mdata_structured.uns['true_factor_loadings'] = true_factor_loadings
    
    for fidx in range(n_latent_factors):
        mdata_structured.obs[f'sim_factor_{fidx+1}'] = latent_factors[:, fidx]
    
    first_view = list(mdata_structured.mod.keys())[0]
    if 'batch' in mdata_structured.mod[first_view].obs.columns:
        mdata_structured.obs['batch'] = mdata_structured.mod[first_view].obs['batch'].values
    
    return mdata_structured



def generate_batch_effects(
    mdata: mu.MuData,
    batch_column: str = 'batch',
    effect_strength: float = 0.2,
    random_state: int = 42
) -> mu.MuData:
    """
    Add reproducible batch effects to synthetic data.
    """
    if batch_column not in mdata.obs.columns:
        warnings.warn(f"Column '{batch_column}' not found. No batch effects added.")
        return mdata
    
    rng = np.random.default_rng(random_state)
    mdata_batch = mdata.copy()
    batches = mdata.obs[batch_column].unique()
    
    for view_name, view_data in mdata_batch.mod.items():
        current_expression = view_data.X.toarray() if hasattr(view_data.X, 'toarray') else view_data.X
        for batch in batches:
            mask = mdata.obs[batch_column] == batch
            batch_effects = rng.normal(0, effect_strength, size=(mask.sum(), view_data.n_vars))
            current_expression[mask, :] *= np.exp(batch_effects)
        mdata_batch.mod[view_name].X = np.maximum(current_expression, 0)
    
    return mdata_batch


def simulate_missing_data(
    mdata: mu.MuData,
    missing_rate: float = 0.1,
    view_specific_rates: Optional[Dict[str, float]] = None,
    random_state: int = 42
) -> mu.MuData:
    """
    Simulate reproducible missing data in views.
    """
    rng = np.random.default_rng(random_state)
    mdata_missing = mdata.copy()
    
    for view_name, view_data in mdata_missing.mod.items():
        rate = view_specific_rates.get(view_name, missing_rate) if view_specific_rates else missing_rate
        current_expression = view_data.X.toarray() if hasattr(view_data.X, 'toarray') else view_data.X
        missing_mask = rng.binomial(1, rate, size=current_expression.shape)
        current_expression[missing_mask.astype(bool)] = 0
        mdata_missing.mod[view_name].X = current_expression
    
    return mdata_missing