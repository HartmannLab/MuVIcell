"""Test configuration and fixtures for MuVIcell tests."""

import pytest
import numpy as np
import pandas as pd
from muvicell.synthetic import generate_synthetic_data


@pytest.fixture
def sample_mudata():
    """Generate sample MuData for testing."""
    return generate_synthetic_data(
        n_cells=50,
        view_configs={
            'rna': {'n_vars': 20, 'sparsity': 0.3},
            'protein': {'n_vars': 15, 'sparsity': 0.2}
        },
        random_state=42
    )


@pytest.fixture
def processed_mudata(sample_mudata):
    """Generate preprocessed MuData for testing."""
    from muvicell.preprocessing import preprocess_for_muvi
    return preprocess_for_muvi(sample_mudata, subset_hvg=False)


@pytest.fixture
def muvi_results(processed_mudata):
    """Generate MuVI results for testing."""
    from muvicell.muvi_runner import run_muvi
    # Use mock results for testing since MuVI might not be installed
    import muon as mu
    mdata = processed_mudata.copy()
    
    # Add mock MuVI results
    n_cells = mdata.n_obs
    n_factors = 5
    
    # Mock factor scores
    mdata.obsm['X_muvi'] = np.random.normal(0, 1, size=(n_cells, n_factors))
    
    # Mock factor loadings for each view
    for view_name, view_data in mdata.mod.items():
        n_vars = view_data.n_vars
        mdata.mod[view_name].varm['muvi_loadings'] = np.random.normal(
            0, 0.5, size=(n_vars, n_factors)
        )
    
    # Mock variance explained
    mdata.uns['muvi_variance_explained'] = {
        view_name: np.random.uniform(0.05, 0.3, size=n_factors)
        for view_name in mdata.mod.keys()
    }
    
    return mdata