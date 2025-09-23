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
