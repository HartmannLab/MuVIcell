"""Tests for MuVIcell synthetic data generation."""

import pytest
import numpy as np
import pandas as pd
import muon as mu
from muvicell.synthetic import (
    generate_synthetic_data,
    add_realistic_structure,
    generate_batch_effects,
    simulate_missing_data
)


class TestSyntheticDataGeneration:
    """Test synthetic data generation functions."""
    
    def test_generate_default_synthetic_data(self):
        """Test generation of default synthetic data."""
        mdata = generate_synthetic_data(n_cells=100)
        
        # Check basic structure
        assert isinstance(mdata, mu.MuData)
        assert mdata.n_obs == 100
        assert len(mdata.mod) == 3  # Default 3 views
        
        # Check view dimensions
        expected_vars = [5, 10, 15]
        for i, (view_name, view_data) in enumerate(mdata.mod.items()):
            assert view_data.n_vars == expected_vars[i]
            assert view_data.n_obs == 100
    
    def test_generate_custom_synthetic_data(self):
        """Test generation of custom synthetic data."""
        view_configs = {
            'rna': {'n_vars': 50, 'sparsity': 0.4},
            'protein': {'n_vars': 25, 'sparsity': 0.2}
        }
        
        mdata = generate_synthetic_data(
            n_cells=200,
            view_configs=view_configs
        )
        
        assert mdata.n_obs == 200
        assert len(mdata.mod) == 2
        assert mdata.mod['rna'].n_vars == 50
        assert mdata.mod['protein'].n_vars == 25
    
    def test_synthetic_data_metadata(self):
        """Test that synthetic data contains expected metadata."""
        mdata = generate_synthetic_data(n_cells=50)
        
        # Check observation metadata
        required_obs_cols = ['cell_type', 'condition', 'batch', 'total_counts']
        for col in required_obs_cols:
            assert col in mdata.obs.columns
        
        # Check variable metadata for each view
        for view_name, view_data in mdata.mod.items():
            required_var_cols = ['gene_name', 'highly_variable']
            for col in required_var_cols:
                assert col in view_data.var.columns
    
    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible."""
        mdata1 = generate_synthetic_data(n_cells=100, random_state=42)
        mdata2 = generate_synthetic_data(n_cells=100, random_state=42)
        
        # Check that data is identical
        for view_name in mdata1.mod.keys():
            np.testing.assert_array_equal(
                mdata1.mod[view_name].X,
                mdata2.mod[view_name].X
            )
    
    def test_add_realistic_structure(self):
        """Test adding realistic latent factor structure."""
        mdata = generate_synthetic_data(n_cells=100)
        mdata_structured = add_realistic_structure(
            mdata, 
            n_latent_factors=3,
            factor_variance=[0.3, 0.2, 0.1]
        )
        
        # Check that true factors are stored
        assert 'true_factors' in mdata_structured.obsm
        assert 'true_factor_variance' in mdata_structured.uns
        assert mdata_structured.obsm['true_factors'].shape == (100, 3)
        assert len(mdata_structured.uns['true_factor_variance']) == 3
    
    def test_generate_batch_effects(self):
        """Test generation of batch effects."""
        mdata = generate_synthetic_data(n_cells=100)
        mdata_batch = generate_batch_effects(mdata, effect_strength=0.5)
        
        # Data should be different after batch effects
        for view_name in mdata.mod.keys():
            assert not np.array_equal(
                mdata.mod[view_name].X,
                mdata_batch.mod[view_name].X
            )
    
    def test_simulate_missing_data(self):
        """Test simulation of missing data."""
        mdata = generate_synthetic_data(n_cells=100)
        original_zeros = {}
        
        # Count original zeros
        for view_name, view_data in mdata.mod.items():
            original_zeros[view_name] = np.sum(view_data.X == 0)
        
        mdata_missing = simulate_missing_data(mdata, missing_rate=0.2)
        
        # Should have more zeros after adding missing data
        for view_name, view_data in mdata_missing.mod.items():
            new_zeros = np.sum(view_data.X == 0)
            assert new_zeros >= original_zeros[view_name]
    
    def test_view_specific_missing_rates(self):
        """Test view-specific missing data rates."""
        mdata = generate_synthetic_data(n_cells=100)
        
        view_specific_rates = {
            'view1': 0.1,
            'view2': 0.3,
            'view3': 0.5
        }
        
        mdata_missing = simulate_missing_data(
            mdata, 
            view_specific_rates=view_specific_rates
        )
        
        # Different views should have different amounts of missing data
        zero_counts = {}
        for view_name, view_data in mdata_missing.mod.items():
            zero_counts[view_name] = np.sum(view_data.X == 0)
        
        # View3 should have most zeros, view1 should have least
        assert zero_counts['view3'] > zero_counts['view2'] > zero_counts['view1']
    
    def test_synthetic_data_non_negative(self):
        """Test that synthetic data is non-negative."""
        mdata = generate_synthetic_data(n_cells=100)
        
        for view_name, view_data in mdata.mod.items():
            assert np.all(view_data.X >= 0), f"Negative values found in {view_name}"