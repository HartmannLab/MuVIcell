"""Tests for MuVIcell preprocessing utilities."""

import pytest
import numpy as np
import pandas as pd
import muon as mu
from muvicell.preprocessing import (
    normalize_views,
    filter_views,
    find_highly_variable_genes,
    subset_to_hvg,
    preprocess_for_muvi
)


class TestPreprocessing:
    """Test preprocessing functions."""
    
    def test_normalize_views_default(self, sample_mudata):
        """Test default normalization."""
        mdata_norm = normalize_views(sample_mudata)
        
        # Check that data is modified
        for view_name in sample_mudata.mod.keys():
            original = sample_mudata.mod[view_name].X
            normalized = mdata_norm.mod[view_name].X
            assert not np.array_equal(original, normalized)
        
        # Check that raw data is preserved
        for view_name, view_data in mdata_norm.mod.items():
            assert view_data.raw is not None
    
    def test_normalize_views_custom_config(self, sample_mudata):
        """Test normalization with custom view configurations."""
        view_configs = {
            'rna': {'log_transform': True, 'scale': True},
            'protein': {'log_transform': False, 'scale': False}
        }
        
        mdata_norm = normalize_views(sample_mudata, view_configs=view_configs)
        
        # Both views should be modified (even if differently)
        for view_name in sample_mudata.mod.keys():
            original = sample_mudata.mod[view_name].X
            normalized = mdata_norm.mod[view_name].X
            assert not np.array_equal(original, normalized)
    
    def test_filter_views_default(self, sample_mudata):
        """Test default filtering."""
        original_shapes = {
            view_name: view_data.shape 
            for view_name, view_data in sample_mudata.mod.items()
        }
        
        mdata_filt = filter_views(sample_mudata)
        
        # Check that shapes might be different (depending on filtering)
        for view_name, view_data in mdata_filt.mod.items():
            # Number of observations should be the same or smaller
            assert view_data.n_obs <= original_shapes[view_name][0]
            # Number of variables should be the same or smaller
            assert view_data.n_vars <= original_shapes[view_name][1]
    
    def test_filter_views_custom_params(self, sample_mudata):
        """Test filtering with custom parameters."""
        # Use very permissive filtering
        mdata_filt = filter_views(
            sample_mudata,
            min_cells_per_gene=1,
            min_genes_per_cell=1
        )
        
        # With permissive filtering, shapes should be unchanged or minimally changed
        for view_name, view_data in mdata_filt.mod.items():
            original_shape = sample_mudata.mod[view_name].shape
            # Should have at least 90% of original size
            assert view_data.n_obs >= original_shape[0] * 0.9
            assert view_data.n_vars >= original_shape[1] * 0.9
    
    def test_find_highly_variable_genes(self, sample_mudata):
        """Test finding highly variable genes."""
        mdata_hvg = find_highly_variable_genes(sample_mudata, n_top_genes=10)
        
        # Check that highly_variable column is added
        for view_name, view_data in mdata_hvg.mod.items():
            assert 'highly_variable' in view_data.var.columns
            
            # Should have exactly 10 highly variable genes (or all genes if < 10)
            n_hvg = view_data.var['highly_variable'].sum()
            expected_hvg = min(10, view_data.n_vars)
            assert n_hvg == expected_hvg
    
    def test_find_hvg_view_specific(self, sample_mudata):
        """Test finding HVGs with view-specific numbers."""
        view_specific_n_genes = {
            'rna': 8,
            'protein': 5
        }
        
        mdata_hvg = find_highly_variable_genes(
            sample_mudata,
            view_specific_n_genes=view_specific_n_genes
        )
        
        # Check view-specific numbers
        for view_name, expected_n in view_specific_n_genes.items():
            actual_n = mdata_hvg.mod[view_name].var['highly_variable'].sum()
            assert actual_n == min(expected_n, mdata_hvg.mod[view_name].n_vars)
    
    def test_subset_to_hvg(self, sample_mudata):
        """Test subsetting to highly variable genes."""
        # First find HVGs
        mdata_hvg = find_highly_variable_genes(sample_mudata, n_top_genes=5)
        
        # Then subset
        mdata_subset = subset_to_hvg(mdata_hvg)
        
        # Check that views are subsetted
        for view_name, view_data in mdata_subset.mod.items():
            original_vars = mdata_hvg.mod[view_name].n_vars
            subset_vars = view_data.n_vars
            
            # Should have at most 5 variables (or original number if < 5)
            expected_vars = min(5, original_vars)
            assert subset_vars == expected_vars
    
    def test_subset_to_hvg_warning(self, sample_mudata):
        """Test warning when subsetting without HVG identification."""
        with pytest.warns(UserWarning, match="No highly variable genes found"):
            subset_to_hvg(sample_mudata)
    
    def test_preprocess_for_muvi_full_pipeline(self, sample_mudata):
        """Test complete preprocessing pipeline."""
        mdata_processed = preprocess_for_muvi(
            sample_mudata,
            filter_cells=True,
            filter_genes=True,
            normalize=True,
            find_hvg=True,
            subset_hvg=True
        )
        
        # Check that processing was applied
        for view_name, view_data in mdata_processed.mod.items():
            # Should have raw data stored
            assert view_data.raw is not None
            
            # Should have HVG information
            assert 'highly_variable' in view_data.var.columns
            
            # Data should be different from original
            original = sample_mudata.mod[view_name].X
            processed = view_data.X
            assert not np.array_equal(original, processed)
    
    def test_preprocess_for_muvi_partial_pipeline(self, sample_mudata):
        """Test partial preprocessing pipeline."""
        mdata_processed = preprocess_for_muvi(
            sample_mudata,
            filter_cells=False,
            filter_genes=False,
            normalize=True,
            find_hvg=False,
            subset_hvg=False
        )
        
        # Only normalization should be applied
        for view_name, view_data in mdata_processed.mod.items():
            # Should have raw data stored (from normalization)
            assert view_data.raw is not None
            
            # Should NOT have HVG information
            assert 'highly_variable' not in view_data.var.columns
            
            # Shapes should be unchanged
            original_shape = sample_mudata.mod[view_name].shape
            assert view_data.shape == original_shape