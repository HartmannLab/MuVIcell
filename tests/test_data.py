"""Tests for MuVIcell data handling utilities."""

import pytest
import numpy as np
import pandas as pd
import muon as mu
from muvicell.data import (
    get_view_info,
    validate_muon_data
)


class TestDataHandling:
    """Test data loading and handling functions."""
    
    def test_get_view_info(self, sample_mudata):
        """Test getting view information."""
        view_info = get_view_info(sample_mudata)
        
        # Check return type and structure
        assert isinstance(view_info, pd.DataFrame)
        assert len(view_info) == 2  # rna and protein views
        
        # Check required columns
        required_cols = ['view_name', 'n_obs', 'n_vars', 'var_names']
        for col in required_cols:
            assert col in view_info.columns
        
        # Check data consistency
        for _, row in view_info.iterrows():
            assert row['n_obs'] == sample_mudata.n_obs
            assert len(row['var_names']) <= 5  # Should show max 5 gene names
    
    def test_validate_muon_data_valid(self, sample_mudata):
        """Test validation of valid muon data."""
        validation = validate_muon_data(sample_mudata)
        
        # Check return type
        assert isinstance(validation, dict)
        
        # Check validation keys
        expected_keys = [
            'has_multiple_views',
            'consistent_observations', 
            'views_have_variables',
            'consistent_obs_names'
        ]
        for key in expected_keys:
            assert key in validation
        
        # All validations should pass for synthetic data
        for key, result in validation.items():
            assert result is True, f"Validation failed for {key}"
    
    def test_validate_muon_data_single_view(self, sample_mudata):
        """Test validation with single view (should fail)."""
        # Create single-view data
        single_view_data = mu.MuData({
            'rna': sample_mudata.mod['rna']
        })
        
        validation = validate_muon_data(single_view_data)
        assert validation['has_multiple_views'] is False
    
    def test_validate_muon_data_inconsistent_obs(self, sample_mudata):
        """Test validation with inconsistent observations."""
        # Modify one view to have different observations
        mdata_copy = sample_mudata.copy()
        rna_view = mdata_copy.mod['rna']
        
        # Remove some cells from RNA view
        rna_subset = rna_view[:-10, :].copy()
        mdata_copy.mod['rna'] = rna_subset
        
        validation = validate_muon_data(mdata_copy)
        assert validation['consistent_observations'] is False
    
    def test_validate_muon_data_no_variables(self, sample_mudata):
        """Test validation with view having no variables."""
        mdata_copy = sample_mudata.copy()
        
        # Create view with no variables (empty slice)
        empty_view = mdata_copy.mod['rna'][:, :0].copy()
        mdata_copy.mod['empty'] = empty_view
        
        validation = validate_muon_data(mdata_copy)
        assert validation['views_have_variables'] is False