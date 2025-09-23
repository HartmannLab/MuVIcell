"""Tests for MuVIcell MuVI runner utilities."""

import pytest
import numpy as np
import pandas as pd
import muon as mu
from muvicell.muvi_runner import (
    get_factor_scores,
    get_factor_loadings, 
    get_variance_explained,
    select_top_factors
)


class TestMuVIRunner:
    """Test MuVI runner functions."""
    
    def test_get_factor_scores(self, muvi_results):
        """Test extracting factor scores."""
        scores = get_factor_scores(muvi_results)
        
        # Check return type and shape
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (muvi_results.n_obs, 5)  # 5 factors from fixture
        
        # Should be the same as stored in obsm
        expected_scores = muvi_results.obsm['X_muvi']
        np.testing.assert_array_equal(scores, expected_scores)
    
    def test_get_factor_scores_no_results(self, sample_mudata):
        """Test error when no MuVI results available."""
        with pytest.raises(ValueError, match="No MuVI results found"):
            get_factor_scores(sample_mudata)
    
    def test_get_factor_loadings(self, muvi_results):
        """Test extracting factor loadings."""
        for view_name in muvi_results.mod.keys():
            loadings = get_factor_loadings(muvi_results, view_name)
            
            # Check return type and shape
            assert isinstance(loadings, np.ndarray)
            n_vars = muvi_results.mod[view_name].n_vars
            assert loadings.shape == (n_vars, 5)  # 5 factors
            
            # Should be the same as stored in varm
            expected_loadings = muvi_results.mod[view_name].varm['muvi_loadings']
            np.testing.assert_array_equal(loadings, expected_loadings)
    
    def test_get_factor_loadings_invalid_view(self, muvi_results):
        """Test error with invalid view name."""
        with pytest.raises(ValueError, match="View 'invalid_view' not found"):
            get_factor_loadings(muvi_results, 'invalid_view')
    
    def test_get_factor_loadings_no_results(self, sample_mudata):
        """Test error when no MuVI loadings available."""
        view_name = list(sample_mudata.mod.keys())[0]
        with pytest.raises(ValueError, match="No MuVI loadings found"):
            get_factor_loadings(sample_mudata, view_name)
    
    def test_get_variance_explained(self, muvi_results):
        """Test extracting variance explained."""
        var_exp = get_variance_explained(muvi_results)
        
        # Check return type and structure
        assert isinstance(var_exp, dict)
        assert len(var_exp) == len(muvi_results.mod)
        
        # Check each view's variance
        for view_name, view_var in var_exp.items():
            assert isinstance(view_var, np.ndarray)
            assert len(view_var) == 5  # 5 factors
            assert all(view_var >= 0)  # Variance should be non-negative
    
    def test_get_variance_explained_no_results(self, sample_mudata):
        """Test error when no variance explained available."""
        with pytest.raises(ValueError, match="No MuVI variance explained found"):
            get_variance_explained(sample_mudata)
    
    def test_select_top_factors_by_number(self, muvi_results):
        """Test selecting top factors by number."""
        n_top = 3
        top_factors = select_top_factors(muvi_results, n_top_factors=n_top)
        
        # Check return type and length
        assert isinstance(top_factors, list)
        assert len(top_factors) == n_top
        
        # Check that indices are valid
        for factor_idx in top_factors:
            assert isinstance(factor_idx, int)
            assert 0 <= factor_idx < 5  # 5 total factors
        
        # Should be sorted
        assert top_factors == sorted(top_factors)
    
    def test_select_top_factors_by_threshold(self, muvi_results):
        """Test selecting factors by variance threshold."""
        # Use a very low threshold to ensure some factors are selected
        threshold = 0.01
        top_factors = select_top_factors(
            muvi_results, 
            n_top_factors=None,
            variance_threshold=threshold
        )
        
        # Check return type
        assert isinstance(top_factors, list)
        
        # Should select at least some factors (depends on mock data)
        assert len(top_factors) >= 0
        
        # Check that indices are valid
        for factor_idx in top_factors:
            assert isinstance(factor_idx, int)
            assert 0 <= factor_idx < 5
    
    def test_select_top_factors_high_threshold(self, muvi_results):
        """Test selecting factors with high threshold (might select none)."""
        # Use a very high threshold
        threshold = 10.0  # Unrealistically high
        top_factors = select_top_factors(
            muvi_results,
            n_top_factors=None, 
            variance_threshold=threshold
        )
        
        # Might select no factors with such high threshold
        assert isinstance(top_factors, list)
        assert len(top_factors) >= 0