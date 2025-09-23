"""Tests for MuVIcell analysis utilities."""

import pytest
import numpy as np
import pandas as pd
import muon as mu
from muvicell.analysis import (
    characterize_factors,
    calculate_factor_correlations,
    identify_factor_associations,
    cluster_cells_by_factors,
    calculate_factor_distances,
    summarize_factor_activity
)


class TestAnalysis:
    """Test analysis functions."""
    
    def test_characterize_factors(self, muvi_results):
        """Test factor characterization."""
        factor_genes = characterize_factors(
            muvi_results, 
            top_genes_per_factor=10,
            loading_threshold=0.1
        )
        
        # Check return type and structure
        assert isinstance(factor_genes, dict)
        assert len(factor_genes) == len(muvi_results.mod)
        
        # Check each view's results
        for view_name, results_df in factor_genes.items():
            assert isinstance(results_df, pd.DataFrame)
            
            if len(results_df) > 0:  # If any genes pass threshold
                required_cols = ['factor', 'gene', 'loading', 'abs_loading']
                for col in required_cols:
                    assert col in results_df.columns
                
                # Check that absolute loadings are >= threshold
                assert all(results_df['abs_loading'] >= 0.1)
    
    def test_calculate_factor_correlations(self, muvi_results):
        """Test factor correlation calculation."""
        factor_corr = calculate_factor_correlations(muvi_results)
        
        # Check return type and structure
        assert isinstance(factor_corr, pd.DataFrame)
        
        # Should be square matrix
        n_factors = muvi_results.obsm['X_muvi'].shape[1]
        assert factor_corr.shape == (n_factors, n_factors)
        
        # Diagonal should be 1.0 (correlation with self)
        np.testing.assert_array_almost_equal(
            np.diag(factor_corr), 
            np.ones(n_factors)
        )
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            factor_corr.values,
            factor_corr.values.T
        )
    
    def test_identify_factor_associations(self, muvi_results):
        """Test factor-metadata associations."""
        associations = identify_factor_associations(muvi_results)
        
        # Check return type
        assert isinstance(associations, pd.DataFrame)
        
        if len(associations) > 0:
            # Check required columns
            required_cols = ['factor', 'metadata', 'test', 'statistic', 'p_value', 'n_valid']
            for col in required_cols:
                assert col in associations.columns
            
            # Check that p-values are valid
            assert all(0 <= associations['p_value']) and all(associations['p_value'] <= 1)
            
            # Check that corrected p-values exist
            assert 'p_value_corrected' in associations.columns
    
    def test_identify_factor_associations_specific_columns(self, muvi_results):
        """Test factor associations with specific metadata columns."""
        # Test with specific columns
        associations = identify_factor_associations(
            muvi_results,
            metadata_columns=['cell_type', 'condition']
        )
        
        if len(associations) > 0:
            # Should only contain specified metadata columns
            unique_metadata = associations['metadata'].unique()
            for metadata in unique_metadata:
                assert metadata in ['cell_type', 'condition']
    
    def test_cluster_cells_by_factors(self, muvi_results):
        """Test cell clustering based on factors."""
        n_clusters = 3
        cluster_labels = cluster_cells_by_factors(
            muvi_results,
            n_clusters=n_clusters,
            random_state=42
        )
        
        # Check return type and shape
        assert isinstance(cluster_labels, np.ndarray)
        assert len(cluster_labels) == muvi_results.n_obs
        
        # Check number of unique clusters
        unique_clusters = np.unique(cluster_labels)
        assert len(unique_clusters) <= n_clusters  # Might be fewer if some clusters are empty
        
        # Check cluster labels are valid integers
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)
    
    def test_cluster_cells_specific_factors(self, muvi_results):
        """Test clustering with specific factors."""
        factors_to_use = [0, 2]  # Use only factors 0 and 2
        cluster_labels = cluster_cells_by_factors(
            muvi_results,
            n_clusters=4,
            factors_to_use=factors_to_use,
            random_state=42
        )
        
        # Should still return valid clustering
        assert len(cluster_labels) == muvi_results.n_obs
        assert len(np.unique(cluster_labels)) <= 4
    
    def test_calculate_factor_distances(self, muvi_results):
        """Test pairwise distance calculation."""
        distances = calculate_factor_distances(muvi_results, metric='euclidean')
        
        # Check return type and shape
        assert isinstance(distances, np.ndarray)
        n_cells = muvi_results.n_obs
        assert distances.shape == (n_cells, n_cells)
        
        # Distance matrix should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)
        
        # Diagonal should be zero (distance to self)
        np.testing.assert_array_almost_equal(np.diag(distances), np.zeros(n_cells))
        
        # All distances should be non-negative
        assert np.all(distances >= 0)
    
    def test_calculate_factor_distances_different_metrics(self, muvi_results):
        """Test distance calculation with different metrics."""
        metrics = ['euclidean', 'cosine', 'manhattan']
        
        for metric in metrics:
            distances = calculate_factor_distances(muvi_results, metric=metric)
            assert isinstance(distances, np.ndarray)
            
            n_cells = muvi_results.n_obs
            assert distances.shape == (n_cells, n_cells)
    
    def test_summarize_factor_activity_overall(self, muvi_results):
        """Test overall factor activity summary."""
        summary = summarize_factor_activity(muvi_results)
        
        # Check return type and structure
        assert isinstance(summary, pd.DataFrame)
        
        n_factors = muvi_results.obsm['X_muvi'].shape[1]
        assert len(summary) == n_factors
        
        # Check required columns
        required_cols = ['factor', 'mean_activity', 'std_activity', 'min_activity', 'max_activity']
        for col in required_cols:
            assert col in summary.columns
        
        # Check that statistics make sense
        assert all(summary['std_activity'] >= 0)  # Standard deviation should be non-negative
        assert all(summary['min_activity'] <= summary['max_activity'])  # Min <= Max
    
    def test_summarize_factor_activity_by_group(self, muvi_results):
        """Test factor activity summary by group."""
        summary = summarize_factor_activity(muvi_results, group_by='cell_type')
        
        # Check return type
        assert isinstance(summary, pd.DataFrame)
        
        if len(summary) > 0:
            # Check required columns for group-wise summary
            required_cols = ['group', 'factor', 'mean_activity', 'std_activity', 'n_cells']
            for col in required_cols:
                assert col in summary.columns
            
            # Check that all groups are represented
            unique_groups = summary['group'].unique()
            expected_groups = muvi_results.obs['cell_type'].dropna().unique()
            for group in unique_groups:
                assert group in expected_groups
            
            # Check that n_cells makes sense
            assert all(summary['n_cells'] > 0)
    
    def test_summarize_factor_activity_invalid_group(self, muvi_results):
        """Test factor activity summary with invalid group column."""
        with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
            summarize_factor_activity(muvi_results, group_by='invalid_column')