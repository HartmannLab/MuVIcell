"""
Basic tests for MuVIcell package.
"""

import pytest
import pandas as pd
import numpy as np
from muvicell import MuVIcellAnalyzer, load_data
from muvicell.utils import generate_sample_data, validate_data, calculate_correlation_matrix


class TestMuVIcellAnalyzer:
    """Test suite for MuVIcellAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = MuVIcellAnalyzer()
        assert analyzer.data is None
        assert analyzer.results == {}
        assert analyzer.metadata == {}
        
        # Test with data
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        analyzer_with_data = MuVIcellAnalyzer(data=data)
        assert analyzer_with_data.data is not None
        assert len(analyzer_with_data.data) == 3
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        data = generate_sample_data(n_samples=50, n_features=3, random_state=42)
        analyzer = MuVIcellAnalyzer(data=data)
        
        processed = analyzer.preprocess_data(normalize=True, remove_outliers=False)
        
        # Check that numeric columns are normalized (mean ~0, std ~1)
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(processed[col].mean()) < 0.1  # Should be close to 0
            assert abs(processed[col].std() - 1.0) < 0.1  # Should be close to 1
    
    def test_analyze_cell_features(self):
        """Test cell feature analysis."""
        data = generate_sample_data(n_samples=30, n_features=5, n_cell_types=3, random_state=42)
        analyzer = MuVIcellAnalyzer(data=data)
        
        feature_cols = ['feature_1', 'feature_2', 'feature_3']
        results = analyzer.analyze_cell_features(
            cell_type_col='cell_type',
            feature_cols=feature_cols,
            method='mean'
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert len(results) == 3  # 3 cell types
        
        # Check that all cell types are present
        for cell_type in data['cell_type'].unique():
            assert cell_type in results
            assert len(results[cell_type]) == len(feature_cols)
    
    def test_analyze_cell_features_errors(self):
        """Test error handling in cell feature analysis."""
        data = generate_sample_data(n_samples=20, n_features=3, random_state=42)
        analyzer = MuVIcellAnalyzer(data=data)
        
        # Test missing cell type column
        with pytest.raises(ValueError, match="Cell type column 'nonexistent' not found"):
            analyzer.analyze_cell_features(
                cell_type_col='nonexistent',
                feature_cols=['feature_1'],
                method='mean'
            )
        
        # Test missing feature columns
        with pytest.raises(ValueError, match="Feature columns not found"):
            analyzer.analyze_cell_features(
                cell_type_col='cell_type',
                feature_cols=['nonexistent_feature'],
                method='mean'
            )
        
        # Test unsupported method
        with pytest.raises(ValueError, match="Unsupported method"):
            analyzer.analyze_cell_features(
                cell_type_col='cell_type',
                feature_cols=['feature_1'],
                method='unsupported'
            )
    
    def test_get_summary(self):
        """Test summary generation."""
        data = generate_sample_data(n_samples=20, n_features=3, random_state=42)
        analyzer = MuVIcellAnalyzer(data=data)
        
        summary = analyzer.get_summary()
        
        assert 'data_shape' in summary
        assert 'data_columns' in summary
        assert 'results_available' in summary
        assert 'analysis_metadata' in summary
        
        assert summary['data_shape'] == data.shape
        assert len(summary['data_columns']) == len(data.columns)


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = generate_sample_data(n_samples=100, n_features=5, n_cell_types=3, random_state=42)
        
        assert len(data) == 100
        assert 'cell_type' in data.columns
        assert 'sample_id' in data.columns
        assert len(data['cell_type'].unique()) == 3
        assert len([col for col in data.columns if col.startswith('feature_')]) == 5
    
    def test_validate_data(self):
        """Test data validation."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [1, np.nan, 3, 4],
            'c': ['x', 'y', 'z', 'w']
        })
        
        # Basic validation
        report = validate_data(data)
        assert report['shape'] == (4, 3)
        assert report['validation_passed'] is True
        assert report['missing_values']['b'] == 1
        assert report['duplicate_rows'] == 0
        
        # Validation with required columns
        report_with_req = validate_data(data, required_columns=['a', 'c'])
        assert report_with_req['validation_passed'] is True
        
        # Validation with missing required columns
        report_missing = validate_data(data, required_columns=['a', 'missing_col'])
        assert report_missing['validation_passed'] is False
        assert len(report_missing['issues']) > 0
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],  # Perfect correlation with a
            'c': [5, 4, 3, 2, 1],   # Perfect negative correlation with a
            'd': ['x', 'y', 'z', 'w', 'v']  # Non-numeric
        })
        
        corr_matrix = calculate_correlation_matrix(data)
        
        assert corr_matrix.shape == (3, 3)  # Only numeric columns
        assert corr_matrix.loc['a', 'b'] > 0.99  # Should be ~1
        assert corr_matrix.loc['a', 'c'] < -0.99  # Should be ~-1
    
    def test_calculate_correlation_matrix_no_numeric(self):
        """Test correlation matrix with no numeric columns."""
        data = pd.DataFrame({
            'a': ['x', 'y', 'z'],
            'b': ['p', 'q', 'r']
        })
        
        with pytest.raises(ValueError, match="No numeric columns found"):
            calculate_correlation_matrix(data)


class TestLoadData:
    """Test suite for data loading functionality."""
    
    def test_load_data_error_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data("test.unsupported")


if __name__ == "__main__":
    pytest.main([__file__])