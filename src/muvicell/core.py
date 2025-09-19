"""
Core functionality for MuVIcell analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MuVIcellAnalyzer:
    """
    Main analyzer class for MuVIcell operations.
    
    This class provides functionality for analyzing multicellular coordination
    and cell-type specific features from biological data.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the MuVIcell analyzer.
        
        Args:
            data: Optional pandas DataFrame containing the input data
        """
        self.data = data
        self.results = {}
        self.metadata = {}
        
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            filepath: Path to the data file
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath, **kwargs)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(filepath, **kwargs)
            elif filepath.endswith('.parquet'):
                self.data = pd.read_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
            logger.info(f"Loaded data with shape {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, 
                       normalize: bool = True,
                       remove_outliers: bool = False,
                       outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            normalize: Whether to normalize the data
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier removal
            
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        processed_data = self.data.copy()
        
        # Normalize numeric columns
        if normalize:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = (processed_data[numeric_cols] - 
                                          processed_data[numeric_cols].mean()) / processed_data[numeric_cols].std()
        
        # Remove outliers
        if remove_outliers:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            z_scores = np.abs((processed_data[numeric_cols] - processed_data[numeric_cols].mean()) / 
                             processed_data[numeric_cols].std())
            processed_data = processed_data[(z_scores < outlier_threshold).all(axis=1)]
        
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        self.data = processed_data
        return processed_data
    
    def analyze_cell_features(self, 
                            cell_type_col: str,
                            feature_cols: List[str],
                            method: str = "mean") -> Dict[str, Any]:
        """
        Analyze cell-type specific features.
        
        Args:
            cell_type_col: Column name containing cell type information
            feature_cols: List of feature column names to analyze
            method: Analysis method ('mean', 'median', 'std')
            
        Returns:
            Dictionary containing analysis results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if cell_type_col not in self.data.columns:
            raise ValueError(f"Cell type column '{cell_type_col}' not found in data")
        
        missing_cols = [col for col in feature_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
        
        results = {}
        
        if method == "mean":
            results = self.data.groupby(cell_type_col)[feature_cols].mean()
        elif method == "median":
            results = self.data.groupby(cell_type_col)[feature_cols].median()
        elif method == "std":
            results = self.data.groupby(cell_type_col)[feature_cols].std()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.results['cell_features'] = results
        logger.info(f"Analyzed cell features for {len(results)} cell types")
        return results.to_dict()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current analysis.
        
        Returns:
            Dictionary containing summary information
        """
        summary = {
            'data_shape': self.data.shape if self.data is not None else None,
            'data_columns': list(self.data.columns) if self.data is not None else None,
            'results_available': list(self.results.keys()),
            'analysis_metadata': self.metadata
        }
        
        return summary