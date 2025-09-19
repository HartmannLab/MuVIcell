"""
Utility functions for MuVIcell package.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Convenient function to load data from various file formats.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments passed to pandas read functions
        
    Returns:
        Loaded DataFrame
    """
    try:
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath, **kwargs)
        elif filepath.endswith('.tsv'):
            data = pd.read_csv(filepath, sep='\t', **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
            
        logger.info(f"Loaded data with shape {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data(data: pd.DataFrame, 
                 required_columns: Optional[List[str]] = None,
                 check_missing: bool = True) -> Dict[str, Any]:
    """
    Validate input data and return validation report.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        check_missing: Whether to check for missing values
        
    Returns:
        Dictionary containing validation results
    """
    validation_report = {
        'shape': data.shape,
        'columns': list(data.columns),
        'data_types': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict() if check_missing else {},
        'duplicate_rows': data.duplicated().sum(),
        'validation_passed': True,
        'issues': []
    }
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            validation_report['validation_passed'] = False
            validation_report['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check for excessive missing values
    if check_missing:
        high_missing = data.isnull().sum()
        high_missing_cols = high_missing[high_missing > len(data) * 0.5].index.tolist()
        if high_missing_cols:
            validation_report['issues'].append(f"Columns with >50% missing values: {high_missing_cols}")
    
    logger.info(f"Data validation completed. Issues found: {len(validation_report['issues'])}")
    return validation_report


def visualize_results(data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     hue_col: Optional[str] = None,
                     plot_type: str = "scatter",
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualizations for analysis results.
    
    Args:
        data: DataFrame containing the data to plot
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        hue_col: Optional column name for color grouping
        plot_type: Type of plot ('scatter', 'box', 'violin', 'bar')
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "scatter":
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
    elif plot_type == "box":
        sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
    elif plot_type == "violin":
        sns.violinplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
    elif plot_type == "bar":
        sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    ax.set_title(f"{plot_type.capitalize()} plot: {y_col} vs {x_col}")
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def calculate_correlation_matrix(data: pd.DataFrame,
                               method: str = "pearson",
                               min_periods: int = 1) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        data: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of observations required per pair
        
    Returns:
        Correlation matrix DataFrame
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("No numeric columns found in the data")
    
    corr_matrix = numeric_data.corr(method=method, min_periods=min_periods)
    
    logger.info(f"Calculated {method} correlation matrix for {len(numeric_data.columns)} features")
    return corr_matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                           figsize: Tuple[int, int] = (12, 10),
                           annot: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap visualization of correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        figsize: Figure size tuple
        annot: Whether to annotate the heatmap with correlation values
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle to show only lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=annot, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f' if annot else None,
                cbar_kws={"shrink": .8},
                ax=ax)
    
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {save_path}")
    
    return fig


def generate_sample_data(n_samples: int = 100,
                        n_features: int = 5,
                        n_cell_types: int = 3,
                        random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Generate sample data for testing and examples.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of feature columns
        n_cell_types: Number of different cell types
        random_state: Random seed for reproducibility
        
    Returns:
        Generated sample DataFrame
    """
    if random_state:
        np.random.seed(random_state)
    
    # Generate feature data
    features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate cell types
    cell_types = [f'cell_type_{i+1}' for i in range(n_cell_types)]
    cell_type_col = np.random.choice(cell_types, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(features, columns=feature_names)
    data['cell_type'] = cell_type_col
    data['sample_id'] = [f'sample_{i+1}' for i in range(n_samples)]
    
    logger.info(f"Generated sample data with {n_samples} samples, {n_features} features, {n_cell_types} cell types")
    return data