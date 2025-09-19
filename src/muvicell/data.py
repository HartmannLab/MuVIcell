"""
Data loading and handling utilities for MuVIcell.

This module provides functions for loading and handling muon data objects
for multi-view integration analysis.
"""

import muon as mu
from pathlib import Path
from typing import Union, Dict, List, Optional
import pandas as pd


def load_muon_data(path: Union[str, Path]) -> mu.MuData:
    """
    Load muon data from file.
    
    Parameters
    ----------
    path : str or Path
        Path to the muon data file (.h5mu format)
        
    Returns
    -------
    mu.MuData
        Loaded muon data object
        
    Examples
    --------
    >>> mdata = load_muon_data("data.h5mu")
    """
    return mu.read_h5mu(path)


def save_muon_data(mdata: mu.MuData, path: Union[str, Path]) -> None:
    """
    Save muon data to file.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object to save
    path : str or Path
        Output path for the muon data file (.h5mu format)
        
    Examples
    --------
    >>> save_muon_data(mdata, "output.h5mu")
    """
    mdata.write_h5mu(path)


def get_view_info(mdata: mu.MuData) -> pd.DataFrame:
    """
    Get information about views in the muon data object.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object
        
    Returns
    -------
    pd.DataFrame
        DataFrame with view information including names, dimensions, and features
        
    Examples
    --------
    >>> info = get_view_info(mdata)
    >>> print(info)
    """
    view_info = []
    for view_name, view_data in mdata.mod.items():
        info = {
            'view_name': view_name,
            'n_obs': view_data.n_obs,
            'n_vars': view_data.n_vars,
            'var_names': list(view_data.var_names)[:5]  # First 5 variable names
        }
        view_info.append(info)
    
    return pd.DataFrame(view_info)


def validate_muon_data(mdata: mu.MuData) -> Dict[str, bool]:
    """
    Validate muon data object for MuVI analysis.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object to validate
        
    Returns
    -------
    Dict[str, bool]
        Dictionary with validation results
        
    Examples
    --------
    >>> validation = validate_muon_data(mdata)
    >>> if all(validation.values()):
    ...     print("Data is valid for MuVI analysis")
    """
    validation_results = {}
    
    # Check if there are at least 2 views
    validation_results['has_multiple_views'] = len(mdata.mod) >= 2
    
    # Check if all views have the same observations
    obs_counts = [view.n_obs for view in mdata.mod.values()]
    validation_results['consistent_observations'] = len(set(obs_counts)) == 1
    
    # Check if views have variables
    validation_results['views_have_variables'] = all(
        view.n_vars > 0 for view in mdata.mod.values()
    )
    
    # Check if observation names are consistent across views
    obs_names_sets = [set(view.obs_names) for view in mdata.mod.values()]
    validation_results['consistent_obs_names'] = len(set(
        frozenset(s) for s in obs_names_sets
    )) == 1
    
    return validation_results