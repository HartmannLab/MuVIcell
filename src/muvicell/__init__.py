"""
MuVIcell: From cell-type stratified features to multicellular coordinated programs

A Python package for multi-view integration and analysis of single-cell data
using MuVI (Multi-View Integration), built on top of muon, liana, and plotnine.
"""

from . import data
from . import preprocessing  
from . import muvi_runner
from . import analysis
from . import visualization
from . import synthetic

# Import main MuVI functions directly for convenience
from .muvi_runner import run_muvi, setup_muvi_model, get_factor_scores, get_factor_loadings, get_variance_explained

__version__ = "0.1.0"
__author__ = "HartmannLab"

__all__ = [
    "data",
    "preprocessing", 
    "muvi_runner",
    "analysis",
    "visualization",
    "synthetic",
    # Direct MuVI functions
    "run_muvi",
    "setup_muvi_model", 
    "get_factor_scores",
    "get_factor_loadings",
    "get_variance_explained"
]
