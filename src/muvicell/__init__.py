"""
MuVIcell: From cell-type stratified features to multicellular coordinated programs

A Python package for multi-view integration and analysis of single-cell data
using MuVI (Multi-View Integration), built on top of muon, liana, and plotnine.
"""

from . import data
from . import preprocessing  
from . import analysis
from . import visualization
from . import synthetic

__version__ = "0.1.1"
__author__ = "Loan Vulliard (@Hartmann Lab)"

__all__ = [
    "data",
    "preprocessing", 
    "muvi_runner",
    "analysis",
    "visualization",
    "synthetic",
]
