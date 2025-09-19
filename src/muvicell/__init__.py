"""
MuVIcell: From cell-type stratified features to multicellular coordinated programs.

A Python package for analyzing multicellular coordination and cell-type specific features.
"""

__version__ = "0.1.0"
__author__ = "HartmannLab"
__email__ = "contact@hartmannlab.org"
__license__ = "GPL-3.0"

from .core import MuVIcellAnalyzer
from .utils import load_data, visualize_results

__all__ = [
    "MuVIcellAnalyzer",
    "load_data", 
    "visualize_results",
]