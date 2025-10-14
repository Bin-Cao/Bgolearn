"""
BgolearnFuns - Core Functions Module

This module contains the core implementation functions for Bayesian Global Optimization.
It provides specialized classes for different optimization scenarios including
maximization, minimization, classification boundary exploration, and performance evaluation.

Modules:
--------
- BGOmax: Global maximization optimization functions
- BGOmin: Global minimization optimization functions  
- BGOclf: Classification boundary exploration for active learning
- BGO_eval: Performance evaluation and efficiency testing tools

Key Features:
- Multiple acquisition functions (EI, UCB, PoI, PES, Knowledge_G, etc.)
- Support for both homogeneous and heterogeneous noise handling
- Parallel processing capabilities for improved performance
- Comprehensive uncertainty quantification
- Cross-validation and model evaluation tools

Author: Bin Cao
Institution: Hong Kong University of Science and Technology (Guangzhou)
"""

# Import core optimization classes
from .BGOmax import Global_max
from .BGOmin import Global_min
from .BGOclf import Boundary
from .BGO_eval import BGO_Efficient

# Define public API
__all__ = [
    'Global_max',
    'Global_min', 
    'Boundary',
    'BGO_Efficient'
]
