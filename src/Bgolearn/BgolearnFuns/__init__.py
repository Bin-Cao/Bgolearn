"""Core optimization and evaluation classes used by Bgolearn.

This subpackage exposes the implementation classes for minimization,
maximization, classification boundary exploration, and optimization-efficiency
evaluation.

Author: Dr.Bin Cao (https://bin-cao.github.io/)
Institution: Hong Kong University of Science and Technology (Guangzhou)
"""

from .BGO_eval import BGO_Efficient
from .BGOclf import Boundary
from .BGOmax import Global_max
from .BGOmin import Global_min

__all__ = [
    "Global_max",
    "Global_min",
    "Boundary",
    "BGO_Efficient",
]
