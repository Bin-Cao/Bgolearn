"""Top-level package metadata for Bgolearn.

Bgolearn provides Bayesian global optimization tools for materials discovery,
including regression optimization, classification boundary exploration, model
evaluation, and uncertainty-aware candidate recommendation.

Author: Dr.Bin Cao (https://bin-cao.github.io/)
Institution: Hong Kong University of Science and Technology (Guangzhou)
Email: bcao686@connect.hkust-gz.edu.cn
Documentation: https://bgolearn.netlify.app/
"""

from .BGOsampling import Bgolearn

__description__ = "A Bayesian global optimization package"
__documents__ = "https://bgolearn.netlify.app/"
__author__ = "Dr.Bin Cao (https://bin-cao.github.io/)"
__author_email__ = "bcao686@connect.hkust-gz.edu.cn"
__paper__ = "https://doi.org/10.1038/s41524-026-02226-3"
__url__ = "https://bgolearn.netlify.app/"

__all__ = ["Bgolearn"]
