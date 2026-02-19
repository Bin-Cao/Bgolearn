
# Bgolearn

**Bgolearn** is a unified Python framework for Bayesian Global Optimization (BGO) designed to accelerate data-efficient discovery in materials science and related scientific domains.

The library provides a structured and extensible implementation of surrogate modeling, acquisition strategies, and uncertainty quantification, enabling principled optimization of expensive black-box functions under limited evaluation budgets.

---

## Overview

Bayesian global optimization has become a fundamental methodology for scientific discovery tasks where experimental or computational evaluations are costly. Bgolearn aims to standardize and simplify the implementation of such workflows by integrating:

* surrogate modeling
* acquisition function design
* uncertainty estimation
* candidate recommendation
* iterative active learning loops

The framework emphasizes reproducibility, modularity, and compatibility with scientific computing environments.

---

## Core Capabilities

### 1. Surrogate Modeling

Bgolearn supports multiple regression models for approximating unknown objective functions, including:

* Gaussian Process regression
* Random Forest regression
* Gradient boosting models
* Bootstrap-based ensemble estimators

These models enable predictive mean estimation together with uncertainty quantification.

### 2. Acquisition Functions

The framework provides commonly used acquisition strategies for balancing exploration and exploitation:

* Expected Improvement (EI)
* Upper Confidence Bound (UCB)
* Probability of Improvement (PI)

Users may extend the acquisition interface to incorporate customized strategies.

### 3. Uncertainty Quantification

For non-Gaussian surrogate models, Bgolearn incorporates bootstrap-based uncertainty estimation, enabling principled decision-making beyond GP-based approaches.

### 4. Single- and Multi-Objective Optimization

The framework supports single-objective optimization and can be extended to multi-objective scenarios through companion implementations.

---

## Installation

```bash
pip install Bgolearn
```

---

## Minimal Example

```python
from Bgolearn.BGOsampling import Bgolearn

# Training data
X_train, y_train = ...
X_virtual = ...

# Initialize and fit
bgo = Bgolearn()
model = bgo.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_virtual
)

# Acquisition step
score, recommendation = model.UCB()

print(recommendation)
```

For detailed API usage and workflow examples, refer to the project documentation.

---

## Scientific Reference

If Bgolearn contributes to your research, please cite:

```bibtex
@article{cao2026bgolearn,
  title        = {Bgolearn: A Unified Bayesian Optimization Framework for Accelerating Materials Discovery},
  author       = {Cao, Bin and Xiong, Jie and Ma, Jiaxuan and et al.},
  journal      = {arXiv preprint},
  year         = {2026},
  url          = {https://arxiv.org/abs/2601.06820}
}
```

---

## Design Principles

Bgolearn is developed according to the following principles:

* Methodological transparency
* Reproducibility of optimization workflows
* Modular architecture for extensibility
* Compatibility with scientific Python ecosystems
* Clear separation between modeling, acquisition, and evaluation layers

---

## License

Released under the MIT License.

---

## Maintainer

Bin Cao
PhD Candidate
Hong Kong University of Science and Technology (Guangzhou)
Email: [bcao686@connect.hkust-gz.edu.cn](mailto:bcao686@connect.hkust-gz.edu.cn)

