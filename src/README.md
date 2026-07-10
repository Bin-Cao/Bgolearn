# Bgolearn

Bgolearn is a Bayesian global optimization package for accelerating materials
discovery. It provides practical optimization workflows for costly experiments,
including regression-based candidate recommendation, classification boundary
exploration, cross-validation diagnostics, and several acquisition functions.

Author and maintainer: Dr.Bin Cao (https://bin-cao.github.io/)

Documentation: https://bgolearn.netlify.app/

Repository: https://github.com/Bin-Cao/Bgolearn

## Featured Introduction

[Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) is Chapter 1 of the Springer book *An Introduction to Materials Informatics* by Prof. Tong-Yi Zhang, Academician of the Chinese Academy of Sciences. The active-learning examples and results in this chapter are implemented with and depend on Bgolearn.

## Features

- Bayesian global optimization for materials design and discovery.
- Single-objective minimization and maximization workflows.
- Classification-mode active learning for decision-boundary exploration.
- Acquisition functions including EI, EI with plugin, augmented EI, EQI, UCB,
  PoI, PES, and Knowledge Gradient.
- Built-in surrogate choices for SVM, Random Forest, AdaBoost, and MLP models.
- Gaussian process modeling with homogeneous or heterogeneous noise support.
- Optional cross-validation reports and virtual-sample prediction exports.

## Installation

```bash
pip install Bgolearn
```

For local development from this repository:

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd

from Bgolearn.BGOsampling import Bgolearn

data = pd.read_csv("data.csv")
virtual_samples = pd.read_csv("virtual_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

optimizer = Bgolearn()
model = optimizer.fit(
    data_matrix=X,
    Measured_response=y,
    virtual_samples=virtual_samples,
    Mission="Regression",
    min_search=True,
)

scores, candidates = model.EI()
print(candidates)
```

## Main API

### `Bgolearn.fit`

Fits a Bayesian optimization workflow and returns an acquisition-function model.

Common parameters:

- `data_matrix`: measured feature matrix.
- `Measured_response`: measured target values.
- `virtual_samples`: candidate samples to rank.
- `Mission`: `"Regression"` or `"Classification"`.
- `Kriging_model`: `None`, a built-in model name, or a custom model class with
  a `fit_pre` method.
- `opt_num`: number of candidates to recommend.
- `min_search`: `True` for minimization and `False` for maximization.
- `CV_test`: `False`, `"LOOCV"`, or an integer for k-fold cross-validation.

### Custom Surrogate Model

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class CustomKrigingModel:
    def fit_pre(self, xtrain, ytrain, xtest):
        model = GaussianProcessRegressor(kernel=RBF(), normalize_y=True)
        model.fit(xtrain, ytrain)
        mean, std = model.predict(xtest, return_std=True)
        return mean, std
```

Use it with:

```python
model = optimizer.fit(
    data_matrix=X,
    Measured_response=y,
    virtual_samples=virtual_samples,
    Kriging_model=CustomKrigingModel,
)
```

## Classification Mode

```python
model = optimizer.fit(
    data_matrix=X,
    Measured_response=labels,
    virtual_samples=virtual_samples,
    Mission="Classification",
    Classifier="RandomForest",
)

scores, candidates = model.Entropy()
```

Available classifiers include `GaussianProcess`, `LogisticRegression`,
`NaiveBayes`, `SVM`, and `RandomForest`.

## Citation

If Bgolearn supports your research, please cite:

Cao B. et al., "Bgolearn: A Unified Bayesian Optimization Framework for
Accelerating Materials Discovery", npj Computational Materials.
https://doi.org/10.1038/s41524-026-02226-3

Related introduction: Tong-Yi Zhang, "Bayesian Global Optimization", Chapter 1
of *An Introduction to Materials Informatics*. The chapter's active-learning
examples and results depend on Bgolearn.
https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1

## Support

Questions, issues, pull requests, and research collaborations are welcome.

Contact: bcao686@connect.hkust-gz.edu.cn
