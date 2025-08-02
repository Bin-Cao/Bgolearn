

# Bgolearn

[**🔗 Report**](https://cmc2025.scimeeting.cn/cn/web/speaker-detail/27167?user_id=ZXvycJpgjG2WSbabyEmiSA_d_d) | [**Homepage**](http://bgolearn.caobin.asia/) | [**BgoFace UI**](https://github.com/Bgolearn/BgoFace)

![PyPI Downloads](https://static.pepy.tech/badge/bgolearn)
🤝🤝🤝 Please star ⭐️ this project to support open-source development! For questions or collaboration, contact: **Dr. Bin Cao** ([bcao686@connect.hkust-gz.edu.cn](mailto:bcao686@connect.hkust-gz.edu.cn))

📊 [Usage Statistics (pepy)](https://www.pepy.tech/projects/Bgolearn)

---

## 🎓 Overview

**Bgolearn** is a lightweight and extensible Python package for **Bayesian global optimization**, built for accelerating materials discovery and design. It provides out-of-the-box support for regression and classification tasks, implements various acquisition strategies, and offers a seamless pipeline for virtual screening, active learning, and multi-objective optimization.

> 📦 Official PyPI: [`pip install Bgolearn`](https://pypi.org/project/Bgolearn/)
> 🎥 Code tutorial (BiliBili): [Watch here](https://www.bilibili.com/video/BV1LTtLeaEZp)
> 🚀 Colab Demo: [Run it online](https://colab.research.google.com/drive/1OSc-phxm7QLOm8ceGJiIMGGz9riuwP6Q?usp=sharing)

---


## 📈 Download Statistics

* [Bgolearn](https://pepy.tech/projects/Bgolearn?timeRange=threeMonths&category=version)
* [BgoKit](https://pepy.tech/projects/BgoKit?timeRange=threeMonths&category=version)
* [MultiBgolearn](https://pepy.tech/projects/multibgolearn?timeRange=threeMonths&category=version)
---
## ✨ Key Features

### ✅ One-Line Installation

```bash
pip install Bgolearn
```

### ✅ Update to Latest Version

```bash
pip install --upgrade Bgolearn
```

### ✅ Quick Check

```bash
pip show Bgolearn
```

---

## 🧪 Getting Started

```python
import Bgolearn.BGOsampling as BGOS
import pandas as pd

# Load characterized dataset
data = pd.read_csv('data.csv')
x = data.iloc[:, :-1]   # features
y = data.iloc[:, -1]    # response

# Load virtual samples
vs = pd.read_csv('virtual_data.csv')

# Instantiate and run model
Bgolearn = BGOS.Bgolearn()
Mymodel = Bgolearn.fit(data_matrix=x, Measured_response=y, virtual_samples=vs)

# Get result using Expected Improvement
Mymodel.EI()
```

---

## 🔧 Multi-Objective Optimization

> Install the extension toolkit:

```bash
pip install BgoKit
```

```python
from BgoKit import ToolKit

Model = ToolKit.MultiOpt(vs, [score_1, score_2])
Model.BiSearch()
Model.plot_distribution()
```

📓 See detailed demo: [Multi-objective Example](https://github.com/Bin-Cao/Bgolearn/blob/main/Template/%E4%B8%AD%E6%96%87%E7%A4%BA%E4%BE%8B/%E5%A4%9A%E7%9B%AE%E6%A0%87%E5%AE%9E%E7%8E%B0/%E5%A4%9A%E7%9B%AE%E6%A0%87.ipynb)

<img src="https://github.com/Bin-Cao/Bgolearn/assets/86995074/41c90c29-364c-47cc-aefe-4433f7d93e23" width="300" height="300">

---

## 🧠 Supported Algorithms

### 🔹 For Regression

* Expected Improvement (EI)
* Augmented Expected Improvement (AEI)
* Expected Quantile Improvement (EQI)
* Upper Confidence Bound (UCB)
* Probability of Improvement (PI)
* Predictive Entropy Search (PES)
* Knowledge Gradient (KG)
* Reinterpolation EI (REI)
* Expected Improvement with Plugin

### 🔹 For Classification

* Least Confidence
* Margin Sampling
* Entropy-based approach

---

## 🖥️ User Interface

The graphical frontend of Bgolearn is developed as [**BgoFace**](https://github.com/Bgolearn/BgoFace), providing no-code access to its backend algorithms.

---

## 📚 Technical Innovations

### 🧩 Rich Bayesian Acquisition Functions

Supports a broad range of acquisition strategies (EI, UCB, KG, PES, etc.) for both single and multi-objective optimization. Works well with sparse and high-dimensional datasets common in material science.

### 🤝 Multi-Objective Expansion

Use **BgoKit** and **MultiBgolearn** to implement Pareto optimization across multiple target properties (e.g., strength & ductility), enabling parallel evaluation across virtual samples.

### 🔄 Integrated Active Learning

Incorporates adaptive sampling in an active learning loop—experiment → prediction → update—to accelerate optimization using fewer experiments.

---

## 📌 Academic Impact

### 2025

1. **Nano Letters**: *Self-Driving Laboratory under UHV*
   [Link](https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.5c02445?casa_token=DycwWKxkjjQAAAAA:_qVVZ56VuzbHDnLmJ_-8mUtHatu9S8rOXE78HHGjmNhADLlr7qr-4rPWsAuIOVide29eEy6gOfvzC3do)

2. **Small**: *ML-Engineered Nanozyme System for Anti-Tumor Therapy*
   [Link](https://onlinelibrary.wiley.com/doi/10.1002/smll.202408750?utm_source=chatgpt.com)

3. **Computational Materials Science**: *Mg-Ca-Zn Alloy Optimization*
   [Link](https://www.sciencedirect.com/science/article/pii/S0927025625000084)

4. **Measurement**: *Foaming Agent Optimization in EPB Shield Construction*
   [Link](https://www.sciencedirect.com/science/article/pii/S0263224124013940)

5. **Intelligent Computing**: *Metasurface Design via Bayesian Learning*
   [Link](https://spj.science.org/doi/pdf/10.34133/icomputing.0135)

### 2024

6. **Materials & Design**: *Lead-Free Solder Alloys via Active Learning*
   [Link](https://www.sciencedirect.com/science/article/pii/S0264127524002946)

7. **npj Computational Materials**: *MLMD Platform with Bgolearn Backend*
   [Link](https://www.nature.com/articles/s41524-024-01243-4)

---

## 📦 License

Released under the [MIT License](https://opensource.org/licenses/MIT).
💼 Free for academic and commercial use. Please cite relevant publications if used in research.

---

## 🤝 Contributing & Collaboration

We welcome community contributions and research collaborations:

* Submit issues for bug reports, ideas, or suggestions
* Submit pull requests for code contributions
* Contact Bin Cao ([bcao686@connect.hkust-gz.edu.cn](mailto:bcao686@connect.hkust-gz.edu.cn)) for collaborations


---


``` javascript
Signature:
Bgolearn.fit(
    data_matrix,
    Measured_response,
    virtual_samples,
    Mission='Regression',
    Classifier='GaussianProcess',
    noise_std=None,
    Kriging_model=None,
    opt_num=1,
    min_search=True,
    CV_test=False,
    Dynamic_W=False,
    seed=42,
)

================================================================

:param data_matrix: data matrix of training dataset, X .

:param Measured_response: response of tarining dataset, y.

:param virtual_samples: designed virtual samples.

:param Mission: str, default 'Regression', the mission of optimization.  Mission = 'Regression' or 'Classification'

:param Classifier: if  Mission == 'Classification', classifier is used.
        if user isn't applied one, Bgolearn will call a pre-set classifier.
        default, Classifier = 'GaussianProcess', i.e., Gaussian Process Classifier.
        five different classifiers are pre-setd in Bgolearn:
        'GaussianProcess' --> Gaussian Process Classifier (default)
        'LogisticRegression' --> Logistic Regression
        'NaiveBayes' --> Naive Bayes Classifier
        'SVM' --> Support Vector Machine Classifier
        'RandomForest' --> Random Forest Classifier

:param noise_std: float or ndarray of shape (n_samples,), default=None
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian.
        measurement noise on the training observations.

        if noise_std is not None, a noise value will be estimated by maximum likelihood
        on training dataset.

:param Kriging_model (default None):
        str, Kriging_model = 'SVM', 'RF', 'AdaB', 'MLP'
        The  machine learning models will be implemented: Support Vector Machine (SVM), 
        Random Forest(RF), AdaBoost(AdaB), and Multi-Layer Perceptron (MLP).
        The estimation uncertainity will be determined by Boostsrap sampling.
    or  
        a user defined callable Kriging model, has an attribute of <fit_pre>
        if user isn't applied one, Bgolearn will call a pre-set Kriging model
        atribute <fit_pre> : 
        input -> xtrain, ytrain, xtest ; 
        output -> predicted  mean and std of xtest

        e.g. (take GaussianProcessRegressor in sklearn):
        class Kriging_model(object):
            def fit_pre(self,xtrain,ytrain,xtest):
                # instantiated model
                kernel = RBF()
                mdoel = GaussianProcessRegressor(kernel=kernel).fit(xtrain,ytrain)
                # defined the attribute's outputs
                mean,std = mdoel.predict(xtest,return_std=True)
                return mean,std    

        e.g. (MultiModels estimations):
        class Kriging_model(object):
            def fit_pre(self,xtrain,ytrain,xtest):
                # instantiated model
                pre_1 = SVR(C=10).fit(xtrain,ytrain).predict(xtest) # model_1
                pre_2 = SVR(C=50).fit(xtrain,ytrain).predict(xtest) # model_2
                pre_3 = SVR(C=80).fit(xtrain,ytrain).predict(xtest) # model_3
                model_1 , model_2 , model_3  can be changed to any ML models you desire
                # defined the attribute's outputs
                stacked_array = np.vstack((pre_1,pre_2,pre_3))
                means = np.mean(stacked_array, axis=0)
                std = np.sqrt(np.var(stacked_array), axis=0)
                return mean, std    

:param opt_num: the number of recommended candidates for next iteration, default 1. 

:param min_search: default True -> searching the global minimum ;
                           False -> searching the global maximum.

:param CV_test: 'LOOCV' or an int, default False (pass test) 
        if CV_test = 'LOOCV', LOOCV will be applied,
        elif CV_test = int, e.g., CV_test = 10, 10 folds cross validation will be applied.

:return: 1: array; potential of each candidate. 2: array/float; recommended candidate(s).
File:      ~/miniconda3/lib/python3.9/site-packages/Bgolearn/BGOsampling.py
Type:      method
```
