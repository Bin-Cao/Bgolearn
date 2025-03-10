
[![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/Bgolearn/)
# Python package - Bgolearn 



![Screen Shot 2022-07-11 at 9 13 28 AM](https://user-images.githubusercontent.com/86995074/178176016-8a79db81-fcfb-4af0-9b1c-aa4e6a113b5e.png)

## 为材料设计而生！
## （ A Bayesian global optimization package for material design ）Version 1, Jul, 2022


Reference paper : V. Picheny, T. Wagner, and D. Ginsbourger. “A Benchmark of Kriging-Based Infill Criteria for Noisy Optimization”. In: Structural and Multidisciplinary Optimization 48.3 (Sept. 2013), pp. 607–626. issn: 1615-1488. 


Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Content
Bgolearn guides subsequent material design based on existed experimental data. Which includes: 1.Expected Improvement algorithm, 2.Expected improvement with “plugin”，3.Augmented Expected Improvement，4.Expected Quantile Improvement，5.Reinterpolation Expected Improvement， 6.Upper confidence bound，7.Probability of Improvement，8.Predictive Entropy Search，9.Knowledge Gradient, a total of nine Utility Functions. Predictive Entropy Search，Knowledge Gradient are implemented based on Monte Carlo simulation.（贝叶斯优化设计，根据已有的实验数据对后续材料设计作出指导，本算法包共包括：期望最大化算法，期望最大化算法改进（考虑数据噪声），上确界方法，期望提升方法，熵搜索，知识梯度方法等在内的共计9种贝叶斯采样方法。其中熵搜索和知识梯度方法基于蒙特卡洛实现）

## Installing / 安装
pip install Bgolearn 

## Updating / 更新
pip install --upgrade Bgolearn

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

