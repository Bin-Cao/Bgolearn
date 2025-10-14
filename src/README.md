

# Bgolearn

**A Bayesian Global Optimization Package for Material Design**
First released: July 2022
Official website: [https://bgolearn.netlify.app/](https://bgolearn.netlify.app/)

---

## 为材料设计而生！

Bgolearn is a Python-based **Bayesian Global Optimization (BGO)** toolkit designed to accelerate **material discovery and experimental design**.
It provides a suite of efficient acquisition functions to guide the next-step experiments based on existing data, helping researchers achieve less trial and more discovery.

---

## Reference

V. Picheny, T. Wagner, and D. Ginsbourger.
“A Benchmark of Kriging-Based Infill Criteria for Noisy Optimization.”
*Structural and Multidisciplinary Optimization*, 48(3), 607–626 (2013).
ISSN: 1615-1488.

---

## Core Features

Bgolearn implements nine utility (acquisition) functions for Bayesian optimization, covering both classical and Monte Carlo–based methods:

1. **Expected Improvement (EI)**
2. **Expected Improvement with “Plugin”** (noise-handling version)
3. **Augmented Expected Improvement (AEI)**
4. **Expected Quantile Improvement (EQI)**
5. **Reinterpolation Expected Improvement (REI)**
6. **Upper Confidence Bound (UCB)**
7. **Probability of Improvement (PI)**
8. **Predictive Entropy Search (PES)** (Monte Carlo based)
9. **Knowledge Gradient (KG)** (Monte Carlo based)

贝叶斯全局优化算法包 Bgolearn 可基于已有实验数据对后续材料设计提供指导。
包含 9 种采样方法：期望提升（EI）、改进期望提升（含噪声）、增强期望提升、分位期望提升、再插值期望提升、上置信界、提升概率、熵搜索与知识梯度等。
其中熵搜索与知识梯度方法基于蒙特卡洛仿真实现。

---

## Installation

```bash
pip install Bgolearn
```

---

## Updating

```bash
pip install --upgrade Bgolearn
```

---

## Compatibility

Written in Python, Bgolearn runs smoothly on Windows, Linux, and macOS.

---

## Citation

If you use Bgolearn in your research, please cite the corresponding paper once available or link to this repository.
