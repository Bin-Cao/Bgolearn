<h1 align="center">Bgolearn</h1>

<p align="center">
  A unified Bayesian optimization framework for accelerating materials discovery.
</p>

<p align="center">
  <a href="https://pypi.org/project/bgolearn/"><img src="https://img.shields.io/pypi/v/bgolearn?style=flat-square&label=PyPI" alt="PyPI"></a>
  <a href="https://doi.org/10.1038/s41524-026-02226-3"><img src="https://img.shields.io/badge/DOI-10.1038%2Fs41524--026--02226--3-blue?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Bin-Cao/Bgolearn?style=flat-square" alt="License"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/stargazers"><img src="https://img.shields.io/github/stars/Bin-Cao/Bgolearn?style=flat-square" alt="Stars"></a>
  <a href="https://bgolearn.netlify.app/"><img src="https://img.shields.io/badge/docs-online-2f6f9f?style=flat-square" alt="Documentation"></a>
</p>

<p align="center">
  <strong>Language:</strong>
  <a href="README.md">English</a> |
  <a href="docs/README_zh.md">简体中文</a> |
  <a href="docs/README_ja.md">日本語</a> |
  <a href="docs/README_ko.md">한국어</a> |
  <a href="docs/README_de.md">Deutsch</a>
</p>

<p align="center">
  <strong>Playground:</strong>
  <a href="https://bin-cao.github.io/Bgolearn/">Interactive Bayesian optimization game</a>
</p>

---

## 📺 Media Coverage

### Featured on Dragon TV · Shanghai Media Group (SMG)

Our work was featured on **Dragon TV (东方卫视)**, highlighting its research and applications.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/ea83c704-e9b4-4cd2-be7d-1cfc6a1dcfac"
    alt="Featured on Dragon TV"
    width="85%"
  />
</p>

<p align="center">
  <sub>Featured on Dragon TV (东方卫视), Shanghai Media Group (SMG).</sub>
</p>
---

## Featured Introduction

[**Bayesian Global Optimization**](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) is Chapter 1 of the Springer book *An Introduction to Materials Informatics* by Prof. Tong-Yi Zhang, Academician of the Chinese Academy of Sciences. The active-learning examples and results in this chapter are implemented with and depend on **Bgolearn**.

## Overview

**Bgolearn** is a research-oriented Python framework for **Bayesian Global Optimization (BGO)**. It is designed for data-driven materials discovery, experimental design, and virtual screening, where each new measurement can be costly and every recommendation should be traceable.

The framework brings surrogate modeling, uncertainty-aware acquisition, active learning, and candidate ranking into a single workflow. It supports both regression and classification tasks, so researchers can move from small experimental datasets to the next most informative material candidates with less custom glue code.

## Highlights

- Unified workflows for regression, classification, active learning, and virtual screening.
- Multiple surrogate models, including Gaussian process, SVM, random forest, AdaBoost, and MLP-based options.
- Acquisition functions for noisy and noise-free optimization, including EI, AEI, EQI, REI, UCB, PoI, PES, and KG.
- Candidate recommendation for minimization, maximization, and multi-objective research workflows.
- A lightweight local web interface for users who prefer an interactive workflow.

## Resources

| Resource | Link |
| --- | --- |
| Paper | [npj Computational Materials](https://doi.org/10.1038/s41524-026-02226-3) |
| Book chapter | [Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| Manual | [bgolearn.netlify.app](https://bgolearn.netlify.app/) |
| Video tutorial | [Bilibili](https://www.bilibili.com/video/BV1LTtLeaEZp) |
| Conference report | [CMC 2025](https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842) |
| Multi-object module | [MultiBgolearn](https://github.com/Bin-Cao/MultiBgolearn) |
| Official GUI | [BgoFace](https://github.com/Bgolearn/BgoFace) |
| Example code and data | [CodeDemo](https://github.com/Bgolearn/CodeDemo) |

## Representative Applications

Bgolearn has supported a growing set of materials science and engineering studies. The following papers are a representative subset:

### 2026

| Venue | Application |
| --- | --- |
| Surfaces and Interfaces | [Photochemical synthesis of WS2 thin films](https://www.sciencedirect.com/science/article/pii/S2468023026013799) |
| JPhys Materials | [Design of TaNbMoVW refractory high-entropy alloys](https://iopscience.iop.org/article/10.1088/2515-7639/ae44d1/meta) |
| Chemical Science | [Optimizing on-surface reactions](https://pubs.rsc.org/sc/article/17/22/11114/1231470/A-dual-mode-large-language-model-assistant-for-on) |
| Springer Nature | [Book chapter: Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| Journal of Materials Informatics | [Optimizing SciBERT hyperparameters](https://www.oaepublish.com/articles/jmi.2025.78) |
| Science Bulletin | [Discovering ultra-durable and highly active catalysts](https://www.sciencedirect.com/science/article/pii/S2095927325012678) |
| Aggregate | [Discovering G-quadruplex deep-eutectic circularly polarized luminescence materials](https://onlinelibrary.wiley.com/doi/pdf/10.1002/agt2.70307) |
| Materials Science in Semiconductor Processing | [Substitutional Hf-doped p-type MoS2 via pulsed-laser synthesis](https://www.sciencedirect.com/science/article/pii/S1369800126005263) |

### 2025

| Venue | Application |
| --- | --- |
| Nano Letters | [Self-driving laboratory under UHV](https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.5c02445?casa_token=DycwWKxkjjQAAAAA:_qVVZ56VuzbHDnLmJ_-8mUtHatu9S8rOXE78HHGjmNhADLlr7qr-4rPWsAuIOVide29eEy6gOfvzC3do) |
| Small | [ML-engineered nanozyme system for anti-tumor therapy](https://onlinelibrary.wiley.com/doi/10.1002/smll.202408750?utm_source=chatgpt.com) |
| Computational Materials Science | [Mg-Ca-Zn alloy optimization](https://www.sciencedirect.com/science/article/pii/S0927025625000084) |
| Measurement | [Foaming agent optimization in EPB shield construction](https://www.sciencedirect.com/science/article/pii/S0263224124013940) |
| Intelligent Computing | [Metasurface design via Bayesian learning](https://spj.science.org/doi/pdf/10.34133/icomputing.0135) |

### 2024

| Venue | Application |
| --- | --- |
| Materials & Design | [Lead-free solder alloys via active learning](https://www.sciencedirect.com/science/article/pii/S0264127524002946) |
| npj Computational Materials | [MLMD platform with Bgolearn backend](https://www.nature.com/articles/s41524-024-01243-4) |

## Installation

Install from PyPI:

```bash
pip install Bgolearn
```

Upgrade to the latest release:

```bash
pip install --upgrade Bgolearn
```

Check the installed version:

```bash
pip show Bgolearn
```

## Run the Interface

Clone the repository and start the local UI:

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
python bgolearn_ui.py
```

Then open:

```text
http://127.0.0.1:8787
```

## Citation

If Bgolearn supports your research, please cite:

```bibtex
@article{Cao2026Bgolearn,
  author    = {Bin Cao and Jie Xiong and Jiaxuan Ma and Yuan Tian and Yirui Hu and Mengwei He and Longhan Zhang and Jiayu Wang and Jian Hui and Li Liu and Dezhen Xue and Turab Lookman and Jun Wang and Tong-Yi Zhang},
  title     = {Bgolearn: a unified Bayesian optimization framework for accelerating materials discovery},
  journal   = {npj Computational Materials},
  year      = {2026},
  doi       = {10.1038/s41524-026-02226-3},
  issn      = {2057-3960},
  publisher = {Springer Nature},
  url       = {https://doi.org/10.1038/s41524-026-02226-3}
}
```

## Funding

**Bgolearn** was selected for the [Open-Source Artificial Intelligence Support Program (2025)](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html) supported by the Shanghai Municipal Commission of Economy and Informatization.

Project material: [figures/funding.png](figures/funding.png)

## License

Bgolearn is released under the MIT License.
