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

---

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
| Manual | [bgolearn.netlify.app](https://bgolearn.netlify.app/) |
| Video tutorial | [Bilibili](https://www.bilibili.com/video/BV1LTtLeaEZp) |
| Conference report | [CMC 2025](https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842) |
| Multi-object module | [MultiBgolearn](https://github.com/Bin-Cao/MultiBgolearn) |
| Official GUI | [BgoFace](https://github.com/Bgolearn/BgoFace) |
| Example code and data | [CodeDemo](https://github.com/Bgolearn/CodeDemo) |

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
