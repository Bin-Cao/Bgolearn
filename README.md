
<p align="center">
  <strong>Language:</strong>
  <a href="README.md">English</a> | <a href="docs/README_zh.md">简体中文</a> | <a href="docs/README_ja.md">日本語</a>
</p>



# Bgolearn

### A Unified Bayesian Optimization Framework for Accelerating Materials Discovery

<p align="center">

  <!-- ===== Links ===== -->
  <a href="https://doi.org/10.48550/arXiv.2601.06820">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" />
  </a>

  <a href="https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842">
    <img src="https://img.shields.io/badge/Conference-Report-6A5ACD?style=for-the-badge" />
  </a>

  <a href="https://bgolearn.netlify.app/">
    <img src="https://img.shields.io/badge/Docs-Documentation-1E90FF?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>

  <a href="https://bgolearn-chi.netlify.app/">
    <img src="https://img.shields.io/badge/Docs-中文手册-008080?style=for-the-badge" />
  </a>

  <a href="https://www.bilibili.com/video/BV1LTtLeaEZp">
    <img src="https://img.shields.io/badge/Bilibili-Video-FF69B4?style=for-the-badge" />
  </a>

</p>

<p align="center">

  <!-- ===== Package / Stats ===== -->
  <a href="https://pypi.org/project/bgolearn/">
    <img src="https://img.shields.io/pypi/v/bgolearn?style=for-the-badge&logo=pypi" />
  </a>

  <a href="https://pepy.tech/projects/bgolearn">
    <img src="https://img.shields.io/badge/Downloads-Total-4CAF50?style=for-the-badge" />
  </a>

  <a href="https://github.com/Bin-Cao/Bgolearn">
    <img src="https://img.shields.io/github/stars/Bin-Cao/Bgolearn?style=for-the-badge&logo=github" />
  </a>

  <a href="https://github.com/Bin-Cao/Bgolearn/issues">
    <img src="https://img.shields.io/github/issues/Bin-Cao/Bgolearn?style=for-the-badge&logo=github" />
  </a>

  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Bin-Cao/Bgolearn?style=for-the-badge" />
  </a>

</p>




> [!TIP]
> **Bgolearn is among the first unified Bayesian optimization frameworks purpose-built for the materials science community.**  
> While most Bayesian optimization libraries were originally developed for generic machine learning or black-box optimization, Bgolearn systematically bridges Bayesian optimization, active learning, and materials discovery within a lightweight yet extensible framework. It integrates single-objective and multi-objective optimization, uncertainty-aware surrogate modeling, acquisition-driven experiment recommendation, and automated virtual screening into a unified workflow tailored for real-world materials design. By dramatically reducing experimental search costs while maintaining high discovery efficiency, Bgolearn represents an important step toward fully autonomous, AI-driven materials optimization and closed-loop scientific discovery.  




## About

**Bgolearn** is a research-oriented Python framework for **Bayesian Global Optimization (BGO)**, developed to accelerate data-driven materials discovery and scientific design.

The framework provides:

* Unified regression and classification modeling
* Modular acquisition functions
* Multi-objective optimization
* Active learning workflows
* Virtual screening pipelines

Bgolearn emphasizes reproducibility, extensibility, and research-grade rigor, making it suitable for both academic research and industrial applications.



---

## Run the Interface


<img width="1316" height="505" alt="Screenshot 2026-03-10 at 19 42 51" src="https://github.com/user-attachments/assets/25601b30-19d4-40e4-b2a7-c566dfba64c9" />


1. Open the terminal.

2. Clone the repository:

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
````

3. Navigate to the project directory:

```bash
cd Bgolearn
```

4. Launch the UI:

```bash
python bgolearn_ui.py
```

This will start the Bgolearn user interface.

```bash
http://127.0.0.1:8787
```

---


## Installation

Install from PyPI:

```bash
pip install Bgolearn
```

Upgrade to the latest version:

```bash
pip install --upgrade Bgolearn
```

Check installed version:

```bash
pip show Bgolearn
```

---

## Citation

If you use Bgolearn in your research, please cite:

```
@article{cao2026bgolearn,
  title        = {Bgolearn: a Unified Bayesian Optimization Framework for Accelerating Materials Discovery},
  author       = {Cao, Bin and Xiong, Jie and Ma, Jiaxuan and Tian, Yuan and Hu, Yirui and He, Mengwei and Zhang, Longhan and Wang, Jiayu and Hui, Jian and Liu, Li and Xue, Dezhen and Lookman, Turab and Zhang, Tong-Yi},
  journal      = {arXiv preprint arXiv:2601.06820},
  year         = {2026},
  eprint       = {2601.06820},
  archivePrefix= {arXiv},
  primaryClass = {cond-mat.mtrl-sci},
  doi          = {https://doi.org/10.48550/arXiv.2601.06820}
}
```

---

## Funding

**Bgolearn** is selected for the [Open-Source Artificial Intelligence Support Program (2025)](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html) by the **Shanghai Municipal Commission of Economy and Informatization (上海市经信委).**

Project material:
[https://github.com/Bin-Cao/Bgolearn/blob/main/figures/funding.png](https://github.com/Bin-Cao/Bgolearn/blob/main/figures/funding.png)


---

## Contributors

<a href="https://github.com/Bin-Cao/Bgolearn/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bin-Cao/Bgolearn" />
</a>

