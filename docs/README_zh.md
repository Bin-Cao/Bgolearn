<h1 align="center">Bgolearn</h1>

<p align="center">
  面向材料发现的统一贝叶斯优化框架。
</p>

<p align="center">
  <a href="https://pypi.org/project/bgolearn/"><img src="https://img.shields.io/pypi/v/bgolearn?style=flat-square&label=PyPI" alt="PyPI"></a>
  <a href="https://doi.org/10.1038/s41524-026-02226-3"><img src="https://img.shields.io/badge/DOI-10.1038%2Fs41524--026--02226--3-blue?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Bin-Cao/Bgolearn?style=flat-square" alt="License"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/stargazers"><img src="https://img.shields.io/github/stars/Bin-Cao/Bgolearn?style=flat-square" alt="Stars"></a>
  <a href="https://bgolearn.netlify.app/"><img src="https://img.shields.io/badge/docs-online-2f6f9f?style=flat-square" alt="Documentation"></a>
</p>

<p align="center">
  <strong>语言：</strong>
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_de.md">Deutsch</a>
</p>

---

## 项目概览

**Bgolearn** 是一个面向科研场景的 Python 框架，聚焦 **贝叶斯全局优化（Bayesian Global Optimization, BGO）**。它服务于数据驱动的材料发现、实验设计与虚拟筛选，尤其适合实验代价较高、样本规模有限、需要可解释推荐依据的研究流程。

Bgolearn 将代理模型、不确定性评估、采集函数、主动学习与候选样本排序整合为统一流程。框架同时支持回归与分类任务，使研究者能够从已有实验数据出发，更高效地筛选下一轮最有价值的材料候选。

## 核心特性

- 统一支持回归、分类、主动学习与虚拟筛选流程。
- 支持高斯过程、SVM、随机森林、AdaBoost、MLP 等多种代理模型。
- 提供适用于有噪声与无噪声优化的采集函数，包括 EI、AEI、EQI、REI、UCB、PoI、PES 与 KG。
- 支持最小化、最大化以及面向多目标研究的候选推荐。
- 提供轻量级本地 Web 界面，便于交互式使用。

## 资源

| 资源 | 链接 |
| --- | --- |
| 论文 | [npj Computational Materials](https://doi.org/10.1038/s41524-026-02226-3) |
| 英文文档 | [bgolearn.netlify.app](https://bgolearn.netlify.app/) |
| 中文手册 | [bgolearn-chi.netlify.app](https://bgolearn-chi.netlify.app/) |
| 视频教程 | [Bilibili](https://www.bilibili.com/video/BV1LTtLeaEZp) |
| 会议报告 | [CMC 2025](https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842) |
| 多目标模块 | [MultiBgolearn](https://github.com/Bin-Cao/MultiBgolearn) |
| 官方图形界面 | [BgoFace](https://github.com/Bgolearn/BgoFace) |
| 示例代码与数据 | [CodeDemo](https://github.com/Bgolearn/CodeDemo) |

## 安装

通过 PyPI 安装：

```bash
pip install Bgolearn
```

升级到最新版本：

```bash
pip install --upgrade Bgolearn
```

查看已安装版本：

```bash
pip show Bgolearn
```

## 运行界面

克隆仓库并启动本地界面：

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
python bgolearn_ui.py
```

随后在浏览器中打开：

```text
http://127.0.0.1:8787
```

## 引用

如果 Bgolearn 支持了你的研究，请引用：

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

## 资助

**Bgolearn** 入选由上海市经济和信息化委员会支持的 [开源人工智能支持计划（2025）](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html)。

项目材料：[figures/funding.png](../figures/funding.png)

## 许可证

Bgolearn 基于 MIT License 发布。
