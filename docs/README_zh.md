
# Bgolearn

### 用于加速材料发现的统一贝叶斯优化框架

<p align="center">
  <a href="https://pypi.org/project/bgolearn/">
    <img src="https://img.shields.io/pypi/v/bgolearn.svg" />
  </a>
  <a href="https://pepy.tech/projects/bgolearn">
    <img src="https://static.pepy.tech/personalized-badge/bgolearn?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads" />
  </a>
  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-black.svg" />
  </a>
</p>

---

## 关于

**Bgolearn** 是一个面向科研的 Python 框架，专注于 **贝叶斯全局优化（Bayesian Global Optimization, BGO）**，旨在加速数据驱动的材料发现与科学设计。

该框架提供以下功能：

* 统一的回归与分类建模  
* 模块化的采集函数（acquisition functions）  
* 多目标优化  
* 主动学习工作流  
* 虚拟筛选流程  

Bgolearn 强调可复现性、可扩展性以及研究级严谨性，适用于学术研究与工业应用。

---

## 论文与资源

**Bgolearn: a Unified Bayesian Optimization Framework for Accelerating Materials Discovery**

* 论文: [https://doi.org/10.48550/arXiv.2601.06820](https://doi.org/10.48550/arXiv.2601.06820)
* 会议报告: [https://cmc2025.scimeeting.cn/cn/web/speaker-detail/27167](https://cmc2025.scimeeting.cn/cn/web/speaker-detail/27167)
* 文档: [https://bgolearn.netlify.app/](https://bgolearn.netlify.app/)
* 中文手册: [https://bgolearn-chi.netlify.app/](https://bgolearn-chi.netlify.app/)
* 视频教程: [https://www.bilibili.com/video/BV1LTtLeaEZp](https://www.bilibili.com/video/BV1LTtLeaEZp)

---

## 框架

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd8ff5a6-d3c5-4727-a88a-0dbdddda1dba"
       width="480"
       alt="Bgolearn workflow"/>
</p>

---

## 运行界面

<img width="1316" height="505" alt="Screenshot 2026-03-10 at 19 42 51" src="https://github.com/user-attachments/assets/25601b30-19d4-40e4-b2a7-c566dfba64c9" />

1. 打开终端。

2. 克隆仓库：

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
````

3. 进入项目目录：

```bash
cd Bgolearn
```

4. 启动界面：

```bash
python bgolearn_ui.py
```

这将启动 Bgolearn 用户界面。

```bash
http://127.0.0.1:8787
```

---

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

---

## 引用

如果你在研究中使用了 Bgolearn，请引用：

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

## 资助

**Bgolearn** 入选由 **上海市经济和信息化委员会（上海市经信委）** 支持的
[开源人工智能支持计划（2025）](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html)。

项目材料：
[https://github.com/Bin-Cao/Bgolearn/blob/main/figures/funding.png](https://github.com/Bin-Cao/Bgolearn/blob/main/figures/funding.png)

---

## 联系方式

<table>
  <tr>
    <td width="160" align="center" valign="top">
      <img src="https://github.com/user-attachments/assets/7e77bd5a-42d6-45db-b8e6-2c82cac81b9d"
           width="140"
           style="border-radius: 50%;" />
    </td>
    <td valign="top">
      <b>Bin Cao</b><br>
      博士研究生<br>
      香港科技大学（广州）<br>
      导师：张统一 教授<br><br>
      Email: <a href="mailto:bcao686@connect.hkust-gz.edu.cn">bcao686@connect.hkust-gz.edu.cn</a><br>
      主页: <a href="https://bin-cao.github.io/">https://bin-cao.github.io/</a>
    </td>
  </tr>
</table>

---

## 许可证

基于 MIT License 发布。
可免费用于学术与商业用途。

