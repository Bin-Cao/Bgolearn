
# Bgolearn

### 材料探索を加速する統合ベイズ最適化フレームワーク

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

## 概要

**Bgolearn** は、**ベイズグローバル最適化（Bayesian Global Optimization, BGO）** のための研究指向のPythonフレームワークであり、データ駆動型の材料探索および科学設計の加速を目的として開発されています。

本フレームワークは以下の機能を提供します：

* 回帰および分類の統合モデリング  
* モジュール化された獲得関数（acquisition functions）  
* 多目的最適化  
* アクティブラーニングのワークフロー  
* バーチャルスクリーニングパイプライン  

Bgolearnは、再現性・拡張性・研究水準の厳密性を重視しており、学術研究および産業応用の双方に適しています。

---

## 論文および関連資料

**Bgolearn: a Unified Bayesian Optimization Framework for Accelerating Materials Discovery**

* 論文: [https://doi.org/10.48550/arXiv.2601.06820](https://doi.org/10.48550/arXiv.2601.06820)
* 会議発表: [https://cmc2025.scimeeting.cn/cn/web/speaker-detail/27167](https://cmc2025.scimeeting.cn/cn/web/speaker-detail/27167)
* ドキュメント: [https://bgolearn.netlify.app/](https://bgolearn.netlify.app/)
* 中文マニュアル: [https://bgolearn-chi.netlify.app/](https://bgolearn-chi.netlify.app/)
* 動画チュートリアル: [https://www.bilibili.com/video/BV1LTtLeaEZp](https://www.bilibili.com/video/BV1LTtLeaEZp)

---

## フレームワーク

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd8ff5a6-d3c5-4727-a88a-0dbdddda1dba"
       width="480"
       alt="Bgolearn workflow"/>
</p>

---

## インターフェースの実行

<img width="1316" height="505" alt="Screenshot 2026-03-10 at 19 42 51" src="https://github.com/user-attachments/assets/25601b30-19d4-40e4-b2a7-c566dfba64c9" />

1. ターミナルを開きます。

2. リポジトリをクローンします：

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
````

3. プロジェクトディレクトリに移動します：

```bash
cd Bgolearn
```

4. UIを起動します：

```bash
python bgolearn_ui.py
```

これにより、Bgolearnのユーザーインターフェースが起動します。

```bash
http://127.0.0.1:8787
```

---

## インストール

PyPIからインストール：

```bash
pip install Bgolearn
```

最新版へアップグレード：

```bash
pip install --upgrade Bgolearn
```

インストール済みバージョンの確認：

```bash
pip show Bgolearn
```

---

## 引用

Bgolearnを研究で使用する場合は、以下を引用してください：

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

## 資金支援

**Bgolearn** は、**上海市経済・情報化委員会（上海市经信委）** による
[オープンソース人工知能支援プログラム（2025）](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html) に採択されています。

プロジェクト資料：
[https://github.com/Bin-Cao/Bgolearn/blob/main/figures/funding.png](https://github.com/Bin-Cao/Bgolearn/blob/main/figures/funding.png)

---



## ライセンス

MITライセンスの下で公開されています。
学術・商用いずれの用途でも自由に利用可能です。

