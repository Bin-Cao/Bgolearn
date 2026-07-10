<h1 align="center">Bgolearn</h1>

<p align="center">
  材料探索を加速する統合ベイズ最適化フレームワーク。
</p>

<p align="center">
  <a href="https://pypi.org/project/bgolearn/"><img src="https://img.shields.io/pypi/v/bgolearn?style=flat-square&label=PyPI" alt="PyPI"></a>
  <a href="https://doi.org/10.1038/s41524-026-02226-3"><img src="https://img.shields.io/badge/DOI-10.1038%2Fs41524--026--02226--3-blue?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Bin-Cao/Bgolearn?style=flat-square" alt="License"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/stargazers"><img src="https://img.shields.io/github/stars/Bin-Cao/Bgolearn?style=flat-square" alt="Stars"></a>
  <a href="https://bgolearn.netlify.app/"><img src="https://img.shields.io/badge/docs-online-2f6f9f?style=flat-square" alt="Documentation"></a>
</p>

<p align="center">
  <strong>言語：</strong>
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_de.md">Deutsch</a>
</p>

<p align="center">
  <strong>Playground:</strong>
  <a href="https://bin-cao.github.io/Bgolearn/">ベイズ最適化インタラクティブゲーム</a>
</p>

---

## 注目の紹介

[**Bayesian Global Optimization**](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) は、中国科学院院士である Tong-Yi Zhang 教授による Springer 書籍 *An Introduction to Materials Informatics* の第 1 章です。本章のアクティブラーニングの例と結果は **Bgolearn** によって実装され、Bgolearn に依存しています。

## 概要

**Bgolearn** は、**ベイズグローバル最適化（Bayesian Global Optimization, BGO）** のための研究指向 Python フレームワークです。データ駆動型の材料探索、実験設計、バーチャルスクリーニングを対象としており、測定コストが高く、限られたデータから次の候補を慎重に選ぶ必要がある研究に適しています。

本フレームワークは、代理モデル、不確実性評価、獲得関数、アクティブラーニング、候補ランキングを一つのワークフローとして統合します。回帰と分類の両方に対応しているため、既存の実験データから次に評価すべき有望な材料候補を効率よく抽出できます。

## 主な特徴

- 回帰、分類、アクティブラーニング、バーチャルスクリーニングを統一的に扱えます。
- Gaussian process、SVM、random forest、AdaBoost、MLP など複数の代理モデルを利用できます。
- EI、AEI、EQI、REI、UCB、PoI、PES、KG など、ノイズの有無に応じた獲得関数を備えています。
- 最小化、最大化、多目的研究ワークフローにおける候補推薦を支援します。
- コードを書かずに試せる軽量なローカル Web インターフェースを提供します。

## リソース

| リソース | リンク |
| --- | --- |
| 論文 | [npj Computational Materials](https://doi.org/10.1038/s41524-026-02226-3) |
| 書籍章 | [Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| マニュアル | [bgolearn.netlify.app](https://bgolearn.netlify.app/) |
| 動画チュートリアル | [Bilibili](https://www.bilibili.com/video/BV1LTtLeaEZp) |
| 会議発表 | [CMC 2025](https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842) |
| 多目的モジュール | [MultiBgolearn](https://github.com/Bin-Cao/MultiBgolearn) |
| 公式 GUI | [BgoFace](https://github.com/Bgolearn/BgoFace) |
| サンプルコードとデータ | [CodeDemo](https://github.com/Bgolearn/CodeDemo) |

## 代表的な応用例

Bgolearn は、材料科学および工学分野のさまざまな研究で活用されています。以下は代表的な論文の一部です：

### 2026

| 掲載誌・出版元 | 応用内容 |
| --- | --- |
| Surfaces and Interfaces | [WS2 薄膜の光化学合成](https://www.sciencedirect.com/science/article/pii/S2468023026013799) |
| JPhys Materials | [TaNbMoVW 耐火高エントロピー合金の設計](https://iopscience.iop.org/article/10.1088/2515-7639/ae44d1/meta) |
| Chemical Science | [表面反応の最適化](https://pubs.rsc.org/sc/article/17/22/11114/1231470/A-dual-mode-large-language-model-assistant-for-on) |
| Springer Nature | [書籍章：Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| Journal of Materials Informatics | [SciBERT ハイパーパラメータの最適化](https://www.oaepublish.com/articles/jmi.2025.78) |
| Science Bulletin | [超高耐久・高活性触媒の発見](https://www.sciencedirect.com/science/article/pii/S2095927325012678) |
| Aggregate | [G-quadruplex 深共晶円偏光発光材料の発見](https://onlinelibrary.wiley.com/doi/pdf/10.1002/agt2.70307) |
| Materials Science in Semiconductor Processing | [パルスレーザー合成による置換型 Hf ドープ p 型 MoS2](https://www.sciencedirect.com/science/article/pii/S1369800126005263) |

### 2025

| 掲載誌・出版元 | 応用内容 |
| --- | --- |
| Nano Letters | [超高真空下の自律実験室](https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.5c02445?casa_token=DycwWKxkjjQAAAAA:_qVVZ56VuzbHDnLmJ_-8mUtHatu9S8rOXE78HHGjmNhADLlr7qr-4rPWsAuIOVide29eEy6gOfvzC3do) |
| Small | [抗腫瘍治療に向けた機械学習設計ナノザイムシステム](https://onlinelibrary.wiley.com/doi/10.1002/smll.202408750?utm_source=chatgpt.com) |
| Computational Materials Science | [Mg-Ca-Zn 合金の最適化](https://www.sciencedirect.com/science/article/pii/S0927025625000084) |
| Measurement | [EPB シールド工法における発泡剤最適化](https://www.sciencedirect.com/science/article/pii/S0263224124013940) |
| Intelligent Computing | [ベイズ学習によるメタサーフェス設計](https://spj.science.org/doi/pdf/10.34133/icomputing.0135) |

### 2024

| 掲載誌・出版元 | 応用内容 |
| --- | --- |
| Materials & Design | [アクティブラーニングによる鉛フリーはんだ合金設計](https://www.sciencedirect.com/science/article/pii/S0264127524002946) |
| npj Computational Materials | [Bgolearn バックエンドを用いた MLMD プラットフォーム](https://www.nature.com/articles/s41524-024-01243-4) |

## インストール

PyPI からインストールします：

```bash
pip install Bgolearn
```

最新版へアップグレードします：

```bash
pip install --upgrade Bgolearn
```

インストール済みバージョンを確認します：

```bash
pip show Bgolearn
```

## インターフェースの起動

リポジトリをクローンし、ローカル UI を起動します：

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
python bgolearn_ui.py
```

その後、ブラウザで開きます：

```text
http://127.0.0.1:8787
```

## 引用

研究で Bgolearn を使用した場合は、以下を引用してください：

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

## 資金支援

**Bgolearn** は、上海市経済・情報化委員会が支援する [オープンソース人工知能支援プログラム（2025）](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html) に採択されています。

プロジェクト資料：[figures/funding.png](../figures/funding.png)

## ライセンス

Bgolearn は MIT License の下で公開されています。
