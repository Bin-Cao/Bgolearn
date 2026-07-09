<h1 align="center">Bgolearn</h1>

<p align="center">
  Ein einheitliches Bayesian-Optimization-Framework zur Beschleunigung der Materialentdeckung.
</p>

<p align="center">
  <a href="https://pypi.org/project/bgolearn/"><img src="https://img.shields.io/pypi/v/bgolearn?style=flat-square&label=PyPI" alt="PyPI"></a>
  <a href="https://doi.org/10.1038/s41524-026-02226-3"><img src="https://img.shields.io/badge/DOI-10.1038%2Fs41524--026--02226--3-blue?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Bin-Cao/Bgolearn?style=flat-square" alt="License"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/stargazers"><img src="https://img.shields.io/github/stars/Bin-Cao/Bgolearn?style=flat-square" alt="Stars"></a>
  <a href="https://bgolearn.netlify.app/"><img src="https://img.shields.io/badge/docs-online-2f6f9f?style=flat-square" alt="Documentation"></a>
</p>

<p align="center">
  <strong>Sprache:</strong>
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_de.md">Deutsch</a>
</p>

---

## Überblick

**Bgolearn** ist ein forschungsorientiertes Python-Framework für **Bayesian Global Optimization (BGO)**. Es wurde für datengetriebene Materialentdeckung, experimentelles Design und virtuelles Screening entwickelt, also für Szenarien, in denen Messungen teuer sind und jede Empfehlung nachvollziehbar sein muss.

Das Framework bündelt Surrogatmodellierung, Unsicherheitsbewertung, Akquisitionsfunktionen, Active Learning und Kandidatenranking in einem konsistenten Workflow. Es unterstützt sowohl Regressions- als auch Klassifikationsaufgaben und hilft Forschenden, aus vorhandenen Versuchsdaten die vielversprechendsten nächsten Materialkandidaten abzuleiten.

## Kernfunktionen

- Einheitliche Workflows für Regression, Klassifikation, Active Learning und virtuelles Screening.
- Mehrere Surrogatmodelle, darunter Gaussian process, SVM, random forest, AdaBoost und MLP-basierte Modelle.
- Akquisitionsfunktionen für Optimierung mit und ohne Rauschen, darunter EI, AEI, EQI, REI, UCB, PoI, PES und KG.
- Kandidatenempfehlungen für Minimierung, Maximierung und multiobjektive Forschungsabläufe.
- Eine leichtgewichtige lokale Weboberfläche für interaktives Arbeiten.

## Ressourcen

| Ressource | Link |
| --- | --- |
| Paper | [npj Computational Materials](https://doi.org/10.1038/s41524-026-02226-3) |
| Handbuch | [bgolearn.netlify.app](https://bgolearn.netlify.app/) |
| Playground | [Interaktives Spiel zur Bayes'schen Optimierung](https://bin-cao.github.io/Bgolearn/) |
| Video-Tutorial | [Bilibili](https://www.bilibili.com/video/BV1LTtLeaEZp) |
| Konferenzbericht | [CMC 2025](https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842) |
| Multi-Objective-Modul | [MultiBgolearn](https://github.com/Bin-Cao/MultiBgolearn) |
| Offizielle GUI | [BgoFace](https://github.com/Bgolearn/BgoFace) |
| Beispielcode und Daten | [CodeDemo](https://github.com/Bgolearn/CodeDemo) |

## Repräsentative Anwendungen

Bgolearn wurde bereits in mehreren Studien aus Materialwissenschaft und Ingenieurwesen eingesetzt. Die folgende Liste zeigt eine repräsentative Auswahl:

### 2026

| Publikation | Anwendung |
| --- | --- |
| Surfaces and Interfaces | [Photochemische Synthese von WS2-Dünnfilmen](https://www.sciencedirect.com/science/article/pii/S2468023026013799) |
| JPhys Materials | [Design von refraktären TaNbMoVW-High-Entropy-Legierungen](https://iopscience.iop.org/article/10.1088/2515-7639/ae44d1/meta) |
| Chemical Science | [Optimierung von On-Surface-Reaktionen](https://pubs.rsc.org/sc/article/17/22/11114/1231470/A-dual-mode-large-language-model-assistant-for-on) |
| Springer Nature | [Buchkapitel: Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| Journal of Materials Informatics | [Optimierung von SciBERT-Hyperparametern](https://www.oaepublish.com/articles/jmi.2025.78) |
| Science Bulletin | [Entdeckung ultra-beständiger und hochaktiver Katalysatoren](https://www.sciencedirect.com/science/article/pii/S2095927325012678) |
| Aggregate | [Entdeckung von G-Quadruplex Deep-Eutectic-Materialien mit zirkular polarisierter Lumineszenz](https://onlinelibrary.wiley.com/doi/pdf/10.1002/agt2.70307) |
| Materials Science in Semiconductor Processing | [Substitutionell Hf-dotiertes p-Typ-MoS2 durch Pulslasersynthese](https://www.sciencedirect.com/science/article/pii/S1369800126005263) |

### 2025

| Publikation | Anwendung |
| --- | --- |
| Nano Letters | [Self-driving Laboratory unter UHV-Bedingungen](https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.5c02445?casa_token=DycwWKxkjjQAAAAA:_qVVZ56VuzbHDnLmJ_-8mUtHatu9S8rOXE78HHGjmNhADLlr7qr-4rPWsAuIOVide29eEy6gOfvzC3do) |
| Small | [ML-entwickeltes Nanozym-System für Anti-Tumor-Therapie](https://onlinelibrary.wiley.com/doi/10.1002/smll.202408750?utm_source=chatgpt.com) |
| Computational Materials Science | [Optimierung von Mg-Ca-Zn-Legierungen](https://www.sciencedirect.com/science/article/pii/S0927025625000084) |
| Measurement | [Optimierung von Schaummitteln im EPB-Schildvortrieb](https://www.sciencedirect.com/science/article/pii/S0263224124013940) |
| Intelligent Computing | [Metasurface-Design durch Bayesian Learning](https://spj.science.org/doi/pdf/10.34133/icomputing.0135) |

### 2024

| Publikation | Anwendung |
| --- | --- |
| Materials & Design | [Bleifreie Lotlegierungen durch Active Learning](https://www.sciencedirect.com/science/article/pii/S0264127524002946) |
| npj Computational Materials | [MLMD-Plattform mit Bgolearn-Backend](https://www.nature.com/articles/s41524-024-01243-4) |

## Installation

Installation über PyPI:

```bash
pip install Bgolearn
```

Upgrade auf die neueste Version:

```bash
pip install --upgrade Bgolearn
```

Installierte Version prüfen:

```bash
pip show Bgolearn
```

## Oberfläche Starten

Repository klonen und lokale UI starten:

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
python bgolearn_ui.py
```

Danach im Browser öffnen:

```text
http://127.0.0.1:8787
```

## Zitieren

Wenn Bgolearn Ihre Forschung unterstützt, zitieren Sie bitte:

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

## Förderung

**Bgolearn** wurde für das [Open-Source Artificial Intelligence Support Program (2025)](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html) ausgewählt, das von der Shanghai Municipal Commission of Economy and Informatization unterstützt wird.

Projektmaterial: [figures/funding.png](../figures/funding.png)

## Lizenz

Bgolearn wird unter der MIT License veröffentlicht.
