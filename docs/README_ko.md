<h1 align="center">Bgolearn</h1>

<p align="center">
  소재 발견을 가속하기 위한 통합 베이지안 최적화 프레임워크.
</p>

<p align="center">
  <a href="https://pypi.org/project/bgolearn/"><img src="https://img.shields.io/pypi/v/bgolearn?style=flat-square&label=PyPI" alt="PyPI"></a>
  <a href="https://doi.org/10.1038/s41524-026-02226-3"><img src="https://img.shields.io/badge/DOI-10.1038%2Fs41524--026--02226--3-blue?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Bin-Cao/Bgolearn?style=flat-square" alt="License"></a>
  <a href="https://github.com/Bin-Cao/Bgolearn/stargazers"><img src="https://img.shields.io/github/stars/Bin-Cao/Bgolearn?style=flat-square" alt="Stars"></a>
  <a href="https://bgolearn.netlify.app/"><img src="https://img.shields.io/badge/docs-online-2f6f9f?style=flat-square" alt="Documentation"></a>
</p>

<p align="center">
  <strong>언어:</strong>
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_de.md">Deutsch</a>
</p>

<p align="center">
  <strong>Playground:</strong>
  <a href="https://bin-cao.github.io/Bgolearn/">베이지안 최적화 인터랙티브 게임</a>
</p>

---

## 주요 소개

[**Bayesian Global Optimization**](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1)는 중국과학원 원사인 Tong-Yi Zhang 교수의 Springer 도서 *An Introduction to Materials Informatics* 제1장입니다. 이 장의 능동 학습 예제와 결과는 **Bgolearn**으로 구현되며 Bgolearn에 의존합니다.

## 개요

**Bgolearn**은 **Bayesian Global Optimization (BGO)** 을 위한 연구 중심 Python 프레임워크입니다. 데이터 기반 소재 발견, 실험 설계, 가상 스크리닝을 위해 설계되었으며, 측정 비용이 높고 제한된 데이터에서 다음 후보를 신중하게 선택해야 하는 연구 환경에 적합합니다.

이 프레임워크는 대리 모델, 불확실성 평가, 획득 함수, 능동 학습, 후보 순위를 하나의 워크플로로 통합합니다. 회귀와 분류 작업을 모두 지원하므로 기존 실험 데이터에서 다음 단계의 유망한 소재 후보를 효율적으로 추천할 수 있습니다.

## 주요 기능

- 회귀, 분류, 능동 학습, 가상 스크리닝을 위한 통합 워크플로.
- Gaussian process, SVM, random forest, AdaBoost, MLP 등 다양한 대리 모델 지원.
- EI, AEI, EQI, REI, UCB, PoI, PES, KG 등 노이즈 조건을 고려한 획득 함수 제공.
- 최소화, 최대화, 다목적 연구 워크플로를 위한 후보 추천.
- 코드 작성 부담을 줄여 주는 가벼운 로컬 Web 인터페이스 제공.

## 리소스

| 리소스 | 링크 |
| --- | --- |
| 논문 | [npj Computational Materials](https://doi.org/10.1038/s41524-026-02226-3) |
| 도서 장 | [Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| 매뉴얼 | [bgolearn.netlify.app](https://bgolearn.netlify.app/) |
| 동영상 튜토리얼 | [Bilibili](https://www.bilibili.com/video/BV1LTtLeaEZp) |
| 학회 발표 | [CMC 2025](https://cmc2025.scimeeting.cn/cn/web/xue-shu-xin/27167?abstract_id=3726842) |
| 다목적 모듈 | [MultiBgolearn](https://github.com/Bin-Cao/MultiBgolearn) |
| 공식 GUI | [BgoFace](https://github.com/Bgolearn/BgoFace) |
| 예제 코드와 데이터 | [CodeDemo](https://github.com/Bgolearn/CodeDemo) |

## 대표 응용 사례

Bgolearn은 소재 과학 및 공학 분야의 여러 연구에 활용되고 있습니다. 아래는 대표 논문의 일부입니다:

### 2026

| 저널/출판처 | 응용 분야 |
| --- | --- |
| Surfaces and Interfaces | [WS2 박막의 광화학 합성](https://www.sciencedirect.com/science/article/pii/S2468023026013799) |
| JPhys Materials | [TaNbMoVW 내화 고엔트로피 합금 설계](https://iopscience.iop.org/article/10.1088/2515-7639/ae44d1/meta) |
| Chemical Science | [표면 반응 최적화](https://pubs.rsc.org/sc/article/17/22/11114/1231470/A-dual-mode-large-language-model-assistant-for-on) |
| Springer Nature | [도서 장: Bayesian Global Optimization](https://link.springer.com/chapter/10.1007/978-981-95-6091-2_1) |
| Journal of Materials Informatics | [SciBERT 하이퍼파라미터 최적화](https://www.oaepublish.com/articles/jmi.2025.78) |
| Science Bulletin | [초고내구성 고활성 촉매 발견](https://www.sciencedirect.com/science/article/pii/S2095927325012678) |
| Aggregate | [G-quadruplex deep-eutectic 원형 편광 발광 소재 발견](https://onlinelibrary.wiley.com/doi/pdf/10.1002/agt2.70307) |
| Materials Science in Semiconductor Processing | [펄스 레이저 합성을 통한 치환형 Hf 도핑 p형 MoS2](https://www.sciencedirect.com/science/article/pii/S1369800126005263) |

### 2025

| 저널/출판처 | 응용 분야 |
| --- | --- |
| Nano Letters | [초고진공 환경의 자율 실험실](https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.5c02445?casa_token=DycwWKxkjjQAAAAA:_qVVZ56VuzbHDnLmJ_-8mUtHatu9S8rOXE78HHGjmNhADLlr7qr-4rPWsAuIOVide29eEy6gOfvzC3do) |
| Small | [항종양 치료를 위한 ML 기반 나노자임 시스템](https://onlinelibrary.wiley.com/doi/10.1002/smll.202408750?utm_source=chatgpt.com) |
| Computational Materials Science | [Mg-Ca-Zn 합금 최적화](https://www.sciencedirect.com/science/article/pii/S0927025625000084) |
| Measurement | [EPB 쉴드 시공의 발포제 최적화](https://www.sciencedirect.com/science/article/pii/S0263224124013940) |
| Intelligent Computing | [베이지안 학습 기반 메타표면 설계](https://spj.science.org/doi/pdf/10.34133/icomputing.0135) |

### 2024

| 저널/출판처 | 응용 분야 |
| --- | --- |
| Materials & Design | [능동 학습 기반 무연 솔더 합금 설계](https://www.sciencedirect.com/science/article/pii/S0264127524002946) |
| npj Computational Materials | [Bgolearn 백엔드를 사용한 MLMD 플랫폼](https://www.nature.com/articles/s41524-024-01243-4) |

## 설치

PyPI에서 설치합니다:

```bash
pip install Bgolearn
```

최신 버전으로 업그레이드합니다:

```bash
pip install --upgrade Bgolearn
```

설치된 버전을 확인합니다:

```bash
pip show Bgolearn
```

## 인터페이스 실행

저장소를 클론하고 로컬 UI를 실행합니다:

```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
python bgolearn_ui.py
```

그다음 브라우저에서 엽니다:

```text
http://127.0.0.1:8787
```

## 인용

연구에 Bgolearn을 사용했다면 다음 논문을 인용해 주세요:

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

## 지원

**Bgolearn**은 Shanghai Municipal Commission of Economy and Informatization이 지원하는 [Open-Source Artificial Intelligence Support Program (2025)](https://www.sheitc.sh.gov.cn/cyfz/20250728/e571042d40384fcf859a347eb99e10df.html)에 선정되었습니다.

프로젝트 자료: [figures/funding.png](../figures/funding.png)

## 라이선스

Bgolearn은 MIT License로 배포됩니다.
