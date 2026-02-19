
# PyWPEM

<p align="center">
  <img src="https://github.com/Bin-Cao/TCGPR/assets/86995074/28f69830-4ece-43b3-a887-e78fdb25bcab" width="140" alt="PyWPEM Logo"/>
</p>

<p align="center">
  <strong>Python Toolkit for X-ray Diffraction Simulation, Analysis, and AI-driven Structure Refinement</strong>
</p>

<p align="center">
  <a href="https://pyxplore.netlify.app/">Documentation</a> ·
  <a href="https://arxiv.org/abs/2602.16372v1">Paper (arXiv)</a> ·
  <a href="https://www.pepy.tech/projects/PyXplore">Download Statistics</a>
</p>

---

## Overview

**PyWPEM** is a modular Python framework for **X-ray diffraction (XRD) simulation, decomposition, quantitative analysis, and AI-assisted structure refinement**.

It integrates:

* Physics-based diffraction modeling
* EM-based Bragg optimization
* Structure graph construction
* Extinction and Wyckoff analysis
* Amorphous phase quantification
* AI-driven structural refinement

The toolkit is designed for reproducible scientific workflows in materials characterization and AI for Science research.

---

## Key Features

* **XRD Simulation**
  Accurate diffraction pattern generation from crystallographic information.

* **Peak Decomposition & Quantitative Analysis**
  WPEM-based decomposition and volume fraction determination.

* **Bragg Law Optimization (EM Framework)**
  Expectation-Maximization-based parameter solving.

* **Extinction & Wyckoff Handling**
  Symmetry-aware preprocessing and structural filtering.

* **Graph-Based Structure Representation**
  Crystal graph construction for downstream machine learning tasks.

* **Amorphous Structure Analysis**
  RDF-based quantitative evaluation.

* **Multi-modal Extension**
  Integrated modules for XAS and XPS analysis.

---

## Architecture Overview

```text
PyWPEM/
├── WPEM.py
├── XRDSimulation/
├── EMBraggOpt/
├── Refinement/
├── StructureOpt/
├── GraphStructure/
├── Extinction/
├── Amorphous/
├── Background/
├── Plot/
├── DecomposePlot/
├── WPEMXAS/
├── WPEMXPS/
└── refs/
```

The design follows a **physics-consistent, modular architecture**, enabling independent or pipeline-based execution.

---

## Tables & Figures

<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" />
</p>

<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" />
</p>

---

## Scientific Reference

If you use **PyWPEM** in your research, please cite:

```bibtex
@article{cao2026wpem,
  title={AI-Driven Structure Refinement of X-ray Diffraction},
  author={Bin Cao, Qian Zhang, Zhenjie Feng, Taolue Zhang, Jiaqiang Huang, Lu-Tao Weng, Tong-Yi Zhang},
  journal={arXiv preprint},
  year={2026},
  url={https://arxiv.org/abs/2602.16372v1}
}
```

---

## Maintainer

<table>
  <tr>
    <td width="150" align="center">
      <img src="https://github.com/user-attachments/assets/7e77bd5a-42d6-45db-b8e6-2c82cac81b9d" width="130" style="border-radius: 50%;" />
    </td>
    <td>
      <strong>Bin Cao</strong><br>
      PhD Candidate<br>
      Hong Kong University of Science and Technology (Guangzhou)<br><br>
      Research Area: AI for Science · Intelligent Crystal Structure Analysis<br><br>
      Email: <a href="mailto:bcao686@connect.hkust-gz.edu.cn">bcao686@connect.hkust-gz.edu.cn</a><br>
      Homepage: <a href="https://www.caobin.asia/">https://www.caobin.asia/</a>
    </td>
  </tr>
</table>



---

## Contributing

We welcome contributions from the community.

* Report bugs via Issues
* Propose features
* Submit pull requests
* Contact for academic collaboration

Please ensure code readability, documentation clarity, and scientific correctness before submission.

---

## License

This project is released under the MIT License.

Free for academic and commercial use.
Please cite related publications when used in scientific research.

---

## Contributors

[![Contributors](https://contrib.rocks/image?repo=Bin-Cao/PyWPEM\&v=6)](https://github.com/Bin-Cao/PyWPEM/graphs/contributors)



