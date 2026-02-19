# Bgolearn

## Documentation
* Paper: [https://arxiv.org/pdf/2601.06820](https://arxiv.org/pdf/2601.06820)
* English Manual: [https://bgolearn.netlify.app/](https://bgolearn.netlify.app/)
* 中文手册: [https://bgolearn-chi.netlify.app/](https://bgolearn-chi.netlify.app/)


---



**Bgolearn** is a unified Bayesian Global Optimization framework for data-efficient scientific discovery, with a particular focus on materials informatics and experimental design.

The framework provides a structured implementation of Gaussian Process Regression together with uncertainty-aware acquisition strategies, enabling adaptive sampling under limited experimental or computational budgets. Bgolearn is designed to support reproducible, closed-loop optimization workflows in AI-driven scientific research.

---

<p align="center">
<img width="500" height="650" src="https://github.com/user-attachments/assets/c52b1645-f971-4597-8f85-7012c9f5168e" />
</p>

---

## Core Capabilities

### Data-Efficient Optimization

Bgolearn is tailored for small-data scientific scenarios where each evaluation is costly. It integrates:

* Gaussian Process surrogate modeling
* Principled uncertainty quantification
* Acquisition strategies balancing exploration and exploitation

This design enables rapid convergence while minimizing experimental or simulation cost.

---

### Single-Objective Optimization

For intensity-driven or performance-maximization tasks, Bgolearn supports standard and robust acquisition functions, including:

* Expected Improvement
* Upper Confidence Bound
* Probability of Improvement

These strategies ensure stable and reproducible optimization in high-cost experimental settings.

---

### Multi-Objective Optimization

Bgolearn extends naturally to multi-objective scenarios, providing:

* Simultaneous optimization of competing objectives
* Pareto front exploration
* Uncertainty-aware trade-off analysis

This capability is particularly suitable for structure–property co-optimization and materials design problems involving conflicting targets.

---

## Citation

If Bgolearn contributes to your research, please cite:

```bibtex
@article{cao2026bgolearn,
  title={Bgolearn: A Unified Bayesian Optimization Framework for Accelerating Materials Discovery},
  author={Cao, Bin and Xiong, Jie and Ma, Jiaxuan and Tian, Yuan and Hu, Yirui and He, Mengwei and Zhang, Longhan and Wang, Jiayu and Hui, Jian and Liu, Li and others},
  journal={arXiv preprint arXiv:2601.06820},
  year={2026}
}
```

---

## License

Released under the MIT License.

---

## Contributors

[![Contributors](https://contrib.rocks/image?repo=Bin-Cao/Bgolearn)](https://github.com/Bin-Cao/Bgolearn/graphs/contributors)

