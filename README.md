# Bayesian Robust Aggregation for Federated Learning (BRA-FL)

This repository contains the official implementation of the paper:

<div align="center">
<h2 style="border-bottom:0px none #000; margin-bottom: 0;"><a href="link-to-arxiv">Bayesian Robust Aggregation for Federated Learning</a></h2>

  <p style="border-bottom:0px none #000; margin-bottom: 0;">
    <a href="https://scholar.google.com/citations?user=XNBDwTkAAAAJ&hl=en">Aleksandr Karakulev</a> | 
    <a href="https://usamazf.github.io/">Usama Zafar</a> | 
    <a href="https://scholar.google.se/citations?user=PIGlWyYAAAAJ&hl=en">Salman Toor</a> | 
    <a href="https://www.prashantsingh.se/">Prashant Singh</a>
  </p>
  <p><a href="https://www.uu.se/en">Uppsala University</a></p>
</div>

## 🧠 Overview

Federated Learning (FL) enables decentralized training of machine learning models across multiple clients (e.g., hospitals, mobile devices) without sharing raw data. However, FL is vulnerable to **Byzantine attacks**, where malicious clients submit poisoned model updates to compromise the global model.

We propose **Bayesian Robust Aggregation** — a simple yet powerful framework for robust aggregation in FL based on:

<!-- - Bayesian inference under the **Huber $\epsilon$-contamination model** -->

- Adaptive, **parameter-free**, and **generalizable** design
- Applicability to **i.i.d.** and **non-i.i.d.** settings
- Strong defense against **sign-flip**, **backdoor**, and other adversarial attacks

## 📈 Highlights

- 🛡️ Robust to diverse adversarial behaviors
- ⚙️ No tuning of thresholds or trimming parameters
- 📦 Easy to plug into existing FL pipelines
- 📊 Outperforms or matches state-of-the-art under various threat models

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/SciML-FL/bra-fl
cd bra-fl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run an experiment

**TO BE ADDED**

## 📊 Example Results

**TO BE ADDED**

## 🧪 Method Summary

<!-- * We formulate robust aggregation as **likelihood maximization** under the Huber contamination model. -->

- We assume each client update is either:

  - **Benign**: aligned with the true gradient direction
  - **Malicious**: arbitrary or adversarial noise

- We perform **inference over a binary latent variable** for each client to estimate the true mean update vector robustly.

## 🤝 Acknowledgements

This work was supported by Uppsala University. The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725. We thank the open-source community for tools and baselines used in this project.

## 📬 Contact

For questions or collaborations:

- Aleksandr Karakulev — [aleksandr.karakulev@it.uu.se](mailto:aleksandr.karakulev@it.uu.se)
- Usama Zafar — [usama.zafar@it.uu.se](mailto:usama.zafar@it.uu.se)

## 📄 Citation

If you use this work, please cite:

**TO BE UPDATED**

```bibtex
@article{karakulev2025brafl,
  title={Bayesian Robust Aggregation for Federated Learning},
  author={Aleksandr Karakulev and Usama Zafar and Salman Toor and Prashant Singh},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
}
```
