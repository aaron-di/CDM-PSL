<h1 align="center"> CDM-PSL: Expensive Multi-Objective Bayesian Optimization Based on Diffusion Models </h1>

This repository includes the code for the paper “_**Expensive Multi-Objective Bayesian Optimization Based on Diffusion Models**_”, which has been accepted at **AAAI 2025**. The paper is available at: https://ojs.aaai.org/index.php/AAAI/article/view/34913

## 💥 News 💥

- 🎉🎉🎉 **[Dec 10, 2024]** Our paper is accepted at AAAI 2025!
  
## Abstract

**CDM-PSL (Composite Diffusion Model-Based Pareto Set Learning)** is a novel algorithm designed to address the challenges of expensive multi-objective Bayesian optimization (EMOBO). While MOBO has shown great potential in solving expensive multi-objective optimization problems (EMOPs), it struggles to model complex distributions of the Pareto optimal solutions when only a limited number of function evaluations are available.

Existing Pareto set learning methods often exhibit instability in expensive scenarios, leading to significant deviations between the obtained solution set and the true Pareto set (PS). To overcome these challenges, CDM-PSL introduces **Composite Diffusion Model** to enhance optimization performance.

## Overview

<img src="./assets/CDM_PSL.png">

## 🚀 Getting Started

### Install dependencies

```
cd CDM-PSL/cdm_psl
pip install -r requirements.txt
```

### Run CDM-PSL on DTLZ2

```
bash example_dtlz2.sh
```

### File Structure
```
cdm_psl/
├── 📂 evolution/         
├── 📂 learning/          
├── 📂 mobo/               
├── 📜 data_enhancement.py # Data augmentation utilities
├── 📜 diffusion.py        # [Core] Diffusion model implementation
├── 🗒️ example_dtlz2.sh    # Example execution scripts
├── 📜 lhs.py              # Latin Hypercube Sampling utilities
├── 📜 problem.py          # Problem definition
├── 📜 run_d20.py          # [Main] Main algorithm workflow
└── 📜 utils.py            # Common utility functions
```


## Citation

If you find our work helpful, please cite the following BibTeX entry:

```
@inproceedings{li2025expensive,
  title={Expensive multi-objective bayesian optimization based on diffusion models},
  author={Li, Bingdong and Di, Zixiang and Lu, Yongfan and Qian, Hong and Wang, Feng and Yang, Peng and Tang, Ke and Zhou, Aimin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={25},
  pages={27063--27071},
  year={2025}
}
```
