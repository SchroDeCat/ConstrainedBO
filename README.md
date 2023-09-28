
# Constrained Bayesian Optimization

The project includes implementation for COBALT and the baselines. The main components of the src are structured as follows:

```cmd
📦src
 ┣ 📂SCBO
 ┣ 📂cmes_ibo
 ┣ 📂models
 ┣ 📂opt
 ┣ 📂utils
```

1. src/SCBO contains the implementation of [David Eriksson and Matthias Poloczek. Scalable constrained Bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 730–738. PMLR, 2021.](https://doi.org/10.48550/arxiv.2002.08526) adapted from [BoTorch tutorial](https://botorch.org/tutorials/scalable_constrained_bo).

2. src/cmes-ibo contains the implementation of [Takeno, Shion, Tomoyuki Tamura, Kazuki Shitara, and Masayuki Karasuyama. "Sequential and parallel constrained max-value entropy search via information lower bound." In International Conference on Machine Learning, pp. 20960-20986. PMLR, 2022.](https://proceedings.mlr.press/v162/takeno22a.html) acquired from [GitHub Repo](https://github.com/takeuchi-lab/CMES-IBO).

3. src/models, src/opt, and src/utils contain the main components of our implementation.

## 1. Environment

We've exported the conda environment into ./enviroment.yml. 

The following command allows restoring the conda environment from the yml file.

```cmd
conda env create -f environment.yml
```

We tested the execution on M1-chip Mac with OS 13.5.2 (22G91) and 16G memory.

## 2. Instruction

We provide commands to run the scripts for testing and visualization. ***Note*** that a specific folder structure might be required to store the results.

### (1) Tests & Visualization

```cmd
$ Scan different thresholds on Rastrigin-1D-1C
python test_portion.py

$ Plot the simple regret and distribution of the objective value
python visualize_scan.py

$ Run all tests on different CBO tasks
python test_general_iclr.py

$ Plot the simple regret curves from the CBO tasks.
python visualize_res_iclr.py
```

### (2) Others

The illustrations in Figure 2 are generated by ./visualize_cbo_noisy_targets.ipynb
