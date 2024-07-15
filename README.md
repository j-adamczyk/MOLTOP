# Molecular Topological Profile (MOLTOP)

Code for paper "Molecular Topological Profile (MOLTOP) - Simple and Strong Baseline for Molecular Graph Classification"
J. Adamczyk, W. Czech.

In this paper, we present Molecular Topological Profile (MOLTOP), a strong baseline for molecular graph classification.
It is based on our previous Local Topological Profile (LTP), but introduces atom and bond features, and uses different
topological descriptors.

Install requirements using one of:
- Poetry and `poetry.lock` (recommended)
- venv and `requirements.txt`

To reproduce experiment results from the main paper, run:
- main results: `python main_moltop.py`
- statistical tests: `python scripts/wilcoxon.py`
- baselines:
  - `python main_baselines.py`
  - `python main_ecfp.py`
- results on LRGB peptides-func: `python main_lrgb.py`
- feature importances: `python main_feature_importances.py`

For specific settings and CLI arguments, see the respective scripts. If you want to use MOLTOP for new datasets,
`feature_extraction.py` contains all code for calculating MOLTOP features for individual graphs. However, next steps,
like dropping constant features and setting Random Forest hyperparameters, are in `main_moltop.py`.

For reproduce results and examples from the supplementary material, run:
- DPCC and Paclitaxel examples: `python scripts/dpcc_and_paclitaxel.py`
- molecule sizes: `python scripts/dataset_mols_sizes.py`
- graph kernels: `python main_graph_kernels.py`
- decalin and bicyclopentyl examples: `python scripts/decalin_and_bicyclopentyl.py`
- 1-WL examples:
  - `python expressivity/1_wl_simple.py`
  - `python expressivity/1_wl_molecular.py`
- 3-WL examples:
  - `python expressivity/3_wl.py`
  - `python expressivity/3_regular_small.py`
  - `python scripts/3_regular_molecules.py`
