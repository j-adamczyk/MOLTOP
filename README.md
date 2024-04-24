# Molecular Topological Profile (MOLTOP)

Code for paper "Molecular Topological Profile (MOLTOP) - Simple and Strong Baseline for Molecular Graph Classification"

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

For specific settings and CLI arguments, see the respective scripts.

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
