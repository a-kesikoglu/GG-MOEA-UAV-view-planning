# GG-MOEA: evolutionary view planning for UAV photogrammetry

Code, processed data and results for the article

> A. Kesikoğlu, *Evolutionary view planning for UAV photogrammetry with geometric guidance and multiple objectives*, submitted to *Swarm and Evolutionary Computation*.

The repository contains the four-objective view planning model on 2.5D surface models, the geometry-guided many-objective evolutionary algorithm (GG-MOEA), the eight competitor implementations or wrappers, the experiment drivers and the scripts that produce every table and figure of the article.

## Layout

```
src/
  config.py              camera, mission and scene definitions
  data_loader.py         swissSURFACE3D download (swisstopo STAC), cropping, surface targets
  photogrammetry.py      projection, local GSD, prism-model visibility, objectives, clearance (Numba)
  problem.py             scene wrapper and pymoo problem
  operators.py           flight patterns, SNBI, candidate pool, viewpoint alignment (VCA), block DE, lifting
  engineering.py         decoupled two-stage greedy plan (Greedy-2S)
  metrics.py             HV, IGD+, Holm correction, Friedman ranks, Nemenyi critical difference
  algorithms/runner.py   GG-MOEA and ablation variants, MOEA/D, CCMO, pymoo competitors
scripts/
  run_experiments.py     common | random | ablation | sensitivity (parallel, resumable)
  analyze.py             reference sets, HV and IGD+ of all runs
  stats_paired.py        matched-pair and rank-sum tests, Friedman ranks (instance and scene blocks)
  engineering.py         reference plans and selected missions in the prism model
  resolution_dense.py    resolution-conditioned coverage and dense-target evaluation
  robustness.py          re-scoring of stored sets, tour experiment, objective dependence
  profile_runs.py        single-run time profile
  tables2.py             LaTeX table bodies (written to results/latex)
  figures.py, fig_scenes.py, fig_framework.py   figures (written to results/figures)
data/processed/          surface targets of the four scenes, the dense target sets and the
                         shared SNBI initial populations (snbi_cache)
results/tables/          indicators of all runs and all statistics reported in the article
```

The final approximation sets and checkpoints of all 2240 runs (`results/raw_runs`, about 250 MB) are provided as `raw_runs.zip` in the release assets. Unzip it in the repository root to rerun the analysis without repeating the optimization.

## Installation

Python 3.12 was used.

```
pip install -r requirements.txt
```

## Reproduction

All commands are run from the repository root. The processed scenes are included; `python src/data_loader.py` rebuilds them from the swisstopo tiles, which are downloaded automatically, and `python src/data_loader.py dense` rebuilds the dense target sets.

```
python scripts/run_experiments.py common 15        # 9 algorithms x 8 instances x 20 seeds
python scripts/run_experiments.py random 15
python scripts/run_experiments.py ablation 15
python scripts/run_experiments.py sensitivity 15
python scripts/analyze.py
python scripts/stats_paired.py
python scripts/engineering.py
python scripts/resolution_dense.py
python scripts/robustness.py
python scripts/profile_runs.py
python scripts/tables2.py
python scripts/figures.py
python scripts/fig_scenes.py
python scripts/fig_framework.py
```

The second argument of `run_experiments.py` is the number of parallel worker processes. The SNBI initial populations in `data/processed/snbi_cache` are loaded when present, so all algorithms start from exactly the populations used in the article; deleting the folder regenerates them.

## Data

The digital surface models are swissSURFACE3D Raster tiles of the Federal Office of Topography swisstopo, available as open data under the swisstopo terms of use. The country outline used in the location map comes from Natural Earth (public domain).

## Citation

See `CITATION.cff`. Please cite the article once it is published.

## Licence

The code is released under the MIT licence (see `LICENSE`). The swisstopo data remain subject to their own terms of use.
