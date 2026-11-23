# GG-MOEA: evolutionary view planning for UAV photogrammetry

Code, processed data and results for the article

> A. Kesikoğlu, *Multi-objective UAV photogrammetric flight planning on open digital surface models*, manuscript submitted for publication.

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
  extra_checks.py        HV of the initial populations and gains, leave-one-algorithm-out reference
                         sets, duplicate stations, Jacobian against on-axis GSD
  run_additional.py      single-component initial populations, random start with N=100, parameter
                         settings on the independent scene
  validate_model.py      local GSD against closed forms and finite differences; visibility against
                         brute-force intersection with all prisms
  mission_checks.py      grid variants, path statistics, duplicates, distinct-view coverage, image
                         graphs, full-projection precision, GSD comparison, operating points of all runs
  surface_sensitivity.py target density and voxel offsets, raster resolution, DSM perturbation
  extra_stats.py         normalisation bounds, density-balanced IGD+, equal-CPU comparison,
                         feasibility of random starts, single-component starts, run-level supplement
  figures_extra.py       additional figures of the article
  consistency_checks.py  turning term with coinciding stations, closed-prism clearance, GSD bound,
                         precision on a common well-conditioned target set, reference-set coverage,
                         IGD+ grid variants, construction times
  turn_impact.py         effect of the turning-term definition on HV, ranks and point C
  update_f2.py           f2 of the assessed missions with the turning term of the article
  pointc_inside.py       operating point C with the duplicate-free oblique grid as reference
  further_checks.py      distinct-view coverage at g*, closed-prism clearance and HV, evaluations and
                         final-set sizes, DSM perturbation and raster resolution by component, oblique
                         grid flown along its lines
  rerun_f2.py            GG-MOEA runs on T4 and T8 with the turning term of the article (same seeds)
  rerun_f2_compare.py    HV and operating points of these runs against the stored runs
  profile_runs.py        single-run time profile
  tables2.py             LaTeX table bodies (written to results/latex)
  tables3.py             LaTeX table bodies of the mission, run-level and model checks (results/latex)
  figures.py, fig_scenes.py, fig_framework.py   figures (written to results/figures)
data/processed/          surface targets of the four scenes, the dense target sets and the
                         shared SNBI initial populations (snbi_cache)
results/tables/          indicators of all runs and all statistics reported in the article
```

The final approximation sets and checkpoints of all runs (`results/raw_runs`) are provided as `raw_runs.zip` in the release assets (release v1.0). Unzip it in the repository root to rerun the analysis without repeating the optimization.

## Installation

Python 3.12 was used.

```
pip install -r requirements.txt
```

## Quick example

`python example.py` runs GG-MOEA for 3000 evaluations on the urban scene with 50 images and prints the indicators of its best mission (a few minutes on one core).

## Reproduction

`python reproduce.py` recomputes all statistics and tables from the run records in `results/raw_runs` (unzip `raw_runs.zip` first) without repeating the optimization. The individual steps are listed below.

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
python scripts/extra_checks.py
python scripts/run_additional.py components,random100,sens_t7 15
python scripts/rerun_f2.py 15
python scripts/rerun_f2_compare.py
python scripts/validate_model.py
python scripts/mission_checks.py plans prec gsd runsel
python scripts/surface_sensitivity.py density res mc
python scripts/extra_stats.py norm igdbal cpu randfeas comp supp senst7
python scripts/figures_extra.py
python scripts/consistency_checks.py turn clear gsdtol prec refnew igdgrid timing
python scripts/turn_impact.py
python scripts/pointc_inside.py
python scripts/further_checks.py distinct clear comp res mc gridline
python scripts/profile_runs.py
python scripts/tables2.py
python scripts/tables3.py
python scripts/figures.py
python scripts/fig_scenes.py
python scripts/fig_framework.py
```

The second argument of `run_experiments.py` is the number of parallel worker processes. The SNBI initial populations in `data/processed/snbi_cache` are loaded when present, so all algorithms start from exactly the populations used in the article; deleting the folder regenerates them.

## Data

The digital surface models are swissSURFACE3D Raster tiles of the Federal Office of Topography swisstopo, available as open data under the swisstopo terms of use. The release year of every tile is fixed in `src/config.py`, and `data/tiles.sha256` lists the SHA-256 checksum of each tile file; `src/data_loader.py` downloads only these files from the swisstopo STAC API (collection `ch.swisstopo.swisssurface3d-raster`, items `swisssurface3d-raster_<year>_<tile>`, assets `*_0.5_2056_5728.tif`) and stops if a checksum differs. The tiles are cropped to the scene bounds in LV95 with LN02 heights, cells without data are filled from the nearest valid cell, and the 0.5 m cells are aggregated to the processing resolution by their block maximum. The country outline used in the location map comes from Natural Earth (public domain).

## Citation

See `CITATION.cff`. Please cite the article once it is published.

## Licence

The code is released under the MIT licence (see `LICENSE`). The swisstopo data remain subject to their own terms of use.
