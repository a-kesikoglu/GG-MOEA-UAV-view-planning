"""Recompute all statistics, checks and tables of the article from the stored run records.

The optimization runs are not repeated. Unzip raw_runs.zip from the release assets into the repository
root first, so that results/raw_runs/<experiment>/*.pkl exist. Each step writes to results/tables.
Steps that evaluate missions in the prism model (mission, surface and further checks) take several minutes.
"""
import subprocess
import sys

STEPS = [
    ["scripts/analyze.py"],
    ["scripts/stats_paired.py"],
    ["scripts/engineering.py"],
    ["scripts/resolution_dense.py"],
    ["scripts/robustness.py"],
    ["scripts/extra_checks.py"],
    ["scripts/validate_model.py"],
    ["scripts/mission_checks.py", "plans", "prec", "gsd", "runsel"],
    ["scripts/surface_sensitivity.py", "density", "res", "mc"],
    ["scripts/extra_stats.py", "norm", "igdbal", "cpu", "randfeas", "comp", "supp", "senst7"],
    ["scripts/consistency_checks.py", "turn", "clear", "gsdtol", "prec", "refnew", "igdgrid"],
    ["scripts/turn_impact.py"],
    ["scripts/pointc_inside.py"],
    ["scripts/further_checks.py", "distinct", "clear", "comp", "res", "mc", "gridline"],
    ["scripts/tables2.py"],
    ["scripts/tables3.py"],
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for step in STEPS:
        if only and not any(o in step[0] for o in only):
            continue
        print(">>", " ".join(step), flush=True)
        subprocess.run([sys.executable] + step, check=True)
