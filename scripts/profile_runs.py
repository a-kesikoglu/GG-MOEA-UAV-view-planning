"""Single-threaded time profile of GG-MOEA by component (setup, evaluation, alignment,
variation, selection) on every instance, with a reduced budget of 5000 evaluations."""
import cProfile
import os
import pstats
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numba

numba.set_num_threads(1)
from src.problem import Scene
from src.algorithms.runner import run
from scripts.run_experiments import INSTANCES

GROUPS = {
    "setup": ["snbi_init"],
    "evaluation": ["evaluate_population"],
    "alignment": ["align"],
    "variation": ["__call__"],
    "selection": ["_do"],
}


def cum(stats, name, path_hint=""):
    tot = 0.0
    for (fn, line, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        if func == name and path_hint in fn:
            tot = max(tot, ct)
    return tot


rows = []
for inst, (key, N, _) in INSTANCES.items():
    s = Scene(key)
    run("GG-MOEA", s, N, 600, 99)   # compile
    pr = cProfile.Profile()
    pr.enable()
    rec = run("GG-MOEA", s, N, 5000, 99)
    pr.disable()
    st = pstats.Stats(pr)
    total = sum(v[2] for v in st.stats.values())
    setup = cum(st, "snbi_init")
    ev = sum(v[3] for k, v in st.stats.items() if k[2] == "evaluate")
    al = cum(st, "align", "operators")
    var = cum(st, "__call__", "operators") - al
    sel = cum(st, "do", "survival")
    rows.append(dict(instance=inst, total=total, setup=setup, evaluation=ev, alignment=al, variation=var,
                     selection=sel, other=total - setup - ev - al - var - sel))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join("results", "tables", "profile.csv"), index=False)
