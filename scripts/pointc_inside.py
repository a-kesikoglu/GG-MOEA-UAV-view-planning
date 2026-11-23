"""Point C defined against Grid-Oblique-i (no duplicated stations) instead of Grid-Oblique."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.problem import Scene
from scripts.run_experiments import INSTANCES

TAB = os.path.join("results", "tables")
df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
pl = pd.read_csv(os.path.join(TAB, "plans.csv"))
rows = []
for inst, (key, N, _) in INSTANCES.items():
    s = Scene(key)
    e = pl[pl.instance == inst].set_index("plan")
    com = df[(df.exp == "common") & (df.instance == inst) & (df.alg == "GG-MOEA")].sort_values("HV")
    seed = int(com.iloc[len(com) // 2].seed)
    rec = pickle.load(open(os.path.join("results", "raw_runs", "common", f"GG-MOEA__{inst}__{seed}.pkl"), "rb"))
    from scripts.turn_impact import corrected
    F, X = corrected(rec), rec["X"].astype(float)
    for ref in ("Grid-Oblique", "Grid-Oblique-i"):
        cap = e.loc[ref, "f2"]
        ok = F[:, 1] <= cap
        if not ok.any():
            rows.append(dict(instance=inst, ref=ref, dC3=np.nan, dgamma=np.nan))
            continue
        k = np.flatnonzero(ok)[np.argmin(F[ok, 0])]
        V, QL, D, CA, U, GS = s.visibility(X[k])
        A = s.A
        c3 = 100 * A[V.sum(0) >= 3].sum() / A.sum()
        rows.append(dict(instance=inst, ref=ref, dC3=c3 - e.loc[ref, "C3_opt"],
                         dgamma=np.degrees(F[k, 3]) - e.loc[ref, "f4"]))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(TAB, "check_pointc_inside.csv"), index=False)
print(d.pivot(index="instance", columns="ref", values="dC3").round(1))
print(d.pivot(index="instance", columns="ref", values="dgamma").round(1))
