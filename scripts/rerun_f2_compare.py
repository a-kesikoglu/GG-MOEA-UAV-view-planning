"""Compare the GG-MOEA runs on T4 and T8 searched with the corrected f2 against the stored runs
re-scored with the corrected f2: HV per seed and the operating points C and Q of the median run."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import hv
from src.problem import Scene
from scripts.turn_impact import corrected
from scripts.run_experiments import INSTANCES
from scripts.mission_checks import vis, coverage

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")


def load(exp, inst, seed):
    with open(os.path.join(RAW, exp, f"GG-MOEA__{inst}__{seed}.pkl"), "rb") as fh:
        return pickle.load(fh)


def main():
    ref = pickle.load(open(os.path.join(TAB, "reference_sets.pkl"), "rb"))
    base = pd.read_csv(os.path.join(TAB, "missions.csv"))
    rows, pts = [], []
    for inst in ("T4", "T8"):
        r = ref[inst]
        key = INSTANCES[inst][0]
        s, sd = Scene(key), Scene(key + "_d5k")
        cap = base[(base.instance == inst) & (base.plan == "Grid-Oblique")].f2.iloc[0]
        hv_old, hv_new = [], []
        recs = {}
        for seed in range(1, 21):
            a, b = load("common", inst, seed), load("rerun_f2", inst, seed)
            Fa = corrected(a)
            hv_old.append(hv(Fa, a["CV"], r["ideal"], r["nadir"]))
            hv_new.append(hv(b["F"], b["CV"], r["ideal"], r["nadir"]))
            recs[seed] = (a, Fa, b)
        p = stats.wilcoxon(np.array(hv_new) - np.array(hv_old)).pvalue
        rows.append(dict(instance=inst, hv_old=np.mean(hv_old), hv_new=np.mean(hv_new), sd_old=np.std(hv_old),
                         sd_new=np.std(hv_new), med_diff=np.median(np.array(hv_new) - np.array(hv_old)), p=p))
        for lab, hvs, idx in (("stored", hv_old, 0), ("rerun", hv_new, 2)):
            o = np.argsort(hvs)
            seed = int(o[len(o) // 2]) + 1
            rec = recs[seed][idx]
            F = recs[seed][1] if idx == 0 else rec["F"]
            X = rec["X"].astype(float)
            feas = rec["CV"] <= 0
            for name, k in (("Q", int(np.flatnonzero(feas)[np.argmin(F[feas, 0])])),
                            ("C", int(np.flatnonzero(feas & (F[:, 1] <= cap))[np.argmin(F[feas & (F[:, 1] <= cap), 0])]))):
                V, QL, D, CA, U, GS = vis(sd, X[k])
                c = coverage(sd, V, GS)
                pts.append(dict(instance=inst, source=lab, seed=seed, point=name, f1=F[k, 0], f2=F[k, 1],
                                f4=np.degrees(F[k, 3]), C3_dense=c["C3"], G1_dense=c["G1"], C3g_dense=c["C3g"]))
    d = pd.DataFrame(rows)
    q = pd.DataFrame(pts)
    d.to_csv(os.path.join(TAB, "rerun_f2_hv.csv"), index=False)
    q.to_csv(os.path.join(TAB, "rerun_f2_points.csv"), index=False)
    print(d.round(4).to_string())
    print(q.round(3).to_string())


if __name__ == "__main__":
    main()
