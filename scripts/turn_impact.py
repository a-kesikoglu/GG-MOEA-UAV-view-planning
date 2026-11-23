"""Effect of removing zero-length legs before summing the turning angles of f2 on the HV results
and on the selection of operating point C."""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.photogrammetry import build_tour
from src.metrics import hv
from scripts.consistency_checks import turn_terms
from scripts.run_experiments import INSTANCES, COMPETITORS
from scripts.stats_paired import friedman

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")
ALGS = ["GG-MOEA"] + COMPETITORS
W = C.MISSION["w_turn"] / 1000.0


def corrected(rec):
    F = rec["F"].copy()
    N = rec["N"]
    for k, x in enumerate(rec["X"].astype(float)):
        pos = np.ascontiguousarray(x.reshape(N, 5)[:, :3])
        a, b = turn_terms(pos, build_tour(pos))
        F[k, 1] += W * (b - a)
    return F


def main():
    ref = pickle.load(open(os.path.join(TAB, "reference_sets.pkl"), "rb"))
    rows = []
    for inst in INSTANCES:
        r = ref[inst]
        for a in ALGS:
            for s in range(1, 21):
                rec = pickle.load(open(os.path.join(RAW, "common", f"{a.replace('/', '-')}__{inst}__{s}.pkl"), "rb"))
                F2 = corrected(rec)
                rows.append(dict(instance=inst, alg=a, seed=s, HV=hv(rec["F"], rec["CV"], r["ideal"], r["nadir"]),
                                 HVc=hv(F2, rec["CV"], r["ideal"], r["nadir"])))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "check_turn_hv.csv"), index=False)
    print("max |HV change|", (d.HVc - d.HV).abs().max())
    for col in ("HV", "HVc"):
        M = d.groupby(["instance", "alg"])[col].mean().unstack()[ALGS].loc[list(INSTANCES)].values
        mr, chi, p, *_ = friedman(M)
        print(col, dict(zip(ALGS, np.round(mr, 2))))
    flips = 0
    for inst in INSTANCES:
        g = d[d.instance == inst]
        base = g[g.alg == "GG-MOEA"].set_index("seed")
        for a in COMPETITORS:
            o = g[g.alg == a].set_index("seed")
            s0 = np.sign(np.median(o.HV - base.HV)) * (stats.wilcoxon(o.HV - base.HV).pvalue < 0.05)
            s1 = np.sign(np.median(o.HVc - base.HVc)) * (stats.wilcoxon(o.HVc - base.HVc).pvalue < 0.05)
            flips += int(s0 != s1)
    print("unadjusted test outcomes changed:", flips)
    # operating point C on the median-HV run of GG-MOEA
    base = pd.read_csv(os.path.join(TAB, "missions.csv"))
    plans = pd.read_csv(os.path.join(TAB, "plans.csv"))
    chk = pd.read_csv(os.path.join(TAB, "check_turn.csv")) if os.path.exists(os.path.join(TAB, "check_turn.csv")) else None
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    for inst in INSTANCES:
        com = df[(df.exp == "common") & (df.instance == inst) & (df.alg == "GG-MOEA")].sort_values("HV")
        seed = int(com.iloc[len(com) // 2].seed)
        rec = pickle.load(open(os.path.join(RAW, "common", f"GG-MOEA__{inst}__{seed}.pkl"), "rb"))
        F, F2 = rec["F"], corrected(rec)
        cap_old = base[(base.instance == inst) & (base.plan == "Grid-Oblique")].f2.iloc[0]
        from src.problem import Scene
        from src.operators import double_grid_oblique
        s = Scene(INSTANCES[inst][0])
        g = double_grid_oblique(s, INSTANCES[inst][1]).reshape(-1, 5)
        pos = np.ascontiguousarray(g[:, :3])
        a_, b_ = turn_terms(pos, build_tour(pos))
        cap_new = cap_old + W * (b_ - a_)
        ok0, ok1 = F[:, 1] <= cap_old, F2[:, 1] <= cap_new
        c0 = np.flatnonzero(ok0)[np.argmin(F[ok0, 0])] if ok0.any() else None
        c1 = np.flatnonzero(ok1)[np.argmin(F2[ok1, 0])] if ok1.any() else None
        print(inst, "cap", round(cap_old, 3), "->", round(cap_new, 3), "C index", c0, "->", c1,
              "" if c0 == c1 else f"f1 {F[c0, 0]:.3f} -> {F2[c1, 0]:.3f}")


if __name__ == "__main__":
    main()
