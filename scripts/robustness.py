"""Robustness checks on the final approximation sets of the common-initialization experiment.

1. Ranking under modified model parameters: the stored approximation sets are re-evaluated with
   beta* = 15 or 35 deg and with path weights (w_c, w_t) halved or doubled; HV is recomputed with a
   reference set built from the re-evaluated sets and algorithms are ranked by instance means.
2. Tour construction: for the median-HV GG-MOEA run of every instance, the length-based tour is
   compared with a tour improved by 2-opt moves on the full weighted path cost.
3. Objective dependence: Spearman correlations of the objectives over the union of final sets.
"""
import glob
import math
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import hv, nondominated, friedman_ranks
from src.photogrammetry import build_tour, path_costs, clearance_violation
from scripts.run_experiments import INSTANCES, COMPETITORS

TAB = os.path.join("results", "tables")
ALGS = ["GG-MOEA"] + COMPETITORS
SETTINGS = {
    "default": {},
    "beta15": {8: math.radians(15)},
    "beta35": {8: math.radians(35)},
    "w_half": {12: 1.0, 13: 2.5},
    "w_double": {12: 4.0, 13: 10.0},
}


def _reeval(args):
    path, setting = args
    import numba
    numba.set_num_threads(1)
    from src.problem import Scene
    with open(path, "rb") as fh:
        r = pickle.load(fh)
    s = Scene(r["scene"])
    for k, v in SETTINGS[setting].items():
        s.prm[k] = v
    X = r["X"].astype(float)
    if (r["CV"] > 0).all():
        return r["instance"], r["alg"], r["seed"], setting, np.empty((0, 4))
    F, CV, _ = s.evaluate(X)
    return r["instance"], r["alg"], r["seed"], setting, F[CV <= 0]


def ranking_robustness(nw=15):
    files = sorted(glob.glob(os.path.join("results", "raw_runs", "common", "*.pkl")))
    tasks = [(f, st) for f in files for st in SETTINGS if st != "default"]
    with Pool(nw) as p:
        res = p.map(_reeval, tasks, chunksize=4)
    # default setting uses the stored objective vectors
    for f in files:
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        res.append((r["instance"], r["alg"], r["seed"], "default", r["F"][r["CV"] <= 0]))
    rows = []
    for st in SETTINGS:
        for inst in INSTANCES:
            sets = [(a, sd, F) for (i, a, sd, s_, F) in res if i == inst and s_ == st]
            allF = np.vstack([F for _, _, F in sets if len(F)])
            R = nondominated(allF)
            ideal, nadir = R.min(0), R.max(0)
            for a, sd, F in sets:
                h = hv(F, np.zeros(len(F)), ideal, nadir) if len(F) else 0.0
                rows.append(dict(setting=st, instance=inst, alg=a, seed=sd, HV=h))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "robustness_hv.csv"), index=False)
    out = []
    for st in SETTINGS:
        M = df[df.setting == st].groupby(["instance", "alg"]).HV.mean().unstack()[ALGS].values
        r, chi, p = friedman_ranks(M, higher_better=True)
        out.append(dict(setting=st, **{a: rk for a, rk in zip(ALGS, r)}, p=p))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(TAB, "robustness_ranks.csv"), index=False)
    print(out.round(2).to_string(index=False))


@njit(cache=True)
def weighted_two_opt(pos, dirs, tour, w_c, w_t, max_pass):
    best_f2, f4, L = path_costs(pos, dirs, tour, w_c, w_t)
    N = tour.shape[0]
    t = tour.copy()
    for _ in range(max_pass):
        improved = False
        for i in range(0, N - 1):
            for k in range(i + 1, N):
                cand = t.copy()
                lo, hi = i, k
                while lo < hi:
                    tmp = cand[lo]
                    cand[lo] = cand[hi]
                    cand[hi] = tmp
                    lo += 1
                    hi -= 1
                f2, f4c, Lc = path_costs(pos, dirs, cand, w_c, w_t)
                if f2 < best_f2 - 1e-9:
                    best_f2 = f2
                    t = cand
                    improved = True
        if not improved:
            break
    return t, best_f2


def tour_experiment():
    from src.problem import Scene
    from src import config as C
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        s = Scene(key)
        d = df[(df.exp == "common") & (df.instance == inst) & (df.alg == "GG-MOEA")].sort_values("HV")
        seed = int(d.iloc[len(d) // 2]["seed"])
        with open(os.path.join("results", "raw_runs", "common", f"GG-MOEA__{inst}__{seed}.pkl"), "rb") as fh:
            r = pickle.load(fh)
        for x in r["X"].astype(float):
            P = x.reshape(N, 5)
            pos = np.ascontiguousarray(P[:, :3])
            co = np.cos(P[:, 3])
            dirs = np.column_stack([co * np.cos(P[:, 4]), co * np.sin(P[:, 4]), np.sin(P[:, 3])])
            t0 = build_tour(pos)
            f2a, f4a, La = path_costs(pos, dirs, t0, C.MISSION["w_climb"], C.MISSION["w_turn"])
            t1, f2b = weighted_two_opt(pos, dirs, t0, C.MISSION["w_climb"], C.MISSION["w_turn"], 20)
            _, f4b, Lb = path_costs(pos, dirs, t1, C.MISSION["w_climb"], C.MISSION["w_turn"])
            cva = clearance_violation(pos, t0, s.Zc, s.res, C.MISSION["h_safe"])
            cvb = clearance_violation(pos, t1, s.Zc, s.res, C.MISSION["h_safe"])
            rows.append(dict(instance=inst, f2_len=f2a, f2_w=f2b, red=100 * (f2a - f2b) / f2a,
                             f4_len=math.degrees(f4a), f4_w=math.degrees(f4b), cv_len=cva, cv_w=cvb))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "tour_experiment.csv"), index=False)
    print(out.groupby("instance")[["red", "f4_len", "f4_w", "cv_len", "cv_w"]].agg(["mean", "max"]).round(2))


def objective_dependence():
    rows = []
    for inst in INSTANCES:
        Fs = []
        for f in glob.glob(os.path.join("results", "raw_runs", "common", f"*__{inst}__*.pkl")):
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            Fs.append(r["F"][r["CV"] <= 0])
        F = np.vstack(Fs)
        rho = stats.spearmanr(F).correlation
        rows.append(dict(instance=inst, n=len(F), **{f"r{a+1}{b+1}": rho[a, b] for a in range(4) for b in range(a + 1, 4)},
                         nd_share=len(nondominated(F)) / len(F)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "objective_dependence.csv"), index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1:] or ["rank", "tour", "dep"]
    if "dep" in what:
        objective_dependence()
    if "tour" in what:
        tour_experiment()
    if "rank" in what:
        ranking_robustness()
