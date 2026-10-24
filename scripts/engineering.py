"""Geometric assessment of single missions.

For each instance: conventional patterns, the decoupled greedy plan (raw and lifted), the best
competitor of the common-initialization experiment and GG-MOEA at two operating points
(C: best f1 with f2 not above the oblique grid; Q: best f1), taken from the median-HV run.
Indicators are computed in the prism model with the optimization targets and with two denser,
target sets (about 5000 targets, offsets 0.5 m and 0.1 m).
"""
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.problem import Scene
from src.engineering import greedy_two_stage
from src.operators import grid_nadir, double_grid_oblique, lift
from scripts.run_experiments import INSTANCES

TAB = os.path.join("results", "tables")
SIGMA_PX = 0.5


def median_run(df, inst, alg):
    d = df[(df.exp == "common") & (df.instance == inst) & (df.alg == alg)].sort_values("HV")
    return int(d.iloc[len(d) // 2]["seed"])


def load_run(alg, inst, seed):
    with open(os.path.join("results", "raw_runs", "common", f"{alg.replace('/', '-')}__{inst}__{seed}.pkl"), "rb") as fh:
        return pickle.load(fh)


def views(scene, x):
    V, QL, D, CA, U, GS = scene.visibility(np.ascontiguousarray(x, dtype=float))
    return V, D, U, GS


def point_sigma(V, D, U, j):
    idx = np.flatnonzero(V[:, j])
    if len(idx) < 2:
        return np.nan, np.nan
    sig_t = SIGMA_PX * C.CAMERA["pix"] / C.CAMERA["f"]
    Nm = np.zeros((3, 3))
    for i in idx:
        u = U[i, j]
        Nm += (np.eye(3) - np.outer(u, u)) / (sig_t * D[i, j]) ** 2
    ev = np.linalg.eigvalsh(Nm)
    if ev[0] <= 1e-12 * ev[-1]:
        return np.nan, np.inf
    return math.sqrt(np.trace(np.linalg.inv(Nm))), ev[-1] / ev[0]


def graph_stats(V, x, min_shared=5):
    N = V.shape[0]
    S = V.astype(int) @ V.T.astype(int)
    adj = (S >= min_shared) & ~np.eye(N, dtype=bool)
    used = V.sum(1) > 0
    seen, comps = np.zeros(N, bool), []
    for s0 in np.flatnonzero(used):
        if seen[s0]:
            continue
        stack, size = [s0], 0
        seen[s0] = True
        while stack:
            a = stack.pop()
            size += 1
            for b in np.flatnonzero(adj[a] & ~seen):
                seen[b] = True
                stack.append(b)
        comps.append(size)
    P = x.reshape(-1, 5)
    co = np.cos(P[:, 3])
    d = np.column_stack([co * np.cos(P[:, 4]), co * np.sin(P[:, 4]), np.sin(P[:, 3])])
    dist = np.linalg.norm(P[:, None, :3] - P[None, :, :3], axis=2)
    ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    dup = (dist < 5.0) & (ang < 10.0) & ~np.eye(N, dtype=bool)
    return dict(n_comp=len(comps), largest=100.0 * max(comps) / N if comps else 0.0,
                idle=100.0 * np.mean(~used), dup=100.0 * np.mean(dup.any(1)))


def assess(scene, x):
    F, CV, _ = scene.evaluate(x[None])
    V, D, U, GS = views(scene, x)
    A = scene.A
    nv = V.sum(0)
    steep = np.abs(scene.n[:, 2]) < 0.5
    g = np.where(V, GS, np.inf).min(0)
    seen = nv >= 1
    out = dict(f1=F[0, 0], f2=F[0, 1], f3=F[0, 2], f4=math.degrees(F[0, 3]), CV=CV[0],
               C3=100 * A[nv >= 3].sum() / A.sum(), C3s=100 * A[(nv >= 3) & steep].sum() / A[steep].sum(),
               C1=100 * A[seen].sum() / A.sum(),
               gsd=100 * np.average(g[seen], weights=A[seen]) if seen.any() else np.nan)
    out.update(graph_stats(V, x))
    return out, V, D, U, g


def dense_c3(scene, x):
    V, D, U, GS = views(scene, x)
    nv = V.sum(0)
    return 100 * scene.A[nv >= 3].sum() / scene.A.sum()


def main():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    rows, keep = [], {}
    for inst, (key, N, _) in INSTANCES.items():
        s = Scene(key)
        sd5, sd1 = Scene(key + "_d5k"), Scene(key + "_d5k01")
        plans = {"Grid-Nadir": grid_nadir(s, N), "Grid-Oblique": double_grid_oblique(s, N)}
        g = greedy_two_stage(s, N)
        plans["Greedy-2S"] = g
        if s.evaluate(g[None])[1][0] > 0:
            plans["Greedy-2S, lifted"] = lift(s, g.copy())[0]
        cap = s.evaluate(plans["Grid-Oblique"][None])[0][0, 1]
        com = df[(df.exp == "common") & (df.instance == inst)]
        rival = com[com.alg != "GG-MOEA"].groupby("alg").HV.mean().idxmax()
        for alg in (rival, "GG-MOEA"):
            rec = load_run(alg, inst, median_run(df, inst, alg))
            F, X = rec["F"], rec["X"].astype(float)
            ok = F[:, 1] <= cap
            if ok.any():
                plans[f"{alg} (C)"] = X[np.flatnonzero(ok)[np.argmin(F[ok, 0])]]
            plans[f"{alg} (Q)"] = X[np.argmin(F[:, 0])]
        res = {}
        for name, x in plans.items():
            r, V, D, U, gbest = assess(s, x)
            r.update(C3_d5=dense_c3(sd5, x), C3_d1=dense_c3(sd1, x))
            res[name] = (r, V, D, U, gbest)
        # precision on the targets observed at least twice by every plan
        common = np.ones(s.M, bool)
        for name, (r, V, D, U, gb) in res.items():
            common &= V.sum(0) >= 2
        for name, (r, V, D, U, gb) in res.items():
            sig = np.array([point_sigma(V, D, U, j)[0] for j in np.flatnonzero(common)])
            sig_all = np.array([point_sigma(V, D, U, j)[0] for j in range(s.M)])
            r.update(sig50=100 * np.nanmedian(sig), sig90=100 * np.nanpercentile(sig, 90),
                     common_share=100 * s.A[common].sum() / s.A.sum(),
                     sig50_own=100 * np.nanmedian(sig_all))
            rows.append(dict(instance=inst, plan=name, cap=cap, **r))
            keep[(inst, name)] = dict(x=plans[name], nv=V.sum(0), gsd=gb)
            print(inst, name, {k: round(float(v), 2) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "missions.csv"), index=False)
    with open(os.path.join(TAB, "missions.pkl"), "wb") as fh:
        pickle.dump(keep, fh)


if __name__ == "__main__":
    main()
