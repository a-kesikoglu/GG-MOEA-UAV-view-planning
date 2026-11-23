"""Additional checks of the common-start experiment and of the mission assessment.

init      HV and IGD+ of the shared SNBI initial populations and the gain of every algorithm from
          this population to its final set (same reference set and normalisation as the article)
loo       leave-one-algorithm-out reference sets: HV of the common-start runs and Friedman ranks
          when the solutions of one algorithm are removed from the reference set
dup       effect of removing near-duplicate stations from the cost-matched GG-MOEA mission on T6
gsd       local GSD from the projection Jacobian against the on-axis approximation D p / (c cos a)
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.metrics import hv, igd_plus, nondominated
from src.problem import Scene
from scripts.run_experiments import INSTANCES, COMPETITORS
from scripts.stats_paired import friedman

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")
ALGS = ["GG-MOEA"] + COMPETITORS
CACHE = os.path.join("data", "processed", "snbi_cache")


def init_hv():
    with open(os.path.join(TAB, "reference_sets.pkl"), "rb") as fh:
        ref = pickle.load(fh)
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    com = df[df.exp == "common"]
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        s = Scene(key)
        rf = ref[inst]
        for seed in range(1, 21):
            z = np.load(os.path.join(CACHE, f"{key}_N{N}_P100_s{seed}.npz"))
            F, CV, _ = s.evaluate(z["X"])
            rows.append(dict(instance=inst, seed=seed, fe0=int(z["fe0"]) + len(z["X"]),
                             HV0=hv(F, CV, rf["ideal"], rf["nadir"]),
                             IGD0=igd_plus(F, CV, rf["R"], rf["ideal"], rf["nadir"]),
                             feas0=float(np.mean(CV <= 0))))
        print(inst, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "initial_population.csv"), index=False)
    m = com.merge(d, on=["instance", "seed"])
    m["dHV"] = m.HV - m.HV0
    m["dIGD"] = m.IGD0 - m.IGDp
    g = m.groupby(["instance", "alg"])[["HV0", "HV", "dHV", "IGD0", "IGDp", "dIGD"]].mean().reset_index()
    g.to_csv(os.path.join(TAB, "gain_from_initial.csv"), index=False)
    print(d.groupby("instance")[["fe0", "HV0", "IGD0", "feas0"]].mean().round(3))
    print(g.pivot(index="instance", columns="alg", values="dHV")[ALGS].round(3))


def load_final(exp):
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, exp, "*.pkl"))):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        out.append(dict(exp=exp, alg=r["alg"], instance=r["instance"], seed=r["seed"], F=r["F"], CV=r["CV"]))
    return out


def loo():
    runs = []
    for e in ("common", "random", "ablation", "sensitivity"):
        runs += load_final(e)
    family = {a: {a} for a in COMPETITORS}
    family["GG-MOEA"] = {"GG-MOEA", "w/o VCA", "w/o block", "CR=0.1", "CR=0.5", "p=0.05", "p=0.2", "P=50", "P=150"}
    rows, ranks = [], []
    for out_alg in ["none"] + ALGS:
        drop = family.get(out_alg, set())
        M = []
        for inst in INSTANCES:
            Fs = [r["F"][r["CV"] <= 0] for r in runs if r["instance"] == inst and r["alg"] not in drop
                  and (r["CV"] <= 0).any()]
            R = nondominated(np.vstack(Fs))
            ideal, nadir = R.min(0), R.max(0)
            vals = {}
            for a in ALGS:
                h = [hv(r["F"], r["CV"], ideal, nadir) for r in runs
                     if r["exp"] == "common" and r["instance"] == inst and r["alg"] == a]
                vals[a] = np.mean(h)
                rows.append(dict(excluded=out_alg, instance=inst, alg=a, HV=vals[a]))
            M.append([vals[a] for a in ALGS])
        mr, chi, p, ff, pf, _ = friedman(np.array(M))
        best = {inst: max(ALGS, key=lambda a: [r["HV"] for r in rows if r["excluded"] == out_alg
                                                and r["instance"] == inst and r["alg"] == a][0])
                for inst in INSTANCES}
        ranks.append(dict(excluded=out_alg, **{a: v for a, v in zip(ALGS, mr)},
                          gg_best=sum(b == "GG-MOEA" for b in best.values())))
        print(out_alg, np.round(mr, 2), ranks[-1]["gg_best"], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "loo_hv.csv"), index=False)
    pd.DataFrame(ranks).to_csv(os.path.join(TAB, "loo_ranks.csv"), index=False)


def dedup(x, dist=5.0, ang=10.0):
    P = x.reshape(-1, 5)
    co = np.cos(P[:, 3])
    d = np.column_stack([co * np.cos(P[:, 4]), co * np.sin(P[:, 4]), np.sin(P[:, 3])])
    keep = []
    for i in range(len(P)):
        dup = False
        for k in keep:
            if np.linalg.norm(P[i, :3] - P[k, :3]) < dist and np.degrees(np.arccos(np.clip(d[i] @ d[k], -1, 1))) < ang:
                dup = True
                break
        if not dup:
            keep.append(i)
    return P[keep].ravel(), len(P) - len(keep)


def mission_stats(scene, x):
    from scripts.engineering import assess
    from scripts.resolution_dense import shares
    r, V, D, U, g = assess(scene, x)
    ang = []
    for j in np.flatnonzero(V.sum(0) >= 2):
        idx = np.flatnonzero(V[:, j])
        u = U[idx, j]
        a = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))[np.triu_indices(len(idx), 1)]
        ang.append(a[np.argmin(np.abs(a - np.degrees(C.MISSION["beta_opt"])))])
    return dict(N=len(x) // 5, f1=r["f1"], f2=r["f2"], f3=r["f3"], C3=r["C3"], C3s=r["C3s"],
                beta_med=float(np.median(ang)), **{k + "_d5": v for k, v in shares(Scene(scene.key + "_d5k"), x).items()})


def dup():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rows = []
    for inst, name in (("T6", "GG-MOEA (C)"), ("T6", "GG-MOEA (Q)"), ("T6", "Grid-Oblique")):
        key = INSTANCES[inst][0]
        s = Scene(key)
        x = ms[(inst, name)]["x"]
        xd, nrem = dedup(x)
        for tag, xx in (("all", x), ("dedup", xd)):
            rows.append(dict(instance=inst, plan=name, version=tag, removed=nrem if tag == "dedup" else 0,
                             **mission_stats(s, xx)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "duplicates_T6.csv"), index=False)


def gsd():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    cam, mis = C.CAMERA, C.MISSION
    g_star, K = mis["gsd_target"], mis["k_min"]
    rows = []
    for (inst, name), rec in ms.items():
        if not (name.startswith("Grid") or name.endswith("(Q)")):
            continue
        s = Scene(INSTANCES[inst][0])
        V, QL, D, CA, U, GS = s.visibility(np.ascontiguousarray(rec["x"], dtype=float))
        simple = np.where(V, D * cam["pix"] / (cam["f"] * np.maximum(CA, 1e-9)), np.inf)
        jac = np.where(V, GS, np.inf)
        ratio = (jac / simple)[V]

        def f1(G):
            rho = np.where(np.isfinite(G), np.exp(-np.maximum(0.0, G / g_star - 1.0)), 0.0)
            top = -np.sort(-rho, axis=0)[:K]
            return 1.0 - np.sum(s.A * top.mean(0)) / s.A.sum()

        def g1(G):
            return 100 * s.A[(G <= g_star).sum(0) >= 1].sum() / s.A.sum()

        rows.append(dict(instance=inst, plan=name, n_views=int(V.sum()), ratio_med=float(np.median(ratio)),
                         ratio_p90=float(np.percentile(ratio, 90)), share_gt10=float(np.mean(ratio > 1.1) * 100),
                         f1_jac=f1(jac), f1_simple=f1(simple), G1_jac=g1(jac), G1_simple=g1(simple)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "gsd_comparison.csv"), index=False)


if __name__ == "__main__":
    for part in sys.argv[1:] or ["init", "dup", "gsd", "loo"]:
        dict(init=init_hv, loo=loo, dup=dup, gsd=gsd)[part]()
