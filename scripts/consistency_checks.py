"""Consistency checks of the model implementation and of the reported indicators.

turn      turning term of f2 with zero-length legs removed before the angles are summed
clear     clearance with a closed prism model (contacts with cell edges and corners count)
gsdtol    on-axis approximation against the Jacobian GSD: size of the apparent underestimates
prec      precision on the targets that are well conditioned in every plan
refnew    whether the additional runs extend the fixed reference sets
igdgrid   density-balanced IGD+ ranks for several grid widths and offsets
timing    time of the candidate pool, of 20 and 100 greedy constructions and of lifting
"""
import glob
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.problem import Scene
from src.photogrammetry import build_tour, cell_of
from src.metrics import igd_plus, normalise, nondominated
from scripts.run_experiments import INSTANCES, COMPETITORS
from scripts.stats_paired import friedman

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")
ALGS = ["GG-MOEA"] + COMPETITORS
MIS = C.MISSION


# ------------------------------------------------------------------------------------------
@njit(cache=True)
def turn_terms(pos, tour):
    """Sum of turning angles: (a) as in the model (pairs with a zero-length leg skipped),
    (b) with zero-length legs removed first."""
    N = tour.shape[0]
    a = 0.0
    for k in range(1, N - 1):
        o, p, q = tour[k - 1], tour[k], tour[k + 1]
        e = pos[p] - pos[o]
        d = pos[q] - pos[p]
        n1 = math.sqrt((e * e).sum())
        n2 = math.sqrt((d * d).sum())
        if n1 > 1e-9 and n2 > 1e-9:
            c = (e * d).sum() / (n1 * n2)
            a += math.acos(min(1.0, max(-1.0, c)))
    b = 0.0
    prev = np.zeros(3)
    have = False
    for k in range(N - 1):
        d = pos[tour[k + 1]] - pos[tour[k]]
        n2 = math.sqrt((d * d).sum())
        if n2 <= 1e-9:
            continue
        if have:
            n1 = math.sqrt((prev * prev).sum())
            c = (prev * d).sum() / (n1 * n2)
            b += math.acos(min(1.0, max(-1.0, c)))
        prev = d
        have = True
    return a, b


def part_turn():
    rows = []
    for exp in ("common", "random", "ablation", "sensitivity"):
        for f in glob.glob(os.path.join(RAW, exp, "*.pkl")):
            r = pickle.load(open(f, "rb"))
            N = r["N"]
            for x, F, cv in zip(r["X"].astype(float), r["F"], r["CV"]):
                pos = np.ascontiguousarray(x.reshape(N, 5)[:, :3])
                t = build_tour(pos)
                a, b = turn_terms(pos, t)
                if abs(a - b) > 1e-9:
                    rows.append(dict(exp=exp, alg=r["alg"], instance=r["instance"], seed=r["seed"],
                                     df2_km=MIS["w_turn"] * (b - a) / 1000, f2=F[1], feasible=cv <= 0))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "check_turn.csv"), index=False)
    tot = sum(len(pickle.load(open(f, "rb"))["F"]) for exp in ("common",) for f in glob.glob(os.path.join(RAW, exp, "*.pkl")))
    print("solutions affected:", len(d), "of which common:", int((d.exp == "common").sum()) if len(d) else 0,
          "common total", tot)
    if len(d):
        print(d.groupby(["exp", "alg"]).agg(n=("df2_km", "size"), med=("df2_km", "median"), mx=("df2_km", "max"),
                                            rel=("f2", "median")).round(4))
    # assessed missions
    with open(os.path.join(TAB, "plans.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    out = []
    for (inst, name), rec in ms.items():
        x = rec["x"]
        N = len(x) // 5
        pos = np.ascontiguousarray(x.reshape(N, 5)[:, :3])
        a, b = turn_terms(pos, build_tour(pos))
        if abs(a - b) > 1e-9:
            out.append((inst, name, round(MIS["w_turn"] * (b - a) / 1000, 4)))
    print("assessed missions affected:", out)


# ------------------------------------------------------------------------------------------
@njit(cache=True)
def clip_cell(ax_, ay_, dx, dy, x0, x1, y0, y1):
    t0, t1 = 0.0, 1.0
    for pv, qv in ((-dx, ax_ - x0), (dx, x1 - ax_), (-dy, ay_ - y0), (dy, y1 - ay_)):
        if abs(pv) < 1e-15:
            if qv < 0:
                return 1.0, 0.0
        else:
            t = qv / pv
            if pv < 0:
                if t > t0:
                    t0 = t
            else:
                if t < t1:
                    t1 = t
    return t0, t1


@njit(cache=True)
def seg_deficit_closed(Zc, res, ax_, ay_, az_, bx_, by_, bz_, h):
    """Largest deficit over every cell whose closed footprint the segment touches."""
    nr, nc = Zc.shape
    dx, dy, dz = bx_ - ax_, by_ - ay_, bz_ - az_
    c0 = max(int(math.floor(min(ax_, bx_) / res)) - 1, 0)
    c1 = min(int(math.floor(max(ax_, bx_) / res)) + 1, nc - 1)
    r0 = max(int(math.floor(min(ay_, by_) / res)) - 1, 0)
    r1 = min(int(math.floor(max(ay_, by_) / res)) + 1, nr - 1)
    vmax = 0.0
    L = math.hypot(dx, dy)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            # quick rejection by distance of the cell centre to the planimetric segment
            cx, cy = (c + 0.5) * res, (r + 0.5) * res
            if L > 0:
                tt = ((cx - ax_) * dx + (cy - ay_) * dy) / (L * L)
                tt = min(1.0, max(0.0, tt))
                px, py = ax_ + tt * dx, ay_ + tt * dy
            else:
                px, py = ax_, ay_
            if abs(px - cx) > 0.75 * res or abs(py - cy) > 0.75 * res:
                continue
            t0, t1 = clip_cell(ax_, ay_, dx, dy, c * res, (c + 1) * res, r * res, (r + 1) * res)
            if t0 > t1:
                continue
            zlow = min(az_ + dz * t0, az_ + dz * t1)
            v = Zc[r, c] + h - zlow
            if v > vmax:
                vmax = v
    return vmax


@njit(cache=True)
def cv_closed(pos, tour, Zc, res, h):
    nr, nc = Zc.shape
    cv = 0.0
    for i in range(pos.shape[0]):
        # a station on a cell edge or corner is checked against all cells it touches
        x, y = pos[i, 0], pos[i, 1]
        best = -1e30
        for ddx in (-1e-9, 1e-9):
            for ddy in (-1e-9, 1e-9):
                r, c = cell_of(x + ddx, y + ddy, res, nr, nc)
                if Zc[r, c] > best:
                    best = Zc[r, c]
        v = best + h - pos[i, 2]
        if v > 0:
            cv += v
    for k in range(tour.shape[0] - 1):
        a, b = tour[k], tour[k + 1]
        cv += seg_deficit_closed(Zc, res, pos[a, 0], pos[a, 1], pos[a, 2], pos[b, 0], pos[b, 1], pos[b, 2], h)
    return cv


def part_clear():
    rows = []
    scenes = {k: Scene(k) for k in ("urban", "cliff", "dam", "gorge")}
    for exp in ("common", "components", "random100", "sens_t7", "ablation"):
        for f in glob.glob(os.path.join(RAW, exp, "*.pkl")):
            r = pickle.load(open(f, "rb"))
            s = scenes[r["scene"]]
            N = r["N"]
            for x, cv in zip(r["X"].astype(float), r["CV"]):
                if cv > 0:
                    continue
                pos = np.ascontiguousarray(x.reshape(N, 5)[:, :3])
                c2 = cv_closed(pos, build_tour(pos), s.Zc, s.res, MIS["h_safe"])
                rows.append(dict(exp=exp, alg=r["alg"], instance=r["instance"], seed=r["seed"], cv_closed=c2))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "check_clearance_runs.csv"), index=False)
    bad = d[d.cv_closed > 0]
    print("feasible final solutions checked:", len(d), "violating under the closed model:", len(bad))
    if len(bad):
        print(bad.groupby(["exp", "alg"]).agg(n=("cv_closed", "size"), mx=("cv_closed", "max")).round(4))
    with open(os.path.join(TAB, "plans.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    out = []
    for (inst, name), rec in ms.items():
        s = scenes[INSTANCES[inst][0]]
        x = rec["x"]
        pos = np.ascontiguousarray(x.reshape(len(x) // 5, 5)[:, :3])
        F, CV, _ = s.evaluate(x[None])
        c2 = cv_closed(pos, build_tour(pos), s.Zc, s.res, MIS["h_safe"])
        out.append(dict(instance=inst, plan=name, cv=float(CV[0]), cv_closed=float(c2)))
    o = pd.DataFrame(out)
    o.to_csv(os.path.join(TAB, "check_clearance_plans.csv"), index=False)
    print(o[(o.cv_closed - o.cv).abs() > 1e-9].to_string())


# ------------------------------------------------------------------------------------------
def part_gsdtol():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    PF = C.CAMERA["pix"] / C.CAMERA["f"]
    rmin, worst = [], None
    for (inst, name), rec in ms.items():
        s = Scene(INSTANCES[inst][0])
        V, QL, D, CA, U, GS = s.visibility(np.ascontiguousarray(rec["x"], float))
        ratio = (D * PF / np.maximum(CA, 1e-9))[V] / GS[V]
        rmin.append(ratio.min())
        nn = np.linalg.norm(s.n, axis=1)
        if worst is None or ratio.min() < worst[0]:
            worst = (ratio.min(), inst, name, float(np.abs(nn - 1).max()))
    print("smallest ratio approximation/Jacobian over all valid views:", worst, "count<1-1e-6:",
          sum(1 for r in rmin if r < 1 - 1e-6))


# ------------------------------------------------------------------------------------------
def part_prec():
    from scripts.mission_checks import sigma_models, vis
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        s = Scene(key)
        plans = {name: rec["x"] for (i, name), rec in ms.items() if i == inst}
        res, common = {}, np.ones(s.M, bool)
        for name, x in plans.items():
            V, QL, D, CA, U, GS = vis(s, x)
            si, sf, ill = sigma_models(s, np.asarray(x, float), V, D, U)
            res[name] = (si, sf, V)
            common &= (V.sum(0) >= 2) & np.isfinite(si) & np.isfinite(sf)
        for name, (si, sf, V) in res.items():
            rows.append(dict(instance=inst, plan=name, share=100 * s.A[common].sum() / s.A.sum(), n=int(common.sum()),
                             sig50_full=100 * np.median(sf[common]), sig90_full=100 * np.percentile(sf[common], 90),
                             sig50_iso=100 * np.median(si[common]), sig90_iso=100 * np.percentile(si[common], 90)))
        # which targets were ill conditioned and why
        for name, (si, sf, V) in res.items():
            bad = np.flatnonzero((V.sum(0) >= 2) & ~np.isfinite(sf))
            if len(bad):
                P = np.asarray(plans[name], float).reshape(-1, 5)[:, :3]
                coinc = []
                for j in bad:
                    idx = np.flatnonzero(V[:, j])
                    coinc.append(np.max(np.linalg.norm(P[idx] - P[idx[0]], axis=1)) < 1e-6)
                print(inst, name, "ill-conditioned targets:", len(bad), "all seen from coinciding stations:", all(coinc))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "precision_common.csv"), index=False)
    print(d.groupby("instance").share.first().round(1).to_dict())
    for pl in ("GG-MOEA (Q)", "Grid-Oblique", "Grid-Nadir"):
        g = d[d.plan == pl]
        print(pl, g.sig50_full.min().round(2), g.sig50_full.max().round(2))
    print("ratio iso/full", (d.sig50_iso / d.sig50_full).min().round(3), (d.sig50_iso / d.sig50_full).max().round(3))
    mx = 0.0
    for inst, g in d.groupby("instance"):
        e = g.set_index("plan")
        for a in e.index:
            for b in e.index:
                if (e.loc[a, "sig50_iso"] - e.loc[b, "sig50_iso"]) * (e.loc[a, "sig50_full"] - e.loc[b, "sig50_full"]) < 0:
                    mx = max(mx, abs(e.loc[a, "sig50_iso"] - e.loc[b, "sig50_iso"]))
        q = e.loc["GG-MOEA (Q)", "sig50_full"]
        oth = [p for p in e.index if "Greedy" in p or (p.endswith("(Q)") and not p.startswith("GG"))]
        print(inst, "max |other - GG-Q|", round(max(abs(e.loc[p, "sig50_full"] - q) for p in oth), 2))
    print("largest iso difference among swapped pairs", round(mx, 3))


# ------------------------------------------------------------------------------------------
def part_refnew():
    with open(os.path.join(TAB, "reference_sets.pkl"), "rb") as fh:
        ref = pickle.load(fh)
    for inst in INSTANCES:
        r = ref[inst]
        Fs = []
        for exp in ("components", "random100", "sens_t7"):
            for f in glob.glob(os.path.join(RAW, exp, f"*__{inst}__*.pkl")):
                rec = pickle.load(open(f, "rb"))
                Fs.append(rec["F"][rec["CV"] <= 0])
        if not Fs:
            continue
        Fn = np.vstack([F for F in Fs if len(F)])
        R = r["R"]
        both = nondominated(np.vstack([R, Fn]))
        new_nd = sum(1 for f in Fn if any(np.all(np.isclose(f, b)) for b in both))
        out_box = int(np.sum(np.any((Fn < r["ideal"] - 1e-12) | (Fn > r["nadir"] + 1e-12), axis=1)))
        print(inst, "new feasible solutions", len(Fn), "non-dominated w.r.t. R", new_nd, "outside ideal-nadir box", out_box)


# ------------------------------------------------------------------------------------------
def part_igdgrid():
    with open(os.path.join(TAB, "reference_sets.pkl"), "rb") as fh:
        ref = pickle.load(fh)
    recs = {(a, i, s): pickle.load(open(os.path.join(RAW, "common", f"{a.replace('/', '-')}__{i}__{s}.pkl"), "rb"))
            for a in ALGS for i in INSTANCES for s in range(1, 21)}
    rows = []
    for h in (0.05, 0.1, 0.2):
        for shift in (0.0, 0.5):
            M = np.zeros((len(INSTANCES), len(ALGS)))
            sizes = []
            for ii, inst in enumerate(INSTANCES):
                r = ref[inst]
                G = normalise(r["R"], r["ideal"], r["nadir"])
                key = np.floor(G / h + shift).astype(int)
                # representative: the point closest to the cell centre (deterministic)
                cen = (key - shift + 0.5) * h
                dist = np.linalg.norm(G - cen, axis=1)
                order = np.lexsort((dist,) + tuple(key.T[::-1]))
                _, first = np.unique(key[order], axis=0, return_index=True)
                Rb = r["R"][order[first]]
                sizes.append(len(Rb))
                for ai, a in enumerate(ALGS):
                    M[ii, ai] = np.mean([igd_plus(recs[(a, inst, s)]["F"], recs[(a, inst, s)]["CV"], Rb, r["ideal"], r["nadir"])
                                         for s in range(1, 21)])
            mr, chi, p, *_ = friedman(-M)
            best = int(np.sum(np.argmin(M, axis=1) == 0))
            rows.append(dict(width=h, shift=shift, sizes=f"{min(sizes)}-{max(sizes)}", gg_rank=mr[0],
                             best_other=ALGS[1 + int(np.argmin(mr[1:]))], best_other_rank=float(np.min(mr[1:])),
                             gg_best_instances=best))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "check_igd_grid.csv"), index=False)


# ------------------------------------------------------------------------------------------
def part_timing():
    from src.operators import CandidatePool, snbi_population, lift
    out = []
    for key in ("urban", "dam"):
        s = Scene(key)
        s.evaluate(np.zeros((1, 250)) + 1.0)
        rng = np.random.default_rng(5)
        t = time.process_time(); pool = CandidatePool(s, rng); t_pool = time.process_time() - t
        t = time.process_time(); [pool.construct(50, rng) for _ in range(20)]; t20 = time.process_time() - t
        t = time.process_time(); X = np.array([pool.construct(50, rng) for _ in range(100)]); t100 = time.process_time() - t
        t = time.process_time(); lift(s, X.copy()); t_lift = time.process_time() - t
        t = time.process_time(); snbi_population(s, 50, 100, rng, pool=pool); t_snbi_wo_pool = time.process_time() - t
        out.append(dict(scene=key, pool_s=t_pool, greedy20_s=t20, greedy100_s=t100, lift100_s=t_lift,
                        snbi_without_pool_s=t_snbi_wo_pool))
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(os.path.join(TAB, "check_timing.csv"), index=False)


if __name__ == "__main__":
    for part in sys.argv[1:]:
        dict(turn=part_turn, clear=part_clear, gsdtol=part_gsdtol, prec=part_prec, refnew=part_refnew,
             igdgrid=part_igdgrid, timing=part_timing)[part]()
