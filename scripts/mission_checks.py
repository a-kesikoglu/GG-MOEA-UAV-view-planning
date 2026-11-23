"""Mission-level analyses of the assessed missions.

plans    additional reference plans (grids at d* above the local mean surface, staggered oblique grid)
         and per-plan indicators: path length, ascent, turning, duplicates, coverage after removing
         duplicates, three-view coverage with distinct directions, connectivity thresholds,
         resolution-conditioned coverage on the dense targets
prec     a priori precision with the full projection model against the isotropic angular model
gsd      Jacobian GSD against the on-axis approximation: per-mission distributions, area that
         meets g* under only one of the two, dependence on the off-axis angle
runsel   operating points C, Q and the rule-based point A selected in every run (GG-MOEA and the
         strongest competitor), evaluated on the optimization and the dense targets
"""
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.problem import Scene
from src.operators import grid_nadir, double_grid_oblique, lift, _grid_dims, _rotate_in_box, _pack
from src.photogrammetry import build_tour, camera_axes
from scripts.run_experiments import INSTANCES
from scripts.turn_impact import corrected

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")
CAM, MIS = C.CAMERA, C.MISSION
GSTAR = MIS["gsd_target"]
PF = CAM["pix"] / CAM["f"]
SIGMA_PX = 0.5


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def load_missions():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        return pickle.load(fh)


def dirs_of(x):
    P = x.reshape(-1, 5)
    co = np.cos(P[:, 3])
    return np.column_stack([co * np.cos(P[:, 4]), co * np.sin(P[:, 4]), np.sin(P[:, 3])])


def duplicates(x, dmax=5.0, amax=10.0):
    P = x.reshape(-1, 5)
    d = dirs_of(x)
    dist = np.linalg.norm(P[:, None, :3] - P[None, :, :3], axis=2)
    ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    return (dist < dmax) & (ang < amax) & ~np.eye(len(P), dtype=bool)


def dedup(x):
    """Keep the first station of every group of near-duplicates (5 m, 10 degrees)."""
    D = duplicates(x)
    keep = []
    for i in range(D.shape[0]):
        if not any(D[i, k] for k in keep):
            keep.append(i)
    return x.reshape(-1, 5)[keep].ravel(), len(keep)


def vis(scene, x):
    return scene.visibility(np.ascontiguousarray(x, dtype=float))


@njit(cache=True)
def distinct3(V, U, cmax):
    """Per target: True if three valid views exist whose rays pairwise enclose at least the angle
    with cosine cmax."""
    N, M = V.shape
    out = np.zeros(M, dtype=np.bool_)
    idx = np.empty(N, dtype=np.int64)
    for j in range(M):
        n = 0
        for i in range(N):
            if V[i, j]:
                idx[n] = i
                n += 1
        if n < 3:
            continue
        found = False
        for a in range(n - 2):
            ia = idx[a]
            for b in range(a + 1, n - 1):
                ib = idx[b]
                cab = U[ia, j, 0] * U[ib, j, 0] + U[ia, j, 1] * U[ib, j, 1] + U[ia, j, 2] * U[ib, j, 2]
                if cab > cmax:
                    continue
                for c in range(b + 1, n):
                    ic = idx[c]
                    cac = U[ia, j, 0] * U[ic, j, 0] + U[ia, j, 1] * U[ic, j, 1] + U[ia, j, 2] * U[ic, j, 2]
                    if cac > cmax:
                        continue
                    cbc = U[ib, j, 0] * U[ic, j, 0] + U[ib, j, 1] * U[ic, j, 1] + U[ib, j, 2] * U[ic, j, 2]
                    if cbc <= cmax:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        out[j] = found
    return out


def graph(V, k):
    N = V.shape[0]
    S = V.astype(np.int32) @ V.T.astype(np.int32)
    adj = (S >= k) & ~np.eye(N, dtype=bool)
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
    return len(comps), (100.0 * max(comps) / N if comps else 0.0)


def path_stats(x):
    P = x.reshape(-1, 5)
    pos = np.ascontiguousarray(P[:, :3])
    t = build_tour(pos)
    d = np.diff(pos[t], axis=0)
    L = np.linalg.norm(d, axis=1)
    up = d[d[:, 2] > 0, 2].sum()
    down = -d[d[:, 2] < 0, 2].sum()
    turn = 0.0
    nz = [k for k in range(len(d)) if L[k] > 1e-9]
    for k0, k1 in zip(nz[:-1], nz[1:]):
        turn += math.degrees(math.acos(np.clip(d[k0] @ d[k1] / (L[k0] * L[k1]), -1, 1)))
    return dict(length_km=L.sum() / 1000, ascent_m=up, descent_m=down, turning_deg=turn,
                zero_legs=int((L <= 1e-9).sum()))


def coverage(scene, V, GS, U=None):
    A = scene.A
    nv = V.sum(0)
    ng = (V & (GS <= GSTAR)).sum(0)
    out = dict(C1=100 * A[nv >= 1].sum() / A.sum(), C3=100 * A[nv >= 3].sum() / A.sum(),
               G1=100 * A[ng >= 1].sum() / A.sum(), C3g=100 * A[ng >= 3].sum() / A.sum())
    steep = np.abs(scene.n[:, 2]) < 0.5
    out["C3s"] = 100 * A[(nv >= 3) & steep].sum() / A[steep].sum()
    if U is not None:
        d5 = distinct3(V, U, math.cos(math.radians(5.0)))
        out["C3d"] = 100 * A[d5].sum() / A.sum()
    return out


# ------------------------------------------------------------------------------------------
# additional reference plans
# ------------------------------------------------------------------------------------------
def _local_mean(scene, radius=40.0):
    k = 2 * int(radius / scene.res) + 1
    return uniform_filter(scene.Z, size=k, mode="nearest")


def _z_at(Zt, scene, x, y):
    nr, nc = Zt.shape
    c = np.clip((x / scene.res).astype(int), 0, nc - 1)
    r = np.clip((y / scene.res).astype(int), 0, nr - 1)
    return Zt[r, c]


def grid_nadir_mean(scene, N):
    """Nadir block at d* above the local mean surface (GSD tuned), lifted for clearance."""
    x = grid_nadir(scene, N).reshape(N, 5)
    x[:, 2] = _z_at(_local_mean(scene), scene, x[:, 0], x[:, 1]) + MIS["d_opt"]
    x[:, 2] = np.clip(x[:, 2], scene.zlo, scene.zhi)
    return lift(scene, x.ravel())[0]


def grid_oblique_mean(scene, N, tilt=math.radians(45.0)):
    x = double_grid_oblique(scene, N, tilt=tilt).reshape(N, 5)
    # height such that the slant range to the local mean surface is d*
    x[:, 2] = _z_at(_local_mean(scene), scene, x[:, 0], x[:, 1]) + MIS["d_opt"] * math.cos(tilt)
    x[:, 2] = np.clip(x[:, 2], scene.zlo, scene.zhi)
    return lift(scene, x.ravel())[0]


def grid_oblique_inside(scene, N, tilt=math.radians(45.0)):
    """Double oblique grid whose stations stay inside the window: on each line the stations are
    spread over the part from which the forward-looking camera aims at the window, so that no
    station is clipped onto the window edge."""
    ex, ey = scene.extent
    h = MIS["d_opt"] * math.cos(tilt)
    off = h * math.tan(tilt)
    xs, ys, ks = [], [], []
    for s, n in enumerate((N // 2, N - N // 2)):
        nl, nper = _grid_dims(n, ex, ey)
        L = ex if s == 0 else ey
        W = ey if s == 0 else ex
        for li in range(nl):
            w = (li + 0.5) / nl * W
            fwd = 1 if li % 2 == 0 else -1
            a, b = (0.0, L - off) if fwd > 0 else (off, L)
            for pi in range(nper):
                t = a + (pi + 0.5) / nper * (b - a)
                if s == 0:
                    xs.append(t); ys.append(w); ks.append(0.0 if fwd > 0 else math.pi)
                else:
                    xs.append(w); ys.append(t); ks.append(0.5 * math.pi if fwd > 0 else -0.5 * math.pi)
    x, y, ka = np.array(xs), np.array(ys), np.array(ks)
    from src.operators import _local_top
    z = _z_at(_local_top(scene), scene, x, y) + h
    om = np.full(len(x), -(0.5 * math.pi - tilt))
    return _pack(scene, x, y, z, om, ka)


def all_plans(inst, ms):
    key, N, _ = INSTANCES[inst]
    s = Scene(key)
    plans = {name: rec["x"] for (i, name), rec in ms.items() if i == inst}
    plans["Grid-Nadir-m"] = grid_nadir_mean(s, N)
    plans["Grid-Oblique-m"] = grid_oblique_mean(s, N)
    plans["Grid-Oblique-i"] = grid_oblique_inside(s, N)
    return s, plans


def part_plans():
    ms = load_missions()
    rows, store = [], {}
    for inst, (key, N, _) in INSTANCES.items():
        s, plans = all_plans(inst, ms)
        sd = Scene(key + "_d5k")
        for name, x in plans.items():
            x = np.asarray(x, dtype=float)
            F, CV, _ = s.evaluate(x[None])
            V, QL, D, CA, U, GS = vis(s, x)
            Vd, _, Dd, _, Ud, GSd = vis(sd, x)
            r = dict(instance=inst, plan=name, f1=F[0, 0], f2=F[0, 1], f3=F[0, 2], f4=math.degrees(F[0, 3]),
                     CV=CV[0], dup=100 * duplicates(x).any(1).mean())
            r.update(path_stats(x))
            r.update({k + "_opt": v for k, v in coverage(s, V, GS, U).items()})
            r.update({k + "_dense": v for k, v in coverage(sd, Vd, GSd, Ud).items()})
            for k in (3, 5, 10):
                nc_, lg = graph(V, k)
                r[f"comp{k}"], r[f"largest{k}"] = nc_, lg
            for k in (5, 25):
                nc_, lg = graph(Vd, k)
                r[f"comp{k}_dense"], r[f"largest{k}_dense"] = nc_, lg
            xd, nkeep = dedup(x)
            Vx, _, _, _, Ux, GSx = vis(s, xd)
            Vxd, _, _, _, Uxd, GSxd = vis(sd, xd)
            r["n_distinct"] = nkeep
            r.update({k + "_opt_dedup": v for k, v in coverage(s, Vx, GSx).items()})
            r.update({k + "_dense_dedup": v for k, v in coverage(sd, Vxd, GSxd).items()})
            rows.append(r)
            store[(inst, name)] = dict(x=x, nv=V.sum(0), ng=(V & (GS <= GSTAR)).sum(0),
                                       nv_d=Vd.sum(0), ng_d=(Vd & (GSd <= GSTAR)).sum(0))
            print(inst, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                               if k in ("f2", "CV", "C3_opt", "C3d_opt", "C3_opt_dedup", "G1_dense", "C3g_dense",
                                        "dup", "n_distinct", "largest5")}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "plans.csv"), index=False)
    with open(os.path.join(TAB, "plans.pkl"), "wb") as fh:
        pickle.dump(store, fh)


# ------------------------------------------------------------------------------------------
# precision: full projection model
# ------------------------------------------------------------------------------------------
def sigma_models(scene, x, V, D, U):
    """Point precision under the isotropic angular model (Eq. 19) and under the full model
    with the Jacobian of the normalised image coordinates."""
    P = x.reshape(-1, 5)
    sig_t = SIGMA_PX * PF
    w_img = 1.0 / sig_t ** 2
    ax = [camera_axes(P[i, 3], P[i, 4]) for i in range(len(P))]
    s_iso = np.full(scene.M, np.nan)
    s_full = np.full(scene.M, np.nan)
    ill = 0
    for j in np.flatnonzero(V.sum(0) >= 2):
        idx = np.flatnonzero(V[:, j])
        N1 = np.zeros((3, 3))
        N2 = np.zeros((3, 3))
        for i in idx:
            u = U[i, j]
            N1 += (np.eye(3) - np.outer(u, u)) / (sig_t * D[i, j]) ** 2
            d0, d1, d2, e0, e1, e2, v0, v1, v2 = ax[i]
            d, e, v = np.array([d0, d1, d2]), np.array([e0, e1, e2]), np.array([v0, v1, v2])
            r = scene.q[j] - P[i, :3]
            zc = r @ d
            xi, yi = (r @ e) / zc, (r @ v) / zc
            B = np.vstack([e - xi * d, v - yi * d]) / zc
            N2 += w_img * B.T @ B
        for Nm, out in ((N1, s_iso), (N2, s_full)):
            ev = np.linalg.eigvalsh(Nm)
            if ev[0] <= 1e-12 * ev[-1]:
                ill += 1
                continue
            out[j] = math.sqrt(np.trace(np.linalg.inv(Nm)))
    return s_iso, s_full, ill


def part_prec():
    ms = load_missions()
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        s = Scene(key)
        plans = {name: rec["x"] for (i, name), rec in ms.items() if i == inst}
        res = {}
        common = np.ones(s.M, bool)
        for name, x in plans.items():
            V, QL, D, CA, U, GS = vis(s, x)
            res[name] = sigma_models(s, np.asarray(x, float), V, D, U)
            common &= V.sum(0) >= 2
        for name, (si, sf, ill) in res.items():
            rows.append(dict(instance=inst, plan=name, sig50_iso=100 * np.nanmedian(si[common]),
                             sig90_iso=100 * np.nanpercentile(si[common], 90),
                             sig50_full=100 * np.nanmedian(sf[common]), sig90_full=100 * np.nanpercentile(sf[common], 90),
                             ratio_med=float(np.nanmedian((sf / si)[common])), ill=ill,
                             n_common=int(common.sum())))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "precision.csv"), index=False)
    for inst, g in df.groupby("instance"):
        same50 = list(g.sort_values("sig50_iso").plan) == list(g.sort_values("sig50_full").plan)
        same90 = list(g.sort_values("sig90_iso").plan) == list(g.sort_values("sig90_full").plan)
        print(inst, "order sigma50 unchanged:", same50, "sigma90 unchanged:", same90)


# ------------------------------------------------------------------------------------------
# GSD: Jacobian against the on-axis approximation
# ------------------------------------------------------------------------------------------
def part_gsd():
    ms = load_missions()
    rows, bins = [], []
    sample = {}
    for (inst, name), rec in ms.items():
        s = Scene(INSTANCES[inst][0])
        x = np.asarray(rec["x"], float)
        V, QL, D, CA, U, GS = vis(s, x)
        Dd = dirs_of(x)
        cos_off = np.einsum("ijk,ik->ij", U, Dd)
        th = np.degrees(np.arccos(np.clip(cos_off, -1, 1)))[V]
        g_ax = (D * PF / np.maximum(CA, 1e-9))[V]
        g_j = GS[V]
        ratio = g_ax / g_j
        Gj = np.where(V, GS, np.inf)
        Ga = np.where(V, D * PF / np.maximum(CA, 1e-9), np.inf)
        okj = (Gj <= GSTAR).sum(0) >= 1
        oka = (Ga <= GSTAR).sum(0) >= 1
        A = s.A
        rows.append(dict(instance=inst, plan=name, n_views=int(V.sum()), over_med=float(np.median(ratio) - 1) * 100,
                         over_q1=float(np.percentile(ratio, 25) - 1) * 100, over_q3=float(np.percentile(ratio, 75) - 1) * 100,
                         under_share=float(np.mean(ratio < 1.0)) * 100,
                         G1_jac=100 * A[okj].sum() / A.sum(), G1_axis=100 * A[oka].sum() / A.sum(),
                         only_jac=100 * A[okj & ~oka].sum() / A.sum(), only_axis=100 * A[oka & ~okj].sum() / A.sum()))
        bins.append(pd.DataFrame(dict(inst=inst, plan=name, theta=th, ratio=ratio)))
        if (inst, name) in (("T5", "GG-MOEA (Q)"), ("T1", "Grid-Nadir"), ("T1", "Grid-Oblique")):
            sample[(inst, name)] = (g_j, g_ax, th)
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "gsd.csv"), index=False)
    B = pd.concat(bins)
    B["bin"] = (B.theta // 4) * 4
    agg = B.groupby("bin").ratio.describe(percentiles=[0.25, 0.5, 0.75])
    agg.to_csv(os.path.join(TAB, "gsd_theta.csv"))
    with open(os.path.join(TAB, "gsd_sample.pkl"), "wb") as fh:
        pickle.dump(dict(sample=sample, theta=agg), fh)
    print(agg)


# ------------------------------------------------------------------------------------------
# operating points selected in every run
# ------------------------------------------------------------------------------------------
def select_points(F, X, cap_f2, ref_f1, ref_f3):
    out = {}
    out["Q"] = int(np.argmin(F[:, 0]))
    ok = F[:, 1] <= cap_f2
    if ok.any():
        out["C"] = int(np.flatnonzero(ok)[np.argmin(F[ok, 0])])
    # rule A: no worse than the oblique grid in observation quality and intersection geometry,
    # at most 10% near-duplicate stations; the cheapest such mission
    dupl = np.array([duplicates(x).any(1).mean() for x in X])
    okA = (F[:, 0] <= ref_f1) & (F[:, 2] <= ref_f3) & (dupl <= 0.10)
    if okA.any():
        out["A"] = int(np.flatnonzero(okA)[np.argmin(F[okA, 1])])
    return out, dupl


def part_runsel():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    base = pd.read_csv(os.path.join(TAB, "missions.csv"))
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        s, sd = Scene(key), Scene(key + "_d5k")
        ob = base[(base.instance == inst) & (base.plan == "Grid-Oblique")].iloc[0]
        com = df[(df.exp == "common") & (df.instance == inst)]
        rival = com[com.alg != "GG-MOEA"].groupby("alg").HV.mean().idxmax()
        for alg in ("GG-MOEA", rival):
            for seed in range(1, 21):
                with open(os.path.join(RAW, "common", f"{alg.replace('/', '-')}__{inst}__{seed}.pkl"), "rb") as fh:
                    rec = pickle.load(fh)
                F, X = corrected(rec), rec["X"].astype(float)
                pts, dupl = select_points(F, X, ob.f2, ob.f1, ob.f3)
                for lab, k in pts.items():
                    x = X[k]
                    V, QL, D, CA, U, GS = vis(s, x)
                    Vd, _, _, _, Ud, GSd = vis(sd, x)
                    c_o, c_d = coverage(s, V, GS), coverage(sd, Vd, GSd)
                    rows.append(dict(instance=inst, alg=alg, role="GG-MOEA" if alg == "GG-MOEA" else "rival",
                                     seed=seed, point=lab, f1=F[k, 0], f2=F[k, 1], f3=F[k, 2],
                                     f4=math.degrees(F[k, 3]), dup=100 * dupl[k],
                                     C3_opt=c_o["C3"], C3_dense=c_d["C3"], G1_dense=c_d["G1"], C3g_dense=c_d["C3g"]))
            print(inst, alg, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "runsel.csv"), index=False)
    summ = out.groupby(["instance", "role", "point"]).agg(
        n=("seed", "size"), f2_med=("f2", "median"), f2_min=("f2", "min"), f2_max=("f2", "max"),
        C3_med=("C3_dense", "median"), C3_min=("C3_dense", "min"), C3_max=("C3_dense", "max"),
        G1_med=("G1_dense", "median"), G1_min=("G1_dense", "min"), G1_max=("G1_dense", "max"),
        C3g_med=("C3g_dense", "median"), C3g_min=("C3g_dense", "min"), C3g_max=("C3g_dense", "max"),
        f4_med=("f4", "median"), dup_med=("dup", "median"), dup_max=("dup", "max"))
    summ.to_csv(os.path.join(TAB, "runsel_summary.csv"))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    for part in sys.argv[1:]:
        dict(plans=part_plans, prec=part_prec, gsd=part_gsd, runsel=part_runsel)[part]()
