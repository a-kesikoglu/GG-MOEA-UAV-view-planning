"""Sensitivity of the mission indicators to the surface representation.

density  target sets with about 500 to 10000 targets and three offsets of the voxel grid
res      raster resolution r = 1, 2 and 4 m on the dam scene (targets rebuilt, missions re-evaluated)
mc       DSM perturbation with the stated accuracy of swissSURFACE3D Raster (10 cm height, 20 cm
         planimetric, 1 sigma): visibility and clearance re-tested for the selected missions
"""
import copy
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.problem import Scene
from src.data_loader import load_dsm
from scripts.run_experiments import INSTANCES

TAB = os.path.join("results", "tables")
GSTAR = C.MISSION["gsd_target"]
PLANS = ("Grid-Oblique", "Greedy-2S", "Greedy-2S, lifted", "GG-MOEA (C)", "GG-MOEA (Q)")


def build_targets(Z, res, m_target, shift=0.0):
    """Same construction as src.data_loader.build_targets, with the voxel grid shifted by
    `shift` times the voxel size along all three axes."""
    nr, nc = Z.shape
    thr = res * np.tan(np.deg2rad(70.0))
    pts, nrm, area = [], [], []
    for axis in (0, 1):
        if axis == 1:
            za, zb = Z[:, :-1], Z[:, 1:]
        else:
            za, zb = Z[:-1, :], Z[1:, :]
        dz = zb - za
        rr, cc = np.nonzero(np.abs(dz) > thr)
        for r, c in zip(rr, cc):
            hi, lo = (zb[r, c], za[r, c]) if dz[r, c] > 0 else (za[r, c], zb[r, c])
            sgn = -1.0 if dz[r, c] > 0 else 1.0
            if axis == 1:
                xe, ye = (c + 1) * res, (r + 0.5) * res
                nv = (sgn, 0.0, 0.0)
            else:
                xe, ye = (c + 0.5) * res, (r + 1) * res
                nv = (0.0, sgn, 0.0)
            nseg = max(1, int(np.ceil((hi - lo) / 3.0)))
            hseg = (hi - lo) / nseg
            for s in range(nseg):
                pts.append((xe, ye, lo + (s + 0.5) * hseg))
                nrm.append(nv)
                area.append(res * hseg)
    gx = np.zeros_like(Z)
    gy = np.zeros_like(Z)
    for g, axis in ((gx, 1), (gy, 0)):
        fwd = np.diff(Z, axis=axis, append=np.nan)
        bwd = np.diff(Z, axis=axis, prepend=np.nan)
        fwd[np.abs(fwd) > thr] = np.nan
        bwd[np.abs(bwd) > thr] = np.nan
        g[:] = np.nan_to_num(np.nanmean(np.stack([fwd, bwd]), axis=0), nan=0.0) / res
    rr, cc = np.mgrid[0:nr, 0:nc]
    nn = np.stack([-gx, -gy, np.ones_like(Z)], axis=-1)
    nl = np.linalg.norm(nn, axis=-1, keepdims=True)
    nn /= nl
    P = np.stack([(cc + 0.5) * res, (rr + 0.5) * res, Z], axis=-1).reshape(-1, 3)
    n_vf = len(pts)
    pts = np.vstack([np.array(pts).reshape(-1, 3), P])
    nrm = np.vstack([np.array(nrm).reshape(-1, 3), nn.reshape(-1, 3)])
    area = np.concatenate([np.array(area), (res * res * nl[..., 0]).ravel()])
    hor = np.abs(nrm[:, 2]) < 0.5
    az = np.round(np.arctan2(nrm[:, 1], nrm[:, 0]) / (np.pi / 2)).astype(int) % 4
    cls = np.where(hor, 1 + az, 0)
    lo, hi = 0.5, 60.0
    for _ in range(50):
        v = 0.5 * (lo + hi)
        key = np.column_stack([np.floor(pts / v + shift).astype(np.int64), cls])
        _, inv = np.unique(key, axis=0, return_inverse=True)
        k = inv.max() + 1
        if k > m_target:
            lo = v
        else:
            hi = v
        if abs(k - m_target) <= 0.01 * m_target:
            break
    inv = inv.ravel()
    A = np.bincount(inv, weights=area)
    Q = np.column_stack([np.bincount(inv, weights=area * pts[:, d]) for d in range(3)]) / A[:, None]
    Nn = np.column_stack([np.bincount(inv, weights=area * nrm[:, d]) for d in range(3)])
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    snap = np.zeros((k, 3))
    d2 = np.sum((pts - Q[inv]) ** 2, axis=1)
    seen = np.zeros(k, bool)
    for idx in np.argsort(d2):
        g = inv[idx]
        if not seen[g]:
            seen[g] = True
            snap[g] = pts[idx]
    stats = dict(total_area=area.sum(), steep_share=100 * area[hor].sum() / area.sum(),
                 vertical_facets=n_vf, vertical_area_share=100 * area[:n_vf].sum() / area.sum())
    return snap, Nn, A, v, stats


def scene_from(base, Z, res, q, n, A):
    s = copy.copy(base)
    rb = int(np.ceil(C.MISSION["buffer"] / res))
    s.Z = np.ascontiguousarray(Z)
    s.Zc = np.ascontiguousarray(maximum_filter(Z, size=2 * rb + 1, mode="nearest"))
    s.res = float(res)
    s.q, s.n, s.A = np.ascontiguousarray(q), np.ascontiguousarray(n), np.ascontiguousarray(A)
    s.M = len(q)
    return s


def indicators(s, x):
    V, QL, D, CA, U, GS = s.visibility(np.ascontiguousarray(x, dtype=float))
    A = s.A
    nv = V.sum(0)
    ng = (V & (GS <= GSTAR)).sum(0)
    F, CV, _ = s.evaluate(np.asarray(x, float)[None])
    return dict(C1=100 * A[nv >= 1].sum() / A.sum(), C3=100 * A[nv >= 3].sum() / A.sum(),
                G1=100 * A[ng >= 1].sum() / A.sum(), C3g=100 * A[ng >= 3].sum() / A.sum(), CV=float(CV[0]),
                f1=float(F[0, 0]))


def scene_missions(key, ms):
    out = {}
    for inst, (k, N, _) in INSTANCES.items():
        if k != key:
            continue
        names = [n for (i, n) in ms if i == inst]
        rival = [n for n in names if n.endswith("(Q)") and not n.startswith("GG")]
        for n in list(PLANS) + rival:
            if (inst, n) in ms:
                out[(inst, n if n not in rival else "Rival (Q)")] = np.asarray(ms[(inst, n)]["x"], float)
    return out


def part_density():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rows = []
    for key in ("urban", "cliff", "dam", "gorge"):
        base = Scene(key)
        d = np.load(os.path.join(C.PROC, f"{key}.npz"))
        Z = d["Z"]           # already shifted by z0
        res = float(d["res"])
        miss = scene_missions(key, ms)
        for m in (500, 1000, 2000, 5000, 10000):
            for shift in (0.0, 1 / 3, 2 / 3):
                q, n, A, vox, st = build_targets(Z, res, m, shift)
                s = scene_from(base, Z, res, q + 0.5 * n, n, A)
                for (inst, name), x in miss.items():
                    t = time.perf_counter()
                    r = indicators(s, x)
                    rows.append(dict(scene=key, instance=inst, plan=name, M_target=m, M=len(q), shift=shift,
                                     voxel=vox, eval_s=time.perf_counter() - t, **r))
                print(key, m, round(shift, 2), len(q), round(vox, 2), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "density.csv"), index=False)
    print(df.groupby(["instance", "plan", "M_target"])[["C3", "G1", "C3g"]].agg(["mean", "min", "max"]).round(1))


def part_res():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    key = "dam"
    base = Scene(key)
    z0_ref = base.z0
    sc = dict(C.SCENES[key])
    miss = scene_missions(key, ms)
    rows = []
    for r in (1.0, 2.0, 4.0):
        sc["res"] = r
        Z, _ = load_dsm(sc)
        q, n, A, vox, st = build_targets(Z, r, 1000)
        z0 = Z.min()
        s = scene_from(base, Z - z0, r, q + 0.5 * n - np.array([0, 0, z0]), n, A)
        steep = np.abs(n[:, 2]) < 0.5
        for (inst, name), x in miss.items():
            xx = x.reshape(-1, 5).copy()
            xx[:, 2] += z0_ref - z0
            ind = indicators(s, xx.ravel())
            rows.append(dict(res=r, instance=inst, plan=name, M=len(q), target_area=A.sum(),
                             steep_target_share=100 * A[steep].sum() / A.sum(), **st, **ind))
        print(r, st, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "resolution.csv"), index=False)
    print(df.round(2).to_string())


def part_mc(n_rep=20):
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rng = np.random.default_rng(11)
    rows = []
    for key in ("urban", "cliff", "dam", "gorge"):
        base = Scene(key)
        miss = scene_missions(key, ms)
        ref = {k: indicators(base, x) for k, x in miss.items()}
        for rep in range(n_rep):
            Zp = base.Z + rng.normal(0.0, 0.10, base.Z.shape)
            s = scene_from(base, Zp, base.res, base.q, base.n, base.A)
            dxy = rng.normal(0.0, 0.20, 2)
            for (inst, name), x in miss.items():
                xx = x.reshape(-1, 5).copy()
                xx[:, :2] -= dxy           # equivalent to shifting the DSM by +dxy
                ind = indicators(s, xx.ravel())
                rows.append(dict(scene=key, instance=inst, plan=name, rep=rep,
                                 dC3=ind["C3"] - ref[(inst, name)]["C3"], dG1=ind["G1"] - ref[(inst, name)]["G1"],
                                 CV=ind["CV"], CV0=ref[(inst, name)]["CV"]))
        print(key, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "mc.csv"), index=False)
    g = df.groupby(["instance", "plan"]).agg(dC3_med=("dC3", "median"), dC3_min=("dC3", "min"), dC3_max=("dC3", "max"),
                                             dG1_min=("dG1", "min"), dG1_max=("dG1", "max"),
                                             CV0=("CV0", "first"), CV_max=("CV", "max"),
                                             viol=("CV", lambda v: int((v > 0).sum())))
    g.to_csv(os.path.join(TAB, "mc_summary.csv"))
    print(g.round(2).to_string())


if __name__ == "__main__":
    for part in sys.argv[1:]:
        dict(density=part_density, res=part_res, mc=part_mc)[part]()
