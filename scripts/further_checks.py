"""Additional checks of the selected missions and of the stored runs.

distinct  three views that meet g* and pairwise enclose at least 5 degrees (same three views), on the
          optimization and the dense targets, with and without near-duplicate stations
clear     effect of the closed prism clearance check on HV and IGD+ of the affected common-start runs
comp      evaluations used and composition of the final sets of every algorithm (common start)
mc        DSM perturbation split into planimetric shift, height noise and both (same random numbers)
res       raster resolution on the dam scene separated into raster, target set and buffer effects
gridline  duplicate-free oblique grid flown in the order of its flight lines
"""
import glob
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.problem import Scene
from src.data_loader import load_dsm
from src.metrics import hv, igd_plus
from src.photogrammetry import build_tour, path_costs, clearance_violation
from scripts.run_experiments import INSTANCES, COMPETITORS
from scripts.mission_checks import vis, distinct3, dedup, grid_oblique_inside, dirs_of
from scripts.surface_sensitivity import scene_from, indicators, scene_missions, build_targets
from scripts.consistency_checks import cv_closed
from scripts.turn_impact import corrected

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")
MIS = C.MISSION
GSTAR = MIS["gsd_target"]
COS5 = math.cos(math.radians(5.0))


def _nd(F):
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        if dom.any():
            keep[i] = False
    return keep


# ------------------------------------------------------------------------------------------
def part_distinct():
    with open(os.path.join(TAB, "plans.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rows = []
    sc = {}
    for (inst, name), rec in ms.items():
        key = INSTANCES[inst][0]
        if key not in sc:
            sc[key] = (Scene(key), Scene(key + "_d5k"))
        r = dict(instance=inst, plan=name)
        for tag, s in zip(("opt", "dense"), sc[key]):
            for dd, x in (("", rec["x"]), ("_dedup", dedup(rec["x"])[0])):
                V, QL, D, CA, U, GS = vis(s, x)
                Vg = V & (GS <= GSTAR)
                A = s.A
                r[f"C3d_{tag}{dd}"] = 100 * A[distinct3(V, U, COS5)].sum() / A.sum()
                r[f"C3gd_{tag}{dd}"] = 100 * A[distinct3(Vg, U, COS5)].sum() / A.sum()
                r[f"C3g_{tag}{dd}"] = 100 * A[Vg.sum(0) >= 3].sum() / A.sum()
        rows.append(r)
        print(inst, name, {k: round(v, 1) for k, v in r.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "distinct.csv"), index=False)


# ------------------------------------------------------------------------------------------
def part_clear():
    d = pd.read_csv(os.path.join(TAB, "check_clearance_runs.csv"))
    bad = d[(d.exp == "common") & (d.cv_closed > 0)][["alg", "instance", "seed"]].drop_duplicates()
    ref = pickle.load(open(os.path.join(TAB, "reference_sets.pkl"), "rb"))
    rows = []
    for _, b in bad.iterrows():
        rec = pickle.load(open(os.path.join(RAW, "common", f"{b.alg.replace('/', '-')}__{b.instance}__{b.seed}.pkl"), "rb"))
        s = Scene(rec["scene"])
        N = rec["N"]
        CV2 = rec["CV"].copy()
        for k, x in enumerate(rec["X"].astype(float)):
            if CV2[k] > 0:
                continue
            pos = np.ascontiguousarray(x.reshape(N, 5)[:, :3])
            c2 = cv_closed(pos, build_tour(pos), s.Zc, s.res, MIS["h_safe"])
            if c2 > 0:
                CV2[k] = c2
        r = ref[b.instance]
        rows.append(dict(alg=b.alg, instance=b.instance, seed=b.seed, removed=int(((CV2 > 0) & (rec["CV"] <= 0)).sum()),
                         HV=hv(rec["F"], rec["CV"], r["ideal"], r["nadir"]), HV_closed=hv(rec["F"], CV2, r["ideal"], r["nadir"]),
                         IGD=igd_plus(rec["F"], rec["CV"], r["R"], r["ideal"], r["nadir"]),
                         IGD_closed=igd_plus(rec["F"], CV2, r["R"], r["ideal"], r["nadir"])))
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(TAB, "clearance_hv.csv"), index=False)
    print(o.to_string())


# ------------------------------------------------------------------------------------------
def part_comp():
    rows = []
    for a in ["GG-MOEA"] + COMPETITORS:
        for f in glob.glob(os.path.join(RAW, "common", f"{a.replace('/', '-')}__T*__*.pkl")):
            r = pickle.load(open(f, "rb"))
            feas = r["CV"] <= 0
            nd = _nd(r["F"][feas]) if feas.any() else np.zeros(0, bool)
            rows.append(dict(alg=a, instance=r["instance"], N=r["N"], fe=r["fe_final"], maxfe=r["maxfe"],
                             final=len(r["F"]), feasible=int(feas.sum()), nd=int(nd.sum())))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "final_sets.csv"), index=False)
    g = d.groupby(["alg", "N"]).agg(fe_min=("fe", "min"), fe_max=("fe", "max"), final_min=("final", "min"),
                                    final_max=("final", "max"), feas_med=("feasible", "median"),
                                    nd_min=("nd", "min"), nd_med=("nd", "median"), nd_max=("nd", "max"))
    g.to_csv(os.path.join(TAB, "final_sets_summary.csv"))
    print(g.to_string())


# ------------------------------------------------------------------------------------------
def part_mc(n_rep=20):
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rows = []
    for key in ("urban", "cliff", "dam", "gorge"):
        base = Scene(key)
        miss = scene_missions(key, ms)
        ref = {k: indicators(base, x) for k, x in miss.items()}
        rng = np.random.default_rng(11)
        for rep in range(n_rep):
            dz = rng.normal(0.0, 0.10, base.Z.shape)
            dxy = rng.normal(0.0, 0.20, 2)
            for mode in ("planimetric", "height", "both"):
                Zp = base.Z + (dz if mode != "planimetric" else 0.0)
                s = scene_from(base, Zp, base.res, base.q, base.n, base.A)
                for (inst, name), x in miss.items():
                    xx = x.reshape(-1, 5).copy()
                    if mode != "height":
                        xx[:, :2] -= dxy
                    ind = indicators(s, xx.ravel())
                    rows.append(dict(scene=key, instance=inst, plan=name, rep=rep, mode=mode,
                                     dC3=ind["C3"] - ref[(inst, name)]["C3"], dG1=ind["G1"] - ref[(inst, name)]["G1"],
                                     CV=ind["CV"], CV0=ref[(inst, name)]["CV"]))
        print(key, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "mc_split.csv"), index=False)
    g = df.groupby(["instance", "plan", "mode"]).agg(dC3_min=("dC3", "min"), dC3_max=("dC3", "max"),
                                                     dG1_min=("dG1", "min"), dG1_max=("dG1", "max"),
                                                     CV_max=("CV", "max"), viol=("CV", lambda v: int((v > 0).sum())))
    g.to_csv(os.path.join(TAB, "mc_split_summary.csv"))
    print(g.round(2).to_string())


# ------------------------------------------------------------------------------------------
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
        z0 = Z.min()
        q, n, A, vox, st = build_targets(Z, r, 1000)
        steep = np.abs(n[:, 2]) < 0.5
        variants = {
            "rebuilt targets": (scene_from(base, Z - z0, r, q + 0.5 * n - np.array([0, 0, z0]), n, A),
                                dict(area=A.sum(), steep_area=A[steep].sum())),
            # targets of the 2 m model kept in world coordinates; only the raster used for visibility changes
            "fixed targets": (scene_from(base, Z - z0, r, base.q + np.array([0, 0, z0_ref - z0]), base.n, base.A),
                              dict(area=base.A.sum(), steep_area=base.A[np.abs(base.n[:, 2]) < 0.5].sum())),
        }
        s_fb = scene_from(base, Z - z0, r, base.q + np.array([0, 0, z0_ref - z0]), base.n, base.A)
        from scipy.ndimage import maximum_filter
        rb = int(round(4.0 / r))       # 4 m buffer at every resolution
        s_fb.Zc = np.ascontiguousarray(maximum_filter(Z - z0, size=2 * rb + 1, mode="nearest"))
        variants["fixed targets, 4 m buffer"] = (s_fb, variants["fixed targets"][1])
        for vname, (s, ar) in variants.items():
            for (inst, name), x in miss.items():
                xx = x.reshape(-1, 5).copy()
                xx[:, 2] += z0_ref - z0
                ind = indicators(s, xx.ravel())
                rows.append(dict(res=r, variant=vname, instance=inst, plan=name, **ar, **ind))
        print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "resolution_split.csv"), index=False)
    print(df.round(2).to_string())


# ------------------------------------------------------------------------------------------
def part_gridline():
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        s = Scene(key)
        x = grid_oblique_inside(s, N)
        P = x.reshape(N, 5)
        pos = np.ascontiguousarray(P[:, :3])
        dirs = np.ascontiguousarray(dirs_of(x))
        for tname, tour in (("decoder", build_tour(pos)), ("line order", np.arange(N))):
            tour = np.ascontiguousarray(tour, dtype=np.int64)
            f2, f4, L = path_costs(pos, dirs, tour, MIS["w_climb"], MIS["w_turn"])
            cv = clearance_violation(pos, tour, s.Zc, s.res, MIS["h_safe"])
            rows.append(dict(instance=inst, tour=tname, f2_km=f2 / 1000, f4_deg=math.degrees(f4), L_km=L / 1000,
                             cv=float(cv)))
        print(rows[-2], rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "gridline.csv"), index=False)


if __name__ == "__main__":
    for part in sys.argv[1:]:
        dict(distinct=part_distinct, clear=part_clear, comp=part_comp, mc=part_mc, res=part_res,
             gridline=part_gridline)[part]()
