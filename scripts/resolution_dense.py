"""Resolution-conditioned coverage and dense-target evaluation of the assessed missions.

For every mission of missions.pkl the area shares are computed on the optimization targets and on the
dense target set (about 5000 targets; the same points, normals and areas evaluated with normal offsets
of 0.5 m and 0.1 m):
G1   area seen by at least one valid view with local GSD <= g*,
C3   area seen by at least three valid views,
C3g  area seen by at least three valid views that each have local GSD <= g*.
Pairs of plans whose order by C3 differs between the optimization and the dense targets are listed.
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.problem import Scene
from scripts.run_experiments import INSTANCES

TAB = os.path.join("results", "tables")
GSTAR = C.MISSION["gsd_target"]


def shares(scene, x):
    V, QL, D, CA, U, GS = scene.visibility(np.ascontiguousarray(x, dtype=float))
    ok = V & (GS <= GSTAR)
    A, tot = scene.A, scene.A.sum()
    return dict(G1=100 * A[ok.sum(0) >= 1].sum() / tot, C3=100 * A[V.sum(0) >= 3].sum() / tot,
                C3g=100 * A[ok.sum(0) >= 3].sum() / tot)


def main():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    rows = []
    for inst, (key, N, _) in INSTANCES.items():
        sets = {"opt": Scene(key), "d5": Scene(key + "_d5k"), "d1": Scene(key + "_d5k01")}
        for (i, name), rec in ms.items():
            if i != inst:
                continue
            r = dict(instance=inst, plan=name)
            for tag, s in sets.items():
                for k, v in shares(s, rec["x"]).items():
                    r[f"{k}_{tag}"] = v
            rows.append(r)
            print(inst, name, {k: round(v, 1) for k, v in r.items() if k not in ("instance", "plan")}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "resolution_dense.csv"), index=False)
    flips = []
    for inst, g in d.groupby("instance"):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            for tag in ("d5", "d1"):
                s0 = np.sign(a["C3_opt"] - b["C3_opt"])
                s1 = np.sign(a[f"C3_{tag}"] - b[f"C3_{tag}"])
                if s0 != s1:
                    flips.append(dict(instance=inst, set=tag, a=a.plan, b=b.plan, a_opt=a["C3_opt"], b_opt=b["C3_opt"],
                                      a_dense=a[f"C3_{tag}"], b_dense=b[f"C3_{tag}"]))
    f = pd.DataFrame(flips)
    f.to_csv(os.path.join(TAB, "dense_order_changes.csv"), index=False)
    print(f.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
