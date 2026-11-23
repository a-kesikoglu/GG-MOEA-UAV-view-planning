"""Re-evaluate f2 and the turning sum of the assessed missions with the corrected turning term."""
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.problem import Scene
from src.photogrammetry import build_tour
from scripts.consistency_checks import turn_terms
from scripts.run_experiments import INSTANCES

TAB = os.path.join("results", "tables")
scenes = {k: Scene(k) for k in ("urban", "cliff", "dam", "gorge")}
ms = pickle.load(open(os.path.join(TAB, "plans.pkl"), "rb"))
p = pd.read_csv(os.path.join(TAB, "plans.csv"))
m = pd.read_csv(os.path.join(TAB, "missions.csv"))
for idx, r in p.iterrows():
    x = ms[(r.instance, r.plan)]["x"]
    s = scenes[INSTANCES[r.instance][0]]
    F, CV, _ = s.evaluate(np.asarray(x, float)[None])
    pos = np.ascontiguousarray(np.asarray(x, float).reshape(-1, 5)[:, :3])
    a, b = turn_terms(pos, build_tour(pos))
    if abs(F[0, 1] - r.f2) > 1e-9:
        print(r.instance, r.plan, round(r.f2, 4), "->", round(F[0, 1], 4))
    p.loc[idx, "f2"] = F[0, 1]
    p.loc[idx, "turning_deg"] = math.degrees(b)
    mm = (m.instance == r.instance) & (m.plan == r.plan)
    if mm.any():
        m.loc[mm, "f2"] = F[0, 1]
p.to_csv(os.path.join(TAB, "plans.csv"), index=False)
m.to_csv(os.path.join(TAB, "missions.csv"), index=False)
