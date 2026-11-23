"""Many-objective viewpoint and path planning problem on a 2.5D DSM scene."""
import os
import math
import numpy as np
from pymoo.core.problem import Problem

from src import config as C
from src.photogrammetry import evaluate_population, visibility_matrix


class Scene:
    def __init__(self, key, proc=C.PROC):
        d = np.load(os.path.join(proc, f"{key}.npz"))
        self.key = key
        self.Z, self.Zc = d["Z"], d["Zc"]
        self.res = float(d["res"])
        self.q, self.n, self.A = d["q"], d["n"], d["A"]
        self.z0 = float(d["z0"])
        self.M = len(self.q)
        nr, nc = self.Z.shape
        self.extent = (nc * self.res, nr * self.res)
        cam, mis = C.CAMERA, C.MISSION
        self.prm = np.array([
            mis["gsd_target"], mis["d_min"], mis["d_max"], math.cos(mis["alpha_max"]),
            0.5 * cam["sw"] / cam["f"], 0.5 * cam["sh"] / cam["f"], cam["pix"] / cam["f"],
            mis["k_min"], mis["beta_opt"], mis["h_safe"], mis["w_climb"], mis["w_turn"]], dtype=np.float64)
        self.zlo = float(self.Zc.min() + mis["h_safe"])
        self.zhi = float(self.Z.max() + mis["z_ceiling"])

    def bounds(self, N):
        lo = np.tile([0.0, 0.0, self.zlo, -0.5 * math.pi, -math.pi], N)
        hi = np.tile([self.extent[0], self.extent[1], self.zhi, 0.0, math.pi], N)
        return lo, hi

    def evaluate(self, X):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        F, CV, Q, used = evaluate_population(X, self.q, self.n, self.A, self.Z, self.Zc, self.res,
                                             self.prm)
        F[:, 1] *= 1e-3      # path cost in km
        self.last_used = used
        return F, CV, Q

    def visibility(self, x):
        return visibility_matrix(np.ascontiguousarray(x, dtype=np.float64), self.q, self.n, self.Z,
                                 self.res, self.prm)

    def clearance_z(self, x, y):
        nr, nc = self.Zc.shape
        c = np.clip((np.asarray(x) / self.res).astype(int), 0, nc - 1)
        r = np.clip((np.asarray(y) / self.res).astype(int), 0, nr - 1)
        return self.Zc[r, c] + C.MISSION["h_safe"]


class ViewPlanningProblem(Problem):
    """Decision vector: N blocks [X, Y, Z, omega, kappa]; four objectives; one aggregated
    clearance constraint (CV <= 0)."""

    def __init__(self, scene, N):
        self.scene, self.N = scene, N
        lo, hi = scene.bounds(N)
        super().__init__(n_var=5 * N, n_obj=4, n_ieq_constr=1, xl=lo, xu=hi)
        self.n_evals = 0

    def _evaluate(self, X, out, *args, **kwargs):
        F, CV, Q = self.scene.evaluate(X)
        self.n_evals += len(X)
        out["F"] = F
        out["G"] = CV[:, None]
        out["Qcov"] = Q
