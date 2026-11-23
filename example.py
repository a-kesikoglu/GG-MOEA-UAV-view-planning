"""Reduced example: GG-MOEA with 3000 evaluations on the urban scene with 50 images.

Runs on one core in a few minutes and prints the objectives and the coverage of the mission with the best
observation quality in the final population, next to the oblique reference grid.
"""
import math

import numpy as np

from src.problem import Scene
from src.algorithms.runner import run
from src.operators import double_grid_oblique


def coverage(scene, x):
    V, QL, D, CA, U, GS = scene.visibility(np.ascontiguousarray(x, dtype=float))
    A = scene.A
    nv = V.sum(0)
    ng = (V & (GS <= 0.04)).sum(0)
    return 100 * A[nv >= 3].sum() / A.sum(), 100 * A[ng >= 1].sum() / A.sum()


if __name__ == "__main__":
    scene = Scene("urban")
    rec = run("GG-MOEA", scene, 50, 3000, seed=1, P=100)
    F, X = rec["F"], rec["X"].astype(float)
    k = int(np.argmin(F[:, 0]))
    grid = double_grid_oblique(scene, 50)
    Fg, CVg, _ = scene.evaluate(grid[None])
    print(f"final feasible non-dominated missions: {len(F)}, CPU {rec['cpu']:.0f} s")
    for name, f, x in (("GG-MOEA, best f1", F[k], X[k]), ("Grid-Oblique", Fg[0], grid)):
        c3, g1 = coverage(scene, x)
        print(f"{name:18s} 1-f1 {100 * (1 - f[0]):5.1f}%  f2 {f[1]:.2f} km  f3 {f[2]:.2f}  "
              f"f4 {math.degrees(f[3]):5.1f} deg  C3 {c3:5.1f}%  G1 {g1:5.1f}%")
