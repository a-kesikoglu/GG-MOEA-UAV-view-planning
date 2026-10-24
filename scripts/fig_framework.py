"""Fig. 1: observation model, viewpoint correspondence alignment and GG-MOEA workflow."""
import os
import sys

import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plotstyle import setup, save, plt
from src import config as C

setup()
fig = plt.figure(figsize=(7.2, 2.9))
axa = fig.add_axes([0.0, 0.02, 0.36, 0.86])
axb = fig.add_axes([0.37, 0.02, 0.24, 0.86])
axc = fig.add_axes([0.63, 0.02, 0.37, 0.86])

# (a) observation geometry in a vertical section
ax = axa
xs = np.array([0, 18, 18, 34, 34, 52, 60, 70, 82, 100])
zs = np.array([4, 4, 30, 30, 4, 4, 9, 7, 12, 10])
ax.fill_between(xs, zs, 0, color="#d9d9d9", lw=0)
ax.plot(xs, zs, color="black", lw=0.8)
cam = np.array([70.0, 62.0])
tgt = np.array([34.0, 20.0])
occ_t = np.array([10.0, 4.0])
ax.plot(*cam, marker="o", color="black", ms=4)
ax.text(cam[0] + 2, cam[1] + 1, "camera $\\mathbf{p}_i$", fontsize=7)
d = (np.array([38.0, 18.0]) - cam); d /= np.linalg.norm(d)
half = np.radians(0.5 * np.degrees(2 * np.arctan(0.5 * C.CAMERA["sh"] / C.CAMERA["f"])))
for s in (-1, 1):
    rot = np.array([[np.cos(s * half), -np.sin(s * half)], [np.sin(s * half), np.cos(s * half)]])
    e = cam + 48 * rot @ d
    ax.plot([cam[0], e[0]], [cam[1], e[1]], color="#E69F00", lw=0.8)
ax.annotate("", xy=cam + 18 * d, xytext=cam, arrowprops=dict(arrowstyle="-|>", color="#E69F00", lw=0.9))
ax.text(cam[0] - 16, cam[1] - 3, "$\\mathbf{d}_i$", color="#E69F00", fontsize=8)
ax.plot([cam[0], tgt[0]], [cam[1], tgt[1]], color="#0072B2", lw=1.0)
ax.plot(*tgt, marker="s", color="#0072B2", ms=3.5)
ax.annotate("", xy=tgt + np.array([12, 0]), xytext=tgt, arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=0.9))
ax.text(tgt[0] + 9, tgt[1] - 5, "$\\mathbf{n}_j$", color="#009E73", fontsize=8)
ang = np.degrees(np.arctan2(cam[1] - tgt[1], cam[0] - tgt[0]))
ax.add_patch(Arc(tgt, 14, 14, theta1=0, theta2=ang, color="black", lw=0.6))
ax.text(tgt[0] + 7.5, tgt[1] + 2.0, "$\\alpha_{ij}$", fontsize=7)
ax.text(tgt[0] - 15, tgt[1] - 1, "$\\mathbf{q}_j$", color="#0072B2", fontsize=8)
ax.plot([cam[0], occ_t[0]], [cam[1], occ_t[1]], color="#D55E00", lw=0.9, ls="--")
ax.plot(*occ_t, marker="x", color="#D55E00", ms=4)
ax.text(2, 9, "occluded", color="#D55E00", fontsize=6.5)
ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.set_aspect("equal"); ax.axis("off")


# (b) viewpoint correspondence alignment (top view)
ax = axb
rng = np.random.default_rng(3)
A = rng.uniform(0, 10, (7, 2))
B = A + rng.normal(0, 0.8, A.shape)
perm = rng.permutation(len(B))
Bp = B[perm]
cost = np.linalg.norm(A[:, None] - Bp[None], axis=2)
_, col = linear_sum_assignment(cost)
for i in range(len(A)):
    ax.plot([A[i, 0], Bp[i, 0]], [A[i, 1] + 12, Bp[i, 1] + 12], color="#999999", lw=0.5, ls=":")
    ax.plot([A[i, 0], Bp[col[i], 0]], [A[i, 1], Bp[col[i], 1]], color="#009E73", lw=0.8)
for off in (0, 12):
    ax.scatter(A[:, 0], A[:, 1] + off, s=12, color="#D55E00", zorder=3)
    ax.scatter(Bp[:, 0], Bp[:, 1] + off, s=12, color="#0072B2", marker="^", zorder=3)
ax.text(5, 22.6, "matched by index $k$", ha="center", fontsize=6.5)
ax.text(5, 11.0, "after alignment $\\sigma^{*}$", ha="center", fontsize=6.5)
ax.scatter([], [], s=12, color="#D55E00", label="$\\mathbf{x}_i$ (target)")
ax.scatter([], [], s=12, color="#0072B2", marker="^", label="$\\mathbf{x}_r$ (donor)")
ax.legend(loc="lower center", fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.12), ncol=2, handletextpad=0.1)
ax.set_xlim(-1, 11); ax.set_ylim(-2, 23.5); ax.axis("off")


# (c) workflow
ax = axc
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
boxes = [
    (5, 9.1, "DSM prisms and surface targets"),
    (5, 7.75, "SNBI: patterns, back-projection, GRASP"),
    (5, 6.4, "Evaluate objectives and clearance"),
    (5, 5.05, "VCA: assignment of donor viewpoints"),
    (5, 3.7, "Block DE/current-to-$p$best/1"),
    (5, 2.35, "Evaluate offspring"),
    (5, 1.0, "NSGA-III selection with feasibility rules"),
]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x - 4.3, y - 0.45), 8.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#f2f2f2", ec="black", lw=0.6))
    ax.text(x, y, t, ha="center", va="center", fontsize=6.6)
for (x0, y0, _), (x1, y1, _) in zip(boxes[:-1], boxes[1:]):
    ax.annotate("", xy=(x1, y1 + 0.47), xytext=(x0, y0 - 0.47), arrowprops=dict(arrowstyle="-|>", lw=0.6))
ax.add_patch(FancyArrowPatch((9.35, 1.0), (9.35, 5.05), connectionstyle="arc3,rad=0", arrowstyle="-|>",
                             mutation_scale=7, lw=0.6))
ax.plot([9.3, 9.35], [1.0, 1.0], color="black", lw=0.6)
ax.text(9.6, 3.0, "next generation", rotation=90, va="center", fontsize=6)

for x, t in ((0.18, "(a) Observation geometry"), (0.49, "(b) Viewpoint correspondence"), (0.815, "(c) GG-MOEA workflow")):
    fig.text(x, 0.95, t, ha="center", fontsize=8)
save(fig, os.path.join("results", "figures", "fig1_framework"))
