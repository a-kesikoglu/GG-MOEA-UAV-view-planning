"""Additional figures of the article.

sites     study areas: locator map, hillshaded heights and surface targets with scale bars
targets   construction of the surface targets on a small DSM patch
conv      convergence with the main competitors highlighted
fronts    f1-f2 projections with the competitor names in every panel
parallel  four objectives of the selected missions (parallel coordinates)
maps      views at the required GSD and difference maps on the dense targets
cdf       cumulative distributions of the finest local GSD for all instances (dense targets)
profiles  clearance profiles along the tours
ablation  matched-pair differences with bootstrap intervals
gsdval    Jacobian GSD against the on-axis approximation (supplement)
density   indicators against the number of targets (supplement)
"""
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, Normalize, ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.plotstyle import setup, plt, ALG_STYLE
from src import plotstyle
from src.problem import Scene
from src.metrics import nondominated
from src.photogrammetry import build_tour
from scripts.run_experiments import INSTANCES, COMPETITORS

setup()
ALG_STYLE["NSDE-R"] = dict(color="#8C564B", marker="h", ls="-")
FIG = os.path.join("results", "figures")
TAB = os.path.join("results", "tables")
INST = list(INSTANCES)
ALGS = ["GG-MOEA"] + COMPETITORS
LAB = {k: f"{k} ({INSTANCES[k][0]}, $N$={INSTANCES[k][1]})" for k in INST}
GSTAR = C.MISSION["gsd_target"]
ORDER8 = ["T1", "T3", "T5", "T7", "T2", "T4", "T6", "T8"]
PLAN_STYLE = {
    "Grid-Nadir": dict(color="#999999", ls=":"),
    "Grid-Oblique": dict(color="#0072B2", ls="--"),
    "Grid-Nadir-m": dict(color="#56B4E9", ls=(0, (1, 1))),
    "Grid-Oblique-m": dict(color="#56B4E9", ls="-."),
    "Greedy-2S": dict(color="#009E73", ls="-."),
    "Rival (Q)": dict(color="#555555", ls=(0, (5, 1))),
    "GG-MOEA (Q)": dict(color="#D55E00", ls="-"),
    "GG-MOEA (A)": dict(color="#E69F00", ls="-"),
    "GG-MOEA (C)": dict(color="#CC79A7", ls="-"),
}


def save(fig, name):
    plotstyle.save(fig, os.path.join(FIG, name))


def missions():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    with open(os.path.join(TAB, "plans.pkl"), "rb") as fh:
        ms2 = pickle.load(fh)
    return ms, ms2


def greedy_name(ms, inst):
    return "Greedy-2S, lifted" if (inst, "Greedy-2S, lifted") in ms else "Greedy-2S"


def rival_name(ms, inst):
    r = [n for (i, n) in ms if i == inst and n.endswith("(Q)") and not n.startswith("GG")]
    return r[0]


def scalebar(ax, extent, length=100, color="black"):
    x0, y0 = 0.06 * extent[0], 0.06 * extent[1]
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.0, solid_capstyle="butt")
    ax.text(x0 + 0.5 * length, y0 + 0.03 * extent[1], f"{length} m", ha="center", va="bottom", fontsize=6.5,
            color=color)
    xa, ya = 0.9 * extent[0], 0.78 * extent[1]
    ax.annotate("", xy=(xa, ya + 0.14 * extent[1]), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))
    ax.text(xa, ya + 0.16 * extent[1], "N", ha="center", va="bottom", fontsize=7, color=color)


# ------------------------------------------------------------------------------------------
def fig_sites():
    SC = [("urban", "Bern old town (urban)"), ("cliff", "Lauterbrunnen valley wall (cliff)"),
          ("dam", "Contra arch dam (dam)"), ("gorge", "Fribourg, Sarine gorge (gorge)")]
    fig = plt.figure(figsize=(5.2, 5.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1, 1], hspace=0.62, wspace=0.45)
    ax = fig.add_subplot(gs[0, 1:3])
    from pyproj import Transformer
    geo = json.load(open(os.path.join(C.RAW, "naturalearth", "switzerland.geojson")))
    ring = np.array(geo["geometry"]["coordinates"][0])
    tr = Transformer.from_crs(4326, 2056, always_xy=True)
    ex, ny = tr.transform(ring[:, 0], ring[:, 1])
    ax.fill(np.array(ex) / 1000, np.array(ny) / 1000, color="#eeeeee", ec="#555555", lw=0.6)
    for k, (key, lab) in enumerate(SC):
        b = C.SCENES[key]["bounds"]
        xc, yc = 0.5 * (b[0] + b[2]) / 1000, 0.5 * (b[1] + b[3]) / 1000
        ax.plot(xc, yc, "o", ms=4, color="#D55E00", mec="black", mew=0.4)
        off = {"urban": (5, 3), "cliff": (6, -9), "dam": (6, -4), "gorge": (-30, -14)}[key]
        ax.annotate(lab.split(" (")[0], (xc, yc), xytext=off, textcoords="offset points", fontsize=6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("E (km, LV95)")
    ax.set_ylabel("N (km, LV95)")
    ax.set_title("(a) Location of the four scenes")
    ax.tick_params(labelsize=6.5)
    letters = "bcdefghi"
    for k, (key, title) in enumerate(SC):
        s = Scene(key)
        Z = s.Z
        ls = LightSource(azdeg=315, altdeg=40)
        hs = ls.hillshade(s.Z, vert_exag=1.0, dx=s.res, dy=s.res)
        ext = [0, s.extent[0], 0, s.extent[1]]
        ax = fig.add_subplot(gs[1, k])
        norm = Normalize(Z.min(), Z.max())
        rgb = ls.shade(Z, cmap=plt.cm.cividis, norm=norm, vert_exag=1.0, blend_mode="soft", dx=s.res, dy=s.res)
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="bilinear")
        scalebar(ax, s.extent, 100, "white")
        ax.set_title(f"({letters[k]}) {title.replace(' (', chr(10) + '(')}", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xticklabels([])
        cb = fig.colorbar(ScalarMappable(norm, plt.cm.cividis), ax=ax, fraction=0.05, pad=0.03, orientation="horizontal")
        cb.ax.tick_params(labelsize=6.5)
        cb.ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        cb.set_label("height above lowest cell (m)", fontsize=5.8)
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(hs, origin="lower", extent=ext, cmap="Greys_r", vmin=-0.3, vmax=1.2, interpolation="bilinear")
        steep = np.abs(s.n[:, 2]) < 0.5
        ax.scatter(s.q[~steep, 0], s.q[~steep, 1], s=0.5, c="#56B4E9", edgecolors="none")
        ax.scatter(s.q[steep, 0], s.q[steep, 1], s=0.5, c="#D55E00", edgecolors="none")
        scalebar(ax, s.extent, 100, "black")
        ax.set_title(f"({letters[k + 4]}) targets,\nsteep share {100 * s.A[steep].sum() / s.A.sum():.0f}%",
                     fontsize=7)
        ax.tick_params(labelsize=6.5)
        if k == 0:
            ax.set_ylabel("y (m)", fontsize=7)
        ax.set_xlabel("x (m)", fontsize=7)
    hs_ = [Line2D([], [], marker="o", ls="none", ms=3, color="#56B4E9", label="target, $|n_z|\\geq0.5$"),
           Line2D([], [], marker="o", ls="none", ms=3, color="#D55E00", label="target, $|n_z|<0.5$ (steep or vertical)")]
    fig.legend(handles=hs_, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_sites")


# ------------------------------------------------------------------------------------------
def facets(Z, res):
    thr = res * np.tan(np.deg2rad(70.0))
    nr, nc = Z.shape
    pts, nrm, area, kind = [], [], [], []
    for axis in (0, 1):
        za, zb = (Z[:, :-1], Z[:, 1:]) if axis == 1 else (Z[:-1, :], Z[1:, :])
        dz = zb - za
        rr, cc = np.nonzero(np.abs(dz) > thr)
        for r, c in zip(rr, cc):
            hi, lo = (zb[r, c], za[r, c]) if dz[r, c] > 0 else (za[r, c], zb[r, c])
            sgn = -1.0 if dz[r, c] > 0 else 1.0
            if axis == 1:
                xe, ye, nv = (c + 1) * res, (r + 0.5) * res, (sgn, 0.0, 0.0)
            else:
                xe, ye, nv = (c + 0.5) * res, (r + 1) * res, (0.0, sgn, 0.0)
            nseg = max(1, int(np.ceil((hi - lo) / 3.0)))
            hseg = (hi - lo) / nseg
            for k in range(nseg):
                pts.append((xe, ye, lo + (k + 0.5) * hseg)); nrm.append(nv); area.append(res * hseg); kind.append(1)
    gx, gy = np.zeros_like(Z), np.zeros_like(Z)
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
    pts = np.vstack([np.array(pts).reshape(-1, 3), P])
    nrm = np.vstack([np.array(nrm).reshape(-1, 3), nn.reshape(-1, 3)])
    area = np.concatenate([np.array(area), (res * res * nl[..., 0]).ravel()])
    kind = np.concatenate([np.array(kind), np.zeros(nr * nc, int)])
    return pts, nrm, area, kind


def fig_targets():
    s = Scene("urban")
    d = np.load(os.path.join(C.PROC, "urban.npz"))
    vox = float(d["voxel"])
    dd = np.load(os.path.join(C.PROC, "urban_d5k.npz"))
    vox5 = float(dd["voxel"])
    pts, nrm, area, kind = facets(s.Z, s.res)
    hor = np.abs(nrm[:, 2]) < 0.5
    az = np.round(np.arctan2(nrm[:, 1], nrm[:, 0]) / (np.pi / 2)).astype(int) % 4
    cls = np.where(hor, 1 + az, 0)
    x0, y0, w = 150.0, 95.0, 60.0
    win = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + w) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + w)
    fig = plt.figure(figsize=(5.2, 2.35))

    def ticks3d(ax):
        ax.tick_params(labelsize=6.5, pad=-2)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_major_locator(plt.MaxNLocator(3))

    # (a) prisms of the patch
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    c0, r0 = int(x0 / s.res), int(y0 / s.res)
    n = int(w / s.res)
    Zp = s.Z[r0:r0 + n, c0:c0 + n]
    step = 3
    Zs = Zp[::step, ::step]
    xx, yy = np.meshgrid(np.arange(Zs.shape[1]) * step * s.res, np.arange(Zs.shape[0]) * step * s.res)
    ax.bar3d(xx.ravel(), yy.ravel(), np.zeros(Zs.size), step * s.res, step * s.res, Zs.ravel(), color="#cccccc",
             edgecolor="#777777", linewidth=0.1, shade=True)
    ax.set_title("(a) DSM as prisms", fontsize=7)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(30, -60); ticks3d(ax)
    # (b) facets with normals
    ax = fig.add_subplot(1, 4, 2, projection="3d")
    P, Nn, K = pts[win], nrm[win], kind[win]
    sub = np.random.default_rng(0).random(len(P)) < 0.25
    for k, col in ((0, "#56B4E9"), (1, "#D55E00")):
        m = sub & (K == k)
        ax.quiver(P[m, 0] - x0, P[m, 1] - y0, P[m, 2], Nn[m, 0], Nn[m, 1], Nn[m, 2], length=2.5, color=col, lw=0.4)
    ax.set_title("(b) top and vertical\nfacets, normals", fontsize=7)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(30, -60); ticks3d(ax)
    # (c) voxel-orientation groups (M about 1000)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    key = np.column_stack([np.floor(pts / vox).astype(np.int64), cls])
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    gi = inv[win]
    cmap = plt.cm.tab20
    ax.scatter(P[:, 0] - x0, P[:, 1] - y0, P[:, 2], s=0.6, c=cmap((gi * 7) % 20), depthshade=False)
    ax.set_title(f"(c) groups, voxel {vox:.1f} m", fontsize=7)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(30, -60); ticks3d(ax)
    # (d) targets of the two sets
    ax = fig.add_subplot(1, 4, 4)
    ls = LightSource(azdeg=315, altdeg=40)
    hs = ls.hillshade(Zp, dx=s.res, dy=s.res)
    ax.imshow(hs, origin="lower", extent=[0, w, 0, w], cmap="Greys_r", vmin=-0.3, vmax=1.3)
    for qq, nn_, mk, col, lab in ((dd["q"], dd["n"], ".", "#0072B2", f"dense set (voxel {vox5:.1f} m)"),
                                  (s.q, s.n, "o", "#D55E00", f"optimization set (voxel {vox:.1f} m)")):
        m = (qq[:, 0] >= x0) & (qq[:, 0] < x0 + w) & (qq[:, 1] >= y0) & (qq[:, 1] < y0 + w)
        ax.scatter(qq[m, 0] - x0, qq[m, 1] - y0, s=6 if mk == "o" else 3, marker=mk, facecolors="none" if mk == "o" else col,
                   edgecolors=col, lw=0.6, label=lab)
    ax.set_xlim(0, w); ax.set_ylim(0, w)
    ax.set_title("(d) target positions", fontsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlabel("x (m)", fontsize=7); ax.tick_params(labelsize=6.5)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0),
               markerscale=1.5)
    fig.subplots_adjust(left=0.0, right=0.99, wspace=0.45, top=0.86, bottom=0.15)
    save(fig, "fig_targets")


# ------------------------------------------------------------------------------------------
MAIN = ["GG-MOEA", "MOEA/D", "AGE-MOEA-II", "CCMO"]


def fig_conv():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    d0 = df[df.exp == "common"]
    init = pd.read_csv(os.path.join(TAB, "initial_population.csv")).set_index(["instance", "seed"])
    fig, axs = plt.subplots(2, 4, figsize=(5.2, 3.9))
    for inst, ax in zip(ORDER8, axs.ravel()):
        for a in [x for x in ALGS if x not in MAIN] + MAIN[::-1]:
            d = d0[(d0.instance == inst) & (d0.alg == a)].sort_values("seed")
            h0 = init.loc[[(inst, s) for s in d.seed], "HV0"].values
            H = np.column_stack([h0, np.vstack(d.hv_hist.values)])
            fe = np.concatenate([[init.loc[(inst, int(d.seed.iloc[0])), "fe0"]], d.fe_hist.values[0]]) / 1000
            if a in MAIN:
                st = ALG_STYLE[a]
                ax.plot(fe, H.mean(0), color=st["color"], ls=st["ls"], lw=1.5 if a == "GG-MOEA" else 1.1, label=a)
                ax.fill_between(fe, H.mean(0) - H.std(0), H.mean(0) + H.std(0), color=st["color"], alpha=0.15, lw=0)
            else:
                ax.plot(fe, H.mean(0), color="#bbbbbb", lw=0.6, label="other algorithms" if a == "NSGA-II" else None)
        ax.set_title(LAB[inst], fontsize=7)
        ax.grid(True)
        ax.tick_params(labelsize=6.5)
    for ax in axs[1]:
        ax.set_xlabel("FEs ($\\times10^3$)")
    for ax in axs[:, 0]:
        ax.set_ylabel("HV")
    h, l = axs[0, 0].get_legend_handles_labels()
    order = [l.index(x) for x in MAIN + ["other algorithms"]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.3)
    save(fig, "fig3_convergence")


def median_rec(df, inst, alg):
    d = df[(df.exp == "common") & (df.instance == inst) & (df.alg == alg)].sort_values("HV")
    seed = int(d.iloc[len(d) // 2]["seed"])
    with open(os.path.join("results", "raw_runs", "common", f"{alg.replace('/', '-')}__{inst}__{seed}.pkl"), "rb") as fh:
        rec = pickle.load(fh)
    from scripts.turn_impact import corrected
    rec["F"] = corrected(rec)   # f2 with the turning rule of the article, as for the selected missions
    return rec


def fig_fronts():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    ms = pd.read_csv(os.path.join(TAB, "missions.csv"))
    fig, axs = plt.subplots(2, 4, figsize=(5.2, 4.3))
    for inst, ax in zip(ORDER8, axs.ravel()):
        com = df[(df.exp == "common") & (df.instance == inst)]
        rivals = com[com.alg != "GG-MOEA"].groupby("alg").HV.mean().sort_values(ascending=False).index[:2]
        grey = {rivals[0]: dict(color="#555555", ls="--"), rivals[1]: dict(color="#999999", ls=":")}
        hs = []
        for a in list(rivals)[::-1] + ["GG-MOEA"]:
            r = median_rec(df, inst, a)
            F = r["F"][r["CV"] <= 0]
            st = ALG_STYLE[a] if a == "GG-MOEA" else grey[a]
            ax.scatter(F[:, 1], 100 * (1 - F[:, 0]), s=3, color=st["color"], alpha=0.3, edgecolors="none")
            F2 = nondominated(F[:, :2])
            o = np.argsort(F2[:, 1])
            hs.append(ax.plot(F2[o, 1], 100 * (1 - F2[o, 0]), color=st["color"], ls=st["ls"], lw=0.9, label=a)[0])
        e = ms[ms.instance == inst].set_index("plan")
        gname = "Greedy-2S, lifted" if "Greedy-2S, lifted" in e.index else "Greedy-2S"
        for m, mk in (("Grid-Nadir", "s"), ("Grid-Oblique", "D"), (gname, "*")):
            ax.scatter(e.loc[m, "f2"], 100 * (1 - e.loc[m, "f1"]), marker=mk, s=22, facecolors="white",
                       edgecolors="black", lw=0.7, zorder=5)
        for m, mk in (("GG-MOEA (C)", "o"), ("GG-MOEA (Q)", "^")):
            if m in e.index:
                ax.scatter(e.loc[m, "f2"], 100 * (1 - e.loc[m, "f1"]), marker=mk, s=22, color="#D55E00",
                           edgecolors="black", lw=0.6, zorder=6)
        ax.legend(handles=hs[::-1], frameon=False, fontsize=6, loc="center left" if inst == "T3" else "best", handlelength=1.4, borderaxespad=0.2,
                  labelspacing=0.25, handletextpad=0.4)
        ax.set_title(LAB[inst], fontsize=7)
        ax.grid(True)
        ax.tick_params(labelsize=6.5)
    for ax in axs[1]:
        ax.set_xlabel("$f_2$ (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("$1-f_1$ (%)")
    hs = [Line2D([], [], marker="s", ls="none", mfc="white", mec="black", label="Grid-Nadir"),
          Line2D([], [], marker="D", ls="none", mfc="white", mec="black", label="Grid-Oblique"),
          Line2D([], [], marker="*", ls="none", mfc="white", mec="black", label="Greedy-2S"),
          Line2D([], [], marker="o", ls="none", color="#D55E00", mec="black", label="GG-MOEA, point C"),
          Line2D([], [], marker="^", ls="none", color="#D55E00", mec="black", label="GG-MOEA, point Q")]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01),
               columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.3)
    save(fig, "fig5_fronts")


def point_A(df, inst):
    from scripts.mission_checks import select_points
    base = pd.read_csv(os.path.join(TAB, "missions.csv"))
    ob = base[(base.instance == inst) & (base.plan == "Grid-Oblique")].iloc[0]
    rec = median_rec(df, inst, "GG-MOEA")
    pts, _ = select_points(rec["F"], rec["X"].astype(float), ob.f2, ob.f1, ob.f3)
    return rec["F"][pts["A"]] if "A" in pts else None


def fig_parallel():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    with open(os.path.join(TAB, "reference_sets.pkl"), "rb") as fh:
        ref = pickle.load(fh)
    ms, _ = missions()
    pl = pd.read_csv(os.path.join(TAB, "missions.csv"))
    fig, axs = plt.subplots(2, 4, figsize=(5.2, 4.2), sharey=True)
    for inst, ax in zip(ORDER8, axs.ravel()):
        r = ref[inst]
        e = pl[pl.instance == inst].set_index("plan")
        rows = [("Grid-Nadir", e.loc["Grid-Nadir"]), ("Grid-Oblique", e.loc["Grid-Oblique"]),
                ("Greedy-2S", e.loc[greedy_name(ms, inst)]), ("Rival (Q)", e.loc[rival_name(ms, inst)])]
        if "GG-MOEA (C)" in e.index:
            rows.append(("GG-MOEA (C)", e.loc["GG-MOEA (C)"]))
        rows.append(("GG-MOEA (Q)", e.loc["GG-MOEA (Q)"]))
        FA = point_A(df, inst)
        for name, row in rows:
            f = np.array([row.f1, row.f2, row.f3, math.radians(row.f4)])
            g = (f - r["ideal"]) / (r["nadir"] - r["ideal"])
            st = PLAN_STYLE[name]
            lab = {"Rival (Q)": "strongest competitor, point Q", "GG-MOEA (C)": "GG-MOEA, point C",
                   "GG-MOEA (Q)": "GG-MOEA, point Q"}.get(name, name)
            ax.plot(range(4), g, color=st["color"], ls=st["ls"], lw=1.1, marker="o", ms=2.5, label=lab)
        if FA is not None:
            g = (FA - r["ideal"]) / (r["nadir"] - r["ideal"])
            st = PLAN_STYLE["GG-MOEA (A)"]
            ax.plot(range(4), g, color=st["color"], ls=st["ls"], lw=1.1, marker="s", ms=2.5, label="GG-MOEA, point A")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["$f_1$", "$f_2$", "$f_3$", "$f_4$"])
        ax.set_title(LAB[inst], fontsize=7)
        ax.grid(True)
        ax.set_ylim(-0.05, 1.45)
        ax.tick_params(labelsize=6.5)
        ax.text(0.03, 0.97, f"competitor: {rival_name(ms, inst).split(' (')[0]}", transform=ax.transAxes,
                fontsize=6, va="top")
    for ax in axs[:, 0]:
        ax.set_ylabel("normalized objective\n(0: ideal, 1: nadir)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=0.3)
    save(fig, "fig_parallel")


# ------------------------------------------------------------------------------------------
def dense_views(cache=os.path.join(TAB, "dense_views.pkl")):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    ms, ms2 = missions()
    out = {}
    for inst, (key, N, _) in INSTANCES.items():
        sd = Scene(key + "_d5k")
        names = ["Grid-Nadir", "Grid-Oblique", "Grid-Nadir-m", "Grid-Oblique-m", greedy_name(ms, inst),
                 rival_name(ms, inst), "GG-MOEA (Q)"]
        for name in names:
            x = ms2[(inst, name)]["x"]
            V, QL, D, CA, U, GS = sd.visibility(np.ascontiguousarray(x, float))
            out[(inst, name)] = dict(nv=V.sum(0), ng=(V & (GS <= GSTAR)).sum(0), g=np.where(V, GS, np.inf).min(0))
        print(inst, flush=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def fig_maps():
    dv = dense_views()
    cmap = ListedColormap(["#d9d9d9", "#fdd49e", "#fc8d59", "#b30000"])
    dmap = ListedColormap(["#eeeeee", "#0072B2", "#E69F00", "#009E73"])
    fig, axs = plt.subplots(2, 3, figsize=(5.2, 3.95))
    for ri, inst in enumerate(["T3", "T5"]):
        key = INSTANCES[inst][0]
        sd = Scene(key + "_d5k")
        ls = LightSource(azdeg=315, altdeg=40)
        hs = ls.hillshade(sd.Z, dx=sd.res, dy=sd.res)
        ext = [0, sd.extent[0], 0, sd.extent[1]]
        o = np.argsort(sd.q[:, 2])
        for ci, name in enumerate(["Grid-Oblique", "GG-MOEA (Q)"]):
            ax = axs[ri, ci]
            ax.imshow(hs, origin="lower", extent=ext, cmap="Greys_r", vmin=-0.5, vmax=1.5)
            cls = np.minimum(dv[(inst, name)]["ng"], 3)
            ax.scatter(sd.q[o, 0], sd.q[o, 1], s=1.6, c=cls[o], cmap=cmap, vmin=-0.5, vmax=3.5, edgecolors="none")
            A = sd.A
            ax.set_title(f"({'ab'[ci] if ri == 0 else 'de'[ci]}) {inst}: {name.replace(' (Q)', ', point Q')}, "
                         f"$G_1$ {100 * A[cls >= 1].sum() / A.sum():.0f}%", fontsize=7)
            scalebar(ax, sd.extent, 100)
            ax.tick_params(labelsize=6.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        a = dv[(inst, "GG-MOEA (Q)")]["ng"] >= 1
        b = dv[(inst, "Grid-Oblique")]["ng"] >= 1
        cat = np.where(a & b, 3, np.where(a, 1, np.where(b, 2, 0)))
        ax = axs[ri, 2]
        ax.imshow(hs, origin="lower", extent=ext, cmap="Greys_r", vmin=-0.5, vmax=1.5)
        ax.scatter(sd.q[o, 0], sd.q[o, 1], s=1.6, c=cat[o], cmap=dmap, vmin=-0.5, vmax=3.5, edgecolors="none")
        A = sd.A
        sh = [100 * A[cat == k].sum() / A.sum() for k in range(4)]
        ax.set_title(f"({'c' if ri == 0 else 'f'}) {inst}: at least one view at $g^*$", fontsize=7)
        ax.text(0.02, 0.98, f"only GG-MOEA {sh[1]:.0f}%\nonly grid {sh[2]:.0f}%\nboth {sh[3]:.0f}%",
                transform=ax.transAxes, fontsize=6.2, va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
        scalebar(ax, sd.extent, 100)
        ax.tick_params(labelsize=6.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    h1 = [Patch(color=cmap(k), label=l) for k, l in enumerate(["0", "1", "2", "$\\geq$3"])]
    h2 = [Patch(color=dmap(k), label=l) for k, l in enumerate(["neither", "only GG-MOEA (Q)", "only Grid-Oblique", "both"])]
    fig.legend(handles=h1, title="views with $g_{ij}\\leq g^*$", loc="lower left", ncol=4, frameon=False, fontsize=7,
               title_fontsize=7, bbox_to_anchor=(0.02, -0.01), columnspacing=1.0, handlelength=1.5)
    fig.legend(handles=h2, loc="lower right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.99, -0.01),
               columnspacing=1.0, handlelength=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.6, w_pad=0.4)
    save(fig, "fig_maps")


def fig_cdf():
    dv = dense_views()
    ms, _ = missions()
    fig, axs = plt.subplots(2, 4, figsize=(5.2, 4.2), sharey=True)
    for inst, ax in zip(ORDER8, axs.ravel()):
        key = INSTANCES[inst][0]
        sd = Scene(key + "_d5k")
        A = sd.A
        txt = []
        for name, lab in (("Grid-Nadir", "Grid-Nadir"), ("Grid-Oblique", "Grid-Oblique"),
                          ("Grid-Oblique-m", "Grid-Oblique-m"), (greedy_name(ms, inst), "Greedy-2S"),
                          (rival_name(ms, inst), "Rival (Q)"), ("GG-MOEA (Q)", "GG-MOEA (Q)")):
            g = dv[(inst, name)]["g"] * 100
            o = np.argsort(g)
            cumA = np.cumsum(A[o]) / A.sum()
            ok = np.isfinite(g[o])
            st = PLAN_STYLE[lab]
            label = lab if lab != "Rival (Q)" else "strongest competitor, Q"
            ax.plot(np.r_[0, g[o][ok]], 100 * np.r_[0, cumA[ok]], color=st["color"], ls=st["ls"], lw=0.9, label=label)
            ng = dv[(inst, name)]["ng"]
            if lab in ("Grid-Oblique", "GG-MOEA (Q)"):
                txt.append(f"{'Grid-Obl.' if lab == 'Grid-Oblique' else 'GG-MOEA'}:\n$G_1$ {100 * A[ng >= 1].sum() / A.sum():.0f}, "
                           f"$C_{{3,g^*}}$ {100 * A[ng >= 3].sum() / A.sum():.0f}")
        ax.axvline(100 * GSTAR, color="black", lw=0.5)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 100)
        ax.text(0.97, 0.03, "\n".join(txt), transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_title(LAB[inst], fontsize=7)
        ax.grid(True)
        ax.tick_params(labelsize=6.5)
    for ax in axs[1]:
        ax.set_xlabel("finest local GSD (cm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("share of total area (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=0.6, w_pad=0.3)
    save(fig, "fig_cdf")


# ------------------------------------------------------------------------------------------
def profile(s, x, step=1.0):
    P = x.reshape(-1, 5)[:, :3]
    t = build_tour(np.ascontiguousarray(P))
    hs = C.MISSION["h_safe"]
    dist, zf, zr, marks = [], [], [], [0.0]
    acc = 0.0
    nr, nc = s.Zc.shape
    for a, b in zip(t[:-1], t[1:]):
        L = np.linalg.norm(P[b] - P[a])
        n = max(2, int(L / step))
        u = np.linspace(0, 1, n)
        pts = P[a][None] + u[:, None] * (P[b] - P[a])[None]
        c = np.clip((pts[:, 0] / s.res).astype(int), 0, nc - 1)
        r = np.clip((pts[:, 1] / s.res).astype(int), 0, nr - 1)
        dist.append(acc + u * L)
        zf.append(pts[:, 2])
        zr.append(s.Zc[r, c] + hs)
        acc += L
        marks.append(acc)
    return np.concatenate(dist), np.concatenate(zf), np.concatenate(zr), np.array(marks)


def fig_profiles():
    ms, _ = missions()
    fig, axs = plt.subplots(2, 3, figsize=(5.2, 3.5), sharey="row")
    for ri, inst in enumerate(["T5", "T6"]):
        s = Scene(INSTANCES[inst][0])
        for ci, (name, lab) in enumerate((("Greedy-2S", "Greedy-2S"), ("Greedy-2S, lifted", "Greedy-2S, lifted"),
                                          ("GG-MOEA (Q)", "GG-MOEA, point Q"))):
            ax = axs[ri, ci]
            d, zf, zr, mk = profile(s, ms[(inst, name)]["x"])
            ax.fill_between(d / 1000, 0, zr - C.MISSION["h_safe"], color="#cccccc", lw=0, label="buffered DSM")
            ax.plot(d / 1000, zr, color="#555555", lw=0.6, label="required height ($Z^b+h_s$)")
            ax.plot(d / 1000, zf, color="#0072B2", lw=0.9, label="flight path")
            bad = zf < zr
            ax.fill_between(d / 1000, zf, zr, where=bad, color="#D55E00", lw=0, label="clearance deficit")
            ax.plot(mk / 1000, np.interp(mk, d, zf), "o", ms=1.6, color="black")
            from src.photogrammetry import clearance_violation
            x = ms[(inst, name)]["x"].reshape(-1, 5)
            cv = clearance_violation(np.ascontiguousarray(x[:, :3]), build_tour(np.ascontiguousarray(x[:, :3])), s.Zc, s.res,
                                     C.MISSION["h_safe"])
            ax.set_title(f"({'abcdef'[3 * ri + ci]}) {inst}: {lab}," + chr(10) + f"CV = {cv:.1f} m", fontsize=7)
            ax.grid(True)
            ax.tick_params(labelsize=6.5)
            if ri == 1:
                ax.set_xlabel("distance along the tour (km)")
        axs[ri, 0].set_ylabel("height (m)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=0.4)
    save(fig, "fig_profiles")


# ------------------------------------------------------------------------------------------
def fig_ablation():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    ab = pd.read_pickle(os.path.join(TAB, "ablation_indicators.pkl"))
    pr = pd.read_csv(os.path.join(TAB, "ablation_HV_paired.csv"))
    comps = [("w/o SNBI", "SNBI"), ("w/o VCA", "VCA"), ("w/o block", "block crossover"), ("NSGA-III", "DE operators")]
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 3, figsize=(5.2, 2.3), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axs[0]
    x = np.arange(len(INST))
    for k, (v, lab) in enumerate(comps):
        for i, inst in enumerate(INST):
            g = ab[(ab.instance == inst) & (ab.alg == "GG-MOEA")].set_index("seed").HV
            w = ab[(ab.instance == inst) & (ab.alg == v)].set_index("seed").HV
            if len(w) == 0:
                continue
            xo = i + (k - 1.5) * 0.2
            if v == "w/o SNBI":
                bs = [rng.choice(g.values, len(g)).mean() - rng.choice(w.values, len(w)).mean() for _ in range(4000)]
                m, lo, hi = g.mean() - w.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
                pts = None
            else:
                rowp = pr[(pr.instance == inst) & (pr.alg == v)].iloc[0]
                m, lo, hi = -rowp.med, -rowp.hi, -rowp.lo
                s_ = g.index.intersection(w.index)
                pts = g.loc[s_].values - w.loc[s_].values
            if pts is not None:
                ax.scatter(np.full(len(pts), xo) + rng.uniform(-0.04, 0.04, len(pts)), pts, s=1.2, color=cols[k],
                           alpha=0.45, edgecolors="none")
            ax.errorbar(xo, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=2.8, capsize=1.5, lw=0.8, color=cols[k],
                        label=lab if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(INST)
    ax.set_ylabel("HV(GG-MOEA) $-$ HV(variant)")
    ax.set_title("(a) Contribution of each component", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ha, la = ax.get_legend_handles_labels()
    ax.grid(True, axis="y")
    for ax, inst, t in ((axs[1], "T3", "(b)"), (axs[2], "T5", "(c)")):
        for v, col, ls in (("GG-MOEA", "#D55E00", "-"), ("w/o VCA", "#009E73", "-.")):
            src = df[(df.exp == "common") & (df.alg == v)] if v == "GG-MOEA" else df[(df.exp == "ablation") & (df.alg == v)]
            d = src[src.instance == inst]
            S = np.vstack([h for h in d.step_hist.values if len(h)])
            fe = d.fe_hist.values[0] / 1000
            ax.plot(fe[1:], np.nanmean(S, 0)[1:], color=col, ls=ls, label=v)
        ax.set_title(f"{t} Mean viewpoint" + chr(10) + f"displacement, {inst}", fontsize=7)
        ax.set_xlabel("FEs ($\\times10^3$)")
        ax.set_ylabel("m")
        ax.tick_params(labelsize=6.5)
        ax.grid(True)
        if inst == "T3":
            ax.legend(frameon=False, fontsize=6.5, loc="center right")
        else:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.8))
    fig.legend(ha, la, frameon=False, fontsize=7, ncol=4, columnspacing=1.2, handletextpad=0.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig9_ablation")


# ------------------------------------------------------------------------------------------
def fig_gsdval():
    d = pickle.load(open(os.path.join(TAB, "gsd_sample.pkl"), "rb"))
    th = d["theta"]
    fig, axs = plt.subplots(1, 2, figsize=(6.7, 2.6))
    ax = axs[0]
    for ((inst, name), (gj, ga, t)), col in zip(d["sample"].items(), ["#999999", "#0072B2", "#D55E00"]):
        ax.scatter(100 * gj, 100 * ga, s=1.0, color=col, alpha=0.4, edgecolors="none",
                   label=f"{inst}, {name.replace(' (Q)', ', point Q')}")
    lim = [0, 14]
    ax.plot(lim, lim, color="black", lw=0.6, label="equality")
    ax.axvline(4, color="black", lw=0.4, ls=":"); ax.axhline(4, color="black", lw=0.4, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("local GSD from the Jacobian (cm)")
    ax.set_ylabel("on-axis approximation (cm)")
    ax.set_title("(a) Valid views of three missions")
    ax.legend(frameon=False, fontsize=6.5, markerscale=4)
    ax.grid(True)
    ax = axs[1]
    c = th.index.values + 2
    ax.fill_between(c, th["25%"], th["75%"], color="#0072B2", alpha=0.25, lw=0, label="interquartile range")
    ax.plot(c, th["50%"], color="#0072B2", marker="o", ms=2.5, label="median")
    tt = np.linspace(0, 42, 100)
    ax.plot(tt, 1 / np.cos(np.radians(tt)) ** 2, color="black", ls="--", lw=0.7,
            label="$1/\\cos^2\\theta$ (plane parallel to the image)")
    ax.set_xlabel("off-axis angle $\\theta$ of the view (degrees)")
    ax.set_ylabel("approximation / Jacobian GSD")
    ax.set_title("(b) All valid views of the 59 assessed missions")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "figS_gsd")


def fig_density():
    d = pd.read_csv(os.path.join(TAB, "density.csv"))
    fig, axs = plt.subplots(3, 4, figsize=(6.7, 5.4), sharex=True)
    plans = ["Grid-Oblique", "Greedy-2S", "Rival (Q)", "GG-MOEA (Q)"]
    for ci, inst in enumerate(["T1", "T3", "T5", "T7"]):
        e = d[d.instance == inst].copy()
        e["plan"] = e.plan.replace({"Greedy-2S, lifted": "Greedy-2S"})
        e = e.drop_duplicates(["plan", "M_target", "shift"], keep="last")
        for ri, ind in enumerate(["C3", "G1", "C3g"]):
            ax = axs[ri, ci]
            for p in plans:
                g = e[e.plan == p].groupby("M_target")[ind]
                st = PLAN_STYLE[p]
                m = g.mean()
                ax.plot(m.index, m.values, color=st["color"], ls=st["ls"], marker="o", ms=2, lw=0.9,
                        label=p.replace("Rival (Q)", "strongest competitor, Q"))
                ax.fill_between(m.index, g.min().values, g.max().values, color=st["color"], alpha=0.15, lw=0)
            ax.set_xscale("log")
            ax.grid(True)
            ax.tick_params(labelsize=6.5)
            if ri == 0:
                ax.set_title(LAB[inst], fontsize=7)
            if ci == 0:
                ax.set_ylabel({"C3": "$C_3$ (%)", "G1": "$G_1$ (%)", "C3g": "$C_{3,g^*}$ (%)"}[ind])
            if ri == 2:
                ax.set_xlabel("number of targets")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "figS_density")


if __name__ == "__main__":
    F = dict(sites=fig_sites, targets=fig_targets, conv=fig_conv, fronts=fig_fronts, parallel=fig_parallel,
             maps=fig_maps, cdf=fig_cdf, profiles=fig_profiles, ablation=fig_ablation, gsdval=fig_gsdval,
             density=fig_density)
    for k in sys.argv[1:] or list(F):
        if k in F:
            F[k]()


def fig_sens():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    se = df[df.exp == "sensitivity"]
    base = df[(df.exp == "common") & (df.alg == "GG-MOEA") & df.instance.isin(["T1", "T5", "T7"]) & (df.seed <= 10)]
    t7 = pd.read_csv(os.path.join(TAB, "sens_t7.csv")).assign(instance="T7")
    se = pd.concat([se[["instance", "alg", "HV"]], base[["instance", "alg", "HV"]], t7[["instance", "alg", "HV"]]])
    fig, axs = plt.subplots(1, 3, figsize=(5.3, 2.0))
    groups = [(["CR=0.1", "CR=0.5", "GG-MOEA"], ["0.1", "0.5", "0.9"], "$\mu_{CR}$"),
              (["p=0.05", "GG-MOEA", "p=0.2"], ["0.05", "0.1", "0.2"], "$p$"),
              (["P=50", "GG-MOEA", "P=150"], ["50", "100", "150"], "$P$")]
    for k, (ax, (names, ticks, xl)) in enumerate(zip(axs, groups)):
        for inst, c, mk in (("T1", "#0072B2", "o"), ("T5", "#D55E00", "s"), ("T7", "#009E73", "^")):
            xs, m, sd = [], [], []
            for x, n in enumerate(names):
                h = se[(se.instance == inst) & (se.alg == n)].HV
                if len(h):
                    xs.append(x); m.append(h.mean()); sd.append(h.std())
            if len(xs) > 1:
                ax.errorbar(xs, m, yerr=sd, color=c, marker=mk, ms=3.5, capsize=2, lw=1,
                            ls="-" if inst != "T7" else "--", label=inst + (" (independent)" if inst == "T7" else ""))
        ax.set_xticks(range(3)); ax.set_xticklabels(ticks); ax.set_xlabel(xl); ax.grid(True)
        ax.set_title("(" + "abc"[k] + ")")
    axs[0].set_ylabel("HV"); axs[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig10_sensitivity")


if __name__ == "__main__" and "sens" in sys.argv:
    fig_sens()
