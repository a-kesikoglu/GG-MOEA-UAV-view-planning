"""Result figures of the manuscript."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plotstyle import setup, save, plt, ALG_STYLE
from src.metrics import nondominated
from src.problem import Scene
from src import config as C
from scripts.run_experiments import INSTANCES, COMPETITORS

setup()
ALG_STYLE["NSDE-R"] = dict(color="#8C564B", marker="h", ls="-")
FIG = os.path.join("results", "figures")
TAB = os.path.join("results", "tables")
INST = list(INSTANCES)
ALGS = ["GG-MOEA"] + COMPETITORS
SCN = {"urban": "urban", "cliff": "cliff", "dam": "dam", "gorge": "gorge"}
LAB = {k: f"{k} ({INSTANCES[k][0]}, $N$={INSTANCES[k][1]})" for k in INST}


def df_all():
    return pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))


def grid8(h=4.6):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, h))
    order = ["T1", "T3", "T5", "T7", "T2", "T4", "T6", "T8"]
    return fig, dict(zip(order, axs.ravel())), axs


def fig_convergence(df):
    d0 = df[df.exp == "common"]
    fig, ax_of, axs = grid8(4.4)
    for inst, ax in ax_of.items():
        for a in ALGS:
            d = d0[(d0.instance == inst) & (d0.alg == a)]
            H = np.vstack(d.hv_hist.values)
            fe = d.fe_hist.values[0] / 1000
            st = ALG_STYLE[a]
            lw = 1.4 if a == "GG-MOEA" else 0.9
            ax.plot(fe, H.mean(0), color=st["color"], ls=st["ls"], lw=lw, label=a)
            ax.fill_between(fe, H.mean(0) - H.std(0), H.mean(0) + H.std(0), color=st["color"], alpha=0.12, lw=0)
        ax.set_title(LAB[inst], fontsize=7)
        ax.grid(True)
        ax.tick_params(labelsize=6)
    for ax in axs[1]:
        ax.set_xlabel("FEs ($\\times10^3$)")
    for ax in axs[:, 0]:
        ax.set_ylabel("HV")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=9, frameon=False, fontsize=6.5, handlelength=2.0,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, os.path.join(FIG, "fig3_convergence"))


def fig_boxplots(df):
    d0 = df[df.exp == "common"]
    fig, ax_of, axs = grid8(4.8)
    for inst, ax in ax_of.items():
        data = [d0[(d0.instance == inst) & (d0.alg == a)].IGDp.values for a in ALGS]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, flierprops=dict(markersize=1.5, markeredgewidth=0.3),
                        medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                        boxprops=dict(lw=0.5))
        for p, a in zip(bp["boxes"], ALGS):
            p.set_facecolor(ALG_STYLE[a]["color"])
            p.set_alpha(0.75)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(ALGS) + 1))
        ax.set_xticklabels(ALGS, rotation=70, ha="right", fontsize=5.5)
        ax.set_title(LAB[inst], fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y")
    for ax in axs[:, 0]:
        ax.set_ylabel("IGD$^+$")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig4_igd"))


def median_rec(df, inst, alg):
    d = df[(df.exp == "common") & (df.instance == inst) & (df.alg == alg)].sort_values("HV")
    seed = int(d.iloc[len(d) // 2]["seed"])
    with open(os.path.join("results", "raw_runs", "common", f"{alg.replace('/', '-')}__{inst}__{seed}.pkl"), "rb") as fh:
        return pickle.load(fh)


def fig_fronts(df):
    ms = pd.read_csv(os.path.join(TAB, "missions.csv"))
    fig, ax_of, axs = grid8(4.9)
    for inst, ax in ax_of.items():
        com = df[(df.exp == "common") & (df.instance == inst)]
        rivals = com[com.alg != "GG-MOEA"].groupby("alg").HV.mean().sort_values(ascending=False).index[:2]
        grey = {rivals[0]: dict(color="#555555", ls="--"), rivals[1]: dict(color="#999999", ls=":")}
        for a in list(rivals)[::-1] + ["GG-MOEA"]:
            r = median_rec(df, inst, a)
            F = r["F"][r["CV"] <= 0]
            st = ALG_STYLE[a] if a == "GG-MOEA" else grey[a]
            ax.scatter(F[:, 1], 100 * (1 - F[:, 0]), s=3, color=st["color"], alpha=0.3, edgecolors="none")
            F2 = nondominated(F[:, :2])
            o = np.argsort(F2[:, 1])
            ax.plot(F2[o, 1], 100 * (1 - F2[o, 0]), color=st["color"], ls=st["ls"], lw=0.9)
        e = ms[ms.instance == inst].set_index("plan")
        gname = "Greedy-2S, lifted" if "Greedy-2S, lifted" in e.index else "Greedy-2S"
        for m, mk in (("Grid-Nadir", "s"), ("Grid-Oblique", "D"), (gname, "*")):
            ax.scatter(e.loc[m, "f2"], 100 * (1 - e.loc[m, "f1"]), marker=mk, s=22, facecolors="white",
                       edgecolors="black", lw=0.7, zorder=5)
        for m, mk in (("GG-MOEA (C)", "o"), ("GG-MOEA (Q)", "^")):
            if m in e.index:
                ax.scatter(e.loc[m, "f2"], 100 * (1 - e.loc[m, "f1"]), marker=mk, s=22, color="#D55E00",
                           edgecolors="black", lw=0.6, zorder=6)
        ax.set_title(LAB[inst], fontsize=7)
        ax.grid(True)
        ax.tick_params(labelsize=6)
    for ax in axs[1]:
        ax.set_xlabel("$f_2$ (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("$1-f_1$ (%)")
    hs = [Line2D([], [], color=ALG_STYLE["GG-MOEA"]["color"], label="GG-MOEA"),
          Line2D([], [], color="#555555", ls="--", label="strongest competitor"),
          Line2D([], [], color="#999999", ls=":", label="second competitor"),
          Line2D([], [], marker="s", ls="none", mfc="white", mec="black", label="Grid-Nadir"),
          Line2D([], [], marker="D", ls="none", mfc="white", mec="black", label="Grid-Oblique"),
          Line2D([], [], marker="*", ls="none", mfc="white", mec="black", label="Greedy-2S"),
          Line2D([], [], marker="o", ls="none", color="#D55E00", mec="black", label="point C"),
          Line2D([], [], marker="^", ls="none", color="#D55E00", mec="black", label="point Q")]
    fig.legend(handles=hs, loc="lower center", ncol=8, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, os.path.join(FIG, "fig5_fronts"))


def fig_operating_points():
    ms = pd.read_csv(os.path.join(TAB, "missions.csv"))
    rows = []
    for inst in INST:
        e = ms[ms.instance == inst].set_index("plan")
        gname = "Greedy-2S, lifted" if "Greedy-2S, lifted" in e.index else "Greedy-2S"
        c = e.loc["GG-MOEA (C)"] if "GG-MOEA (C)" in e.index else None
        q = e.loc["GG-MOEA (Q)"]
        o, g = e.loc["Grid-Oblique"], e.loc[gname]
        rows.append(dict(instance=inst,
                         dC3=c.C3 - o.C3 if c is not None else np.nan,
                         dg=c.f4 - o.f4 if c is not None else np.nan,
                         dQ=100 * ((1 - q.f1) - (1 - g.f1)), df2=100 * (q.f2 - g.f2) / g.f2))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "operating_point_differences.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3))
    x = np.arange(len(INST))
    ax = axs[0]
    ax.bar(x - 0.2, d.dC3, 0.4, color="#0072B2", label="$\\Delta C_3$ (percentage points)")
    ax.bar(x + 0.2, d.dg, 0.4, color="#E69F00", label="$\\Delta\\bar{\\gamma}$ (degrees)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(INST)
    ax.set_title("(a) Point C against Grid-Oblique")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis="y")
    ax = axs[1]
    ax.bar(x - 0.2, d.dQ, 0.4, color="#0072B2", label="$\\Delta(1-f_1)$ (percentage points)")
    ax.bar(x + 0.2, d.df2, 0.4, color="#009E73", label="$\\Delta f_2$ (%)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(INST)
    ax.set_title("(b) Point Q against Greedy-2S (lifted where needed)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig6_operating_points"))


def _frusta(ax, x, depth, every):
    tw, th = 0.5 * C.CAMERA["sw"] / C.CAMERA["f"], 0.5 * C.CAMERA["sh"] / C.CAMERA["f"]
    for k, (X, Y, Zc, om, ka) in enumerate(x.reshape(-1, 5)):
        if k % every:
            continue
        d = np.array([np.cos(om) * np.cos(ka), np.cos(om) * np.sin(ka), np.sin(om)])
        r = np.array([np.sin(ka), -np.cos(ka), 0.0])
        u = np.cross(r, d)
        p = np.array([X, Y, Zc])
        cs = [p + depth * (d + sx * tw * r + sy * th * u) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        for c in cs:
            ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], color="#D55E00", lw=0.35)
        cc = cs + [cs[0]]
        ax.plot([c[0] for c in cc], [c[1] for c in cc], [c[2] for c in cc], color="#D55E00", lw=0.35)


def fig_missions3d():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    from src.photogrammetry import build_tour
    fig = plt.figure(figsize=(7.2, 4.8))
    rows = ["T1", "T7"]
    for ri, inst in enumerate(rows):
        key, N, _ = INSTANCES[inst]
        s = Scene(key)
        gname = "Greedy-2S, lifted" if (inst, "Greedy-2S, lifted") in ms else "Greedy-2S"
        nr, nc = s.Z.shape
        XX, YY = np.meshgrid((np.arange(nc) + 0.5) * s.res, (np.arange(nr) + 0.5) * s.res)
        st = max(1, nr // 100)
        for ci, name in enumerate(["Grid-Oblique", gname, "GG-MOEA (Q)"]):
            x = ms[(inst, name)]["x"]
            ax = fig.add_subplot(2, 3, 3 * ri + ci + 1, projection="3d")
            ax.plot_surface(XX[::st, ::st], YY[::st, ::st], s.Z[::st, ::st], cmap="Greys", linewidth=0,
                            antialiased=False, alpha=0.5, vmin=s.Z.min() - 0.3 * np.ptp(s.Z), vmax=s.Z.max())
            P = x.reshape(-1, 5)
            t = build_tour(np.ascontiguousarray(P[:, :3]))
            ax.plot(P[t, 0], P[t, 1], P[t, 2], color="#0072B2", lw=0.8)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2.5, color="black", depthshade=False)
            _frusta(ax, x, 18.0, 3)
            ax.set_title(f"{inst}: {name.replace('GG-MOEA (Q)', 'GG-MOEA, point Q')}", fontsize=7, pad=0)
            ax.set_xlim(0, s.extent[0]); ax.set_ylim(0, s.extent[1]); ax.set_zlim(s.Z.min(), s.zhi)
            ax.set_box_aspect((1, 1, 0.55))
            ax.view_init(elev=28, azim=-60)
            ax.tick_params(labelsize=5, pad=-2)
    fig.subplots_adjust(wspace=0.0, hspace=0.08, left=0, right=1, top=0.96, bottom=0.02)
    save(fig, os.path.join(FIG, "fig7_missions3d"))


def fig_coverage():
    with open(os.path.join(TAB, "missions.pkl"), "rb") as fh:
        ms = pickle.load(fh)
    fig = plt.figure(figsize=(7.2, 5.6))
    for ri, inst in enumerate(["T3", "T5"]):
        key, N, _ = INSTANCES[inst]
        s = Scene(key)
        for ci, name in enumerate(["Grid-Oblique", "GG-MOEA (Q)"]):
            nv = ms[(inst, name)]["nv"]
            ax = fig.add_subplot(3, 2, 2 * ri + ci + 1, projection="3d")
            unseen = nv == 0
            ax.scatter(s.q[unseen, 0], s.q[unseen, 1], s.q[unseen, 2], s=2, color="#bbbbbb", depthshade=False)
            sc = ax.scatter(s.q[~unseen, 0], s.q[~unseen, 1], s.q[~unseen, 2], s=2.5, c=np.minimum(nv[~unseen], 6),
                            cmap="viridis", vmin=1, vmax=6, depthshade=False)
            ax.set_title(f"{inst}: {name.replace('GG-MOEA (Q)', 'GG-MOEA, point Q')}", fontsize=7, pad=0)
            ax.set_box_aspect((1, 1, 0.6))
            ax.view_init(elev=25, azim=-110 if key == "cliff" else -95)
            ax.tick_params(labelsize=5, pad=-2)
    cax = fig.add_axes([0.93, 0.42, 0.012, 0.45])
    cb = fig.colorbar(sc, cax=cax, ticks=range(1, 7))
    cb.ax.set_yticklabels(["1", "2", "3", "4", "5", r"$\geq$6"])
    cb.set_label("Valid views (grey: none)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    # area-weighted ECDF of the finest local GSD
    for ci, inst in enumerate(["T5", "T7"]):
        key, N, _ = INSTANCES[inst]
        s = Scene(key)
        ax = fig.add_subplot(3, 2, 5 + ci)
        names = [n for (i, n) in ms if i == inst]
        for name, col, ls in (("Grid-Nadir", "#999999", ":"), ("Grid-Oblique", "#0072B2", "--"),
                              ("Greedy-2S, lifted" if "Greedy-2S, lifted" in names else "Greedy-2S", "#009E73", "-."),
                              ("GG-MOEA (Q)", "#D55E00", "-")):
            g = ms[(inst, name)]["gsd"] * 100
            o = np.argsort(g)
            cumA = np.cumsum(s.A[o]) / s.A.sum()
            ok = np.isfinite(g[o])
            ax.plot(g[o][ok], 100 * cumA[ok], color=col, ls=ls, lw=1.0, label=name.replace(" (Q)", ", point Q"))
        ax.axvline(100 * C.MISSION["gsd_target"], color="black", lw=0.5)
        ax.set_xlim(0, 15)
        ax.set_xlabel("Finest local GSD (cm)")
        ax.set_ylabel("Share of total surface area (%)")
        ax.set_title(f"{inst}: area-weighted cumulative distribution", fontsize=7)
        ax.grid(True)
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.subplots_adjust(left=0.07, right=0.91, top=0.96, bottom=0.07, hspace=0.25, wspace=0.18)
    save(fig, os.path.join(FIG, "fig8_coverage"))


def fig_ablation(df):
    ab = pd.read_pickle(os.path.join(TAB, "ablation_indicators.pkl"))
    comps = [("w/o SNBI", "SNBI"), ("w/o VCA", "VCA"), ("w/o block", "block crossover"), ("NSGA-III", "DE variation")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axs[0]
    x = np.arange(len(INST))
    for k, (v, lab) in enumerate(comps):
        diff, err = [], []
        for inst in INST:
            g = ab[(ab.instance == inst) & (ab.alg == "GG-MOEA")].HV
            w = ab[(ab.instance == inst) & (ab.alg == v)].HV
            if len(w) == 0:
                diff.append(np.nan)
                err.append(np.nan)
                continue
            diff.append(g.mean() - w.mean())
            err.append(np.sqrt(g.var() / len(g) + w.var() / len(w)) * 1.96)
        ax.errorbar(x + (k - 1.5) * 0.18, diff, yerr=err, fmt="o", ms=3, capsize=1.5, lw=0.8, label=lab,
                    color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"][k])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(INST)
    ax.set_ylabel("HV(GG-MOEA) $-$ HV(variant)")
    ax.set_title("(a) Contribution of each component")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.grid(True, axis="y")
    for ax, inst, t in ((axs[1], "T3", "(b)"), (axs[2], "T5", "(c)")):
        for v, col, ls in (("GG-MOEA", "#D55E00", "-"), ("w/o VCA", "#009E73", "-.")):
            src = df[(df.exp == "common") & (df.alg == v)] if v == "GG-MOEA" else df[(df.exp == "ablation") & (df.alg == v)]
            d = src[src.instance == inst]
            S = np.vstack([h for h in d.step_hist.values if len(h)])
            fe = d.fe_hist.values[0] / 1000
            ax.plot(fe[1:], np.nanmean(S, 0)[1:], color=col, ls=ls, label=v)
        ax.set_title(f"{t} Mean viewpoint displacement, {inst}")
        ax.set_xlabel("FEs ($\\times10^3$)")
        ax.set_ylabel("m")
        ax.grid(True)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig9_ablation"))


def fig_sensitivity(df):
    se = df[df.exp == "sensitivity"]
    base = df[(df.exp == "common") & (df.alg == "GG-MOEA") & df.instance.isin(["T1", "T5"]) & (df.seed <= 10)]
    se = pd.concat([se, base])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1))
    groups = [(["CR=0.1", "CR=0.5", "GG-MOEA"], ["0.1", "0.5", "0.9"], "$\\mu_{CR}$"),
              (["p=0.05", "GG-MOEA", "p=0.2"], ["0.05", "0.1", "0.2"], "$p$"),
              (["P=50", "GG-MOEA", "P=150"], ["50", "100", "150"], "$P$")]
    for k, (ax, (names, ticks, xl)) in enumerate(zip(axs, groups)):
        for inst, c, mk in (("T1", "#0072B2", "o"), ("T5", "#D55E00", "s")):
            m = [se[(se.instance == inst) & (se.alg == n)].HV.mean() for n in names]
            sd = [se[(se.instance == inst) & (se.alg == n)].HV.std() for n in names]
            ax.errorbar(range(3), m, yerr=sd, color=c, marker=mk, ms=3.5, capsize=2, lw=1, label=inst)
        ax.set_xticks(range(3)); ax.set_xticklabels(ticks); ax.set_xlabel(xl); ax.grid(True)
        ax.set_title("(" + "abc"[k] + ")")
    axs[0].set_ylabel("HV"); axs[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig10_sensitivity"))


if __name__ == "__main__":
    df = df_all()
    which = sys.argv[1:] or ["conv", "box", "fronts", "op", "m3d", "cov", "abl", "sens"]
    if "conv" in which: fig_convergence(df)
    if "box" in which: fig_boxplots(df)
    if "fronts" in which: fig_fronts(df)
    if "op" in which: fig_operating_points()
    if "m3d" in which: fig_missions3d()
    if "cov" in which: fig_coverage()
    if "abl" in which: fig_ablation(df)
    if "sens" in which: fig_sensitivity(df)
