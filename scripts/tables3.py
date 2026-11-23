"""LaTeX table bodies of the mission, run-level and model checks (written to results/latex)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import INSTANCES

TAB = os.path.join("results", "tables")
OUT = os.path.join("results", "latex")
INST = list(INSTANCES)
ALGS = ["GG-MOEA", "NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA-II", "CCMO", "CMOPSO", "NSDE-R"]
ORDER = ["Grid-Nadir", "Grid-Nadir-m", "Grid-Oblique", "Grid-Oblique-m", "Grid-Oblique-i", "Greedy-2S",
         "Greedy-2S, lifted", "RIVAL (Q)", "GG-MOEA (C)", "GG-MOEA (Q)"]


def w(name, txt):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf8") as fh:
        fh.write(txt.rstrip("\n") + "\n")


def plan_rows(d, inst):
    e = d[d.instance == inst].set_index("plan")
    rival = [p for p in e.index if p.endswith("(Q)") and not p.startswith("GG")][0]
    out = []
    for p in ORDER:
        q = rival if p == "RIVAL (Q)" else p
        if q in e.index:
            out.append((q, e.loc[q]))
    return out


def label(p):
    return p.replace(" (Q)", ", Q").replace(" (C)", ", C")


def f(v, dec=1):
    return f"{v:.{dec}f}"


def cv_txt(v):
    return "0" if v <= 0 else (f"{v:.2f}" if v < 0.05 else f(v))


def missions_main(insts):
    d = pd.read_csv(os.path.join(TAB, "plans.csv"))
    x = pd.read_csv(os.path.join(TAB, "distinct.csv")).set_index(["instance", "plan"])
    lines = []
    for inst in insts:
        for k, (p, r) in enumerate(plan_rows(d, inst)):
            g = x.loc[(inst, p)]
            cells = [inst if k == 0 else "", label(p), f(r.f2, 2), cv_txt(r.CV), f(r.C3s_opt), f(r.C3_dense),
                     f(r.G1_dense), f(r.C3g_dense), f(g.C3gd_dense), f(r.dup, 0)]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1])


def quality():
    """Precision on the common well-conditioned targets; the instance label gives the area share and
    the number of common targets."""
    d = pd.read_csv(os.path.join(TAB, "plans.csv"))
    pr = pd.read_csv(os.path.join(TAB, "precision_common.csv"))
    old = pd.read_csv(os.path.join(TAB, "missions.csv"))
    lines = []
    for inst in INST:
        e = d[d.instance == inst].set_index("plan")
        p_ = pr[pr.instance == inst].set_index("plan")
        o = old[old.instance == inst].set_index("plan")
        share = p_.share.iloc[0]
        k = 0
        for p in p_.index:
            r, q = e.loc[p], p_.loc[p]
            cells = [f"{inst} ({share:.0f}\\%, {int(q.n)})" if k == 0 else "", label(p), f(o.loc[p, "gsd"], 2),
                     f(q.sig50_full, 2), f(q.sig90_full, 2), f(q.sig50_iso, 2), f(r.C3d_opt), f(r.C3_opt_dedup),
                     f"{int(r.comp5)}", f(r.largest5, 0), f(r.largest10, 0)]
            lines.append(" & ".join(cells) + r" \\")
            k += 1
        lines.append(r"\midrule")
    return "\n".join(lines[:-1])


def comp():
    d = pd.read_csv(os.path.join(TAB, "final_sets.csv"))
    L = []
    for a in ALGS:
        for N in (50, 100):
            e = d[(d.alg == a) & (d.N == N)]
            fe = f"{int(e.fe.min()):,}".replace(",", "{,}")
            fin = f"{int(e.final.min())}" if e.final.min() == e.final.max() else f"{int(e.final.min())}--{int(e.final.max())}"
            nd = f"{int(e.nd.min())}" if e.nd.min() == e.nd.max() else f"{int(e.nd.min())}--{int(e.nd.max())}"
            L.append(" & ".join([a if N == 50 else "", str(N), fe, fin, nd, f(e.nd.median(), 0)]) + r" \\")
    return "\n".join(L)


def mc_split():
    d = pd.read_csv(os.path.join(TAB, "mc_split.csv"))
    L = []
    for inst in sorted(d.instance.unique()):
        e = d[d.instance == inst]
        names = ["Grid-Oblique", "Greedy-2S, lifted", "Rival (Q)", "GG-MOEA (C)", "GG-MOEA (Q)"]
        k = 0
        for p in names:
            g = e[e.plan == p]
            if g.empty or g.CV0.iloc[0] > 0:
                continue
            b = g[g["mode"] == "both"]
            v = [int((g[g["mode"] == m].CV > 0).sum()) for m in ("planimetric", "height", "both")]
            L.append(" & ".join([inst if k == 0 else "", label(p).replace("Rival", "Competitor"),
                                 f"{b.dC3.min():.1f} to {b.dC3.max():.1f}", f"{b.dG1.min():.1f} to {b.dG1.max():.1f}",
                                 str(v[0]), str(v[1]), str(v[2]), f(b.CV.max(), 2)]).replace("-", "$-$").replace("Grid$-$", "Grid-").replace("Greedy$-$2S", "Greedy-2S").replace("GG$-$MOEA", "GG-MOEA").replace("$-$0.0 ", "0.0 ").replace("$-$0.0 &", "0.0 &") + r" \\")
            k += 1
        L.append(r"\midrule")
    return "\n".join(L[:-1])


def res_split():
    d = pd.read_csv(os.path.join(TAB, "resolution_split.csv"))
    L = []
    for v in ("rebuilt targets", "fixed targets"):
        e = d[d.variant == v]
        for r in (1.0, 2.0, 4.0):
            g = e[e.res == r]
            a = g.iloc[0]
            row = [v if r == 1.0 else "", f"{r:.0f}", f"{a.area / 1e6:.3f}", f"{a.steep_area / 1e6:.3f}"]
            for inst, p in (("T5", "Grid-Oblique"), ("T5", "GG-MOEA (Q)"), ("T6", "Grid-Oblique"), ("T6", "GG-MOEA (Q)")):
                h = g[(g.instance == inst) & (g.plan == p)].iloc[0]
                row.append(f(h.C3))
            row.append(f"{int((g.CV > 0).sum())}")
            L.append(" & ".join(row) + r" \\")
        L.append(r"\midrule")
    return "\n".join(L[:-1])


def gsdbin():
    d = pd.read_csv(os.path.join(TAB, "gsd_theta.csv"))
    L = []
    for _, r in d.iterrows():
        a = int(r["bin"])
        L.append(" & ".join([f"{a}--{a + 4}", f"{int(r['count']):,}".replace(",", "{,}"), f(100 * (r['50%'] - 1)),
                             f(100 * (r['25%'] - 1)), f(100 * (r['75%'] - 1))]) + r" \\")
    return "\n".join(L)


def gridline():
    d = pd.read_csv(os.path.join(TAB, "gridline.csv"))
    L = []
    for inst in sorted(d.instance.unique()):
        a = d[(d.instance == inst) & (d.tour == "decoder")].iloc[0]
        b = d[(d.instance == inst) & (d.tour == "line order")].iloc[0]
        L.append(" & ".join([inst, f(a.L_km, 2), f(a.f2_km, 2), f(a.f4_deg), f(b.L_km, 2), f(b.f2_km, 2), f(b.f4_deg),
                             cv_txt(b.cv)]) + r" \\")
    return "\n".join(L)


def ranks_scene():
    d = pd.read_csv(os.path.join(TAB, "ranks_blocks.csv"))
    L = []
    for met, lab in (("HV", "HV, scene blocks"), ("IGDp", "IGD$^{+}$, scene blocks")):
        e = d[d.metric == met].set_index("alg")["rank4"]
        best = e.idxmin()
        cells = [(r"\textbf{" + f"{e[a]:.2f}" + "}") if a == best else f"{e[a]:.2f}" for a in ALGS]
        L.append(" & ".join([lab] + cells) + r" \\")
    return "\n".join(L)


if __name__ == "__main__":
    w("missionsmain50_body.tex", missions_main(["T1", "T3", "T5", "T7"]))
    w("missionsmain100_body.tex", missions_main(["T2", "T4", "T6", "T8"]))
    w("quality_common_body.tex", quality())
    w("ranks_scene_body.tex", ranks_scene())
    w("final_sets_body.tex", comp())
    w("mc_split_body.tex", mc_split())
    w("res_split_body.tex", res_split())
    w("gsdbin_body.tex", gsdbin())
    w("gridline_body.tex", gridline())
    for n in ("missionsmain50_body.tex", "quality_common_body.tex", "ranks_scene_body.tex", "final_sets_body.tex",
              "mc_split_body.tex", "res_split_body.tex", "gsdbin_body.tex", "gridline_body.tex"):
        print(open(os.path.join(OUT, n), encoding="utf8").read()[:900])
