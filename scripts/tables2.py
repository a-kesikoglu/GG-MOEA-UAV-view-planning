"""LaTeX table bodies of the manuscript."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import INSTANCES, COMPETITORS

TAB = os.path.join("results", "tables")
OUT = os.path.join("results", "latex")
ALGS = ["GG-MOEA"] + COMPETITORS
SYM = {"+": "$+$", "-": "$-$", "~": r"$\approx$"}


def stat_body(name, algs, dec=3, counts=True):
    st = pd.read_csv(os.path.join(TAB, f"{name}.csv"))
    higher = "IGD" not in name
    lines = []
    for inst in INSTANCES:
        d = st[st.instance == inst].set_index("alg")
        if "GG-MOEA" not in d.index:
            continue
        present = [a for a in algs if a in d.index]
        means = d.loc[present, "mean"].values
        best = present[int(np.argmax(means) if higher else np.argmin(means))]
        cells = []
        for a in algs:
            if a not in d.index:
                cells.append("--")
                continue
            c = f"{d.loc[a, 'mean']:.{dec}f}({d.loc[a, 'std']:.{dec}f})"
            if a == best:
                c = r"\textbf{" + c + "}"
            if a != "GG-MOEA":
                c += SYM[d.loc[a, "sym"]]
            cells.append(c)
        lines.append(inst + " & " + " & ".join(cells) + r" \\")
    if counts:
        lines.append(r"\midrule")
        cnt = []
        for a in algs[1:]:
            s = list(st[st.alg == a].sym)
            cnt.append(f"{s.count('+')}/{s.count('-')}/{s.count('~')}")
        lines.append(r"$+/-/\approx$ & -- & " + " & ".join(cnt) + r" \\")
    return "\n".join(lines)


def ranks_body():
    rows = []
    def add(label, path, col):
        d = pd.read_csv(path)
        if "setting" in d.columns:
            d = d.set_index("setting")
            vals = [d.loc[col, a] for a in ALGS]
        else:
            vals = list(d.set_index("alg").loc[ALGS, "mean_rank"])
        best = int(np.argmin(vals))
        cells = [(r"\textbf{%.2f}" % v) if k == best else "%.2f" % v for k, v in enumerate(vals)]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    add("HV", os.path.join(TAB, "friedman_common_HV.csv"), None)
    add(r"HV, reference point 1.2", os.path.join(TAB, "friedman_common_HV12.csv"), None)
    rb = os.path.join(TAB, "robustness_ranks.csv")
    add(r"HV, $\beta^{*}=15^\circ$", rb, "beta15")
    add(r"HV, $\beta^{*}=35^\circ$", rb, "beta35")
    add(r"HV, $w_c,w_t$ halved", rb, "w_half")
    add(r"HV, $w_c,w_t$ doubled", rb, "w_double")
    add(r"IGD$^{+}$", os.path.join(TAB, "friedman_common_IGDp.csv"), None)
    return "\n".join(rows)


PLANS = ["Grid-Nadir", "Grid-Oblique", "Greedy-2S"]


def missions_body(kind):
    d = pd.read_csv(os.path.join(TAB, "missions.csv")).drop(columns=["C3_d5", "C3_d1"])
    d = d.merge(pd.read_csv(os.path.join(TAB, "resolution_dense.csv")), on=["instance", "plan"])
    lines = []
    for inst in INSTANCES:
        e = d[d.instance == inst].set_index("plan")
        names = ["Grid-Nadir", "Grid-Oblique", "Greedy-2S"]
        if "Greedy-2S, lifted" in e.index:
            names.append("Greedy-2S, lifted")
        rival = [p for p in e.index if p.endswith("(Q)") and not p.startswith("GG")][0]
        names += [rival, "GG-MOEA (C)", "GG-MOEA (Q)"]
        first = True
        for n in names:
            if n not in e.index:
                continue
            r = e.loc[n]
            lab = n.replace(" (Q)", ", Q").replace(" (C)", ", C")
            if kind == "mission":
                cv = "0" if r.CV <= 0 else (f"{r.CV:.2f}" if r.CV < 0.05 else f"{r.CV:.1f}")
                cells = [f"{100*(1-r.f1):.1f}", f"{r.f2:.2f}", f"{r.f4:.1f}", cv, f"{r.C3:.1f}", f"{r.C3s:.1f}",
                         f"{r.C3_d5:.1f}", f"{r.G1_d5:.1f}", f"{r.C3g_d5:.1f}"]
            else:
                cells = [f"{r.gsd:.2f}", f"{r.sig50:.2f}", f"{r.sig90:.2f}", f"{int(r.n_comp)}", f"{r.largest:.0f}",
                         f"{r.dup:.0f}"]
            lines.append(((inst + f" ({r.common_share:.0f}\\%)") if (first and kind != "mission") else (inst if first else ""))
                         + " & " + lab + " & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    return "\n".join(lines[:-1])


def profile_body():
    p = pd.read_csv(os.path.join(TAB, "profile.csv"))
    lines = []
    for _, r in p.iterrows():
        search = r.total
        cells = [f"{r.setup:.1f}", f"{search:.1f}", f"{100*r.evaluation/search:.0f}", f"{100*r.alignment/search:.0f}",
                 f"{100*r.variation/search:.0f}", f"{100*r.selection/search:.0f}", f"{100*r.other/search:.0f}"]
        lines.append(r.instance + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    w = lambda n, s: open(os.path.join(OUT, n), "w", encoding="utf8").write(s)
    w("common_HV_body.tex", stat_body("common_HV_paired", ALGS))
    w("common_IGD_body.tex", stat_body("common_IGD_paired", ALGS))
    w("random_HV_body.tex", stat_body("random_HV_tests", ALGS))
    w("ablation_body.tex", stat_body("ablation_HV_paired", ["GG-MOEA", "w/o SNBI", "w/o VCA", "w/o block", "NSGA-III"],
                                     counts=False))
    w("ranks_body.tex", ranks_body())
    w("missions_body.tex", missions_body("mission"))
    w("quality_body.tex", missions_body("quality"))
    w("profile_body.tex", profile_body())
    print(open(os.path.join(OUT, "ranks_body.tex")).read())
    print(open(os.path.join(OUT, "ablation_body.tex")).read())
