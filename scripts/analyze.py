"""Indicators and statistics of the experiments.

A single reference set per instance is built from the final approximation sets of all experiments
(common, random, ablation, sensitivity). HV uses the reference point 1.1 (and 1.2 as a check).
Pairwise tests: two-sided Wilcoxon rank-sum test (normal approximation without continuity
correction, scipy.stats.ranksums) with Holm's correction over the competitors of an instance.
A12 is the probability that a run of the competitor is better than a run of GG-MOEA
(higher HV, lower IGD+). Friedman ranks use instance means (8 blocks).
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import hv, igd_plus, nondominated, holm, vargha_delaney, friedman_ranks, nemenyi_cd
from scripts.run_experiments import INSTANCES, COMPETITORS

RAW = os.path.join("results", "raw_runs")
TAB = os.path.join("results", "tables")
os.makedirs(TAB, exist_ok=True)
INST = list(INSTANCES)
ALGS = ["GG-MOEA"] + COMPETITORS


def load(exp):
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, exp, "*.pkl"))):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        r["exp"] = exp
        r["key"] = r["alg"]
        out.append(r)
    return out


def build_reference(runs):
    ref = {}
    for inst in INST:
        Fs = [r["F"][r["CV"] <= 0] for r in runs if r["instance"] == inst and (r["CV"] <= 0).any()]
        R = nondominated(np.vstack(Fs))
        ref[inst] = dict(R=R, ideal=R.min(0), nadir=R.max(0))
    return ref


def outside_share(F, CV, rf, refpt):
    F = F[CV <= 0]
    if len(F) == 0:
        return np.nan
    G = (F - rf["ideal"]) / np.maximum(rf["nadir"] - rf["ideal"], 1e-12)
    return float(np.mean(np.any(G >= refpt, axis=1)))


def indicators(runs, ref):
    rows = []
    for r in runs:
        rf = ref[r["instance"]]
        rows.append(dict(
            exp=r["exp"], alg=r["key"], instance=r["instance"], seed=r["seed"],
            HV=hv(r["F"], r["CV"], rf["ideal"], rf["nadir"]),
            HV12=hv(r["F"], r["CV"], rf["ideal"], rf["nadir"], ref=1.2),
            IGDp=igd_plus(r["F"], r["CV"], rf["R"], rf["ideal"], rf["nadir"]),
            out11=outside_share(r["F"], r["CV"], rf, 1.1),
            nnd=len(r["F"]), feasible=bool((r["CV"] <= 0).any()),
            cpu=r["cpu"], wall=r.get("wall", np.nan), setup=r.get("setup_cpu", 0.0),
            fe_final=r.get("fe_final", np.nan), maxfe=r["maxfe"],
            hv_hist=np.array([hv(F, C, rf["ideal"], rf["nadir"]) for F, C in zip(r["F_hist"], r["CV_hist"])]),
            fe_hist=r["fe"], feas_hist=r["feas_hist"], step_hist=r.get("step_hist", np.array([]))))
    return pd.DataFrame(rows)


def pairwise(df, metric, higher, algs):
    rows = []
    for inst in INST:
        d = df[df.instance == inst]
        base = d[d.alg == "GG-MOEA"][metric].values
        if len(base) == 0:
            continue
        raw, comp = [], []
        for a in algs[1:]:
            v = d[d.alg == a][metric].values
            if len(v) == 0 or len(base) == 0:
                continue
            raw.append(stats.ranksums(v, base).pvalue)
            a12 = vargha_delaney(v, base) if higher else vargha_delaney(-v, -base)
            comp.append((a, v, a12))
        adj = holm(raw) if raw else []
        rows.append(dict(instance=inst, alg="GG-MOEA", mean=base.mean(), std=base.std(ddof=1),
                         median=np.median(base), n=len(base), p_raw=np.nan, p_holm=np.nan, A12=np.nan, sym=""))
        for (a, v, a12), p, pa in zip(comp, raw, adj):
            better = (np.median(v) > np.median(base)) if higher else (np.median(v) < np.median(base))
            sym = ("+" if better else "-") if pa < 0.05 else "~"
            rows.append(dict(instance=inst, alg=a, mean=v.mean(), std=v.std(ddof=1), median=np.median(v), n=len(v),
                             p_raw=p, p_holm=pa, A12=a12, sym=sym))
    return pd.DataFrame(rows)


def latex(st, algs, higher, path, dec=3, counts=True):
    lines = []
    for inst in INST:
        d = st[st.instance == inst].set_index("alg")
        present = [a for a in algs if a in d.index]
        if not present:
            continue
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
                c += {"+": "$+$", "-": "$-$", "~": r"$\approx$"}[d.loc[a, "sym"]]
            cells.append(c)
        lines.append(inst + " & " + " & ".join(cells) + r" \\")
    if counts:
        lines.append(r"\midrule")
        cnt = []
        for a in algs[1:]:
            s = list(st[st.alg == a].sym)
            cnt.append(f"{s.count('+')}/{s.count('-')}/{s.count('~')}")
        lines.append(r"$+/-/\approx$ & -- & " + " & ".join(cnt) + r" \\")
    with open(path, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))


def friedman_table(df, metric, higher, algs, tag):
    inst = [i for i in INST if i in set(df.instance)]
    M = df.groupby(["instance", "alg"])[metric].mean().unstack()[algs].loc[inst].values
    r, chi, p = friedman_ranks(M, higher_better=higher)
    cd = nemenyi_cd(len(algs), len(inst))
    out = pd.DataFrame(dict(alg=algs, mean_rank=r))
    out["chi2"], out["p"], out["cd"] = chi, p, cd
    out.to_csv(os.path.join(TAB, f"friedman_{tag}.csv"), index=False)
    return out


def main():
    exps = ["common", "random", "ablation", "sensitivity"]
    runs = []
    if "reuse" not in sys.argv:
        for e in exps:
            runs += load(e)
    if "reuse" in sys.argv:
        df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
        df["alg"] = df.alg.replace({"MOEA-D": "MOEA/D", "w-o VCA": "w/o VCA", "w-o block": "w/o block"})
    else:
        ref = build_reference(runs)
        with open(os.path.join(TAB, "reference_sets.pkl"), "wb") as fh:
            pickle.dump(ref, fh)
        df = indicators(runs, ref)
    df.to_pickle(os.path.join(TAB, "indicators_all.pkl"))
    com = df[df.exp == "common"]
    rnd = df[df.exp == "random"]
    for metric, higher in (("HV", True), ("IGDp", False)):
        st = pairwise(com, metric, higher, ALGS)
        st.to_csv(os.path.join(TAB, f"common_{metric}_stats.csv"), index=False)
        latex(st, ALGS, higher, os.path.join(TAB, f"common_{metric}_table.tex"))
        fr = friedman_table(com, metric, higher, ALGS, f"common_{metric}")
        print("common", metric, fr.round(3).to_string(index=False))
    st = pairwise(rnd, "HV", True, ALGS)
    st.to_csv(os.path.join(TAB, "random_HV_stats.csv"), index=False)
    latex(st, ALGS, True, os.path.join(TAB, "random_HV_table.tex"))
    fr = friedman_table(rnd, "HV", True, ALGS, "random_HV")
    print("random HV", fr.round(3).to_string(index=False))
    # HV with reference point 1.2 as a check of the ranking
    fr12 = friedman_table(com, "HV12", True, ALGS, "common_HV12")
    print("common HV(1.2)", fr12.round(3).to_string(index=False))
    # ablation: complete algorithm and variants on all instances
    ab = pd.concat([com[com.alg.isin(["GG-MOEA", "NSGA-III"])], df[df.exp == "ablation"],
                    rnd[rnd.alg == "GG-MOEA"].assign(alg="w/o SNBI")])
    ab.to_pickle(os.path.join(TAB, "ablation_indicators.pkl"))
    abl = ["GG-MOEA", "w/o SNBI", "w/o VCA", "w/o block", "NSGA-III"]
    st = pairwise(ab, "HV", True, abl)
    st.to_csv(os.path.join(TAB, "ablation_HV_stats.csv"), index=False)
    latex(st, abl, True, os.path.join(TAB, "ablation_table.tex"), counts=False)
    summ = df.groupby(["exp", "instance", "alg"])[["HV", "IGDp", "out11", "nnd", "cpu", "setup", "fe_final"]].agg(
        ["mean", "std"]).round(4)
    summ.to_csv(os.path.join(TAB, "summary_all.csv"))
    print(df.groupby(["exp"])[["out11"]].mean())


if __name__ == "__main__":
    main()
