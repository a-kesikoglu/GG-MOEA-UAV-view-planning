"""Matched-pair statistics for the experiments that share the initial population.

In the common-start experiment, the ablation variants w/o VCA and w/o block, the seeded NSGA-III and
the sensitivity variants of mu_CR and p, every instance-seed pair starts from the same SNBI population.
These runs are compared with GG-MOEA by the two-sided Wilcoxon signed-rank test on the seed-level
differences (exact null distribution when there are no ties or zero differences, normal approximation
otherwise), Holm's correction over the comparisons of an instance, the median paired difference with a
percentile bootstrap interval (10,000 resamples) and the matched-pairs rank-biserial correlation.
Comparisons with different initial populations (w/o SNBI, the random-start experiment, P = 50 and 150)
use the two-sided Wilcoxon rank-sum test.
Algorithm ranks are computed with the eight instances as blocks and, as a check of the dependence
between the two budgets of a scene, with the four scenes as blocks (mean of the two instance ranks).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import holm, vargha_delaney, nemenyi_cd
from scripts.run_experiments import INSTANCES, COMPETITORS

TAB = os.path.join("results", "tables")
INST = list(INSTANCES)
ALGS = ["GG-MOEA"] + COMPETITORS
SCENE = {i: INSTANCES[i][0] for i in INST}
RNG = np.random.default_rng(0)


def paired(base, other, higher):
    """base, other: Series indexed by seed. Returns statistics of other - base."""
    s = base.index.intersection(other.index)
    d = other.loc[s].values - base.loc[s].values
    if not higher:
        d = -d                                       # positive: competitor better
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(n=len(d), med=0.0, lo=0.0, hi=0.0, p=1.0, rrb=0.0)
    p = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue
    r = stats.rankdata(np.abs(nz))
    rrb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    bs = np.median(d[RNG.integers(0, len(d), (10000, len(d)))], axis=1)
    return dict(n=len(d), med=float(np.median(d)), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), p=float(p), rrb=float(rrb))


def compare(df, metric, higher, algs, paired_algs):
    rows = []
    for inst in INST:
        d = df[df.instance == inst]
        b = d[d.alg == algs[0]].set_index("seed")[metric]
        if len(b) == 0:
            continue
        rows.append(dict(instance=inst, alg=algs[0], mean=b.mean(), std=b.std(ddof=1), n=len(b)))
        part = []
        for a in algs[1:]:
            o = d[d.alg == a].set_index("seed")[metric]
            if len(o) == 0:
                continue
            a12 = vargha_delaney(o.values, b.values) if higher else vargha_delaney(-o.values, -b.values)
            if a in paired_algs:
                r = paired(b, o, higher)
                r["test"] = "signed-rank"
            else:
                r = dict(n=len(o), med=np.nan, lo=np.nan, hi=np.nan, rrb=np.nan,
                         p=float(stats.ranksums(o.values, b.values).pvalue), test="rank-sum")
            better = (o.median() > b.median()) if higher else (o.median() < b.median())
            if a in paired_algs:
                better = r["med"] > 0 if r["med"] != 0 else better
            part.append(dict(instance=inst, alg=a, mean=o.mean(), std=o.std(ddof=1), A12=a12, better=better, **r))
        ph = holm([x["p"] for x in part])
        for x, q in zip(part, ph):
            x["p_holm"] = q
            x["sym"] = ("+" if x["better"] else "-") if q < 0.05 else "~"
            rows.append(x)
    return pd.DataFrame(rows)


def body(st, algs, higher, dec=3, counts=True):
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
            if a != algs[0]:
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
    return "\n".join(lines)


def friedman(M):
    """M: blocks x algorithms (higher is better). Returns mean ranks, chi2, p, Iman-Davenport F and p."""
    R = np.vstack([stats.rankdata(-row) for row in M])
    n, k = R.shape
    mr = R.mean(0)
    chi = 12 * n / (k * (k + 1)) * (np.sum(mr ** 2) - k * (k + 1) ** 2 / 4)
    p = stats.chi2.sf(chi, k - 1)
    ff = (n - 1) * chi / (n * (k - 1) - chi)
    pf = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))
    return mr, chi, p, ff, pf, R


def rank_tables(com):
    out = []
    for metric, higher in (("HV", True), ("IGDp", False), ("HV12", True)):
        M = com.groupby(["instance", "alg"])[metric].mean().unstack()[ALGS].loc[INST].values
        if not higher:
            M = -M
        mr, chi, p, ff, pf, R = friedman(M)
        cd8 = nemenyi_cd(len(ALGS), len(INST))
        # scene blocks: mean of the two instance ranks of a scene, re-ranked within the scene
        scenes = sorted(set(SCENE.values()), key=list(SCENE.values()).index)
        Rs = np.vstack([R[[k for k, i in enumerate(INST) if SCENE[i] == s]].mean(0) for s in scenes])
        mr4, chi4, p4, ff4, pf4, _ = friedman(-Rs)
        cd4 = nemenyi_cd(len(ALGS), len(scenes))
        for a, r8, r4 in zip(ALGS, mr, mr4):
            sig8 = (r8 - mr[0]) > cd8 if a != "GG-MOEA" else False
            sig4 = (r4 - mr4[0]) > cd4 if a != "GG-MOEA" else False
            out.append(dict(metric=metric, alg=a, rank8=r8, rank4=r4, sig8=sig8, sig4=sig4,
                            chi8=chi, p8=p, F8=ff, pF8=pf, cd8=cd8, chi4=chi4, p4=p4, F4=ff4, pF4=pf4, cd4=cd4))
    return pd.DataFrame(out)


def main():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    com = df[df.exp == "common"]
    for metric, higher, tag in (("HV", True, "common_HV"), ("IGDp", False, "common_IGD")):
        st = compare(com, metric, higher, ALGS, set(COMPETITORS))
        st.to_csv(os.path.join(TAB, f"{tag}_paired.csv"), index=False)
        print(tag, st[st.alg != "GG-MOEA"].groupby("alg").sym.apply(lambda s: "".join(s)).to_string())
    rnd = df[df.exp == "random"]
    st = compare(rnd, "HV", True, ALGS, set())
    st.to_csv(os.path.join(TAB, "random_HV_tests.csv"), index=False)
    ab = pd.read_pickle(os.path.join(TAB, "ablation_indicators.pkl"))
    abl = ["GG-MOEA", "w/o SNBI", "w/o VCA", "w/o block", "NSGA-III"]
    st = compare(ab, "HV", True, abl, {"w/o VCA", "w/o block", "NSGA-III"})
    st.to_csv(os.path.join(TAB, "ablation_HV_paired.csv"), index=False)
    print("ablation", st[st.alg != "GG-MOEA"].groupby("alg").sym.apply(lambda s: "".join(s)).to_string())
    se = pd.concat([df[df.exp == "sensitivity"],
                    com[(com.alg == "GG-MOEA") & (com.seed <= 10) & com.instance.isin(["T1", "T5"])]])
    se = se[se.instance.isin(["T1", "T5"])]
    sv = ["GG-MOEA", "CR=0.1", "CR=0.5", "p=0.05", "p=0.2", "P=50", "P=150"]
    st = compare(se, "HV", True, sv, {"CR=0.1", "CR=0.5", "p=0.05", "p=0.2"})
    st.to_csv(os.path.join(TAB, "sensitivity_paired.csv"), index=False)
    print(st[["instance", "alg", "mean", "p_holm", "sym"]].round(4).to_string(index=False))
    rk = rank_tables(com)
    rk.to_csv(os.path.join(TAB, "ranks_blocks.csv"), index=False)
    print(rk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
