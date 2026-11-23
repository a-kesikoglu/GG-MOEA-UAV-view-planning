"""Additional optimization-level checks.

norm     ideal and nadir points of the reference sets and the share of final solutions outside the
         reference box (ignored by HV)
igdbal   IGD+ with a density-balanced reference subset (one point per cell of a 0.1 grid in the
         normalised objective space) and the resulting Friedman ranks
cpu      equal-CPU comparison: HV of GG-MOEA at the evaluation count at which its CPU time equals the
         mean CPU time of a competitor run
randfeas feasibility during the random-start runs
comp     single-component initial populations and their final HV
supp     run-level supplementary table of the paired comparisons
"""
import glob
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import hv, igd_plus, normalise, holm
from src.problem import Scene
from scripts.run_experiments import INSTANCES, COMPETITORS
from scripts.stats_paired import friedman

TAB = os.path.join("results", "tables")
RAW = os.path.join("results", "raw_runs")
ALGS = ["GG-MOEA"] + COMPETITORS
INST = list(INSTANCES)


def refsets():
    with open(os.path.join(TAB, "reference_sets.pkl"), "rb") as fh:
        return pickle.load(fh)


def load(exp, name, inst, seed):
    with open(os.path.join(RAW, exp, f"{name.replace('/', '-')}__{inst}__{seed}.pkl"), "rb") as fh:
        return pickle.load(fh)


def part_norm():
    ref = refsets()
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    com = df[df.exp == "common"]
    rows = []
    for inst in INST:
        r = ref[inst]
        row = dict(instance=inst, n_R=len(r["R"]))
        for k, lab in enumerate(("f1", "f2", "f3", "f4")):
            row[f"ideal_{lab}"], row[f"nadir_{lab}"] = r["ideal"][k], r["nadir"][k]
        for a in ALGS:
            row[f"out_{a}"] = 100 * com[(com.instance == inst) & (com.alg == a)].out11.mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "norm.csv"), index=False)
    print(out.round(3).to_string())


def thin(R, ideal, nadir, h=0.1):
    G = normalise(R, ideal, nadir)
    key = np.floor(G / h).astype(int)
    _, idx = np.unique(key, axis=0, return_index=True)
    return R[np.sort(idx)]


def part_igdbal():
    ref = refsets()
    rows = []
    for inst in INST:
        r = ref[inst]
        Rb = thin(r["R"], r["ideal"], r["nadir"])
        for a in ALGS:
            for seed in range(1, 21):
                rec = load("common", a, inst, seed)
                rows.append(dict(instance=inst, alg=a, seed=seed, n_R=len(r["R"]), n_Rb=len(Rb),
                                 IGDp_bal=igd_plus(rec["F"], rec["CV"], Rb, r["ideal"], r["nadir"])))
        print(inst, len(r["R"]), len(Rb), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "igd_balanced.csv"), index=False)
    M = d.groupby(["instance", "alg"]).IGDp_bal.mean().unstack()[ALGS].loc[INST]
    mr, chi, p, *_ = friedman(-M.values)
    print(M.round(3).to_string())
    print("mean ranks", dict(zip(ALGS, np.round(mr, 2))), "chi2", round(chi, 2), "p", p)
    pd.DataFrame(dict(alg=ALGS, rank=mr)).to_csv(os.path.join(TAB, "igd_balanced_ranks.csv"), index=False)


def part_cpu():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    com = df[df.exp == "common"]
    rows = []
    for inst in INST:
        d = com[com.instance == inst]
        gg = d[d.alg == "GG-MOEA"]
        c_gg = gg.cpu.mean()
        H = np.vstack(gg.hv_hist.values)
        fe = np.asarray(gg.fe_hist.iloc[0], float)
        maxfe = gg.maxfe.iloc[0]
        for a in COMPETITORS:
            o = d[d.alg == a]
            frac = o.cpu.mean() / c_gg
            fe_eq = frac * maxfe
            hv_eq = float(np.interp(fe_eq, fe, H.mean(0)))
            rows.append(dict(instance=inst, alg=a, cpu_ratio=c_gg / o.cpu.mean(), fe_equal_cpu=fe_eq,
                             HV_gg_equal_cpu=hv_eq, HV_gg_final=gg.HV.mean(), HV_comp=o.HV.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "cpu.csv"), index=False)
    best = out.loc[out.groupby("instance").HV_comp.idxmax()]
    print(best.round(3).to_string())
    print("GG-MOEA at equal CPU above every competitor on:",
          [i for i, g in out.groupby("instance") if (g.HV_gg_equal_cpu > g.HV_comp).all()])


def part_randfeas():
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    ran = df[df.exp == "random"]
    rows = []
    for (inst, a), g in ran.groupby(["instance", "alg"]):
        first = []
        for fe, fh in zip(g.fe_hist, g.feas_hist):
            k = np.flatnonzero(np.asarray(fh) > 0)
            first.append(fe[k[0]] if len(k) else np.nan)
        rows.append(dict(instance=inst, alg=a, runs=len(g), runs_feasible=int(g.feasible.sum()),
                         first_feasible_checkpoint_med=float(np.nanmedian(first)) if np.isfinite(first).any() else np.nan,
                         first_feasible_max=float(np.nanmax(first)) if np.isfinite(first).any() else np.nan,
                         feas_share_final=float(np.mean([fh[-1] for fh in g.feas_hist])),
                         IGDp_nan=int(g.IGDp.isna().sum())))
    out = pd.DataFrame(rows)
    # feasibility of the uniform random initial populations of GG-MOEA (N=50) and of the new N=100 runs
    init = []
    for inst in ("T1", "T3", "T5", "T7"):
        key, N, _ = INSTANCES[inst]
        s = Scene(key)
        lo, hi = s.bounds(N)
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            X = rng.uniform(lo, hi, (100, len(lo)))
            F, CV, _ = s.evaluate(X)
            init.append(dict(instance=inst, seed=seed, feas=float(np.mean(CV <= 0)), cv_med=float(np.median(CV)),
                             C1_mean=float(np.mean(1 - F[:, 0]))))
    for f in glob.glob(os.path.join(RAW, "random100", "*.pkl")):
        rec = pickle.load(open(f, "rb"))
        init.append(dict(instance=rec["instance"], seed=rec["seed"], feas=float(np.mean(rec["CV0"] <= 0)),
                         cv_med=float(np.median(rec["CV0"])), C1_mean=float(np.mean(1 - rec["F0"][:, 0]))))
    ini = pd.DataFrame(init)
    out.to_csv(os.path.join(TAB, "randfeas.csv"), index=False)
    ini.to_csv(os.path.join(TAB, "random_init.csv"), index=False)
    print(out.to_string())
    print(ini.groupby("instance")[["feas", "cv_med", "C1_mean"]].agg(["mean", "min", "max"]).round(3))


def hv_of(rec_F, rec_CV, r):
    return hv(rec_F, rec_CV, r["ideal"], r["nadir"])


def part_comp():
    """Single-component initial populations (10 runs on T1, T3, T5, T7) against the full SNBI run
    with the same seeds, and the N=100 random-start runs."""
    ref = refsets()
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "components", "*.pkl"))) + \
            sorted(glob.glob(os.path.join(RAW, "random100", "*.pkl"))):
        rec = pickle.load(open(f, "rb"))
        r = ref[rec["instance"]]
        F0, CV0 = rec["F0"], rec["CV0"]
        rows.append(dict(exp=rec["exp"], variant=rec["alg"] if rec["exp"] == "components" else "random",
                         instance=rec["instance"], seed=rec["seed"],
                         feas0=100 * np.mean(CV0 <= 0), HV0=hv_of(F0, CV0, r), C1_0=100 * np.max(1 - F0[:, 0]),
                         HV=hv_of(rec["F"], rec["CV"], r), cpu=rec["cpu"], init_cpu=rec.get("init_cpu")))
    # full SNBI: initial population from the cache, final HV from the common-start runs
    from src.algorithms.runner import snbi_init
    for inst in ("T1", "T3", "T5", "T7", "T2", "T4", "T6", "T8"):
        key, N, _ = INSTANCES[inst]
        s = Scene(key)
        for seed in range(1, 11):
            X, fe0 = snbi_init(s, N, 100, np.random.default_rng(seed), seed)
            F0, CV0, _ = s.evaluate(X)
            g = df[(df.exp == "common") & (df.alg == "GG-MOEA") & (df.instance == inst) & (df.seed == seed)]
            rows.append(dict(exp="snbi", variant="SNBI", instance=inst, seed=seed, feas0=100 * np.mean(CV0 <= 0),
                             HV0=hv_of(F0, CV0, ref[inst]), C1_0=100 * np.max(1 - F0[:, 0]), HV=float(g.HV.iloc[0]),
                             cpu=float(g.cpu.iloc[0]), init_cpu=float(g.setup.iloc[0])))
    # lifting removed: initial statistics only
    from src.algorithms.runner import component_init
    for inst in ("T1", "T3", "T5", "T7"):
        key, N, _ = INSTANCES[inst]
        s = Scene(key)
        for kind in ("pattern", "backproj", "greedy"):
            for seed in range(1, 4):
                X, _ = component_init(s, N, 100, np.random.default_rng(seed), kind, do_lift=False)
                F0, CV0, _ = s.evaluate(X)
                rows.append(dict(exp="nolift", variant=f"init={kind}, no lift", instance=inst, seed=seed,
                                 feas0=100 * np.mean(CV0 <= 0), HV0=hv_of(F0, CV0, ref[inst]),
                                 C1_0=100 * np.max(1 - F0[:, 0]), HV=np.nan, cpu=np.nan, init_cpu=np.nan))
        print(inst, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "components.csv"), index=False)
    summ = out.groupby(["instance", "variant"])[["feas0", "HV0", "C1_0", "HV", "init_cpu"]].agg(["mean", "std"])
    summ.to_csv(os.path.join(TAB, "components_summary.csv"))
    print(summ.round(3).to_string())
    # rank-sum tests against the full SNBI start (same seeds 1-10)
    tests = []
    for inst, g in out[out.exp.isin(["components", "snbi", "random100"])].groupby("instance"):
        base = g[g.variant == "SNBI"].HV.values
        part = []
        for v, h in g[g.variant != "SNBI"].groupby("variant"):
            part.append(dict(instance=inst, variant=v, mean=h.HV.mean(), base=base.mean(),
                             p=stats.ranksums(h.HV.values, base).pvalue))
        if part:
            for x, q in zip(part, holm([x["p"] for x in part])):
                x["p_holm"] = q
                tests.append(x)
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(TAB, "components_tests.csv"), index=False)
    print(t.round(4).to_string())


def part_supp():
    d = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    com = d[d.exp == "common"][["instance", "alg", "seed", "HV", "IGDp", "cpu", "fe_final"]].copy()
    com["initial_population"] = com.instance + "-seed" + com.seed.astype(str)
    gg = com[com.alg == "GG-MOEA"].set_index(["instance", "seed"])
    com["HV_diff_vs_GG"] = com.apply(lambda r: r.HV - gg.loc[(r.instance, r.seed), "HV"], axis=1)
    com["IGDp_diff_vs_GG"] = com.apply(lambda r: r.IGDp - gg.loc[(r.instance, r.seed), "IGDp"], axis=1)
    com.to_csv(os.path.join(TAB, "supplement_runs_common.csv"), index=False)
    for m in ("HV", "IGD"):
        p = pd.read_csv(os.path.join(TAB, f"common_{m}_paired.csv"))
        p.to_csv(os.path.join(TAB, f"supplement_paired_{m}.csv"), index=False)
    print(len(com))


if __name__ == "__main__":
    for part in sys.argv[1:]:
        dict(norm=part_norm, igdbal=part_igdbal, cpu=part_cpu, randfeas=part_randfeas, comp=part_comp,
             supp=part_supp).get(part, lambda: None)()


def part_senst7():
    """mu_CR = 0.5 and P = 150 on T7 (independent scene) against the default (seeds 1-10)."""
    ref = refsets()["T7"]
    df = pd.read_pickle(os.path.join(TAB, "indicators_all.pkl"))
    base = df[(df.exp == "common") & (df.alg == "GG-MOEA") & (df.instance == "T7") & (df.seed <= 10)].set_index("seed").HV
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "sens_t7", "*.pkl"))):
        rec = pickle.load(open(f, "rb"))
        rows.append(dict(alg=rec["alg"], seed=rec["seed"], HV=hv_of(rec["F"], rec["CV"], ref)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TAB, "sens_t7.csv"), index=False)
    out = []
    for a, g in d.groupby("alg"):
        g = g.set_index("seed").HV
        if a.startswith("CR"):
            s_ = g.index.intersection(base.index)
            p = stats.wilcoxon(g.loc[s_] - base.loc[s_]).pvalue
        else:
            p = stats.ranksums(g.values, base.values).pvalue
        out.append(dict(alg=a, mean=g.mean(), std=g.std(), base=base.mean(), base_std=base.std(), p=p))
    o = pd.DataFrame(out)
    o["p_holm"] = holm(o.p.values)
    o.to_csv(os.path.join(TAB, "sens_t7_tests.csv"), index=False)
    print(o.round(4).to_string())


if __name__ == "__main__" and "senst7" in sys.argv:
    part_senst7()
