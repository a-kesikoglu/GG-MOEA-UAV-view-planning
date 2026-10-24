"""Run the experiments in parallel.

usage: python scripts/run_experiments.py {common|ablation|random|sensitivity} [n_workers]
common      all algorithms start from the same SNBI population (primary comparison)
ablation    GG-MOEA without correspondence alignment or without block crossover
random      all algorithms start from uniform random populations
sensitivity mu_CR, p and P of GG-MOEA
Each run is stored as results/raw_runs/<exp>/<config>__<instance>__<seed>.pkl and skipped if present.
"""
import itertools
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INSTANCES = {  # name: (scene, N, MaxFEs)
    "T1": ("urban", 50, 30000), "T2": ("urban", 100, 40000),
    "T3": ("cliff", 50, 30000), "T4": ("cliff", 100, 40000),
    "T5": ("dam", 50, 30000), "T6": ("dam", 100, 40000),
    "T7": ("gorge", 50, 30000), "T8": ("gorge", 100, 40000),
}
COMPETITORS = ["NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA-II", "CCMO", "CMOPSO", "NSDE-R"]


def configs(exp):
    insts = list(INSTANCES)
    if exp == "common":
        cf = [("GG-MOEA", "GG-MOEA", {})] + [(a, a, dict(seeded=True)) for a in COMPETITORS]
        return cf, insts, range(1, 21)
    if exp == "ablation":
        return [("w/o VCA", "w/o VCA", {}), ("w/o block", "w/o block", {})], insts, range(1, 21)
    if exp == "random":
        cf = [("GG-MOEA", "GG-MOEA", dict(init="random"))] + [(a, a, {}) for a in COMPETITORS]
        return cf, ["T1", "T3", "T5", "T7"], range(1, 11)
    if exp == "sensitivity":
        cf = []
        for c in (0.1, 0.5):
            cf.append((f"CR={c}", "GG-MOEA", dict(cr_mean=c)))
        for pb in (0.05, 0.2):
            cf.append((f"p={pb}", "GG-MOEA", dict(p_best=pb)))
        for P in (50, 150):
            cf.append((f"P={P}", "GG-MOEA", dict(P=P)))
        return cf, ["T1", "T5"], range(1, 11)
    raise ValueError(exp)


def job(task):
    exp, name, alg, kw, inst, seed = task
    out = os.path.join("results", "raw_runs", exp, f"{name.replace('/', '-')}__{inst}__{seed}.pkl")
    if os.path.exists(out):
        return out, 0.0
    import warnings
    warnings.filterwarnings("ignore")
    import numba
    numba.set_num_threads(1)
    from src.problem import Scene
    from src.algorithms.runner import run
    scene, N, fe = INSTANCES[inst]
    kw = dict(kw)
    P = kw.pop("P", 100)
    t = time.time()
    rec = run(alg, Scene(scene), N, fe, seed, P=P, **kw)
    rec["wall"] = time.time() - t
    rec.update(alg=name, instance=inst, scene=scene, N=N, maxfe=fe, seed=seed, exp=exp)
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh)
    os.replace(tmp, out)
    return out, rec["wall"]


if __name__ == "__main__":
    exp = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    os.makedirs(os.path.join("results", "raw_runs", exp), exist_ok=True)
    cf, insts, seeds = configs(exp)
    tasks = [(exp, n, a, kw, i, s) for i, (n, a, kw), s in itertools.product(
        sorted(insts, key=lambda k: -INSTANCES[k][1]), cf, seeds)]
    t0 = time.time()
    done = 0
    with Pool(nw, maxtasksperchild=20) as pool:
        for out, w in pool.imap_unordered(job, tasks):
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} runs, elapsed {(time.time()-t0)/60:.1f} min", flush=True)
