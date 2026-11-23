"""Additional experiments: single-component initial populations, random start with N=100
and two parameter settings on the independent scene.

usage: python scripts/run_additional.py {components|random100|sens_t7} [n_workers]
"""
import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import job


def tasks(exp):
    if exp == "components":
        cf = [(f"init={k}", "GG-MOEA", dict(init=k)) for k in ("pattern", "backproj", "greedy")]
        return [(exp, n, a, kw, i, s) for (n, a, kw), i, s in itertools.product(cf, ["T1", "T3", "T5", "T7"], range(1, 11))]
    if exp == "random100":
        return [(exp, "GG-MOEA", "GG-MOEA", dict(init="random"), i, s)
                for i, s in itertools.product(["T2", "T4", "T6", "T8"], range(1, 11))]
    if exp == "sens_t7":
        cf = [("CR=0.5", "GG-MOEA", dict(cr_mean=0.5)), ("P=150", "GG-MOEA", dict(P=150))]
        return [(exp, n, a, kw, "T7", s) for (n, a, kw), s in itertools.product(cf, range(1, 11))]
    raise ValueError(exp)


if __name__ == "__main__":
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    T = []
    for e in sys.argv[1].split(","):
        os.makedirs(os.path.join("results", "raw_runs", e), exist_ok=True)
        T += tasks(e)
    T.sort(key=lambda t: -int(t[4][1]) % 2)  # N=100 instances first
    done = 0
    with Pool(nw, maxtasksperchild=10) as pool:
        for out, w in pool.imap_unordered(job, T):
            done += 1
            print(f"{done}/{len(T)} {os.path.basename(out)} {w:.0f}s", flush=True)
