"""GG-MOEA rerun on T4 and T8 with the corrected turning rule of f2 (same seeds and initial populations).

usage: python scripts/rerun_f2.py [n_workers]
"""
import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import job

if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    os.makedirs(os.path.join("results", "raw_runs", "rerun_f2"), exist_ok=True)
    T = [("rerun_f2", "GG-MOEA", "GG-MOEA", {}, i, s) for i, s in itertools.product(["T4", "T8"], range(1, 21))]
    done = 0
    with Pool(nw, maxtasksperchild=4) as pool:
        for out, w in pool.imap_unordered(job, T):
            done += 1
            print(f"{done}/{len(T)} {os.path.basename(out)} {w:.0f}s", flush=True)
