"""Experiment runners for GG-MOEA, its ablation variants and the competitor algorithms.
Every runner returns a record with the approximation set at regular evaluation checkpoints."""
import math
import time

import numpy as np
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.algorithms.moo.spea2 import SPEA2Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from src.problem import ViewPlanningProblem
from src.operators import snbi_population, GGVariation, CandidatePool, lift

N_CHECK = 20
_NDS = NonDominatedSorting()


def ref_dirs(P, n_obj=4):
    return get_reference_directions("energy", n_obj, P, seed=1)


def current_set(F, CV):
    feas = CV <= 0
    if feas.any():
        Ff = F[feas]
        nd = _NDS.do(Ff, only_non_dominated_front=True)
        return Ff[nd], np.zeros(len(nd)), np.flatnonzero(feas)[nd]
    k = int(np.argmin(CV))
    return F[[k]], CV[[k]], np.array([k])


class Recorder:
    def __init__(self, maxfe):
        self.maxfe = maxfe
        self.marks = [int(round(maxfe * (k + 1) / N_CHECK)) for k in range(N_CHECK)]
        self.fe, self.F, self.CV, self.nfeas, self.step = [], [], [], [], []
        self.t0 = time.process_time()
        self.fe_last = 0
        self.cur_step = np.nan

    def log(self, fe, F, CV):
        self.fe_last = max(self.fe_last, int(fe))
        while self.marks and fe >= self.marks[0]:
            self.step.append(self.cur_step)
            Fo, Co, _ = current_set(F, CV)
            self.fe.append(self.marks.pop(0))
            self.F.append(Fo.astype(np.float32))
            self.CV.append(Co.astype(np.float32))
            self.nfeas.append(float(np.mean(CV <= 0)))

    def record(self, X, F, CV, **extra):
        Fo, Co, idx = current_set(F, CV)
        out = dict(fe=np.array(self.fe), F_hist=self.F, CV_hist=self.CV, feas_hist=np.array(self.nfeas),
                   X=X[idx].astype(np.float32), F=Fo, CV=Co, cpu=time.process_time() - self.t0,
                   fe_final=self.fe_last, step_hist=np.array(self.step))
        out.update(extra)
        return out


# ----------------------------------------------------------------------------------------
# Standard genetic operators (SBX and polynomial mutation)
# ----------------------------------------------------------------------------------------
def sbx(a, b, lo, hi, rng, eta=20.0, pc=0.9):
    c1, c2 = a.copy(), b.copy()
    if rng.random() > pc:
        return c1, c2
    n = len(a)
    do = (rng.random(n) < 0.5) & (np.abs(a - b) > 1e-12)
    y1, y2 = np.minimum(a, b), np.maximum(a, b)
    u = rng.random(n)
    rngw = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = np.where(u <= 1 / alpha, (u * alpha) ** (1 / (eta + 1)), (1 / (2 - u * alpha)) ** (1 / (eta + 1)))
        ch1 = 0.5 * ((y1 + y2) - bq * (y2 - y1))
        beta = 1.0 + 2.0 * (hi - y2) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = np.where(u <= 1 / alpha, (u * alpha) ** (1 / (eta + 1)), (1 / (2 - u * alpha)) ** (1 / (eta + 1)))
        ch2 = 0.5 * ((y1 + y2) + bq * (y2 - y1))
    ch1, ch2 = np.clip(ch1, lo, hi), np.clip(ch2, lo, hi)
    swap = rng.random(n) < 0.5
    ch1, ch2 = np.where(swap, ch2, ch1), np.where(swap, ch1, ch2)
    c1[do], c2[do] = ch1[do], ch2[do]
    del rngw
    return c1, c2


def pm(x, lo, hi, rng, eta=20.0, pm_rate=None):
    n = len(x)
    pm_rate = 1.0 / n if pm_rate is None else pm_rate
    y = x.copy()
    m = rng.random(n) < pm_rate
    if not m.any():
        return y
    xl, xu, xv = lo[m], hi[m], y[m]
    d1 = (xv - xl) / (xu - xl)
    d2 = (xu - xv) / (xu - xl)
    u = rng.random(m.sum())
    mp = 1.0 / (eta + 1.0)
    dq = np.where(u <= 0.5,
                  (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** mp - 1,
                  1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** mp)
    y[m] = np.clip(xv + dq * (xu - xl), xl, xu)
    return y


def ranking(F, CV):
    """Order: feasible solutions by non-domination rank, then infeasible by violation."""
    n = len(F)
    rank = np.full(n, n, dtype=float)
    feas = CV <= 0
    if feas.any():
        _, r = _NDS.do(F[feas], return_rank=True)
        rank[feas] = r
    key = np.where(feas, rank, n + CV / (CV.max() + 1e-12))
    return key


def tournament(key, rng, k):
    a = rng.integers(len(key), size=k)
    b = rng.integers(len(key), size=k)
    return np.where(key[a] < key[b], a, np.where(key[b] < key[a], b, np.where(rng.random(k) < 0.5, a, b)))


def ga_offspring(X, key, lo, hi, rng, n_off):
    par = tournament(key, rng, n_off + (n_off % 2))
    Y = []
    for i in range(0, len(par), 2):
        c1, c2 = sbx(X[par[i]], X[par[i + 1]], lo, hi, rng)
        Y += [pm(c1, lo, hi, rng), pm(c2, lo, hi, rng)]
    return np.array(Y[:n_off]), par[:n_off]


def snbi_init(scene, N, P, rng, seed=None):
    import json
    import os
    cache = None
    if seed is not None:
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "processed", "snbi_cache")
        os.makedirs(d, exist_ok=True)
        cache = os.path.join(d, f"{scene.key}_N{N}_P{P}_s{seed}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            rng.bit_generator.state = json.loads(str(z["state"]))
            return z["X"].copy(), int(z["fe0"])
    pool = CandidatePool(scene, rng)
    fe0 = int(math.ceil(len(pool.cand) / N))       # candidate scoring counted as evaluations
    X = snbi_population(scene, N, P, rng, pool=pool)
    if cache is not None:
        tmp = cache + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, X=X, fe0=fe0, state=json.dumps(rng.bit_generator.state))
        os.replace(tmp, cache)
    return X, fe0


# ----------------------------------------------------------------------------------------
# GG-MOEA and ablation variants (NSGA-III environmental selection)
# ----------------------------------------------------------------------------------------
def run_gg(scene, N, maxfe, seed, P=100, init="snbi", variation="de", align=True, cr_mean=0.9, p_best=0.1,
           block=True):
    rng = np.random.default_rng(seed)
    prob = ViewPlanningProblem(scene, N)
    lo, hi = prob.xl, prob.xu
    rec = Recorder(maxfe)
    fe = 0
    t_setup = time.process_time()
    if init == "snbi":
        X, fe = snbi_init(scene, N, P, rng, seed)
    else:
        X = rng.uniform(lo, hi, (P, len(lo)))
    F, CV, Q = scene.evaluate(X)
    U = scene.last_used.copy()
    fe += P
    rec.log(fe, F, CV)
    surv = ReferenceDirectionSurvival(ref_dirs(P))
    setup_cpu = time.process_time() - t_setup
    var = GGVariation(scene, N, use_align=align, cr_mean=cr_mean, p_best=p_best, block=block)
    while fe < maxfe:
        key = ranking(F, CV)
        n_off = min(P, maxfe - fe)
        if variation == "de":
            order = np.argsort(key, kind="stable")
            Y = var(X, order, rng)[:n_off]
            rec.cur_step = var.last_step
        else:
            Y, _ = ga_offspring(X, key, lo, hi, rng, n_off)
        Fy, CVy, Qy = scene.evaluate(Y)
        Uy = scene.last_used.copy()
        fe += len(Y)
        Xa, Fa, CVa, Qa = np.vstack([X, Y]), np.vstack([F, Fy]), np.concatenate([CV, CVy]), np.vstack([Q, Qy])
        Ua = np.vstack([U, Uy])
        pop = Population.new("X", Xa, "F", Fa, "G", CVa[:, None],
                             "idx", np.arange(len(Xa)))
        sel = surv.do(prob, pop, n_survive=P, random_state=rng).get("idx").astype(int)
        X, F, CV, Q, U = Xa[sel], Fa[sel], CVa[sel], Qa[sel], Ua[sel]
        rec.log(fe, F, CV)
    return rec.record(X, F, CV, setup_cpu=setup_cpu)


# ----------------------------------------------------------------------------------------
# pymoo competitors
# ----------------------------------------------------------------------------------------
def run_pymoo(name, scene, N, maxfe, seed, P=100, seeded=False):
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.core.callback import Callback
    prob = ViewPlanningProblem(scene, N)
    rec = Recorder(maxfe)
    ops = dict(crossover=SBX(prob=0.9, eta=20), mutation=PM(eta=20), eliminate_duplicates=False)
    fe0 = 0
    if seeded:
        X0, fe0 = snbi_init(scene, N, P, np.random.default_rng(seed), seed)
        ops["sampling"] = X0
        prob.n_evals = fe0
    rd = ref_dirs(P)
    if name == "NSGA-II":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        alg = NSGA2(pop_size=P, **ops)
    elif name == "NSGA-III":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        alg = NSGA3(ref_dirs=rd, pop_size=P, **ops)
    elif name == "RVEA":
        from pymoo.algorithms.moo.rvea import RVEA
        alg = RVEA(ref_dirs=rd, pop_size=P, alpha=2.0, adapt_freq=0.1, **ops)
    elif name == "AGE-MOEA-II":
        from pymoo.algorithms.moo.age2 import AGEMOEA2
        alg = AGEMOEA2(pop_size=P, **ops)
    elif name == "NSDE-R":
        from pymoo.algorithms.moo.nsder import NSDER
        kw = dict(sampling=ops["sampling"]) if seeded else {}
        alg = NSDER(ref_dirs=rd, pop_size=P, variant="DE/rand/1/bin", CR=0.7, **kw)
    elif name == "CMOPSO":
        from pymoo.algorithms.moo.cmopso import CMOPSO
        alg = CMOPSO(pop_size=P, sampling=ops["sampling"]) if seeded else CMOPSO(pop_size=P)
    else:
        raise ValueError(name)

    class CB(Callback):
        def notify(self, algorithm):
            pop = algorithm.pop
            rec.log(prob.n_evals, pop.get("F"), pop.get("CV")[:, 0])

    term = ("n_gen", int(math.ceil((maxfe - fe0) / P))) if name == "RVEA" else ("n_evals", maxfe - fe0)
    res = minimize(prob, alg, term, seed=seed, callback=CB(), verbose=False)
    pop = res.algorithm.pop
    if name == "CMOPSO":
        el = res.algorithm.elites
        pop = Population.merge(pop, el) if el is not None and len(el) else pop
    return rec.record(pop.get("X"), pop.get("F"), pop.get("CV")[:, 0])


# ----------------------------------------------------------------------------------------
# MOEA/D with PBI decomposition and feasibility-rule replacement
# ----------------------------------------------------------------------------------------
def run_moead(scene, N, maxfe, seed, P=100, T=20, delta=0.9, nr=2, theta=5.0, seeded=False):
    rng = np.random.default_rng(seed)
    prob = ViewPlanningProblem(scene, N)
    lo, hi = prob.xl, prob.xu
    rec = Recorder(maxfe)
    W = ref_dirs(P)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(-1), axis=1)[:, :T]
    fe = 0
    if seeded:
        X, fe = snbi_init(scene, N, P, rng, seed)
    else:
        X = rng.uniform(lo, hi, (P, len(lo)))
    F, CV, _ = scene.evaluate(X)
    fe += P
    rec.log(fe, F, CV)
    z = F.min(0)
    znad = F.max(0)

    def pbi(f, w):
        g = (f - z) / np.maximum(znad - z, 1e-12)
        d1 = g @ w
        d2 = np.linalg.norm(g - np.outer(d1, w) if g.ndim > 1 else g - d1 * w, axis=-1)
        return d1 + theta * d2

    while fe < maxfe:
        n_off = min(P, maxfe - fe)
        order = rng.permutation(P)[:n_off]
        pools, Y = [], []
        for i in order:
            pool = B[i] if rng.random() < delta else np.arange(P)
            a, b = rng.choice(pool, 2, replace=False)
            c, _ = sbx(X[a], X[b], lo, hi, rng)
            Y.append(pm(c, lo, hi, rng))
            pools.append(pool)
        Y = np.array(Y)
        Fy, CVy, _ = scene.evaluate(Y)
        fe += len(Y)
        feas_all = np.concatenate([CV, CVy]) <= 0
        if feas_all.any():
            z = np.minimum(z, np.vstack([F, Fy])[feas_all].min(0))
        Fe = F[CV <= 0]
        znad = Fe.max(0) if len(Fe) > 1 else np.maximum(znad, Fy.max(0))
        for k, i in enumerate(order):
            c = 0
            for j in rng.permutation(pools[k]):
                if c >= nr:
                    break
                if CVy[k] <= 0 and CV[j] <= 0:
                    better = pbi(Fy[k], W[j]) <= pbi(F[j], W[j])
                else:
                    better = CVy[k] < CV[j]
                if better:
                    X[j], F[j], CV[j] = Y[k], Fy[k], CVy[k]
                    c += 1
        rec.log(fe, F, CV)
    return rec.record(X, F, CV)


# ----------------------------------------------------------------------------------------
# CCMO: coevolution of a constrained and an unconstrained population (SPEA2 selection)
# ----------------------------------------------------------------------------------------
def run_ccmo(scene, N, maxfe, seed, P=100, seeded=False):
    rng = np.random.default_rng(seed)
    prob = ViewPlanningProblem(scene, N)
    lo, hi = prob.xl, prob.xu
    rec = Recorder(maxfe)
    fe = 0
    if seeded:
        X1, fe = snbi_init(scene, N, P, rng, seed)
    else:
        X1 = rng.uniform(lo, hi, (P, len(lo)))
    F1, CV1, _ = scene.evaluate(X1)
    X2, F2, CV2 = X1.copy(), F1.copy(), CV1.copy()
    fe += P
    rec.log(fe, F1, CV1)
    s1, s2 = SPEA2Survival(normalize=True), SPEA2Survival(normalize=True)

    def select(surv, X, F, CV, constrained):
        c = CV if constrained else np.zeros_like(CV)
        pop = Population.new("X", X, "F", F, "G", c[:, None],
                             "idx", np.arange(len(X)))
        return surv.do(prob, pop, n_survive=P, random_state=rng).get("idx").astype(int)

    while fe < maxfe:
        n_off = min(P, maxfe - fe)
        h = n_off // 2
        Y1, _ = ga_offspring(X1, ranking(F1, CV1), lo, hi, rng, n_off - h)
        Y2, _ = ga_offspring(X2, ranking(F2, np.zeros_like(CV2)), lo, hi, rng, h) if h else (np.empty((0, len(lo))), None)
        Y = np.vstack([Y1, Y2])
        Fy, CVy, _ = scene.evaluate(Y)
        fe += len(Y)
        Xa, Fa, Ca = np.vstack([X1, Y]), np.vstack([F1, Fy]), np.concatenate([CV1, CVy])
        k = select(s1, Xa, Fa, Ca, True)
        X1, F1, CV1 = Xa[k], Fa[k], Ca[k]
        Xb, Fb, Cb = np.vstack([X2, Y]), np.vstack([F2, Fy]), np.concatenate([CV2, CVy])
        k = select(s2, Xb, Fb, Cb, False)
        X2, F2, CV2 = Xb[k], Fb[k], Cb[k]
        rec.log(fe, F1, CV1)
    return rec.record(X1, F1, CV1)


ALGORITHMS = ["GG-MOEA", "NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA-II", "CCMO", "CMOPSO", "NSDE-R"]

ABLATION = {
    "GG-MOEA": {},
    "w/o SNBI": dict(init="random"),
    "w/o VCA": dict(align=False),
    "w/o block": dict(block=False),
}


def run(alg, scene, N, maxfe, seed, P=100, **kw):
    if alg == "GG-MOEA" or alg in ABLATION:
        opts = dict(ABLATION.get(alg, {}))
        opts.update(kw)
        return run_gg(scene, N, maxfe, seed, P=P, **opts)
    if alg == "MOEA/D":
        return run_moead(scene, N, maxfe, seed, P=P, **kw)
    if alg == "CCMO":
        return run_ccmo(scene, N, maxfe, seed, P=P, **kw)
    return run_pymoo(alg, scene, N, maxfe, seed, P=P, **kw)
