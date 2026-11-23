"""Quality indicators and statistical tests."""
import numpy as np
from scipy import stats
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

_NDS = NonDominatedSorting()


def nondominated(F):
    F = np.unique(np.asarray(F, dtype=float), axis=0)
    return F[_NDS.do(F, only_non_dominated_front=True)]


def normalise(F, ideal, nadir):
    return (np.asarray(F) - ideal) / np.maximum(nadir - ideal, 1e-12)


def hv(F, CV, ideal, nadir, ref=1.1):
    """Hypervolume of the feasible part of F after normalisation; zero if infeasible."""
    F = np.asarray(F)[np.asarray(CV) <= 0]
    if len(F) == 0:
        return 0.0
    G = normalise(F, ideal, nadir)
    G = G[np.all(G < ref, axis=1)]
    if len(G) == 0:
        return 0.0
    return float(HV(ref_point=np.full(G.shape[1], ref))(G))


def igd_plus(F, CV, Rf, ideal, nadir):
    F = np.asarray(F)[np.asarray(CV) <= 0]
    if len(F) == 0:
        return np.nan
    A, R = normalise(F, ideal, nadir), normalise(Rf, ideal, nadir)
    d = np.maximum(A[None, :, :] - R[:, None, :], 0.0)
    return float(np.mean(np.min(np.sqrt((d ** 2).sum(-1)), axis=1)))


def vargha_delaney(a, b):
    """Probability that a value drawn from a is larger than one drawn from b."""
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return (gt + 0.5 * eq) / (len(a) * len(b))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def friedman_ranks(table, higher_better=True):
    """table: instances x algorithms. Returns mean ranks, Friedman statistic and p-value."""
    t = np.asarray(table, float)
    r = stats.rankdata(-t if higher_better else t, axis=1)
    chi, p = stats.friedmanchisquare(*t.T)
    return r.mean(0), chi, p


def nemenyi_cd(k, n, alpha=0.05):
    # critical values q_alpha of the studentised range statistic divided by sqrt(2) (Demsar, 2006)
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    return q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
