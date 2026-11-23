"""Decoupled two-stage baseline and photogrammetric quality assessment in the prism model."""
import math
import numpy as np

from src import config as C
from src.operators import grid_nadir, double_grid_oblique, normal_backprojection


def candidate_views(scene, rng, n_clusters=400):
    cands = []
    for f in (0.6, 0.8, 1.0):
        x = normal_backprojection(scene, n_clusters, rng, d_range=(f, f)).reshape(-1, 5)
        cands.append(x)
    cands.append(grid_nadir(scene, 144).reshape(-1, 5))
    for ang in (0.0, 0.25 * math.pi):
        cands.append(double_grid_oblique(scene, 128, angle=ang).reshape(-1, 5))
    return np.vstack(cands)


def greedy_two_stage(scene, N, seed=0):
    """Stage 1: greedy maximisation of the area-weighted top-k view quality over a candidate set.
    Stage 2: tour construction by the same nearest-neighbour/2-opt routine used in the model."""
    rng = np.random.default_rng(seed)
    cand = candidate_views(scene, rng)
    V, QL, *_ = scene.visibility(cand.ravel())
    Qv = np.where(V, QL, 0.0)                          # candidates x targets
    k = C.MISSION["k_min"]
    top = np.zeros((k, scene.M))                       # sorted descending per target
    chosen = []
    for _ in range(N):
        # gain if candidate c is added: its quality replaces the current k-th best when larger
        gain = (np.maximum(Qv - top[-1][None, :], 0.0) * scene.A[None, :]).sum(1)
        gain[chosen] = -1.0
        c = int(np.argmax(gain))
        chosen.append(c)
        allq = np.vstack([top, Qv[c][None, :]])
        top = -np.sort(-allq, axis=0)[:k]
    return cand[chosen].ravel()


def assess(scene, x, sigma_px=0.5):
    """Photogrammetric indicators of one mission."""
    N = len(x) // 5
    V, QL, D, CA, U, GS = scene.visibility(x)
    F, CV, _ = scene.evaluate(x[None])
    A = scene.A
    steep = np.abs(scene.n[:, 2]) < 0.5
    nv = V.sum(0)
    cam, mis = C.CAMERA, C.MISSION
    gsd = np.where(V, GS, np.inf)
    best_gsd = gsd.min(0)
    seen = nv >= 1
    # best intersection angle and least-squares intersection precision
    sig_t = sigma_px * cam["pix"] / cam["f"]
    beta = np.full(scene.M, np.nan)
    sig3 = np.full(scene.M, np.nan)
    for j in np.flatnonzero(nv >= 2):
        idx = np.flatnonzero(V[:, j])
        u = U[idx, j]
        c = np.clip(u @ u.T, -1, 1)
        ang = np.degrees(np.arccos(c))
        iu = np.triu_indices(len(idx), 1)
        a = ang[iu]
        beta[j] = a[np.argmin(np.abs(a - math.degrees(mis["beta_opt"])))]
        Nm = np.zeros((3, 3))
        for t, i in enumerate(idx):
            w = 1.0 / (sig_t * D[i, j]) ** 2
            Nm += w * (np.eye(3) - np.outer(u[t], u[t]))
        try:
            Cv = np.linalg.inv(Nm)
            sig3[j] = math.sqrt(max(np.trace(Cv), 0.0))
        except np.linalg.LinAlgError:
            pass
    pos = x.reshape(N, 5)[:, :3]
    from src.photogrammetry import build_tour, path_costs
    tour = build_tour(np.ascontiguousarray(pos))
    co = np.cos(x.reshape(N, 5)[:, 3])
    dirs = np.column_stack([co * np.cos(x.reshape(N, 5)[:, 4]), co * np.sin(x.reshape(N, 5)[:, 4]),
                            np.sin(x.reshape(N, 5)[:, 3])])
    f2, f4, L = path_costs(np.ascontiguousarray(pos), dirs, tour, mis["w_climb"], mis["w_turn"])
    m2 = ~np.isnan(sig3)
    return dict(
        f1=F[0, 0], f2=F[0, 1], f3=F[0, 2], f4=F[0, 3], CV=CV[0],
        cover3=100 * A[nv >= 3].sum() / A.sum(),
        cover3_steep=100 * A[(nv >= 3) & steep].sum() / A[steep].sum(),
        cover1=100 * A[seen].sum() / A.sum(),
        gsd_mean=100 * np.average(best_gsd[seen], weights=A[seen]),
        beta_med=float(np.nanmedian(beta)),
        sig_med=100 * float(np.median(sig3[m2])) if m2.any() else np.nan,
        sig_p90=100 * float(np.percentile(sig3[m2], 90)) if m2.any() else np.nan,
        length=L, gimbal=math.degrees(f4), views_per_target=float(np.average(nv, weights=A)),
        nv=nv, best_gsd=best_gsd, sig3=sig3)
