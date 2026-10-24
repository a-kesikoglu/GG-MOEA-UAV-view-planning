"""Problem-specific operators: flight-pattern generators, SNBI initialisation, viewpoint
correspondence alignment, block-wise DE/current-to-pbest/1 and the clearance lift."""
import math
import numpy as np
from numba import njit, prange
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter

from src import config as C
from src.photogrammetry import build_tour, cell_of, segment_deficit


# ----------------------------------------------------------------------------------------
# Conventional flight patterns
# ----------------------------------------------------------------------------------------
def _grid_dims(N, ex, ey):
    best = None
    for r in range(1, N + 1):
        if N % r:
            continue
        c = N // r
        err = abs(math.log((c / r) / (ex / ey)))
        if best is None or err < best[0]:
            best = (err, r, c)
    return best[1], best[2]


def _local_top(scene, radius=40.0):
    k = 2 * int(radius / scene.res) + 1
    return maximum_filter(scene.Zc, size=k, mode="nearest")


def _z_at(Zt, scene, x, y):
    nr, nc = Zt.shape
    c = np.clip((x / scene.res).astype(int), 0, nc - 1)
    r = np.clip((y / scene.res).astype(int), 0, nr - 1)
    return Zt[r, c]


def grid_nadir(scene, N, height=None, angle=0.0, rng=None, jitter=0.0):
    """Lawnmower nadir block with terrain-following height above the local maximum."""
    ex, ey = scene.extent
    height = C.MISSION["d_opt"] if height is None else height
    nrow, ncol = _grid_dims(N, ex, ey)
    u = (np.arange(ncol) + 0.5) / ncol
    v = (np.arange(nrow) + 0.5) / nrow
    U, V = np.meshgrid(u, v)
    x, y = _rotate_in_box(U.ravel(), V.ravel(), angle, ex, ey)
    if rng is not None and jitter > 0:
        x = np.clip(x + rng.normal(0, jitter * ex, N), 0, ex)
        y = np.clip(y + rng.normal(0, jitter * ey, N), 0, ey)
    z = _z_at(_local_top(scene), scene, x, y) + height
    om = np.full(N, -0.5 * math.pi)
    ka = np.full(N, angle)
    return _pack(scene, x, y, z, om, ka)


def double_grid_oblique(scene, N, tilt=math.radians(45.0), angle=0.0, rng=None, jitter=0.0):
    """Two perpendicular line sets; the camera looks forward along each line and alternate
    lines are flown in opposite directions, giving four viewing azimuths."""
    ex, ey = scene.extent
    slant = C.MISSION["d_opt"]
    h = slant * math.cos(tilt)
    xs, ys, ks = [], [], []
    for s, n in enumerate((N // 2, N - N // 2)):
        nl, nper = _grid_dims(n, ex, ey)
        for li in range(nl):
            vv = (li + 0.5) / nl
            for pi in range(nper):
                uu = (pi + 0.5) / nper
                fwd = 1 if li % 2 == 0 else -1
                if s == 0:
                    xs.append(uu); ys.append(vv); ks.append(0.0 if fwd > 0 else math.pi)
                else:
                    xs.append(vv); ys.append(uu); ks.append(0.5 * math.pi if fwd > 0 else -0.5 * math.pi)
    x, y = _rotate_in_box(np.array(xs), np.array(ys), angle, ex, ey)
    ka = np.array(ks) + angle
    # shift the camera backwards so that the optical axis meets the surface below the line
    off = h * math.tan(tilt)
    x = np.clip(x - off * np.cos(ka), 0, ex)
    y = np.clip(y - off * np.sin(ka), 0, ey)
    if rng is not None and jitter > 0:
        x = np.clip(x + rng.normal(0, jitter * ex, N), 0, ex)
        y = np.clip(y + rng.normal(0, jitter * ey, N), 0, ey)
    z = _z_at(_local_top(scene), scene, x, y) + h
    om = np.full(N, -(0.5 * math.pi - tilt))
    return _pack(scene, x, y, z, om, ka)


def _rotate_in_box(u, v, angle, ex, ey):
    x = (u - 0.5) * ex
    y = (v - 0.5) * ey
    ca, sa = math.cos(angle), math.sin(angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return np.clip(xr + 0.5 * ex, 0, ex), np.clip(yr + 0.5 * ey, 0, ey)


def _pack(scene, x, y, z, om, ka):
    z = np.clip(z, scene.zlo, scene.zhi)
    ka = (np.asarray(ka) + math.pi) % (2 * math.pi) - math.pi
    return np.column_stack([x, y, z, om, ka]).ravel()


# ----------------------------------------------------------------------------------------
# Surface-normal back-projection
# ----------------------------------------------------------------------------------------
def normal_backprojection(scene, N, rng, d_range=(0.6, 1.0), n_lloyd=3):
    q, n, A = scene.q, scene.n, scene.A
    feat = np.hstack([q, 30.0 * n])
    w = A / A.sum()
    # area-weighted k-means++ seeding followed by a few Lloyd iterations
    idx = [rng.choice(len(q), p=w)]
    d2 = np.sum((feat - feat[idx[0]]) ** 2, axis=1)
    for _ in range(1, N):
        p = w * d2
        idx.append(rng.choice(len(q), p=p / p.sum()))
        d2 = np.minimum(d2, np.sum((feat - feat[idx[-1]]) ** 2, axis=1))
    cen = feat[idx].copy()
    for _ in range(n_lloyd):
        lab = np.argmin(((feat[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        for c in range(N):
            m = lab == c
            if m.any():
                cen[c] = np.average(feat[m], axis=0, weights=A[m])
    lab = np.argmin(((feat[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
    xs = np.zeros((N, 5))
    for c in range(N):
        m = lab == c
        if not m.any():
            m = np.zeros(len(q), bool); m[idx[c]] = True
        qc = np.average(q[m], axis=0, weights=A[m])
        nc = np.average(n[m], axis=0, weights=A[m]); nc /= np.linalg.norm(nc) + 1e-12
        D = rng.uniform(*d_range) * C.MISSION["d_opt"]
        p = qc + D * nc
        p[0] = np.clip(p[0], 0, scene.extent[0]); p[1] = np.clip(p[1], 0, scene.extent[1])
        p[2] = np.clip(max(p[2], scene.clearance_z(p[0], p[1])), scene.zlo, scene.zhi)
        r = qc - p
        xs[c] = [p[0], p[1], p[2], math.asin(np.clip(r[2] / np.linalg.norm(r), -1, 0)), math.atan2(r[1], r[0])]
    return xs.ravel()


def snbi_population(scene, N, n, rng, frac=(0.3, 0.3, 0.2), pool=None):
    lo, hi = scene.bounds(N)
    out = []
    n_grid = int(round(frac[0] * n))
    n_bp = int(round(frac[1] * n))
    n_gr = int(round(frac[2] * n)) if pool is not None else 0
    for k in range(n_grid):
        ang = rng.uniform(0, math.pi)
        h = rng.uniform(0.8, 1.2)
        if k % 2 == 0:
            x = grid_nadir(scene, N, height=h * C.MISSION["d_opt"], angle=ang, rng=rng, jitter=0.03)
        else:
            x = double_grid_oblique(scene, N, tilt=math.radians(rng.uniform(30, 60)), angle=ang, rng=rng, jitter=0.03)
        x = x.reshape(N, 5)
        x[:, 3] = np.clip(x[:, 3] + rng.normal(0, math.radians(5), N), lo[3], hi[3])
        x[:, 4] = x[:, 4] + rng.normal(0, math.radians(5), N)
        out.append(x.ravel())
    for _ in range(n_bp):
        out.append(normal_backprojection(scene, N, rng))
    for _ in range(n_gr):
        out.append(pool.construct(N, rng))
    while len(out) < n:
        x = rng.uniform(lo, hi).reshape(N, 5)
        x[:, 2] = np.maximum(x[:, 2], scene.clearance_z(x[:, 0], x[:, 1]))
        out.append(x.ravel())
    X = np.array(out)
    X = X.reshape(n, N, 5)
    X[:, :, 4] = (X[:, :, 4] + math.pi) % (2 * math.pi) - math.pi
    X = np.clip(X.reshape(n, -1), lo, hi)
    m = n_grid + n_bp + n_gr
    X[:m] = lift(scene, X[:m])
    return X


# ----------------------------------------------------------------------------------------
# Viewpoint correspondence alignment
# ----------------------------------------------------------------------------------------
def align(ref, other, scale):
    """Permute the viewpoints of `other` so that they correspond to those of `ref`
    (linear assignment on the normalised distances between camera positions)."""
    A, B = ref.reshape(-1, 5), other.reshape(-1, 5)
    cost = np.linalg.norm(A[:, None, :3] - B[None, :, :3], axis=2) / scale
    _, col = linear_sum_assignment(cost)
    return B[col].ravel()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------------------
# Clearance lift of viewpoints and of the straight transit segments
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def _path_lift_one(x, Zc, res, h_safe, zhi, iters):
    N = x.shape[0] // 5
    nr, nc = Zc.shape
    for i in range(N):
        r, c = cell_of(x[5 * i], x[5 * i + 1], res, nr, nc)
        if x[5 * i + 2] < Zc[r, c] + h_safe:
            x[5 * i + 2] = min(Zc[r, c] + h_safe, zhi)
    for it in range(iters):
        pos = np.empty((N, 3))
        for i in range(N):
            pos[i, 0] = x[5 * i]
            pos[i, 1] = x[5 * i + 1]
            pos[i, 2] = x[5 * i + 2]
        tour = build_tour(pos)
        changed = False
        for k in range(N - 1):
            a = tour[k]
            b = tour[k + 1]
            vmax = segment_deficit(Zc, res, pos[a, 0], pos[a, 1], pos[a, 2], pos[b, 0], pos[b, 1], pos[b, 2],
                                   h_safe)
            if vmax > 0.0:
                x[5 * a + 2] = min(x[5 * a + 2] + vmax + 0.5, zhi)
                x[5 * b + 2] = min(x[5 * b + 2] + vmax + 0.5, zhi)
                changed = True
        if not changed:
            break
    return x


@njit(parallel=True, cache=True)
def path_lift(X, Zc, res, h_safe, zhi, iters):
    for k in prange(X.shape[0]):
        _path_lift_one(X[k], Zc, res, h_safe, zhi, iters)
    return X


def lift(scene, X, iters=3):
    X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
    return path_lift(X, scene.Zc, scene.res, C.MISSION["h_safe"], scene.zhi, iters)


# ----------------------------------------------------------------------------------------
# Candidate pool and greedy randomised construction
# ----------------------------------------------------------------------------------------
class CandidatePool:
    def __init__(self, scene, rng, n_clusters=300):
        cands = [normal_backprojection(scene, n_clusters, rng, d_range=(f, f)).reshape(-1, 5)
                 for f in (0.6, 0.8, 1.0)]
        cands.append(grid_nadir(scene, 144).reshape(-1, 5))
        for ang in (0.0, 0.25 * math.pi):
            cands.append(double_grid_oblique(scene, 128, angle=ang).reshape(-1, 5))
        self.cand = np.vstack(cands)
        V, QL, *_ = scene.visibility(self.cand.ravel())
        self.Qv = np.where(V, QL, 0.0)
        self.A = scene.A

    def construct(self, N, rng, rcl=5, k=3):
        top = np.zeros((k, self.Qv.shape[1]))
        chosen = []
        for _ in range(N):
            gain = (np.maximum(self.Qv - top[-1][None, :], 0.0) * self.A[None, :]).sum(1)
            gain[chosen] = -1.0
            best = np.argpartition(-gain, rcl)[:rcl]
            c = int(rng.choice(best))
            chosen.append(c)
            top = -np.sort(-np.vstack([top, self.Qv[c][None, :]]), axis=0)[:k]
        return self.cand[chosen].ravel()

# ----------------------------------------------------------------------------------------
# GG-MOEA variation
# ----------------------------------------------------------------------------------------
class GGVariation:
    """Builds offspring with DE/current-to-pbest/1 on correspondence-aligned donors and
    block-wise (whole viewpoint) or variable-wise binomial crossover."""

    def __init__(self, scene, N, use_align=True, p_best=0.1, cr_mean=0.9, block=True):
        self.scene, self.N = scene, N
        self.use_align, self.p_best, self.cr_mean, self.block = use_align, p_best, cr_mean, block
        self.last_step = 0.0
        self.lo, self.hi = scene.bounds(N)
        self.scale = max(scene.extent)

    def __call__(self, X, order, rng):
        n, N = len(X), self.N
        npb = max(2, int(round(self.p_best * n)))
        Y = np.empty_like(X)
        for i in range(n):
            pb = order[rng.integers(npb)]
            r1, r2 = rng.choice(n, 2, replace=False)
            xi = X[i]
            xp, x1, x2 = X[pb], X[r1], X[r2]
            if self.use_align:
                xp, x1, x2 = (align(xi, v, self.scale) for v in (xp, x1, x2))
            F = 0.0
            while F <= 0.0:
                F = 0.5 + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
            F = min(F, 1.0)
            CR = float(np.clip(rng.normal(self.cr_mean, 0.1), 0.0, 1.0))
            d1 = (xp - xi).reshape(N, 5)
            d2 = (x1 - x2).reshape(N, 5)
            d1[:, 4] = wrap(d1[:, 4]); d2[:, 4] = wrap(d2[:, 4])
            v = xi.reshape(N, 5) + F * d1 + F * d2
            if self.block:
                take = rng.random(N) < CR
                take[rng.integers(N)] = True
                u = np.where(take[:, None], v, xi.reshape(N, 5))
            else:
                take = rng.random((N, 5)) < CR
                take.flat[rng.integers(5 * N)] = True
                u = np.where(take, v, xi.reshape(N, 5))
            u[:, 4] = wrap(u[:, 4])
            Y[i] = u.ravel()
        Y = np.clip(Y, self.lo, self.hi)
        dpos = np.linalg.norm((Y - X).reshape(n, N, 5)[:, :, :3], axis=2)
        moved = dpos > 1e-9
        self.last_step = float(dpos[moved].mean()) if moved.any() else 0.0
        return Y
