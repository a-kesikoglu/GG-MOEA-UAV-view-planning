"""Camera geometry, prism-model visibility by grid traversal and mission evaluation.

All kernels operate on a 2.5D DSM stored as Z[row, col] with row 0 at the southern edge;
cell (r, c) covers x in [c*res, (c+1)*res) and y in [r*res, (r+1)*res).
"""
import math
import numpy as np
from numba import njit, prange

TWO_PI = 2.0 * math.pi


@njit(cache=True)
def cell_of(x, y, res, nr, nc):
    c = int(math.floor(x / res))
    r = int(math.floor(y / res))
    if c < 0:
        c = 0
    elif c >= nc:
        c = nc - 1
    if r < 0:
        r = 0
    elif r >= nr:
        r = nr - 1
    return r, c


# ----------------------------------------------------------------------------------------
# Traversal of the DSM prisms (Amanatides and Woo, 1987)
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def ray_visible(Z, res, px, py, pz, qx, qy, qz):
    """True if the segment p->q does not pass below the top of any DSM prism other than
    the prism that contains q."""
    nr, nc = Z.shape
    rq, cq = cell_of(qx, qy, res, nr, nc)
    r, c = cell_of(px, py, res, nr, nc)
    dx = qx - px
    dy = qy - py
    dz = qz - pz
    step_c = 1 if dx > 0 else -1
    step_r = 1 if dy > 0 else -1
    if abs(dx) > 1e-12:
        nxt = (c + (1 if dx > 0 else 0)) * res
        tmx = (nxt - px) / dx
        tdx = res / abs(dx)
    else:
        tmx = 1e30
        tdx = 1e30
    if abs(dy) > 1e-12:
        nxt = (r + (1 if dy > 0 else 0)) * res
        tmy = (nxt - py) / dy
        tdy = res / abs(dy)
    else:
        tmy = 1e30
        tdy = 1e30
    t0 = 0.0
    while True:
        t1 = min(tmx, tmy, 1.0)
        if not (r == rq and c == cq):
            zmin = pz + dz * min(t0, t1) if dz >= 0 else pz + dz * max(t0, t1)
            if zmin < Z[r, c]:
                return False
        else:
            return True
        if t1 >= 1.0:
            return True
        if tmx < tmy:
            c += step_c
            t0 = tmx
            tmx += tdx
        else:
            r += step_r
            t0 = tmy
            tmy += tdy
        if r < 0 or c < 0 or r >= nr or c >= nc:
            return True


# ----------------------------------------------------------------------------------------
# Camera model
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def camera_axes(om, ka):
    co = math.cos(om)
    d0 = co * math.cos(ka)
    d1 = co * math.sin(ka)
    d2 = math.sin(om)
    # right vector is horizontal (zero roll)
    r0 = math.sin(ka)
    r1 = -math.cos(ka)
    r2 = 0.0
    # up = right x d
    u0 = r1 * d2 - r2 * d1
    u1 = r2 * d0 - r0 * d2
    u2 = r0 * d1 - r1 * d0
    return d0, d1, d2, r0, r1, r2, u0, u1, u2


@njit(cache=True, inline="always")
def local_gsd(rx, ry, rz, zc, xc, yc, ax, n0, n1, n2, pf):
    """Coarsest sampling interval on the surface tangent plane (m/px). The 2x2 Jacobian of
    the central projection restricted to the tangent plane is evaluated and its smaller
    singular value gives the coarsest direction. pf = pixel pitch / focal length."""
    # tangent basis of the surface
    if abs(n2) < 0.9:
        t0, t1_, t2 = n1, -n0, 0.0          # n x z
    else:
        t0, t1_, t2 = 0.0, n2, -n1          # n x x
    tn = math.sqrt(t0 * t0 + t1_ * t1_ + t2 * t2)
    a0, a1, a2 = t0 / tn, t1_ / tn, t2 / tn
    b0 = n1 * a2 - n2 * a1
    b1 = n2 * a0 - n0 * a2
    b2 = n0 * a1 - n1 * a0
    iz2 = 1.0 / (zc * zc)
    # derivatives of the normalised image coordinates along a and b
    ad = a0 * ax[0] + a1 * ax[1] + a2 * ax[2]
    bd = b0 * ax[0] + b1 * ax[1] + b2 * ax[2]
    j11 = ((a0 * ax[3] + a1 * ax[4] + a2 * ax[5]) * zc - xc * ad) * iz2
    j12 = ((b0 * ax[3] + b1 * ax[4] + b2 * ax[5]) * zc - xc * bd) * iz2
    j21 = ((a0 * ax[6] + a1 * ax[7] + a2 * ax[8]) * zc - yc * ad) * iz2
    j22 = ((b0 * ax[6] + b1 * ax[7] + b2 * ax[8]) * zc - yc * bd) * iz2
    fr = j11 * j11 + j12 * j12 + j21 * j21 + j22 * j22
    det = abs(j11 * j22 - j12 * j21)
    disc = math.sqrt(max(fr * fr - 4.0 * det * det, 0.0))
    smin = math.sqrt(max(0.5 * (fr - disc), 1e-30))
    return pf / smin


@njit(cache=True, inline="always")
def view_quality_ax(px, py, pz, ax, j, q, nrm, Z, res, prm):
    """Validity and quality of the observation of target j from a camera at (px, py, pz) with
    precomputed axes ax = (d, r, u). The cheapest tests are applied first.
    Returns (valid, quality, D, cos_alpha, ux, uy, uz)."""
    rx = q[j, 0] - px
    ry = q[j, 1] - py
    rz = q[j, 2] - pz
    zc = rx * ax[0] + ry * ax[1] + rz * ax[2]
    if zc <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xc = rx * ax[3] + ry * ax[4] + rz * ax[5]
    if abs(xc) > prm[4] * zc:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    yc = rx * ax[6] + ry * ax[7] + rz * ax[8]
    if abs(yc) > prm[5] * zc:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    D = math.sqrt(rx * rx + ry * ry + rz * rz)
    if D < prm[1] or D > prm[2]:
        return False, 0.0, D, 0.0, 0.0, 0.0, 0.0, 0.0
    ux = rx / D
    uy = ry / D
    uz = rz / D
    ca = -(ux * nrm[j, 0] + uy * nrm[j, 1] + uz * nrm[j, 2])
    if ca < prm[3]:
        return False, 0.0, D, ca, ux, uy, uz, 0.0
    if not ray_visible(Z, res, px, py, pz, q[j, 0], q[j, 1], q[j, 2]):
        return False, 0.0, D, ca, ux, uy, uz, 0.0
    gsd = local_gsd(rx, ry, rz, zc, xc, yc, ax, nrm[j, 0], nrm[j, 1], nrm[j, 2], prm[6])
    ex = gsd / prm[0] - 1.0
    qual = math.exp(-ex) if ex > 0.0 else 1.0
    return True, qual, D, ca, ux, uy, uz, gsd


@njit(cache=True)
def axes_of(P):
    N = P.shape[0]
    AX = np.empty((N, 9))
    for i in range(N):
        d0, d1, d2, r0, r1, r2, u0, u1, u2 = camera_axes(P[i, 3], P[i, 4])
        AX[i, 0] = d0
        AX[i, 1] = d1
        AX[i, 2] = d2
        AX[i, 3] = r0
        AX[i, 4] = r1
        AX[i, 5] = r2
        AX[i, 6] = u0
        AX[i, 7] = u1
        AX[i, 8] = u2
    return AX


@njit(cache=True)
def view_quality(P, j, q, nrm, Z, res, prm):
    d0, d1, d2, r0, r1, r2, u0, u1, u2 = camera_axes(P[3], P[4])
    ax = np.array([d0, d1, d2, r0, r1, r2, u0, u1, u2])
    return view_quality_ax(P[0], P[1], P[2], ax, j, q, nrm, Z, res, prm)


# ----------------------------------------------------------------------------------------
# Tour construction: nearest neighbour + 2-opt on an open path
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def build_tour(pos):
    N = pos.shape[0]
    Dm = np.empty((N, N))
    for a in range(N):
        for b in range(N):
            Dm[a, b] = math.sqrt((pos[a, 0] - pos[b, 0]) ** 2 + (pos[a, 1] - pos[b, 1]) ** 2
                                 + (pos[a, 2] - pos[b, 2]) ** 2)
    start = 0
    best = 1e30
    for a in range(N):
        s = pos[a, 0] + pos[a, 1]
        if s < best:
            best = s
            start = a
    tour = np.empty(N, dtype=np.int64)
    used = np.zeros(N, dtype=np.bool_)
    tour[0] = start
    used[start] = True
    for k in range(1, N):
        last = tour[k - 1]
        bj = -1
        bd = 1e30
        for b in range(N):
            if not used[b] and Dm[last, b] < bd:
                bd = Dm[last, b]
                bj = b
        tour[k] = bj
        used[bj] = True
    improved = True
    it = 0
    while improved and it < 50:
        improved = False
        it += 1
        for i in range(0, N - 2):
            for k in range(i + 1, N):
                a = tour[i]
                b = tour[i + 1]
                c = tour[k]
                if k == N - 1:
                    delta = Dm[a, c] - Dm[a, b]
                else:
                    d = tour[k + 1]
                    delta = Dm[a, c] + Dm[b, d] - Dm[a, b] - Dm[c, d]
                if delta < -1e-9:
                    lo = i + 1
                    hi = k
                    while lo < hi:
                        t = tour[lo]
                        tour[lo] = tour[hi]
                        tour[hi] = t
                        lo += 1
                        hi -= 1
                    improved = True
        # reversal of a leading segment (open start)
        for k in range(1, N - 1):
            a = tour[0]
            b = tour[k]
            c = tour[k + 1]
            delta = Dm[a, c] - Dm[b, c]
            if delta < -1e-9:
                lo = 0
                hi = k
                while lo < hi:
                    t = tour[lo]
                    tour[lo] = tour[hi]
                    tour[hi] = t
                    lo += 1
                    hi -= 1
                improved = True
    return tour


@njit(cache=True)
def path_costs(pos, dirs, tour, w_climb, w_turn):
    """Energy-equivalent path length (min over both travel directions) and mean gimbal rotation."""
    N = tour.shape[0]
    L = 0.0
    up = 0.0
    down = 0.0
    turn = 0.0
    gim = 0.0
    for k in range(N - 1):
        a = tour[k]
        b = tour[k + 1]
        dx = pos[b, 0] - pos[a, 0]
        dy = pos[b, 1] - pos[a, 1]
        dz = pos[b, 2] - pos[a, 2]
        L += math.sqrt(dx * dx + dy * dy + dz * dz)
        if dz > 0:
            up += dz
        else:
            down -= dz
        c = dirs[a, 0] * dirs[b, 0] + dirs[a, 1] * dirs[b, 1] + dirs[a, 2] * dirs[b, 2]
        c = min(1.0, max(-1.0, c))
        gim += math.acos(c)
        if k >= 1:
            o = tour[k - 1]
            ex = pos[a, 0] - pos[o, 0]
            ey = pos[a, 1] - pos[o, 1]
            ez = pos[a, 2] - pos[o, 2]
            n1 = math.sqrt(ex * ex + ey * ey + ez * ez)
            n2 = math.sqrt(dx * dx + dy * dy + dz * dz)
            if n1 > 1e-9 and n2 > 1e-9:
                c = (ex * dx + ey * dy + ez * dz) / (n1 * n2)
                c = min(1.0, max(-1.0, c))
                turn += math.acos(c)
    f2 = L + w_climb * min(up, down) + w_turn * turn
    f4 = gim / (N - 1)
    return f2, f4, L


@njit(cache=True)
def segment_deficit(Zc, res, ax_, ay_, az_, bx_, by_, bz_, h_safe):
    """Largest clearance deficit of the straight segment a->b over the buffered surface.
    The cells crossed by the segment are visited in order (grid traversal) and the lower of
    the two segment heights at the entry and exit of each cell is compared with the cell."""
    nr, nc = Zc.shape
    r, c = cell_of(ax_, ay_, res, nr, nc)
    rb, cb = cell_of(bx_, by_, res, nr, nc)
    dx = bx_ - ax_
    dy = by_ - ay_
    dz = bz_ - az_
    step_c = 1 if dx > 0 else -1
    step_r = 1 if dy > 0 else -1
    if abs(dx) > 1e-12:
        tmx = ((c + (1 if dx > 0 else 0)) * res - ax_) / dx
        tdx = res / abs(dx)
    else:
        tmx = 1e30
        tdx = 1e30
    if abs(dy) > 1e-12:
        tmy = ((r + (1 if dy > 0 else 0)) * res - ay_) / dy
        tdy = res / abs(dy)
    else:
        tmy = 1e30
        tdy = 1e30
    t0 = 0.0
    vmax = 0.0
    for _ in range(nr + nc + 2):
        t1 = min(tmx, tmy, 1.0)
        zlow = az_ + dz * t0 if dz >= 0 else az_ + dz * t1
        v = Zc[r, c] + h_safe - zlow
        if v > vmax:
            vmax = v
        if t1 >= 1.0 or (r == rb and c == cb):
            break
        if tmx < tmy:
            c += step_c
            t0 = tmx
            tmx += tdx
        else:
            r += step_r
            t0 = tmy
            tmy += tdy
        if r < 0 or c < 0 or r >= nr or c >= nc:
            break
    return vmax


@njit(cache=True)
def clearance_violation(pos, tour, Zc, res, h_safe):
    nr, nc = Zc.shape
    N = pos.shape[0]
    cv = 0.0
    for i in range(N):
        r, c = cell_of(pos[i, 0], pos[i, 1], res, nr, nc)
        v = Zc[r, c] + h_safe - pos[i, 2]
        if v > 0:
            cv += v
    for k in range(N - 1):
        a = tour[k]
        b = tour[k + 1]
        cv += segment_deficit(Zc, res, pos[a, 0], pos[a, 1], pos[a, 2], pos[b, 0], pos[b, 1], pos[b, 2], h_safe)
    return cv


# ----------------------------------------------------------------------------------------
# Full mission evaluation
# ----------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def evaluate_one(x, q, nrm, area, Z, Zc, res, prm, Qout, used):
    N = x.shape[0] // 5
    M = q.shape[0]
    kmin = int(prm[7])
    beta_opt = prm[8]
    P = x.reshape(N, 5)
    pos = P[:, :3].copy()
    dirs = np.empty((N, 3))
    for i in range(N):
        co = math.cos(P[i, 3])
        dirs[i, 0] = co * math.cos(P[i, 4])
        dirs[i, 1] = co * math.sin(P[i, 4])
        dirs[i, 2] = math.sin(P[i, 3])
    U = np.empty((N, 3))
    AX = axes_of(P)
    top = np.zeros(kmin)
    atot = 0.0
    s1 = 0.0
    s3 = 0.0
    for j in range(M):
        cnt = 0
        for t in range(kmin):
            top[t] = 0.0
        for i in range(N):
            ok, qual, D, ca, ux, uy, uz, gsd = view_quality_ax(P[i, 0], P[i, 1], P[i, 2], AX[i], j, q, nrm, Z, res, prm)
            if ok:
                used[i] += 1
                U[cnt, 0] = ux
                U[cnt, 1] = uy
                U[cnt, 2] = uz
                cnt += 1
                # keep the kmin largest qualities
                m = 0
                for t in range(1, kmin):
                    if top[t] < top[m]:
                        m = t
                if qual > top[m]:
                    top[m] = qual
        Qj = 0.0
        for t in range(kmin):
            Qj += top[t]
        Qj /= kmin
        Qout[j] = Qj
        phi = 1.0
        if cnt >= 2:
            for a in range(cnt - 1):
                for b in range(a + 1, cnt):
                    c = U[a, 0] * U[b, 0] + U[a, 1] * U[b, 1] + U[a, 2] * U[b, 2]
                    c = min(1.0, max(-1.0, c))
                    v = abs(math.acos(c) - beta_opt) / beta_opt
                    if v < phi:
                        phi = v
        atot += area[j]
        s1 += area[j] * Qj
        s3 += area[j] * phi
    tour = build_tour(pos)
    f2, f4, L = path_costs(pos, dirs, tour, prm[10], prm[11])
    cv = clearance_violation(pos, tour, Zc, res, prm[9])
    return 1.0 - s1 / atot, f2, s3 / atot, f4, cv


@njit(parallel=True, cache=True)
def evaluate_population(X, q, nrm, area, Z, Zc, res, prm):
    n = X.shape[0]
    M = q.shape[0]
    F = np.empty((n, 4))
    CV = np.empty(n)
    Q = np.empty((n, M), dtype=np.float32)
    N = X.shape[1] // 5
    used = np.zeros((n, N), dtype=np.int32)
    for k in prange(n):
        Qk = np.empty(M)
        f1, f2, f3, f4, cv = evaluate_one(X[k], q, nrm, area, Z, Zc, res, prm, Qk, used[k])
        F[k, 0] = f1
        F[k, 1] = f2
        F[k, 2] = f3
        F[k, 3] = f4
        CV[k] = cv
        for j in range(M):
            Q[k, j] = Qk[j]
    return F, CV, Q, used


@njit(parallel=True, cache=True)
def visibility_matrix(x, q, nrm, Z, res, prm):
    """Per-pair validity, quality, range, incidence cosine and unit ray for one mission."""
    N = x.shape[0] // 5
    M = q.shape[0]
    P = x.reshape(N, 5)
    V = np.zeros((N, M), dtype=np.bool_)
    QL = np.zeros((N, M))
    DD = np.zeros((N, M))
    CA = np.zeros((N, M))
    UU = np.zeros((N, M, 3))
    GS = np.zeros((N, M))
    AX = axes_of(P)
    for i in prange(N):
        for j in range(M):
            ok, qual, D, ca, ux, uy, uz, gsd = view_quality_ax(P[i, 0], P[i, 1], P[i, 2], AX[i], j, q, nrm, Z, res, prm)
            V[i, j] = ok
            QL[i, j] = qual
            DD[i, j] = D
            CA[i, j] = ca
            UU[i, j, 0] = ux
            UU[i, j, 1] = uy
            UU[i, j, 2] = uz
            GS[i, j] = gsd
    return V, QL, DD, CA, UU, GS

