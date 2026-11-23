"""Independent checks of the local GSD and of the prism visibility test.

jac  local GSD of Eq. (5) against closed-form special cases and against a finite-difference
     Jacobian obtained by projecting displaced surface points (different computational path)
vis  grid traversal of the visibility test against a brute-force intersection of the segment
     with every prism of the DSM, on synthetic DSMs and on the four scenes
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.photogrammetry import camera_axes, local_gsd, ray_visible
from src.problem import Scene

TAB = os.path.join("results", "tables")
PF = C.CAMERA["pix"] / C.CAMERA["f"]


# ------------------------------------------------------------------------------------------
# Local GSD
# ------------------------------------------------------------------------------------------
def model_gsd(p, om, ka, q, n):
    d0, d1, d2, r0, r1, r2, u0, u1, u2 = camera_axes(om, ka)
    ax = np.array([d0, d1, d2, r0, r1, r2, u0, u1, u2])
    r = q - p
    zc, xc, yc = r @ ax[:3], r @ ax[3:6], r @ ax[6:]
    return local_gsd(r[0], r[1], r[2], zc, xc, yc, ax, n[0], n[1], n[2], PF)


def fd_gsd(p, om, ka, q, n, h=1e-4):
    """Projection of displaced points, Jacobian by central differences, SVD by LAPACK."""
    d0, d1, d2, r0, r1, r2, u0, u1, u2 = camera_axes(om, ka)
    d, e, u = np.array([d0, d1, d2]), np.array([r0, r1, r2]), np.array([u0, u1, u2])

    def proj(x):
        r = x - p
        return np.array([r @ e / (r @ d), r @ u / (r @ d)])
    # tangent basis by Gram-Schmidt from a random vector (independent of the model's choice)
    a = np.cross(n, [0.3, 0.7, 0.2]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    J = np.column_stack([(proj(q + h * t) - proj(q - h * t)) / (2 * h) for t in (a, b)])
    return PF / np.linalg.svd(J, compute_uv=False)[-1]


def look(p, target):
    r = target - p
    return math.asin(r[2] / np.linalg.norm(r)), math.atan2(r[1], r[0])


def gsd_cases():
    rows = []
    D = 150.0
    # (a) target on the optical axis, surface tilted by alpha about the image x axis
    for alpha in (0.0, 30.0, 60.0):
        p = np.array([0.0, 0.0, D])
        q = np.zeros(3)
        om, ka = -0.5 * math.pi, 0.0
        a = math.radians(alpha)
        n = np.array([math.sin(a), 0.0, math.cos(a)])
        exact = D * PF / math.cos(a)
        rows.append(("on axis, nadir", f"surface tilt {alpha:.0f}", exact, model_gsd(p, om, ka, q, n),
                     fd_gsd(p, om, ka, q, n)))
    # (b) plane parallel to the image at depth D, targets at the centre, edge and corner of the image
    hw, hh = 0.5 * C.CAMERA["sw"] / C.CAMERA["f"], 0.5 * C.CAMERA["sh"] / C.CAMERA["f"]
    for lab, (sx, sy) in (("image centre", (0, 0)), ("image edge", (0.95, 0)), ("image corner", (0.95, 0.95))):
        om, ka = math.radians(-30.0), math.radians(40.0)
        d0, d1, d2, r0, r1, r2, u0, u1, u2 = camera_axes(om, ka)
        d, e, u = np.array([d0, d1, d2]), np.array([r0, r1, r2]), np.array([u0, u1, u2])
        p = np.array([10.0, -5.0, 120.0])
        q = p + D * (d + sx * hw * e + sy * hh * u)
        n = -d
        rows.append(("plane parallel to image", lab, D * PF, model_gsd(p, om, ka, q, n), fd_gsd(p, om, ka, q, n)))
    # (c) oblique views of a horizontal plane and of a vertical wall at off-axis image positions
    rng = np.random.default_rng(3)
    for lab, n in (("horizontal plane", np.array([0.0, 0.0, 1.0])), ("vertical wall", np.array([0.0, -1.0, 0.0]))):
        for k in range(3):
            om, ka = math.radians(rng.uniform(-80, -20)), math.radians(rng.uniform(-150, -30))
            d0, d1, d2, r0, r1, r2, u0, u1, u2 = camera_axes(om, ka)
            d, e, u = np.array([d0, d1, d2]), np.array([r0, r1, r2]), np.array([u0, u1, u2])
            p = np.array([0.0, 0.0, 0.0])
            sx, sy = rng.uniform(-0.9, 0.9, 2)
            ray = d + sx * hw * e + sy * hh * u
            q = p + rng.uniform(60, 200) * ray / np.linalg.norm(ray)
            if ray @ n > 0:
                n = -n
            rr = q - p
            ax_dist = np.linalg.norm(rr)
            cos_a = -(rr / ax_dist) @ n
            approx = ax_dist * PF / cos_a
            rows.append((lab + ", oblique camera", f"x={sx:+.2f}, y={sy:+.2f}", np.nan, model_gsd(p, om, ka, q, n),
                         fd_gsd(p, om, ka, q, n), approx))
    out = []
    for r in rows:
        case, pos, exact, mod, fd = r[:5]
        ref = exact if not np.isnan(exact) else fd
        out.append(dict(case=case, position=pos, closed_form_cm=100 * exact, finite_diff_cm=100 * fd,
                        model_cm=100 * mod, rel_err_model=(mod - ref) / ref,
                        on_axis_cm=100 * (r[5] if len(r) > 5 else np.nan)))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(TAB, "validation_gsd.csv"), index=False)
    print(df.to_string())
    return df


# ------------------------------------------------------------------------------------------
# Visibility
# ------------------------------------------------------------------------------------------
@njit(cache=True)
def brute_visible(Z, res, px, py, pz, qx, qy, qz):
    """Segment p->q against every prism [c*res,(c+1)*res] x [r*res,(r+1)*res] x [-inf, Z] except
    the prism that contains q. Liang-Barsky clipping of the planimetric segment to each cell."""
    nr, nc = Z.shape
    rq = min(max(int(math.floor(qy / res)), 0), nr - 1)
    cq = min(max(int(math.floor(qx / res)), 0), nc - 1)
    dx, dy, dz = qx - px, qy - py, qz - pz
    c0 = max(int(math.floor(min(px, qx) / res)) - 1, 0)
    c1 = min(int(math.floor(max(px, qx) / res)) + 1, nc - 1)
    r0 = max(int(math.floor(min(py, qy) / res)) - 1, 0)
    r1 = min(int(math.floor(max(py, qy) / res)) + 1, nr - 1)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r == rq and c == cq:
                continue
            t0, t1 = 0.0, 1.0
            ok = True
            for pv, qv in ((-dx, px - c * res), (dx, (c + 1) * res - px), (-dy, py - r * res), (dy, (r + 1) * res - py)):
                if abs(pv) < 1e-15:
                    if qv < 0:
                        ok = False
                        break
                else:
                    t = qv / pv
                    if pv < 0:
                        if t > t0:
                            t0 = t
                    else:
                        if t < t1:
                            t1 = t
            if not ok or t0 > t1:
                continue
            zlow = min(pz + dz * t0, pz + dz * t1)
            if zlow < Z[r, c]:
                return False
    return True


@njit(parallel=True, cache=True)
def compare(Z, res, P, Q):
    n = P.shape[0]
    a = np.zeros(n, dtype=np.bool_)
    b = np.zeros(n, dtype=np.bool_)
    for k in prange(n):
        a[k] = ray_visible(Z, res, P[k, 0], P[k, 1], P[k, 2], Q[k, 0], Q[k, 1], Q[k, 2])
        b[k] = brute_visible(Z, res, P[k, 0], P[k, 1], P[k, 2], Q[k, 0], Q[k, 1], Q[k, 2])
    return a, b


def synthetic(kind, rng, n=60, res=1.0):
    Z = np.zeros((n, n))
    if kind == "blocks":
        for _ in range(25):
            r, c = rng.integers(0, n - 8, 2)
            h, w = rng.integers(2, 9, 2)
            Z[r:r + h, c:c + w] = rng.uniform(5, 40)
    elif kind == "cliff":
        yy, xx = np.mgrid[0:n, 0:n]
        Z = np.where(xx + 0.3 * yy > 0.6 * n, 60.0, 0.0) + rng.normal(0, 1.0, (n, n))
    elif kind == "noise":
        Z = rng.uniform(0, 30, (n, n))
    return Z, res


def random_pairs(Z, res, rng, k, mode):
    nr, nc = Z.shape
    ex, ey = nc * res, nr * res
    qx, qy = rng.uniform(0, ex, k), rng.uniform(0, ey, k)
    cq = np.clip((qx / res).astype(int), 0, nc - 1)
    rq = np.clip((qy / res).astype(int), 0, nr - 1)
    qz = Z[rq, cq] + rng.uniform(0.0, 2.0, k)
    px, py = rng.uniform(0, ex, k), rng.uniform(0, ey, k)
    pz = Z.max() + rng.uniform(-0.8, 1.0, k) * (Z.max() - Z.min() + 10)
    if mode == "axis":            # segments parallel to the grid axes
        h = rng.random(k) < 0.5
        px = np.where(h, px, qx)
        py = np.where(h, qy, py)
    elif mode == "diagonal":      # 45 degree segments through cell corners
        qx = np.floor(qx / res) * res + 0.5 * res
        qy = np.floor(qy / res) * res + 0.5 * res
        L = rng.integers(3, 25, k) * res * np.sign(rng.normal(size=k))
        px, py = np.clip(qx + L, 0, ex - 1e-9), np.clip(qy + L, 0, ey - 1e-9)
        cq = np.clip((qx / res).astype(int), 0, nc - 1)
        rq = np.clip((qy / res).astype(int), 0, nr - 1)
        qz = Z[rq, cq] + 0.5
    elif mode == "vertical":
        px, py = qx.copy(), qy.copy()
    P = np.column_stack([px, py, pz])
    Q = np.column_stack([qx, qy, qz])
    return P, Q


def vis_checks():
    rng = np.random.default_rng(7)
    rows = []
    for kind in ("blocks", "cliff", "noise"):
        Z, res = synthetic(kind, rng)
        for mode in ("random", "axis", "diagonal", "vertical"):
            P, Q = random_pairs(Z, res, rng, 50000, mode)
            a, b = compare(Z, res, P, Q)
            rows.append(dict(dsm=f"synthetic {kind}", segments=mode, n=len(a), visible=int(b.sum()),
                             agree=int((a == b).sum()), traversal_only_visible=int((a & ~b).sum()),
                             brute_only_visible=int((~a & b).sum())))
            print(rows[-1], flush=True)
    for key in ("urban", "cliff", "dam", "gorge"):
        s = Scene(key)
        nr, nc = s.Z.shape
        k = 40000
        j = rng.integers(0, s.M, k)
        Q = s.q[j]
        ang = rng.uniform(0, 2 * math.pi, k)
        dist = rng.uniform(20, 250, k)
        P = np.column_stack([np.clip(Q[:, 0] + dist * np.cos(ang), 0, nc * s.res - 1e-6),
                             np.clip(Q[:, 1] + dist * np.sin(ang), 0, nr * s.res - 1e-6),
                             Q[:, 2] + rng.uniform(-50, 200, k)])
        a, b = compare(s.Z, s.res, np.ascontiguousarray(P), np.ascontiguousarray(Q))
        rows.append(dict(dsm=f"scene {key}", segments="targets", n=k, visible=int(b.sum()), agree=int((a == b).sum()),
                         traversal_only_visible=int((a & ~b).sum()), brute_only_visible=int((~a & b).sum())))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "validation_visibility.csv"), index=False)
    return df


if __name__ == "__main__":
    parts = sys.argv[1:] or ["jac", "vis"]
    if "jac" in parts:
        gsd_cases()
    if "vis" in parts:
        vis_checks()
