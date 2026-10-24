"""Download swissSURFACE3D Raster tiles, crop the scene windows, and build surface targets
for each benchmark scene."""
import glob
import os
import sys

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swisssurface3d-raster/items"


def download_tiles(bounds, tiles):
    os.makedirs(C.RAW, exist_ok=True)
    have = [f for t in tiles for f in glob.glob(os.path.join(C.RAW, f"*_{t}_0.5_*.tif"))]
    if len(have) >= len(tiles):
        return have
    from pyproj import Transformer
    tr = Transformer.from_crs(2056, 4326, always_xy=True)
    lon0, lat0 = tr.transform(bounds[0], bounds[1])
    lon1, lat1 = tr.transform(bounds[2], bounds[3])
    items = requests.get(STAC, params={"bbox": f"{lon0},{lat0},{lon1},{lat1}", "limit": 100},
                         timeout=120).json()["features"]
    out = []
    for it in items:
        if not any(t in it["id"] for t in tiles):
            continue
        href = [a["href"] for k, a in it["assets"].items() if k.endswith("_0.5_2056_5728.tif")][0]
        path = os.path.join(C.RAW, href.split("/")[-1])
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(requests.get(href, timeout=600).content)
        out.append(path)
    return out


def load_dsm(sc):
    files = download_tiles(sc["bounds"], sc["tiles"])
    srcs = [rasterio.open(f) for f in files]
    z, _ = merge(srcs, bounds=sc["bounds"], nodata=-9999.0)
    z = z[0].astype(np.float64)
    nodata = z < -1000
    if nodata.any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(nodata, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    f = int(round(sc["res"] / 0.5))
    h, w = z.shape[0] // f, z.shape[1] // f
    z = z[:h * f, :w * f].reshape(h, f, w, f).max(axis=(1, 3))   # block maximum
    z = z[::-1, :].copy()                                           # row 0 = south
    return z, float(np.mean(nodata))


def build_targets(Z, res, m_target, seed=0):
    """Surface facets on prism tops (gradient normals) and on vertical steps, clustered into
    about m_target area-weighted targets."""
    nr, nc = Z.shape
    thr = res * np.tan(np.deg2rad(70.0))
    pts, nrm, area = [], [], []
    # vertical facets on step edges between 4-neighbours
    for axis in (0, 1):
        if axis == 1:
            za, zb = Z[:, :-1], Z[:, 1:]
        else:
            za, zb = Z[:-1, :], Z[1:, :]
        dz = zb - za
        rr, cc = np.nonzero(np.abs(dz) > thr)
        for r, c in zip(rr, cc):
            hi, lo = (zb[r, c], za[r, c]) if dz[r, c] > 0 else (za[r, c], zb[r, c])
            sgn = -1.0 if dz[r, c] > 0 else 1.0       # normal points to the lower prism
            if axis == 1:
                xe, ye = (c + 1) * res, (r + 0.5) * res
                nv = (sgn, 0.0, 0.0)
            else:
                xe, ye = (c + 0.5) * res, (r + 1) * res
                nv = (0.0, sgn, 0.0)
            nseg = max(1, int(np.ceil((hi - lo) / 3.0)))
            hseg = (hi - lo) / nseg
            for s in range(nseg):
                pts.append((xe, ye, lo + (s + 0.5) * hseg))
                nrm.append(nv)
                area.append(res * hseg)
    # prism tops with normals from one-sided differences that ignore step edges
    gx = np.zeros_like(Z)
    gy = np.zeros_like(Z)
    for g, axis in ((gx, 1), (gy, 0)):
        fwd = np.diff(Z, axis=axis, append=np.nan) if axis == 1 else np.diff(Z, axis=0, append=np.nan)
        bwd = np.diff(Z, axis=axis, prepend=np.nan)
        fwd[np.abs(fwd) > thr] = np.nan
        bwd[np.abs(bwd) > thr] = np.nan
        g[:] = np.nan_to_num(np.nanmean(np.stack([fwd, bwd]), axis=0), nan=0.0) / res
    rr, cc = np.mgrid[0:nr, 0:nc]
    nn = np.stack([-gx, -gy, np.ones_like(Z)], axis=-1)
    nl = np.linalg.norm(nn, axis=-1, keepdims=True)
    nn /= nl
    P = np.stack([(cc + 0.5) * res, (rr + 0.5) * res, Z], axis=-1).reshape(-1, 3)
    pts = np.vstack([np.array(pts).reshape(-1, 3), P])
    nrm = np.vstack([np.array(nrm).reshape(-1, 3), nn.reshape(-1, 3)])
    area = np.concatenate([np.array(area), (res * res * nl[..., 0]).ravel()])
    n_vert = int(np.sum(np.abs(nrm[:, 2]) < 1e-9))
    # normal classes: up-facing or one of four horizontal directions
    hor = np.abs(nrm[:, 2]) < 0.5
    az = np.round(np.arctan2(nrm[:, 1], nrm[:, 0]) / (np.pi / 2)).astype(int) % 4
    cls = np.where(hor, 1 + az, 0)
    lo, hi = 2.0, 60.0
    for _ in range(40):
        v = 0.5 * (lo + hi)
        key = np.column_stack([np.floor(pts / v).astype(np.int64), cls])
        _, inv = np.unique(key, axis=0, return_inverse=True)
        k = inv.max() + 1
        if k > m_target:
            lo = v
        else:
            hi = v
        if abs(k - m_target) <= 0.01 * m_target:
            break
    inv = inv.ravel()
    A = np.bincount(inv, weights=area)
    Q = np.column_stack([np.bincount(inv, weights=area * pts[:, d]) for d in range(3)]) / A[:, None]
    Nn = np.column_stack([np.bincount(inv, weights=area * nrm[:, d]) for d in range(3)])
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    # snap each cluster centre to the nearest member facet so that it lies on the surface
    best = np.full(k, np.inf)
    snap = np.zeros((k, 3))
    d2 = np.sum((pts - Q[inv]) ** 2, axis=1)
    order = np.argsort(d2)
    seen = np.zeros(k, bool)
    for idx in order:
        g = inv[idx]
        if not seen[g]:
            seen[g] = True
            snap[g] = pts[idx]
    return snap, Nn, A, cls, n_vert, v


def process_scene(key, out_dir=C.PROC):
    sc = C.SCENES[key]
    Z, nodata_frac = load_dsm(sc)
    res = sc["res"]
    q, n, A, _, n_vert, vox = build_targets(Z, res, C.N_TARGETS)
    q_off = q + 0.5 * n                     # small offset along the normal
    rb = int(np.ceil(C.MISSION["buffer"] / res))
    Zc = maximum_filter(Z, size=2 * rb + 1, mode="nearest")
    zmin = Z.min()
    Zl, Zcl = Z - zmin, Zc - zmin
    q_off[:, 2] -= zmin
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{key}.npz"), Z=Zl, Zc=Zcl, res=res, q=q_off, n=n, A=A,
                        z0=zmin, origin=np.array(sc["bounds"][:2]), nodata=nodata_frac, voxel=vox)
    vert = np.abs(n[:, 2]) < 0.5
    print(f"{key}: grid {Z.shape} res {res} m, elev {Z.min():.1f}-{Z.max():.1f} m, relief {Z.max()-Z.min():.1f} m, "
          f"targets {len(q)} (steep/vertical {vert.sum()}), area {A.sum():.0f} m2, "
          f"steep-area share {A[vert].sum()/A.sum():.3f}, voxel {vox:.2f} m, nodata {nodata_frac:.4f}")


def process_dense(key, m_target, offset, tag, out_dir=C.PROC):
    """Independent, denser target set for checking the representation (not used in optimization)."""
    sc = C.SCENES[key]
    Z, _ = load_dsm(sc)
    res = sc["res"]
    q, n, A, _, _, vox = build_targets(Z, res, m_target)
    q_off = q + offset * n
    rb = int(np.ceil(C.MISSION["buffer"] / res))
    Zc = maximum_filter(Z, size=2 * rb + 1, mode="nearest")
    zmin = Z.min()
    q_off[:, 2] -= zmin
    np.savez_compressed(os.path.join(out_dir, f"{key}_{tag}.npz"), Z=Z - zmin, Zc=Zc - zmin, res=res, q=q_off, n=n,
                        A=A, z0=zmin, origin=np.array(sc["bounds"][:2]), nodata=0.0, voxel=vox)
    print(f"{key}_{tag}: {len(q)} targets, voxel {vox:.2f} m")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dense":
        for k in C.SCENES:
            process_dense(k, 5000, 0.5, "d5k")
            process_dense(k, 5000, 0.1, "d5k01")
    else:
        for k in (sys.argv[1:] or C.SCENES.keys()):
            process_scene(k)
