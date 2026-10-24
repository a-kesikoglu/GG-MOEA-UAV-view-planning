"""Fig. 1: study locations and the four DSM windows with their surface targets."""
import json
import os
import sys

import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.plotstyle import setup, save, plt
from src.problem import Scene

setup()
SC = [("urban", "(b) Bern old town"), ("cliff", "(c) Lauterbrunnen valley wall"),
      ("dam", "(d) Contra arch dam"), ("gorge", "(e) Fribourg, Sarine gorge")]
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.42, wspace=0.5)

# (a) locator map
ax = fig.add_subplot(gs[0, 0])
geo = json.load(open(os.path.join(C.RAW, "naturalearth", "switzerland.geojson")))
ring = np.array(geo["geometry"]["coordinates"][0])
tr = Transformer.from_crs(4326, 2056, always_xy=True)
ex, ny = tr.transform(ring[:, 0], ring[:, 1])
ax.fill(np.array(ex) / 1000, np.array(ny) / 1000, color="#eeeeee", ec="#555555", lw=0.6)
for key, lab in SC:
    b = C.SCENES[key]["bounds"]
    xc, yc = 0.5 * (b[0] + b[2]) / 1000, 0.5 * (b[1] + b[3]) / 1000
    ax.plot(xc, yc, "o", ms=4, color="#D55E00", mec="black", mew=0.4)
    off = {"urban": (4, 4), "cliff": (5, 2), "dam": (5, 2), "gorge": (-10, -9)}[key]
    ax.annotate(lab[1], (xc, yc), xytext=off, textcoords="offset points", fontsize=7)
ax.set_aspect("equal")
ax.set_xlabel("E (km, LV95)")
ax.set_ylabel("N (km, LV95)")
ax.set_title("(a) Study sites")
ax.tick_params(labelsize=6)

pos = [gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
for g, (key, title) in zip(pos, SC):
    ax = fig.add_subplot(g)
    s = Scene(key)
    Z = s.Z + s.z0
    ls = LightSource(azdeg=315, altdeg=40)
    norm = Normalize(Z.min(), Z.max())
    rgb = ls.shade(Z, cmap=plt.cm.cividis, norm=norm, vert_exag=2.0, blend_mode="overlay", dx=s.res, dy=s.res)
    ax.imshow(rgb, origin="lower", extent=[0, s.extent[0], 0, s.extent[1]], interpolation="bilinear")
    steep = np.abs(s.n[:, 2]) < 0.5
    ax.scatter(s.q[~steep, 0], s.q[~steep, 1], s=0.9, c="#56B4E9", edgecolors="none", label="gently sloping")
    ax.scatter(s.q[steep, 0], s.q[steep, 1], s=0.9, c="#D55E00", edgecolors="none", label="steep or vertical")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (m)", fontsize=7)
    cb = fig.colorbar(ScalarMappable(norm, plt.cm.cividis), ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("H (m, LN02)", fontsize=6)
    if key in ("urban", "dam"):
        ax.set_ylabel("y (m)", fontsize=7)
ax_leg = fig.add_subplot(gs[1, 2])
ax_leg.axis("off")
h1 = ax_leg.scatter([], [], s=12, c="#56B4E9", label="gently sloping facet ($|\\,n_z|\\geq0.5$)")
h2 = ax_leg.scatter([], [], s=12, c="#D55E00", label="steep or vertical facet ($|\\,n_z|<0.5$)")
ax_leg.legend(handles=[h1, h2], loc="center left", frameon=False, fontsize=7)
save(fig, os.path.join(C.RES, "figures", "fig1_sites"))
