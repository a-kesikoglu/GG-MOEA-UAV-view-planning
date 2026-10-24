"""Global configuration: sensor model, scene definitions and mission parameters."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

# Frame camera: 4/3-inch 20 MP sensor with a 12.29 mm lens (24 mm equivalent).
CAMERA = dict(
    f=12.29e-3,          # focal length [m]
    sw=17.3e-3,          # sensor width [m]
    sh=13.0e-3,          # sensor height [m]
    npx_w=5280,          # image width [px]
)
CAMERA["pix"] = CAMERA["sw"] / CAMERA["npx_w"]   # pixel pitch [m]

MISSION = dict(
    gsd_target=0.04,         # target ground sampling distance [m/px]
    d_min=20.0,              # minimum camera-to-surface range [m]
    d_max=250.0,             # maximum useful range [m]
    alpha_max=np.deg2rad(65.0),  # maximum incidence angle
    k_min=3,                 # views required for multi-view stereo
    beta_opt=np.deg2rad(25.0),   # preferred intersection angle
    h_safe=10.0,             # vertical clearance above buffered DSM [m]
    buffer=3.0,              # horizontal safety buffer [m]
    w_climb=2.0,             # extra cost per metre climbed
    w_turn=5.0,              # cost per radian of heading change [m/rad]
    z_ceiling=160.0,         # maximum flight height above highest surface point [m]
)
MISSION["d_opt"] = MISSION["gsd_target"] * CAMERA["f"] / CAMERA["pix"]

# Scene windows in Swiss LV95 (EPSG:2056) and processing resolution.
SCENES = {
    "urban": dict(name="Bern old town", tiles=["2600-1199"],
                  bounds=(2600300.0, 1199550.0, 2600600.0, 1199850.0), res=1.0),
    "cliff": dict(name="Lauterbrunnen valley wall", tiles=["2635-1160"],
                  bounds=(2635450.0, 1160300.0, 2635850.0, 1160700.0), res=2.0),
    "dam": dict(name="Contra arch dam", tiles=["2708-1116", "2708-1117", "2709-1116", "2709-1117"],
                bounds=(2708650.0, 1116800.0, 2709050.0, 1117200.0), res=2.0),
    # independent test scene, not used during algorithm design
    "gorge": dict(name="Fribourg old town and Sarine gorge",
                  tiles=["2024_2578-1183", "2024_2578-1184", "2024_2579-1183", "2024_2579-1184"],
                  bounds=(2578550.0, 1183500.0, 2578950.0, 1183900.0), res=2.0),
}
N_TARGETS = 1000
