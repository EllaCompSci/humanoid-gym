"""Export the printed dome head (head.scad) as one simulator mesh + mass properties.

    python tools/head/export_head.py --stock path/to/unitree/head_link.STL [--res 1.5]

head_sdf.py mirrors every module of head.scad as a distance field (the parameters at the
top of head_sdf.py are copied from head.scad: keep them in step if you edit the SCAD).
Writes  src/openG1/assets/mesh/open_g1_dome_head.STL  (metres, Unitree head_link frame,
frame + 5 panels + brows + a D455 block) and  tools/head_mass.json , then re-run
tools/make_mjcf.py to update the models.
"""
import argparse, json, os, sys
import numpy as np
from skimage import measure
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from head_sdf import *            # noqa
from voxmesh import write_stl

ap = argparse.ArgumentParser()
ap.add_argument("--stock", required=True, help="Unitree head_link.STL (the collar below CUT_Z is kept)")
ap.add_argument("--res", type=float, default=2.0, help="mesh sampling, mm (coarsened until under MuJoCo's 200k-face STL limit)")
ap.add_argument("--max-faces", type=int, default=195000)
ap.add_argument("--mass-res", type=float, default=1.0)
ap.add_argument("--density", type=float, default=560.0, help="printed parts, kg/m3 (0.56 g/cm3 like the arms)")
a = ap.parse_args()
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

def fields(res):
    g = Grid((-132, -132, 320), (97, 132, 488), res)
    col = collar_sdf(g, a.stock)
    fs = {"printed": lambda p, c: U(frame(p, c), *[panel(p, i) for i in range(N_PANELS)], brow(p, 1), brow(p, -1)),
          "d455": lambda p, c: d455(p)}
    return g, evaluate(g, fs, col)

# ---- mesh (MuJoCo refuses STL files with more than 200k faces)
res = a.res
while True:
    g, F = fields(res)
    u = np.minimum(F["printed"], F["d455"])
    v, f, _, _ = measure.marching_cubes(np.pad(u, 1, constant_values=50), 0.0, spacing=(g.res,)*3)
    if len(f) <= a.max_faces: break
    print(f"{res:.2f} mm -> {len(f)} faces, too many; coarsening"); res = round(res*np.sqrt(len(f)/a.max_faces)*1.02, 2)
v = v - g.res + g.lo
T = v[f]; flip = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() < 0
out = os.path.join(ROOT, "src", "openG1", "assets", "mesh", "open_g1_dome_head.STL")
write_stl(out, v/1000.0, f, flip=flip); print("wrote", out, len(f), f"triangles at {res} mm")

# ---- mass properties: printed parts + D455 (116 g, datasheet) + 2x SG90 (9 g)
g, F = fields(a.mass_res)
xs, ys, zs = g.axes(); X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij"); dv = (g.res/1000)**3
items = [(F["printed"] < 0, a.density*dv)]
cam = F["d455"] < 0; items.append((cam, 0.116/cam.sum()))
for m in (1, -1):
    sv = (X >= SV_BODY_X[0]) & (X <= SV_BODY_X[1]) & (m*Y >= SV_Y0) & (m*Y <= SV_Y0 + SV_H) & (np.abs(Z - SV_SHAFT[1]) <= SV_W/2)
    items.append((sv, 0.009/sv.sum()))
P = np.stack([X, Y, Z], -1)/1000; M = sum(w*k.sum() for k, w in items)
com = sum(w*P[k].sum(0) for k, w in items)/M; I = np.zeros((3, 3))
for k, w in items:
    d = P[k] - com; I += w*(np.eye(3)*(d*d).sum() - d.T@d)
json.dump(dict(mass=M, com=com.tolist(), I=I.tolist(), printed_mass=items[0][0].sum()*items[0][1], d455=0.116, sg90=0.018,
               note="head_link frame (m). printed parts + D455 116 g + 2x SG90"),
          open(os.path.join(ROOT, "tools", "head_mass.json"), "w"), indent=1)
print(f"head mass {M:.3f} kg, com {np.round(com, 4)}  -> tools/head_mass.json ; now run tools/make_mjcf.py")
