"""Build the simulation assets for the printed Open G1 legs, pelvis, waist and framed torso.

Reads the finished hardware STLs (github.com/open3dhumanoid/humanoid, folder STL/) and writes
  * src/openG1/assets/mesh/og1_*.STL   - sim meshes in metres, each in its MuJoCo body frame,
                                         reduced (vertex clustering) to stay well under MuJoCo's
                                         200k-face limit
  * tools/legs_mass.json               - mass, centre of mass and inertia of every leg/pelvis/waist body
                                         (printed parts + 6512 actuators), plus where the actuator
                                         meshes sit on each body

usage:  python tools/legs/build_sim_legs.py /path/to/humanoid/STL

Mass model (edit the constants below if you weigh your parts):
  printed parts   0.56 g/cm3 effective (PLA, ~45 % solid; same as the arms and torso)
  6512 actuator   printed housing/output/motor-shell at the same density
                  + MAD M6C12 150KV motor 257 g + ACT_EXTRA_G (cycloid discs, bearings, pins,
                  encoder, B-G431B-ESC1 driver, screws) lumped at the actuator centre
Actuators that are not designed yet (hip yaw, two ankle actuators in the shin) are added as mass
only, at the positions given in PLANNED below, so the sim weighs what the robot will weigh.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

RHO = 0.56e-3            # g/mm^3
MOTOR_G = 257.0          # MAD M6C12 EEE 150KV (MAD data sheet)
ACT_EXTRA_G = 120.0      # estimate - weigh a finished actuator and put the difference here
FACE_TARGET = 60000      # sim mesh face budget per part (MuJoCo limit is 200000)

GYM = Path(__file__).resolve().parents[2]
OUT_MESH = GYM / "src" / "openG1" / "assets" / "mesh"
OUT_JSON = GYM / "tools" / "legs_mass.json"


# ---------------------------------------------------------------- STL io
def load_stl(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0] if len(b) >= 84 else 0
    if len(b) == 84 + 50 * n:
        a = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        return a["v"].astype(np.float64)
    v = [list(map(float, l.split()[1:4])) for l in b.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
    return np.array(v, dtype=np.float64).reshape(-1, 3, 3)


def save_stl(path, tri, name="og1"):
    tri = np.asarray(tri, dtype=np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" ") + struct.pack("<I", len(tri)) + rec.tobytes())


# ---------------------------------------------------------------- mesh reduction
def _cluster(tri, cell):
    v = tri.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    rep = np.stack([np.bincount(inv, weights=v[:, k]) for k in range(3)], 1) / cnt[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return rep[f[np.sort(keep)]]


def reduce(tri, target=FACE_TARGET):
    """vertex clustering; smallest cell (0.25 mm steps) that brings the mesh under the face target"""
    if len(tri) <= target:
        return tri
    cell = 0.5
    while True:
        out = _cluster(tri, cell)
        if len(out) <= target:
            return out
        cell += 0.25


# ---------------------------------------------------------------- mass properties (closed mesh)
def mass_props(tri, rho=RHO):
    """mass (g), com (mm), inertia about com (g mm^2) of a closed, outward-oriented triangle mesh"""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    d = np.einsum("ij,ij->i", a, np.cross(b, c))           # 6 x signed tet volume
    V = d.sum() / 6
    com = (d[:, None] * (a + b + c)).sum(0) / 24 / V
    # second moments  int x_i x_j dV  over each tet (origin, a, b, c)
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (d * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                            + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j]
                            + a[:, j] * c[:, i] + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum() / 120
    S = S * rho
    m = V * rho
    Io = np.trace(S) * np.eye(3) - S
    I = Io - m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    return m, com, I


def combine(items):
    """items: (m, com, I) in a common frame -> (m, com, I about com)"""
    m = sum(i[0] for i in items)
    c = sum(i[0] * i[1] for i in items) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in items:
        r = ci - c
        I += Ii + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return m, c, I


def moved(props, M):
    m, c, I = props
    R, p = M[:3, :3], M[:3, 3]
    return m, R @ c + p, R @ I @ R.T


def xf(tri, M):
    return tri @ M[:3, :3].T + M[:3, 3]


# ---------------------------------------------------------------- frames (from OpenScad/lib/legs_lib.scad)
def T(v):
    M = np.eye(4); M[:3, 3] = v; return M
def Rx(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4); M[1:3, 1:3] = [[c, -s], [s, c]]; return M
def Ry(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4); M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c; return M
MY = np.diag([1., -1., 1., 1.])
def side(M, s): return M if s > 0 else MY @ M @ MY

J_ROLL = T([0, 52, -30.465]) @ Ry(-0.1749)
J_YAW = T([25.001, 0, -124.12])
J_KNEE = T([-78.273, 2.1489, -177.34]) @ Ry(0.1749)
AZ_TO_X = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])
AZ_TO_Y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.]])
PEL_L = np.array([[1, 0, 0, 0], [0, 0, 1, 64.452], [0, -1, 0, -102.7], [0, 0, 0, 1.]])
PEL_R = np.array([[-1, 0, 0, 0], [0, 0, -1, -64.452], [0, -1, 0, -102.7], [0, 0, 0, 1.]])
ROLL_ACT = J_ROLL @ T([33.5014, 0, 0]) @ AZ_TO_X          # on the hip pitch link (left)
KNEE_ACT = J_KNEE @ T([0, 15.3733, 0]) @ AZ_TO_Y          # on the hip yaw link (left)
WAIST_ACT = T([0, 0, 2.2])                                # in the pelvis, output face up

# body -> actuator frames that are built (meshes shown) and planned (mass only), left side
BUILT = {"pelvis": [PEL_L, PEL_R, WAIST_ACT], "hip_pitch_link": [ROLL_ACT], "hip_roll_link": [],
         "hip_yaw_link": [KNEE_ACT], "knee_link": [], "waist_yaw_link": []}
PLANNED = {"hip_roll_link": [J_YAW @ Rx(np.pi)],                          # hip yaw, output down onto the yaw link
           "knee_link": [T([0, 0, -60]) @ AZ_TO_Y, T([0, 0, -125]) @ AZ_TO_Y]}   # two ankle drives in the shin

PARTS = {"pelvis": ["pelvis_left_half_waist", "pelvis_right_half_waist", "waist_stator_carrier"],
         "hip_pitch_link": ["{s}_hip_pitch_link_integrated"], "hip_roll_link": ["{s}_hip_roll_link_integrated"],
         "hip_yaw_link": ["{s}_hip_yaw_link_knee_integrated"], "knee_link": ["{s}_knee_link_integrated"],
         "waist_yaw_link": ["waist_yaw_link_integrated"]}
ACT_SHOWN = ["Housing_-_Bambu_-_Housing_Front", "Housing_-_Bambu_-_Housing_Middle", "Housing_-_Bambu_-_Housing_Rear",
             "Output_-_Bambu_-_Output_Shaft_Front", "Output_-_Bambu_-_Output_Shaft_Rear"]
ACT_PRINTED = ACT_SHOWN + ["Motor_Shell_-_Motor_Shell"]


def mat_to_posquat(M):
    R = M[:3, :3]
    w = np.sqrt(max(1e-12, 1 + np.trace(R))) / 2
    if w > 1e-3:
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    else:                                                    # 180 deg rotations
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return (M[:3, 3] / 1000).tolist(), (q / np.linalg.norm(q)).tolist()


def main(stl):
    stl = Path(stl)
    legs = stl / "legs"
    OUT_MESH.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---- the 6512 actuator: one shared sim mesh + its mass properties in the actuator frame
    act_tri = np.concatenate([load_stl(legs / "actuator" / f"{n}.stl") for n in ACT_SHOWN])
    shell = [mass_props(load_stl(legs / "actuator" / f"{n}.stl")) for n in ACT_PRINTED]
    printed = combine(shell)
    centre = np.array([0, 0, -16.0])                         # middle of the 32 mm deep actuator
    act = combine([printed, (MOTOR_G, centre, np.diag([1, 1, 1.6]) * MOTOR_G * 28.0 ** 2 / 2),   # motor ~ ring r=28 mm
                   (ACT_EXTRA_G, centre, np.zeros((3, 3)))])
    save_stl(OUT_MESH / "og1_act6512.STL", reduce(act_tri) / 1000)
    report["act6512"] = dict(mass_kg=act[0] / 1000, printed_kg=printed[0] / 1000)

    out = {"_note": "generated by tools/legs/build_sim_legs.py; kg, m, kg m^2, body frames (left side; "
                    "mirror y for the right)", "actuator_6512": {"mass": act[0] / 1000, "com": (act[1] / 1000).tolist()}}
    for body, parts in PARTS.items():
        sides = ["left"] if "{s}" in parts[0] else [None]
        for s in sides:
            items, meshes = [], []
            for p in parts:
                name = p.format(s=s)
                tri = load_stl(legs / f"{name}.stl")
                pr = mass_props(tri)
                if pr[0] <= 0:
                    raise SystemExit(f"{name}: mesh is inside-out or open (volume {pr[0]})")
                items.append(pr)
                meshes.append(name)
                for sd in (["left", "right"] if s else [None]):          # write both sides' sim meshes
                    nm = p.format(s=sd)
                    red = reduce(load_stl(legs / f"{nm}.stl"))
                    save_stl(OUT_MESH / f"og1_{nm}.STL", red / 1000)
                    report[nm] = len(red)
            printed_m = sum(i[0] for i in items)
            for M in BUILT[body] + PLANNED.get(body, []):
                items.append(moved(act, M))
            m, c, I = combine(items)
            out[body] = dict(mass=m / 1000, com=(c / 1000).tolist(), I=(I * 1e-9).tolist(),
                             printed_mass=printed_m / 1000, actuators=len(BUILT[body]),
                             planned_actuators=len(PLANNED.get(body, [])),
                             act_geoms=[mat_to_posquat(M) for M in BUILT[body]],
                             act_geoms_right=[mat_to_posquat(side(M, -1)) for M in BUILT[body]] if s else None)
    # the pelvis actuator at the right hip is given explicitly (PEL_R), not as a mirror
    out["pelvis"]["act_geoms"] = [mat_to_posquat(M) for M in (PEL_L, PEL_R, WAIST_ACT)]

    # ---- framed torso: one visual mesh of the printed panels + trays, torso frame
    tf = sorted((stl / "torso_frame").glob("*.stl"))
    if tf:
        panels = np.concatenate([reduce(load_stl(f), 18000) for f in tf])
        save_stl(OUT_MESH / "og1_torso_frame.STL", panels / 1000)
        report["torso_frame"] = len(panels)

    json.dump(out, open(OUT_JSON, "w"), indent=1)
    for k, v in report.items():
        print(f"  {k:40s} {v}")
    for k, v in out.items():
        if isinstance(v, dict) and "mass" in v:
            print(f"  {k:16s} {v['mass']:.3f} kg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../humanoid/STL")
