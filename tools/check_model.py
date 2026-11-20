"""Static checks of the generated MJCF without MuJoCo: forward kinematics at q = 0, foot height,
total mass, and convex-hull overlaps between collision geoms that MuJoCo would test.

usage: python tools/check_model.py [src/openG1/assets/open_g1.xml]
"""
import itertools
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent / "legs"))
from build_sim_legs import load_stl  # noqa: E402


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame(el):
    M = np.eye(4)
    if el.get("pos"): M[:3, 3] = list(map(float, el.get("pos").split()))
    if el.get("quat"): M[:3, :3] = qmat(np.array(list(map(float, el.get("quat").split()))))
    return M


def walk(xml):
    root = ET.parse(xml).getroot()
    mdir = Path(xml).parent / root.find("compiler").get("meshdir", ".")
    files = {m.get("name"): mdir / m.get("file") for m in root.iter("mesh")}
    bodies, geoms = {}, []

    def rec(b, Mp, parent):
        M = Mp @ frame(b)
        name = b.get("name")
        bodies[name] = dict(M=M, parent=parent, mass=float(b.find("inertial").get("mass")) if b.find("inertial") is not None else 0)
        for g in b.findall("geom"):
            geoms.append(dict(body=name, el=g, M=M @ frame(g)))
        for c in b.findall("body"):
            rec(c, M, name)

    for b in root.find("worldbody").findall("body"):
        rec(b, np.eye(4), "world")
    excl = {frozenset((e.get("body1"), e.get("body2"))) for e in root.iter("exclude")}
    return root, files, bodies, geoms, excl


def main(xml):
    root, files, bodies, geoms, excl = walk(xml)
    print(f"total mass {sum(b['mass'] for b in bodies.values() if b['mass']):.2f} kg, {len(bodies)} bodies")
    # foot contact spheres
    feet = [g for g in geoms if g["el"].get("size") and g["el"].get("type") in (None, "sphere") and "ankle_roll" in g["body"]]
    if feet:
        z = [g["M"][2, 3] - float(g["el"].get("size").split()[0]) for g in feet]
        print(f"foot spheres lowest/highest point: {min(z)*1000:.1f} / {max(z)*1000:.1f} mm above the floor")
    hulls = {}
    for g in geoms:
        el = g["el"]
        is_col = el.get("class") == "collision" or (el.get("class") is None and el.get("contype") != "0")
        if not is_col or not el.get("mesh"):
            continue
        v = load_stl(files[el.get("mesh")]).reshape(-1, 3)
        v = v[ConvexHull(v).vertices]
        hulls.setdefault(g["body"], []).append((el.get("mesh"), v @ g["M"][:3, :3].T + g["M"][:3, 3]))
    allv = np.concatenate([v for hs in hulls.values() for _, v in hs])
    print(f"lowest collision-mesh point {allv[:, 2].min()*1000:.1f} mm")
    bad = 0
    for a, b in itertools.combinations(hulls, 2):
        if bodies[a]["parent"] == b or bodies[b]["parent"] == a or frozenset((a, b)) in excl:
            continue
        for (ma, va), (mb, vb) in itertools.product(hulls[a], hulls[b]):
            ins = (Delaunay(vb).find_simplex(va) >= 0).sum() + (Delaunay(va).find_simplex(vb) >= 0).sum()
            if ins:
                bad += 1
                print(f"  hull overlap at q=0: {a} ({ma}) x {b} ({mb})  [{ins} hull vertices inside]")
    print("no collision-hull overlaps at q=0" if not bad else f"{bad} overlapping pairs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "src/openG1/assets/open_g1.xml"))
