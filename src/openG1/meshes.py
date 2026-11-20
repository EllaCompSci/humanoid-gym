"""Check the model meshes, and re-fetch the Unitree G1 foot meshes (ankle links) if they are missing.

All meshes ship in assets/mesh; the printed og1_* parts are rebuilt with tools/legs/build_sim_legs.py."""
from __future__ import annotations

import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MESH_DIR = ASSET_DIR / "mesh"
MODEL_XML = ASSET_DIR / "open_g1.xml"
UPPER_XML = ASSET_DIR / "open_g1_upper.xml"
ALL_XMLS = [MODEL_XML, UPPER_XML]

# Unitree G1 mesh folders (tried in order). media.githubusercontent.com serves git-LFS files.
SOURCES = [
    "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/meshes/",
    "https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/main/unitree_robots/g1/meshes/",
    "https://media.githubusercontent.com/media/unitreerobotics/unitree_ros/master/robots/g1_description/meshes/",
    "https://media.githubusercontent.com/media/unitreerobotics/unitree_mujoco/main/unitree_robots/g1/meshes/",
]


def required_meshes(xml=None) -> list[str]:
    files = []
    for x in ([Path(xml)] if xml else ALL_XMLS):
        files += [m.get("file") for m in ET.parse(x).getroot().iter("mesh")]
    return sorted(set(files))


def missing_meshes(xml=None) -> list[str]:
    return [f for f in required_meshes(xml) if not (MESH_DIR / f).is_file()]


def check_meshes(xml=None) -> None:
    """Raise a clear error if any mesh the model(s) need is missing."""
    missing = missing_meshes(xml)
    if missing:
        own = [f for f in missing if f.startswith("og1_")]
        if own:
            raise FileNotFoundError(
                f"printed-part mesh(es) missing from {MESH_DIR}: {', '.join(own)}\n"
                "Rebuild them with:   python tools/legs/build_sim_legs.py /path/to/humanoid/STL")
        raise FileNotFoundError(
            f"{len(missing)} Unitree G1 mesh file(s) missing from {MESH_DIR}:\n  "
            + "\n  ".join(missing)
            + "\nFetch them with:   python -m openG1.meshes"
            + "\nor copy them from a local g1_description/meshes folder:   python -m openG1.meshes --from /path/to/meshes"
        )


def _looks_like_stl(data: bytes) -> bool:
    if data.startswith(b"version https://git-lfs"):
        return False                         # LFS pointer, not the mesh
    return len(data) > 84


def fetch(from_dir: str | None = None) -> int:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    missing = missing_meshes()
    if not missing:
        print("All meshes present.")
        return 0
    failed = []
    for name in missing:
        dest = MESH_DIR / name
        if from_dir:
            src = Path(from_dir) / name
            if src.is_file():
                shutil.copy(src, dest)
                print(f"copied   {name}")
                continue
            failed.append(name)
            continue
        for base in SOURCES:
            try:
                with urllib.request.urlopen(base + name, timeout=30) as r:
                    data = r.read()
                if _looks_like_stl(data):
                    dest.write_bytes(data)
                    print(f"fetched  {name}  ({len(data) // 1024} KB)")
                    break
            except Exception:
                continue
        else:
            failed.append(name)
    if failed:
        print("\nCould not get:\n  " + "\n  ".join(failed))
        print("Download unitree_ros (robots/g1_description/meshes) and run:  python -m openG1.meshes --from <that folder>")
        return 1
    print("\nAll meshes present.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    src = args[args.index("--from") + 1] if "--from" in args else None
    sys.exit(fetch(src))
