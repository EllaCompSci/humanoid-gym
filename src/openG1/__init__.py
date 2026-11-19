"""Open G1 - Gymnasium/MuJoCo environments for the 3D-printed Open G1 humanoid."""
from ._display_env import fix_snap_env

fix_snap_env()   # snap apps (VS Code snap) leak GTK settings that crash the MuJoCo viewer

from gymnasium.envs.registration import register

from .meshes import ASSET_DIR, MESH_DIR, check_meshes, missing_meshes

MODEL_XML = str(ASSET_DIR / "open_g1.xml")           # full robot (floating base, legs, waist, torso, arms, head)
UPPER_XML = str(ASSET_DIR / "open_g1_upper.xml")     # torso fixed, head, two 7-DoF arms

register(
    id="OpenG1-Stand-v0",
    entry_point="openG1.envs.open_g1_env:OpenG1StandEnv",
    max_episode_steps=1000,
)
register(
    id="OpenG1UpperBody-Reach-v0",
    entry_point="openG1.envs.upper_body_env:OpenG1UpperBodyReachEnv",
    max_episode_steps=500,
)

__all__ = ["fix_snap_env", "ASSET_DIR", "MESH_DIR", "MODEL_XML", "UPPER_XML", "check_meshes", "missing_meshes"]
