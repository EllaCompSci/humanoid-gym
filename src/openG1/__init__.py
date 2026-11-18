"""Open G1 - Gymnasium/MuJoCo environments for the 3D-printed Open G1 humanoid."""
from pathlib import Path
from gymnasium.envs.registration import register

ASSET_DIR = Path(__file__).parent / "assets"
MODEL_XML = str(ASSET_DIR / "open_g1.xml")

register(
    id="OpenG1-Stand-v0",
    entry_point="openG1.envs.open_g1_env:OpenG1StandEnv",
    max_episode_steps=1000,
)

__all__ = ["ASSET_DIR", "MODEL_XML"]
