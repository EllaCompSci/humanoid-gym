"""Run the Gymnasium env with random actions (python examples/random_policy.py)."""
from openG1._display_env import fix_snap_env
fix_snap_env(reexec=True)   # must run before anything opens a window (VS Code snap fix)

import gymnasium as gym
import openG1  # noqa: F401  (registers the environments)

env = gym.make("OpenG1-Stand-v0", render_mode="human")
obs, info = env.reset(seed=0)
for _ in range(2000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
