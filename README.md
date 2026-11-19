# humanoid-gym

Gymnasium + MuJoCo environments for the **Open G1**: a 3D-printed humanoid with the shape and
kinematics of the Unitree G1, built on printed leg actuators and hobby-servo arms.
Hardware lives in [open3dhumanoid/humanoid](https://github.com/open3dhumanoid/humanoid).

There are two models:
- `open_g1.xml`: the full robot, with floating base, legs, waist, torso, head and arms.
- `open_g1_upper.xml`: the **upper body only**. The torso is fixed to the world and carries the head and both 7-DoF arms.

The full model is the G1 29-DoF MJCF, changed to match the printed robot:

| Part | Change from the G1 model |
|---|---|
| Torso | `torso_link_rev_1_0` shape (rev 1.0, shoulders at z = 0.24778 m). Mass and inertia are for the **framed torso**: 2020 aluminium frame, printed panels and shoulder blocks, 2 shoulder servos, fasteners (2.03 kg, no electronics). |
| Arms | 7-DoF on hobby servos. Masses and inertias are computed from the printed CAD plus servo masses. The servo joints use **position actuators** with servo torque limits and armature, damping and friction values. |
| Joint ranges | The G1 range clipped to the servo's travel, centred on the middle of the G1 range. Fit each servo horn at that centre angle. |
| Head | The printed dome head (`head.scad`) with a RealSense D455, as `open_g1_dome_head.STL`. It is a fixed `head_link` body with its own mass (0.61 kg: printed parts, D455 and 2 brow servos) and a `head_d455` camera at the camera glass. |
| Hand | The rubber hand, fixed to the wrist yaw link (`*_rubber_hand_open3d.STL`). |
| Legs / pelvis / waist | Still the G1 values (kinematics, inertias, torque limits). **TODO:** replace with your measured printed leg actuators. |

## Install

```bash
git clone https://github.com/open3dhumanoid/humanoid-gym
cd humanoid-gym
pip install -e .
```

### Meshes you need to add
The arm, torso, head and hand meshes are included in `src/openG1/assets/mesh/`, so the
**upper-body model needs no downloads**. The full model also needs 16 leg, pelvis and waist
meshes, which are Unitree's own. Fetch them once:

```bash
python -m openG1.meshes                       # downloads from Unitree's GitHub
python -m openG1.meshes --from /path/to/g1_description/meshes   # or copy from a local folder
```

These are the files it needs:

```
pelvis.STL  pelvis_contour_link.STL  waist_yaw_link.STL  waist_roll_link.STL
{left,right}_hip_pitch_link.STL  {left,right}_hip_roll_link.STL  {left,right}_hip_yaw_link.STL
{left,right}_knee_link.STL  {left,right}_ankle_pitch_link.STL  {left,right}_ankle_roll_link.STL
```

### Head camera
Both models have a `head_d455` camera at the D455 glass, looking forward (58° vertical field of view):

```python
import mujoco
r = mujoco.Renderer(env.unwrapped.model, 480, 848)
r.update_scene(env.unwrapped.data, camera="head_d455"); rgb = r.render()
```

## Quick start

```python
import gymnasium as gym
import openG1                      # registers the environments

env = gym.make("OpenG1-Stand-v0", render_mode="human")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
```

To check a model loads and see it in the MuJoCo viewer:

```bash
python examples/check_model.py          # full robot
python examples/check_model.py upper    # upper body only
```

To run the upper body with random actions:

```bash
python examples/upper_body_reach.py
```

## Environments

| ID | Task | Action | Observation |
|---|---|---|---|
| `OpenG1-Stand-v0` | Stay upright | 29 values in [-1, 1], scaled to each actuator's `ctrlrange` (torque for legs and waist, target angle for arm servos) | `qpos[2:]` + `qvel` |
| `OpenG1UpperBody-Reach-v0` | Upper body only: touch a random target with each hand (green = left, red = right) | 14 servo targets in [-1, 1] | `qpos`, `qvel`, hand positions, target positions, target − hand |

**Stand:** the reward is a healthy bonus, plus how upright the torso is, minus a control cost. An
episode ends when the pelvis height leaves 0.45–1.2 m.

**Reach:**
- **Reward:** minus the sum of both hand-to-target distances, minus small control and action-rate costs.
- **Success:** `info["is_success"]` is true when both hands are within 3 cm of their targets.
- **Targets:** each is sampled in front of its shoulder, at x 0.18–0.40 m and z 0.85–1.25 m.
- **Episodes:** 500 steps.

Both environments run control at 50 Hz (timestep 0.002 s × frame skip 10).

## Actuators

| Joint(s) | Actuator | Torque limit | Travel used |
|---|---|---|---|
| Shoulder pitch, shoulder roll | DS3235 (270° version) | 3.43 N·m | 270° |
| Shoulder yaw, elbow | KST X10 Pro | 1.13 N·m | 160° (check your servo) |
| Wrist roll, pitch, yaw | TowerPro MG92B | 0.343 N·m | 160° |
| Hips, knees, ankles, waist | G1 values (placeholder) | 50–139 N·m | G1 |

Hobby servos run their own position loop, so they are modelled as `<position>` actuators.
`kp` is set so the servo reaches full torque at about 6° of error. The armature, damping and
friction values are estimates for geared hobby servos; tune them against your hardware.

## Layout

```
src/openG1/
  __init__.py               # registers OpenG1-Stand-v0, exposes MODEL_XML
  envs/open_g1_env.py       # full robot: OpenG1-Stand-v0
  envs/upper_body_env.py    # upper body: OpenG1UpperBody-Reach-v0
  meshes.py                 # check / fetch the Unitree meshes (python -m openG1.meshes)
  assets/open_g1.xml        # full robot MJCF
  assets/open_g1_upper.xml  # upper-body MJCF
  assets/mesh/              # STL meshes (arm/torso/head/hand included; legs from Unitree)
tools/
  make_mjcf.py              # regenerates open_g1.xml
  mass_properties.json      # printed-link masses / COMs / inertias
  head_mass.json            # dome head mass / COM / inertia
  head/export_head.py       # head.scad -> open_g1_dome_head.STL + head_mass.json
examples/
  check_model.py            # load + summary + viewer
  random_policy.py          # random actions, full robot
  upper_body_reach.py       # random actions, upper-body reach
```

## Regenerating the model
`open_g1.xml` and `open_g1_upper.xml` are both generated by `tools/make_mjcf.py`. It reads `tools/mass_properties.json`, which
was computed from the hardware CAD, and takes the legs from Unitree's `g1_29dof_with_hand.xml`:

```bash
python tools/make_mjcf.py path/to/g1_29dof_with_hand.xml
```

Edit the `SERVO` table in the script to change the torque, travel or armature values.
Regenerate after you change servos, print density or parts.

The head mesh comes from `head.scad`. After you edit the head, copy the changed parameters into the
top of `tools/head/head_sdf.py` and run:

```bash
python tools/head/export_head.py --stock path/to/unitree/head_link.STL
python tools/make_mjcf.py path/to/g1_29dof_with_hand.xml
```

Mass assumptions:
- **Print density:** PLA at about 45% solid, i.e. 0.56 g/cm³.
- **Servos:** DS3235 60 g, X10 Pro 25 g, MG92B 13.8 g.
- **Head:** D455 116 g (datasheet), SG90 9 g each.
- **Torso:** the framed torso is about 2.03 kg (1.05 kg printed at 1.1 g/cm³ for its 3–4 mm walls, 0.61 kg of 2020, servos, fasteners). Add your compute board, drivers and wiring to its inertial in `tools/mass_properties.json`.

## Troubleshooting
**`GLib-GIO-ERROR ... does not contain a key named 'antialiasing'`, then `Aborted (core dumped)`**
when the viewer opens: VS Code installed as a snap leaks its own GTK/GLib settings into
everything it launches. `import openG1` cleans these variables, and the examples restart
themselves with a clean environment. In your own scripts, put this at the very top:

```python
from openG1._display_env import fix_snap_env
fix_snap_env(reexec=True)
```

Or run from a normal terminal, or install VS Code from the `.deb` instead of the snap.

## Credits
Kinematics and meshes are derived from the Unitree G1 description
([unitree_ros](https://github.com/unitreerobotics/unitree_ros)). Check Unitree's licence before
you redistribute their meshes.
