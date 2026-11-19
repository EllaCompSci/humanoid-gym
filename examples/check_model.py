"""Load an Open G1 model, print a summary, and open the passive viewer.

python examples/check_model.py          # full robot
python examples/check_model.py upper    # upper body only
"""
from openG1._display_env import fix_snap_env
fix_snap_env(reexec=True)   # must run before anything opens a window (VS Code snap fix)

import sys

import mujoco
import mujoco.viewer
from openG1 import MODEL_XML, UPPER_XML, check_meshes

xml = UPPER_XML if (len(sys.argv) > 1 and sys.argv[1] == "upper") else MODEL_XML
check_meshes(xml)
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
print(f"{xml}\nbodies {model.nbody}  joints {model.njnt}  actuators {model.nu}  total mass {sum(model.body_mass):.2f} kg")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {name:28s} ctrlrange {model.actuator_ctrlrange[i]}  forcerange {model.actuator_forcerange[i]}")
mujoco.viewer.launch(model, data)
