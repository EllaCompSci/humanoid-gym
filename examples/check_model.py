"""Load the Open G1 model, print a summary, and open the passive viewer (python examples/check_model.py)."""
import mujoco
import mujoco.viewer
from openG1 import MODEL_XML

model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)
print(f"bodies {model.nbody}  joints {model.njnt}  actuators {model.nu}  total mass {sum(model.body_mass):.2f} kg")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {name:28s} ctrlrange {model.actuator_ctrlrange[i]}  forcerange {model.actuator_forcerange[i]}")
mujoco.viewer.launch(model, data)
