"""Generate the Open G1 MJCF (servo arms + printed torso) from the G1 29-DoF model."""
# usage: python tools/make_mjcf.py path/to/unitree/g1_29dof_with_hand.xml
import json, numpy as np
MP=json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"mass_properties.json")))
# hobby servo specs (stall torque N.m, travel deg, armature, damping, frictionloss)
SERVO={"DS3235":dict(tau=3.43,travel=270,arm=0.005,damp=0.15,fric=0.03),   # DS3235-270 version
       "X10PRO":dict(tau=1.13,travel=160,arm=0.002,damp=0.06,fric=0.01),   # CHECK travel for your X10 Pro
       "MG92B": dict(tau=0.343,travel=160,arm=0.0006,damp=0.02,fric=0.005)}
ARM=[("shoulder_pitch","shoulder_pitch_link","DS3235",(0,1,0),(-3.0892,2.6704),(-3.0892,2.6704)),
     ("shoulder_roll","shoulder_roll_link","DS3235",(1,0,0),(-1.5882,2.2515),(-2.2515,1.5882)),
     ("shoulder_yaw","shoulder_yaw_link","X10PRO",(0,0,1),(-2.618,2.618),(-2.618,2.618)),
     ("elbow","elbow_link","X10PRO",(0,1,0),(-1.0472,2.0944),(-1.0472,2.0944)),
     ("wrist_roll","wrist_roll_link","MG92B",(1,0,0),(-1.972222054,1.972222054),(-1.972222054,1.972222054)),
     ("wrist_pitch","wrist_pitch_link","MG92B",(0,1,0),(-1.614429558,1.614429558),(-1.614429558,1.614429558)),
     ("wrist_yaw","wrist_yaw_link","MG92B",(0,0,1),(-1.614429558,1.614429558),(-1.614429558,1.614429558))]
POS_L={"shoulder_pitch":("0.0039563 0.10022 0.24778","0.990264 0.139201 1.38722e-05 -9.86868e-05"),
       "shoulder_roll":("0 0.038 -0.013831","0.990268 -0.139172 0 0"),"shoulder_yaw":("0 0.00624 -0.1032",None),
       "elbow":("0.015783 0 -0.080518",None),"wrist_roll":("0.1 0.00188791 -0.01",None),"wrist_pitch":("0.038 0 0",None),"wrist_yaw":("0.046 0 0",None)}
POS_R={"shoulder_pitch":("0.0039563 -0.10021 0.24778","0.990264 -0.139201 1.38722e-05 9.86868e-05"),
       "shoulder_roll":("0 -0.038 -0.013831","0.990268 0.139172 0 0"),"shoulder_yaw":("0 -0.00624 -0.1032",None),
       "elbow":("0.015783 0 -0.080518",None),"wrist_roll":("0.1 -0.00188791 -0.01",None),"wrist_pitch":("0.038 0 0",None),"wrist_yaw":("0.046 0 0",None)}
f=lambda v:" ".join("%.6g"%x for x in v)
def inertial(link,side):
    d=MP[link]; c=np.array(d["com"]); I=np.array(d["I"])
    if side=="right": c[1]*=-1; I[0,1]*=-1; I[1,0]*=-1; I[1,2]*=-1; I[2,1]*=-1
    return f'<inertial pos="{f(c)}" mass="{d["mass"]:.5f}" fullinertia="{f([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])}"/>'
def joint_range(lim,servo):
    lo,hi=lim; half=np.radians(SERVO[servo]["travel"])/2
    if hi-lo<=2*half: return lo,hi
    mid=(lo+hi)/2; return mid-half,mid+half     # servo centred on the middle of the G1 range
def arm(side):
    P=POS_L if side=="left" else POS_R; ind="        "; s=[]; close=[]; acts=[]
    for i,(jn,link,servo,axis,limL,limR) in enumerate(ARM):
        pos,quat=P[jn]; lo,hi=joint_range(limL if side=="left" else limR,servo); sp=SERVO[servo]
        q=f' quat="{quat}"' if quat else ""
        s.append(f'{ind}<body name="{side}_{link}" pos="{pos}"{q}>')
        s.append(f'{ind}  {inertial(link,side)}')
        s.append(f'{ind}  <joint name="{side}_{jn}_joint" axis="{f(axis)}" range="{lo:.4f} {hi:.4f}" armature="{sp["arm"]}" damping="{sp["damp"]}" frictionloss="{sp["fric"]}"/>')
        s.append(f'{ind}  <geom class="visual" mesh="{side}_{link}"/>')
        if jn in ("shoulder_pitch",): s.append(f'{ind}  <geom class="collision" type="cylinder" size="0.03 0.025" pos="0 {0.04 if side=="left" else -0.04} -0.01" quat="0.707107 0 0.707107 0"/>')
        elif jn=="shoulder_roll": s.append(f'{ind}  <geom class="collision" type="cylinder" size="0.03 0.015" pos="-0.004 {0.006 if side=="left" else -0.006} -0.053"/>')
        else: s.append(f'{ind}  <geom class="collision" mesh="{side}_{link}"/>')
        if jn=="wrist_yaw":
            s.append(f'{ind}  <geom class="visual" mesh="{side}_rubber_hand_open3d"/>')
            s.append(f'{ind}  <geom class="collision" mesh="{side}_rubber_hand_open3d"/>')
            s.append(f'{ind}  <site name="{side}_hand" pos="0.1 0 0" size="0.01" rgba="0 1 0 0.5"/>')
        close.append(f"{ind}</body>"); ind+="  "
        kp=sp["tau"]/0.1
        acts.append(f'    <position name="{side}_{jn}" joint="{side}_{jn}_joint" kp="{kp:.2f}" ctrlrange="{lo:.4f} {hi:.4f}" forcerange="-{sp["tau"]} {sp["tau"]}"/>   <!-- {servo} -->')
    return s+close[::-1], acts
xml=open("final/_template_head.xml").read() if False else None
LEGS=open(__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "g1_29dof_with_hand.xml").read()
legs_block=LEGS[LEGS.index('      <body name="left_hip_pitch_link"'):LEGS.index('      <body name="waist_yaw_link">')]
legs_block=legs_block.replace('<geom type="mesh" contype="0" conaffinity="0" group="1" density="0"','<geom class="visual"').replace('<geom type="mesh" rgba','<geom class="collision" type="mesh" rgba')
aL,actL=arm("left"); aR,actR=arm("right")
t=MP["torso_link"]; I=np.array(t["I"])
HM=json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"head_mass.json")))
def head(ind):
    """printed dome head (head.scad) as a fixed child of the torso, in Unitree's head_link frame"""
    Ih=np.array(HM["I"])
    L=[f'<body name="head_link" pos="0.0039635 0 -0.044">',
       f'  <inertial pos="{f(HM["com"])}" mass="{HM["mass"]:.4f}" fullinertia="{f([Ih[0,0],Ih[1,1],Ih[2,2],Ih[0,1],Ih[0,2],Ih[1,2]])}"/>   <!-- printed dome + D455 + 2x SG90 -->',
       f'  <geom class="visual" mesh="open_g1_dome_head" rgba="0.4 0.4 0.43 1"/>',
       f'  <geom class="collision" mesh="open_g1_dome_head"/>',
       f'  <camera name="head_d455" pos="0.095 0 0.402" xyaxes="0 -1 0 0 0 1" fovy="58"/>   <!-- D455 glass, looking along +x -->',
       f'</body>']
    return "\n".join(ind+l for l in L)
leg_meshes=["pelvis","pelvis_contour_link"]+[f"{s}_{l}" for s in ("left","right") for l in ("hip_pitch_link","hip_roll_link","hip_yaw_link","knee_link","ankle_pitch_link","ankle_roll_link")]+["waist_yaw_link","waist_roll_link"]
ours=["torso_link_rev_1_0","open_g1_dome_head"]+[f"{s}_{l}" for s in ("left","right") for l in ("shoulder_pitch_link","shoulder_roll_link","shoulder_yaw_link","elbow_link","wrist_roll_link","wrist_pitch_link","wrist_yaw_link","rubber_hand_open3d")]
X=f'''<mujoco model="open_g1">
  <!-- Open G1: 3D-printed Unitree-G1-shaped humanoid (github.com/open3dhumanoid/humanoid)
       * legs / pelvis / waist: Unitree G1 29-DoF kinematics, G1 inertias and torque limits
         -> TODO: replace with your Berkeley-style printed leg actuators (torque, armature, masses)
       * head: printed dome head (head.scad) with a RealSense D455; camera 'head_d455'
       * torso: torso_link_rev_1_0, printed (rev 1.0 shoulder height 0.24778 m), 2x DS3235 shoulder servos
       * arms: 7-DoF, hobby servos (DS3235 / KST X10 Pro / MG92B), printed-part masses and inertias
         computed from the CAD at 0.56 g/cm3 (PLA ~45 % solid) + servo masses
       * hand: rubber hand fixed to the wrist yaw link
       Servo joints use position actuators (hobby servos are position controlled); the joint range is the
       G1 range clipped to the servo travel, centred on the middle of the G1 range. -->
  <compiler angle="radian" meshdir="mesh" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1"/></default>
    <default class="collision"><geom type="mesh" group="3" rgba="0.7 0.7 0.7 1"/></default>
  </default>
  <asset>
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>   <!-- copy from unitree g1_description/meshes -->' for m in leg_meshes)}
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in ours)}
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="pelvis" pos="0 0 0.793">
      <inertial pos="0 0 -0.07605" quat="1 0 -0.000399148 0" mass="3.813" diaginertia="0.010549 0.0093089 0.0079184"/>
      <freejoint name="floating_base_joint"/>
      <geom class="visual" rgba="0.2 0.2 0.2 1" mesh="pelvis"/>
      <geom class="visual" mesh="pelvis_contour_link"/>
      <geom class="collision" mesh="pelvis_contour_link"/>
      <site name="imu_in_pelvis" size="0.01" pos="0.04525 0 -0.08339"/>
{legs_block}      <body name="waist_yaw_link">
        <inertial pos="0.003964 0 0.018769" quat="-0.0178291 0.628464 0.0282471 0.777121" mass="0.244" diaginertia="0.000158561 0.000124229 9.67669e-05"/>
        <joint name="waist_yaw_joint" axis="0 0 1" range="-2.618 2.618"/>
        <geom class="visual" mesh="waist_yaw_link"/>
        <body name="waist_roll_link" pos="-0.0039635 0 0.044">
          <inertial pos="0 -0.000236 0.010111" quat="0.99979 0.020492 0 0" mass="0.047" diaginertia="7.515e-06 6.40206e-06 3.98394e-06"/>
          <joint name="waist_roll_joint" axis="1 0 0" range="-0.52 0.52"/>
          <geom class="visual" mesh="waist_roll_link"/>
          <body name="torso_link">
            <inertial pos="{f(t["com"])}" mass="{t["mass"]:.4f}" fullinertia="{f([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])}"/>
            <joint name="waist_pitch_joint" axis="0 1 0" range="-0.52 0.52"/>
            <geom class="visual" mesh="torso_link_rev_1_0"/>
            <geom class="collision" mesh="torso_link_rev_1_0"/>
{head("            ")}
            <site name="imu_in_torso" size="0.01" pos="-0.03959 -0.00224 0.14792"/>
{chr(10).join("    "+l for l in aL)}
{chr(10).join("    "+l for l in aR)}
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- legs + waist: G1 torque limits (TODO: your leg actuators) -->
{chr(10).join(f'    <motor name="{j}" joint="{j}_joint" ctrlrange="-{tq} {tq}"/>' for j,tq in [(f"{s}_{n}",tq) for s in ("left","right") for n,tq in (("hip_pitch",88),("hip_roll",88),("hip_yaw",88),("knee",139),("ankle_pitch",50),("ankle_roll",50))]+[("waist_yaw",88),("waist_roll",50),("waist_pitch",50)])}
    <!-- arms: hobby servos as position actuators; kp saturates at ~6 deg error -->
{chr(10).join(actL)}
{chr(10).join(actR)}
  </actuator>
  <sensor>
    <gyro name="imu-torso-angular-velocity" site="imu_in_torso" noise="5e-4" cutoff="34.9"/>
    <accelerometer name="imu-torso-linear-acceleration" site="imu_in_torso" noise="1e-2" cutoff="157"/>
    <gyro name="imu-pelvis-angular-velocity" site="imu_in_pelvis" noise="5e-4" cutoff="34.9"/>
    <accelerometer name="imu-pelvis-linear-acceleration" site="imu_in_pelvis" noise="1e-2" cutoff="157"/>
  </sensor>
</mujoco>
'''
# the legs block from the G1 file declares joints with pos="0 0 0" ... keep; strip actuatorfrcrange (motor ctrlrange used)
import re
X=re.sub(r' actuatorfrcrange="[^"]*"','',X)
open(__import__("os").path.join(__import__("os").path.dirname(__file__),"..","src","openG1","assets","open_g1.xml"),"w").write(X)
print("written", len(X.splitlines()),"lines")

# ====================================================================== upper body (torso fixed)
TORSO_Z=0.793+0.044   # torso origin height when the full robot stands at its default pose
U=f'''<mujoco model="open_g1_upper">
  <!-- Open G1 upper body: printed torso (fixed to the world), head, and two 7-DoF hobby-servo arms.
       Same arm/torso parameters as open_g1.xml (generated by tools/make_mjcf.py). -->
  <compiler angle="radian" meshdir="mesh" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1"/></default>
    <default class="collision"><geom type="mesh" group="3" rgba="0.7 0.7 0.7 1"/></default>
  </default>
  <asset>
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in ours)}
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="0" conaffinity="0"/>
    <body name="torso_link" pos="0 0 {TORSO_Z}">
      <inertial pos="{f(t["com"])}" mass="{t["mass"]:.4f}" fullinertia="{f([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])}"/>
      <geom class="visual" mesh="torso_link_rev_1_0"/>
      <geom class="collision" mesh="torso_link_rev_1_0"/>
{head("      ")}
      <site name="imu_in_torso" size="0.01" pos="-0.03959 -0.00224 0.14792"/>
{chr(10).join(l[4:] for l in aL)}
{chr(10).join(l[4:] for l in aR)}
    </body>
    <!-- reach targets (moved by the environment) -->
    <body name="target_left" mocap="true" pos="0.3 0.2 1.0"><site name="target_left" type="sphere" size="0.02" rgba="0.1 0.8 0.1 0.6"/></body>
    <body name="target_right" mocap="true" pos="0.3 -0.2 1.0"><site name="target_right" type="sphere" size="0.02" rgba="0.8 0.1 0.1 0.6"/></body>
  </worldbody>
  <contact>
    <exclude body1="torso_link" body2="left_shoulder_pitch_link"/>
    <exclude body1="torso_link" body2="right_shoulder_pitch_link"/>
  </contact>
  <actuator>
{chr(10).join(actL)}
{chr(10).join(actR)}
  </actuator>
  <sensor>
    <gyro name="imu-torso-angular-velocity" site="imu_in_torso"/>
    <framepos name="left_hand_pos" objtype="site" objname="left_hand"/>
    <framepos name="right_hand_pos" objtype="site" objname="right_hand"/>
  </sensor>
</mujoco>
'''
open(__import__("os").path.join(__import__("os").path.dirname(__file__),"..","src","openG1","assets","open_g1_upper.xml"),"w").write(U)
print("written upper body", len(U.splitlines()), "lines")
