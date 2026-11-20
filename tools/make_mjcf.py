"""Generate the Open G1 MJCFs (printed legs + waist + framed torso + head + servo arms) on G1 29-DoF kinematics."""
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
# ---------------------------------------------------------------- printed legs, pelvis and waist
# Kinematics (body positions, joint axes and ranges) come from the G1 file; everything printed
# (meshes, masses, inertias) comes from tools/legs_mass.json + og1_*.STL, made by
# tools/legs/build_sim_legs.py from the hardware repo's STL/legs and STL/torso_frame.
import xml.etree.ElementTree as ET
G1=ET.parse(__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "g1_29dof_with_hand.xml").getroot()
G1B={b.get("name"):b for b in G1.iter("body")}
LM=json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"legs_mass.json")))
# 6512 cycloidal actuator (Berkeley Humanoid Lite design: M6C12 150KV, 15:1 printed cycloidal)
#   tau   peak joint torque = motor peak 1.71 N.m (MAD data) x 15 x ~0.9 efficiency
#   arm   reflected rotor inertia ~ 9e-5 kg m2 x 15^2  (estimate - outer-rotor ring, 28 mm radius)
#   damp, fric  printed-gearbox losses (estimates - identify on your actuator and edit)
LEG_ACT=dict(tau=23.1,arm=0.02,damp=0.1,fric=0.3)
LEG_JOINTS=("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")
OUR_LEG={"hip_pitch_link":"hip_pitch_link_integrated","hip_roll_link":"hip_roll_link_integrated",
         "hip_yaw_link":"hip_yaw_link_knee_integrated","knee_link":"knee_link_integrated"}
LEG_RGBA={"hip_pitch_link":"0.83 0.71 0.2 1","hip_roll_link":"0.44 0.68 0.42 1","hip_yaw_link":"0.78 0.77 0.83 1",
          "knee_link":"0.36 0.5 0.79 1"}
ACT_RGBA="0.88 0.48 0.22 1"
def inertial_lm(d,side="left"):
    c=np.array(d["com"]); I=np.array(d["I"])
    if side=="right": c[1]*=-1; I[0,1]*=-1; I[1,0]*=-1; I[1,2]*=-1; I[2,1]*=-1
    return f'<inertial pos="{f(c)}" mass="{d["mass"]:.4f}" fullinertia="{f([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])}"/>'
def act_geoms(geoms,ind):
    return [f'{ind}<geom class="visual" mesh="og1_act6512" pos="{f(p)}" quat="{f(q)}" rgba="{ACT_RGBA}"/>' for p,q in geoms]
def leg(side):
    s=[]; close=[]; ind="      "
    for jn in LEG_JOINTS:
        link=f"{jn}_link"; b=G1B[f"{side}_{link}"]; j=b.find("joint")
        quat=f' quat="{b.get("quat")}"' if b.get("quat") else ""
        s.append(f'{ind}<body name="{side}_{link}" pos="{b.get("pos")}"{quat}>')
        if link in OUR_LEG:
            d=LM[link]; s.append(f'{ind}  {inertial_lm(d,side)}   <!-- printed {d["printed_mass"]:.3f} kg + {d["actuators"]} built + {d["planned_actuators"]} planned 6512 -->')
        else:
            s.append(f'{ind}  {ET.tostring(b.find("inertial"),encoding="unicode").strip()}   <!-- G1 foot (not printed yet) -->')
        s.append(f'{ind}  <joint name="{side}_{jn}_joint" axis="{j.get("axis")}" range="{j.get("range")}" armature="{LEG_ACT["arm"]}" damping="{LEG_ACT["damp"]}" frictionloss="{LEG_ACT["fric"]}"/>')
        if link in OUR_LEG:
            m=f"og1_{side}_{OUR_LEG[link]}"
            s.append(f'{ind}  <geom class="visual" mesh="{m}" rgba="{LEG_RGBA[link]}"/>')
            if link!="hip_roll_link": s.append(f'{ind}  <geom class="collision" mesh="{m}"/>')
            s+=act_geoms(LM[link]["act_geoms" if side=="left" else "act_geoms_right"] or [],ind+"  ")
        else:
            s.append(f'{ind}  <geom class="visual" mesh="{side}_{link}" rgba="0.2 0.2 0.2 1"/>')
            if jn=="ankle_roll":
                for g in b.findall("geom"):
                    if g.get("type") is None and g.get("size"):      # G1 foot contact spheres
                        s.append(f'{ind}  <geom size="{g.get("size")}" pos="{g.get("pos")}" rgba="0.2 0.2 0.2 1"/>')
        close.append(f"{ind}</body>"); ind+="  "
    return "\n".join(s+close[::-1])
legs_block=leg("left")+"\n"+leg("right")+"\n"
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
leg_meshes=[f"{s}_{l}" for s in ("left","right") for l in ("ankle_pitch_link","ankle_roll_link")]   # Unitree G1 feet
og1=["og1_act6512","og1_pelvis_left_half_waist","og1_pelvis_right_half_waist","og1_waist_stator_carrier",
     "og1_waist_yaw_link_integrated","og1_torso_frame"]+[f"og1_{s}_{p}" for s in ("left","right") for p in OUR_LEG.values()]
ours=["torso_link_rev_1_0","open_g1_dome_head"]+[f"{s}_{l}" for s in ("left","right") for l in ("shoulder_pitch_link","shoulder_roll_link","shoulder_yaw_link","elbow_link","wrist_roll_link","wrist_pitch_link","wrist_yaw_link","rubber_hand_open3d")]
EXCL=[("pelvis","waist_yaw_link"),("pelvis","torso_link")]+[p for sd in ("left","right") for p in [
    ("pelvis",f"{sd}_hip_roll_link"),("pelvis",f"{sd}_hip_yaw_link"),(f"{sd}_hip_pitch_link",f"{sd}_hip_yaw_link"),
    (f"{sd}_hip_roll_link",f"{sd}_knee_link"),(f"{sd}_hip_yaw_link",f"{sd}_ankle_pitch_link"),
    (f"{sd}_knee_link",f"{sd}_ankle_roll_link")]]
CONTACT_EXCLUDES=chr(10).join(f'    <exclude body1="{a}" body2="{b}"/>' for a,b in EXCL)
X=f'''<mujoco model="open_g1">
  <!-- Open G1: 3D-printed Unitree-G1-shaped humanoid (github.com/open3dhumanoid/humanoid)
       * legs / pelvis / waist: printed parts (STL/legs) on G1 kinematics, 6512 cycloidal actuators
         (Berkeley Humanoid Lite design) on every leg joint and the waist yaw, 23-DoF-style yaw-only waist.
         Hip yaw and ankle actuators are not designed yet: they are in the masses as 6512s, the feet
         are the Unitree G1 ankle links. Masses from tools/legs/build_sim_legs.py.
       * head: printed dome head (head.scad) with a RealSense D455; camera 'head_d455'
       * torso: 2020-aluminium frame with printed panels (STL/torso_frame), G1 rev 1.0 shape; collision uses
         the G1 rev 1.0 shell (same outside); 2x DS3235 shoulder servos
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
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>   <!-- Unitree g1_description (BSD-3) -->' for m in leg_meshes)}
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in og1)}
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in ours)}
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="pelvis" pos="0 0 0.793">
      {inertial_lm(LM["pelvis"])}   <!-- printed halves + stator carrier + 3x 6512 -->
      <freejoint name="floating_base_joint"/>
      <geom class="visual" mesh="og1_pelvis_left_half_waist" rgba="0.72 0.72 0.72 1"/>
      <geom class="visual" mesh="og1_pelvis_right_half_waist" rgba="0.56 0.61 0.69 1"/>
      <geom class="visual" mesh="og1_waist_stator_carrier" rgba="0.88 0.75 0.38 1"/>
      <geom class="collision" mesh="og1_pelvis_left_half_waist"/>
      <geom class="collision" mesh="og1_pelvis_right_half_waist"/>
{chr(10).join(act_geoms(LM["pelvis"]["act_geoms"],"      "))}
      <site name="imu_in_pelvis" size="0.01" pos="0.04525 0 -0.08339"/>
{legs_block}      <body name="waist_yaw_link">
        {inertial_lm(LM["waist_yaw_link"])}
        <joint name="waist_yaw_joint" axis="0 0 1" range="-2.618 2.618" armature="{LEG_ACT["arm"]}" damping="{LEG_ACT["damp"]}" frictionloss="{LEG_ACT["fric"]}"/>
        <geom class="visual" mesh="og1_waist_yaw_link_integrated" rgba="0.62 0.71 0.78 1"/>
        <body name="torso_link" pos="-0.0039635 0 0.044">
            <inertial pos="{f(t["com"])}" mass="{t["mass"]:.4f}" fullinertia="{f([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])}"/>
            <geom class="visual" mesh="og1_torso_frame" rgba="0.6 0.6 0.63 1"/>
            <geom class="collision" mesh="torso_link_rev_1_0"/>
{head("            ")}
            <site name="imu_in_torso" size="0.01" pos="-0.03959 -0.00224 0.14792"/>
{chr(10).join("    "+l for l in aL)}
{chr(10).join("    "+l for l in aR)}
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <!-- printed parts nest closely: skip the convex-hull contacts between neighbouring links -->
{CONTACT_EXCLUDES}
  </contact>
  <actuator>
    <!-- legs + waist yaw: 6512 cycloidal actuators, torque control, peak {LEG_ACT["tau"]} N.m -->
{chr(10).join(f'    <motor name="{j}" joint="{j}_joint" ctrlrange="-{LEG_ACT["tau"]} {LEG_ACT["tau"]}"/>' for j in [f"{s}_{n}" for s in ("left","right") for n in LEG_JOINTS]+["waist_yaw"])}
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
{chr(10).join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in ours+["og1_torso_frame","og1_waist_yaw_link_integrated"])}
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="0" conaffinity="0"/>
    <body name="torso_link" pos="0 0 {TORSO_Z}">
      <inertial pos="{f(t["com"])}" mass="{t["mass"]:.4f}" fullinertia="{f([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])}"/>
      <geom class="visual" mesh="og1_torso_frame" rgba="0.6 0.6 0.63 1"/>
      <geom class="collision" mesh="torso_link_rev_1_0"/>
      <geom class="visual" mesh="og1_waist_yaw_link_integrated" pos="0.0039635 0 -0.044" rgba="0.62 0.71 0.78 1"/>
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
