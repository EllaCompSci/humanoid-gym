"""Simulate Open G1 in MuJoCo and express every frame as MediaPipe Pose landmarks (33 points).

For each frame you get exactly what MediaPipe's PoseLandmarker would return for a person standing where
the robot is, seen from a virtual camera:
  world_landmarks  (T, 33, 3)  metres, origin at the mid-hip, camera axes: x right, y down, z away from
                               the camera (MediaPipe's pose_world_landmarks convention)
  image_landmarks  (T, 33, 3)  x, y normalised to [0, 1] across the image, z = depth relative to the hips on
                               roughly the same scale as x (MediaPipe's pose_landmarks convention)
  visibility       (T, 33)     1 if the point projects inside the image, else 0 (self-occlusion not modelled)

The points are computed geometrically from the robot's links (forward kinematics), so they are exact.
Optionally (--mediapipe-model) the frames are also rendered and run through real MediaPipe, so you can
compare what MediaPipe actually sees on the robot with the geometric answer.

Pose sources (pick one):
  --random N            N random joint configurations, base fixed upright (like random_pose_dataset.py)
  --qpos FILE.npz       replay any .npz with a 'qpos' array (random_poses.npz, retarget_mediapipe output, ...)
  --sim SECONDS         physics rollout in OpenG1StandEnv, driven by --policy (Stable-Baselines3 .zip)
                        or, without one, by smooth random actions

Examples:
  python tools/robot_to_mediapipe.py --random 1000 --out robot_mp.npz
  python tools/robot_to_mediapipe.py --sim 10 --render robot_mp.mp4 --out sim_mp.npz
  python tools/robot_to_mediapipe.py --sim 10 --policy ppo_stand.zip --algo PPO --out sim_mp.npz
  python tools/robot_to_mediapipe.py --qpos clip_g1.npz --render check.mp4 --mediapipe-model pose_landmarker_full.task
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = ROOT / "src" / "openG1" / "assets" / "open_g1.xml"

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

# Where each MediaPipe landmark sits on the robot: (body, offset in that body's frame, metres).
# Joints -> origin of the body the joint moves. Face -> front of the dome head. Hands -> the rubber hand
# mesh (fingers along +x, thumb up +z, index at the top, pinky at the bottom, fingertips ~0.15 m out).
# Heel / toe -> the rear / front foot contact spheres.
def _robot_landmarks():
    pts = {
        "nose": ("head_link", (0.097, 0.0, 0.395)),
        "left_ear": ("head_link", (-0.02, 0.127, 0.41)),
        "right_ear": ("head_link", (-0.02, -0.127, 0.41)),
        "mouth_left": ("head_link", (0.088, 0.025, 0.36)),
        "mouth_right": ("head_link", (0.088, -0.025, 0.36)),
    }
    for side, sy in (("left", 1.0), ("right", -1.0)):
        pts.update({
            f"{side}_eye_inner": ("head_link", (0.088, sy * 0.015, 0.42)),
            f"{side}_eye": ("head_link", (0.085, sy * 0.032, 0.422)),
            f"{side}_eye_outer": ("head_link", (0.078, sy * 0.048, 0.42)),
            f"{side}_shoulder": (f"{side}_shoulder_pitch_link", (0, 0, 0)),
            f"{side}_elbow": (f"{side}_elbow_link", (0, 0, 0)),
            f"{side}_wrist": (f"{side}_wrist_roll_link", (0, 0, 0)),
            f"{side}_index": (f"{side}_wrist_yaw_link", (0.15, -sy * 0.01, 0.03)),
            f"{side}_pinky": (f"{side}_wrist_yaw_link", (0.15, -sy * 0.01, -0.035)),
            f"{side}_thumb": (f"{side}_wrist_yaw_link", (0.11, -sy * 0.005, 0.06)),
            f"{side}_hip": (f"{side}_hip_pitch_link", (0, 0, 0)),
            f"{side}_knee": (f"{side}_knee_link", (0, 0, 0)),
            f"{side}_ankle": (f"{side}_ankle_roll_link", (0, 0, 0)),
            f"{side}_heel": (f"{side}_ankle_roll_link", (-0.05, 0, -0.03)),
            f"{side}_foot_index": (f"{side}_ankle_roll_link", (0.12, 0, -0.03)),
        })
    return [pts[n] for n in LANDMARK_NAMES]


ROBOT_LANDMARKS = _robot_landmarks()
# skeleton for drawing (MediaPipe's pose connections)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
               (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
               (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
               (29, 31), (30, 32), (27, 31), (28, 32)]


class VirtualCamera:
    """A MuJoCo free camera (lookat / distance / azimuth / elevation) plus the matching pinhole projection,
    so the rendered image and the projected landmarks line up exactly."""

    def __init__(self, model, width=640, height=480, distance=2.5, azimuth=180.0, elevation=-5.0,
                 lookat_z=0.75, follow=True):
        self.W, self.H = width, height
        self.distance, self.azimuth, self.elevation = distance, azimuth, elevation
        self.lookat_z, self.follow = lookat_z, follow
        self.fovy = model.vis.global_.fovy
        self.f = 0.5 * height / np.tan(np.radians(self.fovy) / 2)
        self.lookat = np.array([0.0, 0.0, lookat_z])

    def update(self, data, pelvis_id):
        if self.follow:
            self.lookat = np.array([data.xpos[pelvis_id, 0], data.xpos[pelvis_id, 1], self.lookat_z])

    def frame(self):
        """Camera position and rotation whose rows are the camera's right / down / forward axes (world)."""
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = self.lookat - self.distance * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return pos, np.stack([right, down, fwd])

    def mjv_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = self.lookat, self.distance, self.azimuth, self.elevation
        return cam

    def params(self):
        return dict(width=self.W, height=self.H, fovy_deg=self.fovy, focal_px=self.f, distance=self.distance,
                    azimuth_deg=self.azimuth, elevation_deg=self.elevation, lookat_z=self.lookat_z)


def landmarks_world(model, data, ids_offsets):
    """(33, 3) landmark positions in MuJoCo world coordinates."""
    out = np.empty((len(ids_offsets), 3))
    for i, (b, off) in enumerate(ids_offsets):
        out[i] = data.xpos[b] + data.xmat[b].reshape(3, 3) @ off
    return out


def to_mediapipe(world_pts, cam, scale=1.0):
    """World points -> (world_landmarks, image_landmarks, visibility) in MediaPipe's conventions."""
    pos, R = cam.frame()
    pc = (world_pts - pos) @ R.T                          # camera coords: right, down, forward (= depth)
    hip = 0.5 * (pc[23] + pc[24])
    world_lm = scale * (pc - hip)                         # hip-centred, metres, camera axes

    z = np.maximum(pc[:, 2], 1e-6)
    u = cam.f * pc[:, 0] / z + cam.W / 2
    v = cam.f * pc[:, 1] / z + cam.H / 2
    zn = (pc[:, 2] - hip[2]) * cam.f / (cam.W * max(hip[2], 1e-6))
    img_lm = np.column_stack([u / cam.W, v / cam.H, zn])
    vis = ((img_lm[:, 0] >= 0) & (img_lm[:, 0] <= 1) & (img_lm[:, 1] >= 0) & (img_lm[:, 1] <= 1)
           & (pc[:, 2] > 0)).astype(float)
    return world_lm, img_lm, vis


# ---------------------------------------------------------------- pose sources
def poses_random(model, n, seed, margin):
    rng = np.random.default_rng(seed)
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = model.jnt_qposadr[hinge]
    lo, hi = model.jnt_range[hinge].T
    lo, hi = lo + margin * (hi - lo), hi - margin * (hi - lo)
    for _ in range(n):
        q = model.qpos0.copy()
        q[qadr] = rng.uniform(lo, hi)
        yield q


def poses_file(path):
    rec = np.load(path)
    if "qpos" not in rec.files:
        raise SystemExit(f"{path} has no 'qpos' array (keys: {rec.files})")
    yield from rec["qpos"]


def poses_sim(seconds, policy_path, algo, seed, hold):
    """Physics rollout in OpenG1StandEnv. Yields qpos after every control step (50 Hz)."""
    sys.path.insert(0, str(ROOT / "src"))
    from openG1.envs.open_g1_env import OpenG1StandEnv
    env = OpenG1StandEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    if policy_path:
        import stable_baselines3 as sb3
        policy = getattr(sb3, algo).load(policy_path, device="cpu")
        act = lambda o: policy.predict(o, deterministic=True)[0]
        print(f"driving the robot with {algo} policy {policy_path}")
    else:
        current = {"a": rng.uniform(-1, 1, env.action_space.shape), "k": 0}

        def act(o):                       # random action, held for `hold` steps, then eased to a new one
            if current["k"] % hold == 0:
                current["target"] = rng.uniform(-1, 1, env.action_space.shape)
            current["a"] += 0.2 * (current["target"] - current["a"])
            current["k"] += 1
            return current["a"].copy()
        print("no --policy: driving the robot with smooth random actions (it will fall and reset)")
    steps = int(round(seconds / env.dt))
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(act(obs))
        yield env.unwrapped.data.qpos.copy()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


# ---------------------------------------------------------------- MediaPipe on rendered frames
class MediaPipeOnRender:
    def __init__(self, model_path, video_mode):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        if not Path(model_path).expanduser().is_file():
            raise SystemExit(f"model file not found: {model_path}\ndownload it with:\n  wget https://storage.googleapis.com/"
                             "mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        self.mp, self.video_mode = mp, video_mode
        opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(Path(model_path).expanduser())),
                                     running_mode=RunningMode.VIDEO if video_mode else RunningMode.IMAGE, num_poses=1)
        self.lm = PoseLandmarker.create_from_options(opts)

    def __call__(self, rgb, ts_ms):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.lm.detect_for_video(img, ts_ms) if self.video_mode else self.lm.detect(img)
        if not res.pose_world_landmarks:
            return np.full((33, 3), np.nan), np.full((33, 3), np.nan), np.zeros(33)
        w = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
        i = np.array([[p.x, p.y, p.z] for p in res.pose_landmarks[0]])
        v = np.array([p.visibility or 0.0 for p in res.pose_landmarks[0]])
        return w, i, v


def draw_overlay(rgb, img_lm, vis, color=(0, 255, 0), detected=None):
    import cv2
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = out.shape[:2]
    for sets, col in (((img_lm, vis), color), ((detected, None), (0, 0, 255))):
        lm, v = sets
        if lm is None or np.isnan(lm).any():
            continue
        px = np.column_stack([lm[:, 0] * W, lm[:, 1] * H]).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(out, tuple(px[a]), tuple(px[b]), col, 2, cv2.LINE_AA)
        for p in px:
            cv2.circle(out, tuple(p), 3, col, -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--random", type=int, metavar="N", help="N random joint configurations")
    src.add_argument("--qpos", help=".npz with a 'qpos' array to replay")
    src.add_argument("--sim", type=float, metavar="SECONDS", help="physics rollout length")
    p.add_argument("--policy", help="Stable-Baselines3 .zip to drive --sim (default: smooth random actions)")
    p.add_argument("--algo", default="PPO", help="SB3 algorithm class for --policy (PPO, SAC, TD3, ...)")
    p.add_argument("--hold", type=int, default=25, help="steps each random action target is held for (--sim)")
    p.add_argument("--margin", type=float, default=0.0,
                   help="--random: keep this fraction of each joint range away from the limits (0.2 = middle 60%%)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--scale", type=float, default=1.0, help="multiply world landmarks (e.g. 1.3 for human size)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-distance", type=float, default=2.5)
    p.add_argument("--cam-azimuth", type=float, default=180.0, help="180 = facing the robot's front")
    p.add_argument("--cam-elevation", type=float, default=-5.0)
    p.add_argument("--render", help="write an .mp4 of the rendered robot with the landmarks drawn on")
    p.add_argument("--mediapipe-model", help="also run real MediaPipe on the rendered frames (needs rendering)")
    p.add_argument("--out", default="robot_mediapipe.npz")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    ids_offsets = [(model.body(b).id, np.array(off, float)) for b, off in ROBOT_LANDMARKS]
    pelvis = model.body("pelvis").id
    cam = VirtualCamera(model, args.width, args.height, args.cam_distance, args.cam_azimuth, args.cam_elevation,
                        follow=args.sim is not None)

    if args.random:
        poses, fps = poses_random(model, args.random, args.seed, args.margin), 30.0
    elif args.qpos:
        poses, fps = poses_file(args.qpos), 30.0
    else:
        poses, fps = poses_sim(args.sim, args.policy, args.algo, args.seed, args.hold), 50.0

    need_render = bool(args.render or args.mediapipe_model)
    renderer = writer = detector = None
    if need_render:
        renderer = mujoco.Renderer(model, args.height, args.width)
    if args.render:
        import cv2
        writer = cv2.VideoWriter(args.render, cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))
    if args.mediapipe_model:
        detector = MediaPipeOnRender(args.mediapipe_model, video_mode=not args.random)

    rec = {k: [] for k in ("qpos", "world_landmarks", "image_landmarks", "visibility", "mujoco_world_points",
                           "mp_world_landmarks", "mp_image_landmarks", "mp_visibility")}
    for t, q in enumerate(poses):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.update(data, pelvis)
        pts = landmarks_world(model, data, ids_offsets)
        wl, il, vis = to_mediapipe(pts, cam, args.scale)
        rec["qpos"].append(data.qpos.copy()); rec["mujoco_world_points"].append(pts)
        rec["world_landmarks"].append(wl); rec["image_landmarks"].append(il); rec["visibility"].append(vis)

        if need_render:
            renderer.update_scene(data, camera=cam.mjv_camera())
            rgb = renderer.render()
            det = None
            if detector:
                mw, mi, mv = detector(rgb, int(t * 1000 / fps))
                rec["mp_world_landmarks"].append(mw); rec["mp_image_landmarks"].append(mi); rec["mp_visibility"].append(mv)
                det = mi
            if writer:
                writer.write(draw_overlay(rgb, il, vis, detected=det))

    if writer:
        writer.release()
    if renderer:
        renderer.close()

    out = {k: np.array(v) for k, v in rec.items() if v}
    T = len(out["qpos"])
    np.savez_compressed(args.out, **out, landmark_names=np.array(LANDMARK_NAMES),
                        timestamp_ms=(np.arange(T) * 1000 / fps).astype(int), fps=fps,
                        camera=np.array(str(cam.params())), scale=args.scale)
    print(f"{T} frames -> {args.out}  (world_landmarks / image_landmarks {out['world_landmarks'].shape})")
    print(f"  landmarks inside the image: {out['visibility'].mean():.1%}")
    if "mp_world_landmarks" in out:
        mw, ok = out["mp_world_landmarks"], ~np.isnan(out["mp_world_landmarks"][:, 0, 0])
        print(f"  MediaPipe found the robot in {ok.mean():.0%} of rendered frames")
        if ok.any():
            err = np.linalg.norm(mw[ok] - out["world_landmarks"][ok], axis=-1).mean(0) * 100
            worst = np.argsort(-err)[:5]
            print(f"  MediaPipe vs geometric world landmarks: mean {err.mean():.1f} cm; worst: "
                  + ", ".join(f"{LANDMARK_NAMES[i]} {err[i]:.0f}" for i in worst))
    if args.render:
        print(f"  video -> {args.render} (green = geometric landmarks, red = MediaPipe's detection)")


if __name__ == "__main__":
    main()
