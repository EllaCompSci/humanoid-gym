"""Retarget MediaPipe Pose (world landmarks) onto Open G1 with inverse kinematics.

Pipeline, per frame:
  1. MediaPipe gives 33 world landmarks (metres, hip-centred, camera axes).
  2. Build a body frame from the human: origin mid-hip, y = right hip -> left hip, z = up towards the
     mid-shoulder, x = forward. This removes camera placement and heading.
  3. Turn every human bone into a unit direction. Matching directions (not positions) removes the size
     difference between person and robot.
  4. Waist yaw is set directly from the twist of the shoulder line against the hip line.
  5. Damped-least-squares IK (MuJoCo Jacobians, joint limits enforced) turns every robot bone
     (shoulder->elbow->wrist->hand, hip->knee->ankle->toe / heel) to point along the human one. The base stays upright at the default
     pose, so a human leaning forward shows up as hip flexion (the robot has no waist pitch).

Not recoverable from MediaPipe and held near the previous / neutral value: wrist roll (needs palm
orientation). Weakly constrained, so noisy: wrist pitch / yaw (short hand bone), ankle roll (heel + toe only),
hip yaw when the knee is straight. Landmarks are smoothed across frames (--smooth, default 0.5) for video.

Usage:
  python tools/retarget_mediapipe.py --selftest
  python tools/retarget_mediapipe.py --video clip.mp4 --model pose_landmarker_full.task --out clip_g1.npz --view
  python tools/retarget_mediapipe.py --webcam 0 --model pose_landmarker_full.task --view
  python tools/retarget_mediapipe.py --image person.jpg --model pose_landmarker_full.task --view
  (on macOS the --view window needs `mjpython` instead of `python`)
  python tools/retarget_mediapipe.py --landmarks landmarks.npy --out g1.npz      # (T, 33, 3) array, already extracted

Model files (pick one): https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = ROOT / "src" / "openG1" / "assets" / "open_g1.xml"

# MediaPipe Pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32

# Robot keypoints: (body, offset in body frame). Origins of the joint bodies, the middle of the index /
# pinky points on the hand and virtual heel / toe points on the sole (same as robot_to_mediapipe.py).
ROBOT_POINTS = {}
for s, side, sy in (("l", "left", 1.0), ("r", "right", -1.0)):
    ROBOT_POINTS.update({
        f"{s}_shoulder": (f"{side}_shoulder_pitch_link", (0, 0, 0)),
        f"{s}_elbow":    (f"{side}_elbow_link", (0, 0, 0)),
        f"{s}_wrist":    (f"{side}_wrist_roll_link", (0, 0, 0)),
        f"{s}_hand":     (f"{side}_wrist_yaw_link", (0.15, -sy * 0.01, -0.0025)),  # mid index/pinky knuckles
        f"{s}_hip":      (f"{side}_hip_pitch_link", (0, 0, 0)),
        f"{s}_knee":     (f"{side}_knee_link", (0, 0, 0)),
        f"{s}_ankle":    (f"{side}_ankle_roll_link", (0, 0, 0)),
        f"{s}_heel":     (f"{side}_ankle_roll_link", (-0.05, 0, -0.03)),
        f"{s}_toe":      (f"{side}_ankle_roll_link", (0.12, 0, -0.03)),
    })

# Bones: (robot parent point, robot child point, human parent landmark(s), human child landmark(s)).
# A tuple of landmarks means "their mean". Ordered so every parent target exists before its child.
BONES = []
for s, (SH, EL, WR, PK, IX, HP, KN, AN, HE, FT) in (
        ("l", (L_SHOULDER, L_ELBOW, L_WRIST, L_PINKY, L_INDEX, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT)),
        ("r", (R_SHOULDER, R_ELBOW, R_WRIST, R_PINKY, R_INDEX, R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT))):
    BONES += [
        (f"{s}_shoulder", f"{s}_elbow", SH, EL),
        (f"{s}_elbow", f"{s}_wrist", EL, WR),
        (f"{s}_wrist", f"{s}_hand", WR, (PK, IX)),
        (f"{s}_hip", f"{s}_knee", HP, KN),
        (f"{s}_knee", f"{s}_ankle", KN, AN),
        (f"{s}_ankle", f"{s}_toe", AN, FT),
        (f"{s}_ankle", f"{s}_heel", AN, HE),
    ]
ANCHORS = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")       # fixed to the robot's torso / pelvis
POINT_WEIGHT = {"elbow": 1.0, "wrist": 1.0, "hand": 0.5, "knee": 1.0, "ankle": 1.0, "toe": 0.5, "heel": 0.5}


def _lm(lm, idx):
    return lm[list(idx)].mean(axis=0) if isinstance(idx, tuple) else lm[idx]


def _vis(vis, idx):
    if vis is None:
        return 1.0
    return float(np.min(vis[list(idx)])) if isinstance(idx, tuple) else float(vis[idx])


def human_body_frame(lm):
    """Rotation whose columns are the human's forward / left / up axes, in MediaPipe coordinates."""
    mid_hip = 0.5 * (lm[L_HIP] + lm[R_HIP])
    y = lm[L_HIP] - lm[R_HIP]
    y /= np.linalg.norm(y)
    up = 0.5 * (lm[L_SHOULDER] + lm[R_SHOULDER]) - mid_hip
    z = up - up.dot(y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.column_stack([x, y, z]), mid_hip


def to_body_frame(lm):
    R, origin = human_body_frame(lm)
    return (lm - origin) @ R


class Retargeter:
    def __init__(self, model, iters=100, damping=1e-4, posture=1e-4, max_step=0.25, tol=1e-5,
                 restarts=6, good_fit=0.02, seed=0):
        """posture: pull towards the previous / neutral pose; good_fit: RMS bone error (m) below which
        no restarts are tried."""
        self.m, self.d = model, mujoco.MjData(model)
        self.iters, self.damping, self.posture, self.max_step, self.tol = iters, damping, posture, max_step, tol
        self.restarts, self.good_fit, self.rng = restarts, good_fit, np.random.default_rng(seed)

        self.points = {k: (model.body(b).id, np.array(off, float)) for k, (b, off) in ROBOT_POINTS.items()}
        self.waist = model.joint("waist_yaw_joint")
        # IK solves every hinge except the waist (set analytically); the base stays put
        self.jids = [j for j in range(model.njnt)
                     if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and j != self.waist.id]
        self.qadr = model.jnt_qposadr[self.jids]
        self.dadr = model.jnt_dofadr[self.jids]
        self.lo, self.hi = model.jnt_range[self.jids].T
        self.joint_names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

        self.q = model.qpos0.copy()          # warm start, carried between frames

    def _point(self, name):
        b, off = self.points[name]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ off

    def robot_points(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        return {k: self._point(k) for k in self.points}

    def targets(self, lm, vis=None):
        """Per-bone unit directions (world frame), weights and waist yaw from one (33, 3) landmark array."""
        h = to_body_frame(lm)
        s = h[L_SHOULDER] - h[R_SHOULDER]
        waist = float(np.clip(np.arctan2(-s[0], s[1]), *self.m.jnt_range[self.waist.id]))
        self.d.qpos[:] = self.m.qpos0
        mujoco.mj_kinematics(self.m, self.d)
        R_base = self.d.xmat[self.m.body("pelvis").id].reshape(3, 3)

        direction, weight = {}, {}
        for p, c, hp, hc in BONES:
            v = _lm(h, hc) - _lm(h, hp)
            n = np.linalg.norm(v)
            direction[(p, c)] = R_base @ (v / n) if n > 1e-6 else None
            weight[(p, c)] = 0.0 if n <= 1e-6 else POINT_WEIGHT[c.split("_", 1)[1]] * min(_vis(vis, hp), _vis(vis, hc))
        return direction, weight, waist

    def chain_targets(self, qpos, direction):
        """Absolute target points: robot anchors at qpos, bones laid out along the human directions
        with the robot's own (current) bone lengths."""
        pts = self.robot_points(qpos)
        tgt = {a: pts[a] for a in ANCHORS}
        for p, c, _, _ in BONES:
            u, bone = direction[(p, c)], pts[c] - pts[p]
            tgt[c] = tgt[p] + (np.linalg.norm(bone) * u if u is not None else bone)
        return tgt

    def _residual(self, q, direction, weight):
        """Weighted bone-direction error: each robot bone (child - parent) vs |bone| * human direction."""
        m, d = self.m, self.d
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp, jc = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        es, Js, ws = [], [], []
        for (p, c), u in direction.items():
            w = weight[(p, c)]
            if u is None or w <= 0:
                continue
            pp, pc = self._point(p), self._point(c)
            mujoco.mj_jac(m, d, jp, None, pp, self.points[p][0])
            mujoco.mj_jac(m, d, jc, None, pc, self.points[c][0])
            bone = pc - pp
            es.append(np.linalg.norm(bone) * u - bone)
            Js.append((jc - jp)[:, self.dadr])
            ws.append(w)
        return es, Js, ws

    def _ik(self, q, direction, weight, q_ref):
        """Damped least squares. A light pull towards q_ref holds joints the landmarks can't see
        (wrist roll, hip yaw on a straight leg) at the previous / neutral value instead of letting them drift."""
        nj = len(self.jids)
        ref = q_ref[self.qadr]
        for it in range(self.iters):
            es, Js, ws = self._residual(q, direction, weight)
            A = (self.damping + self.posture) * np.eye(nj)
            g = self.posture * (ref - q[self.qadr])
            for e, J, w in zip(es, Js, ws):
                A += w * J.T @ J
                g += w * J.T @ e
            dq = np.linalg.solve(A, g)
            dq *= min(1.0, self.max_step / (np.abs(dq).max() + 1e-12))
            q[self.qadr] = np.clip(q[self.qadr] + dq, self.lo, self.hi)
            if np.abs(dq).max() < self.tol:
                break
        es, _, ws = self._residual(q, direction, weight)
        fit = float(sum(w * e @ e for e, w in zip(es, ws)) / max(sum(ws), 1e-9))   # weighted mean sq. error, m^2
        cost = fit + self.posture * float(np.sum((q[self.qadr] - ref) ** 2)) / max(sum(ws), 1e-9)
        return q, fit, cost, it + 1

    def solve(self, lm, vis=None, warm_start=True):
        """Return (qpos, info) for one frame of MediaPipe world landmarks (33, 3).

        Starts from the previous frame's answer (video) and the zero pose; if the result still fits badly
        it retries from a few random starts, because the legs can lock into a wrong local solution.
        """
        m = self.m
        direction, weight, waist = self.targets(lm, vis)
        q_ref = self.q.copy() if warm_start else m.qpos0.copy()
        starts = [self.q.copy()] if warm_start else []
        starts.append(m.qpos0.copy())
        best = None
        for k in range(len(starts) + self.restarts):
            if k < len(starts):
                q0 = starts[k]
            else:
                q0 = m.qpos0.copy()
                q0[self.qadr] = self.rng.uniform(self.lo, self.hi)
            q0[self.waist.qposadr[0]] = waist
            q, fit, cost, iters = self._ik(q0, direction, weight, q_ref)
            if best is None or cost < best[2]:
                best = (q, fit, cost, iters)
            if best[1] < self.good_fit ** 2:          # good enough: don't try other starts
                break
        q, fit, cost, iters = best
        tgt = self.chain_targets(q, direction)
        pts = self.robot_points(q)
        err = {c: float(np.linalg.norm(pts[c] - tgt[c])) for (p, c) in direction if weight[(p, c)] > 0}
        self.q = q.copy()
        return q, {"iters": iters, "rms_bone_error": float(np.sqrt(fit)), "point_error": err,
                   "mean_error": float(np.mean(list(err.values()))) if err else float("nan"),
                   "targets": tgt, "waist_yaw": waist}


# ---------------------------------------------------------------- MediaPipe front end
def make_landmarker(model_path, mode):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                 running_mode=getattr(RunningMode, mode), num_poses=1)
    return mp, PoseLandmarker.create_from_options(opts)


def result_to_array(result):
    """MediaPipe result -> ((33, 3) world landmarks, (33,) visibility), or (None, None) if nobody found."""
    if not result.pose_world_landmarks:
        return None, None
    lms = result.pose_world_landmarks[0]
    xyz = np.array([[p.x, p.y, p.z] for p in lms])
    vis = np.array([p.visibility if p.visibility is not None else 1.0 for p in lms])
    return xyz, vis


def frames_from_source(args):
    """Yield (timestamp_ms, rgb_frame) from an image, a video file or a webcam."""
    import cv2
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"could not read {args.image}")
        yield 0, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return
    src = str(Path(args.video).expanduser()) if args.video else args.webcam
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        if args.video:
            hint = (f"file does not exist: {Path(src).resolve()}" if not Path(src).is_file()
                    else "file exists but OpenCV can't decode it - try converting: ffmpeg -i in.mov -c:v libx264 out.mp4")
        else:
            cams = sorted(str(p) for p in Path("/dev").glob("video*"))
            hint = (f"cameras found: {', '.join(cams)} - try another --webcam index, close other apps using the "
                    f"camera, or check you're in the 'video' group" if cams else
                    "no /dev/video* devices - Linux can't see a camera")
        raise SystemExit(f"could not open video source {src!r}\n  {hint}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, t0, last = 0, time.monotonic(), -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000 / fps) if args.video else int((time.monotonic() - t0) * 1000)
        ts = last = max(ts, last + 1)                 # MediaPipe needs strictly increasing timestamps
        yield ts, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


class Viewer:
    def __init__(self, model):
        import mujoco.viewer
        self.m, self.d = model, mujoco.MjData(model)
        self.v = mujoco.viewer.launch_passive(model, self.d)

    def show(self, qpos):
        if not self.v.is_running():
            return False
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        self.v.sync()
        return True


def nose_forward(lm):
    """How far the nose sits in front of the shoulder line (> 0 for un-mirrored landmarks)."""
    h = to_body_frame(lm)
    y = h[L_SHOULDER] - h[R_SHOULDER]
    fwd = np.cross(y, [0.0, 0.0, 1.0])
    return float((h[NOSE] - 0.5 * (h[L_SHOULDER] + h[R_SHOULDER])) @ fwd / (np.linalg.norm(fwd) + 1e-9))


def check_orientation(lm_seq):
    """Nose should be in front of the shoulders; if not, the landmarks are mirrored (e.g. a flipped selfie feed)."""
    xs = [nose_forward(lm) for lm in lm_seq if lm is not None]
    if xs and np.median(xs) < 0:
        print("WARNING: nose is behind the body in most frames - input looks mirrored; don't flip camera frames.")


def run(args):
    model = mujoco.MjModel.from_xml_path(args.xml)
    rt = Retargeter(model, iters=args.iters)
    viewer = Viewer(model) if args.view else None

    qs, lms, viss, errs, stamps = [], [], [], [], []
    smoothed = None

    def handle(ts, lm, vis):
        nonlocal smoothed
        if lm is not None and args.smooth > 0:
            smoothed = lm if smoothed is None else args.smooth * smoothed + (1 - args.smooth) * lm
            lm = smoothed
        if lm is None:
            q, err = rt.q.copy(), np.nan                     # nobody detected: hold the last pose
        else:
            q, info = rt.solve(lm, vis)
            err = info["mean_error"]
        qs.append(q); errs.append(err); stamps.append(ts)
        lms.append(lm if lm is not None else np.full((33, 3), np.nan))
        viss.append(vis if vis is not None else np.zeros(33))
        if viewer and not viewer.show(q):
            return False
        return True

    if args.landmarks:
        arr = np.load(args.landmarks)
        arr = arr[None] if arr.ndim == 2 else arr
        for t, lm in enumerate(arr):
            if not handle(t, lm[:, :3], lm[:, 3] if lm.shape[1] > 3 else None):
                break
            if viewer:
                time.sleep(1 / 30)
    else:
        if not args.model:
            raise SystemExit("--model path/to/pose_landmarker_*.task is required for image/video/webcam input")
        if not Path(args.model).expanduser().is_file():
            name = Path(args.model).name
            size = next((s for s in ("lite", "full", "heavy") if s in name), "full")
            raise SystemExit(
                f"model file not found: {Path(args.model).expanduser().resolve()}\n"
                f"download it with:\n  wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                f"pose_landmarker_{size}/float16/latest/pose_landmarker_{size}.task")
        args.model = str(Path(args.model).expanduser())
        mode = "IMAGE" if args.image else "VIDEO"
        mp, landmarker = make_landmarker(args.model, mode)
        with landmarker:
            for ts, rgb in frames_from_source(args):
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                res = landmarker.detect(img) if args.image else landmarker.detect_for_video(img, ts)
                lm, vis = result_to_array(res)
                if not handle(ts, lm, vis):
                    break

    check_orientation([l for l in lms if not np.isnan(l).any()])
    errs = np.array(errs, float)
    print(f"{len(qs)} frames, detected in {np.mean(~np.isnan(errs)):.0%}, "
          f"mean IK point error {np.nanmean(errs) * 100 if np.any(~np.isnan(errs)) else float('nan'):.1f} cm")
    if args.out:
        np.savez_compressed(args.out, qpos=np.array(qs), joint_pos=np.array(qs)[:, 7:],
                            joint_names=np.array(rt.joint_names), landmarks=np.array(lms),
                            visibility=np.array(viss), ik_error=errs, timestamp_ms=np.array(stamps))
        print(f"wrote {args.out}")
    if viewer and args.image:
        print("close the viewer window to exit")
        while viewer.show(qs[-1]):
            time.sleep(0.05)


# ---------------------------------------------------------------- self test
def fake_mediapipe(rt, qpos, rng, scale=1.3, noise=0.01, yaw=None):
    """Make MediaPipe-style world landmarks from a robot pose: person 1.3x bigger, seen by a camera at a
    random heading (or the given yaw), 1 cm noise. Only the landmarks the retargeter uses (plus the nose) are filled."""
    pts = rt.robot_points(qpos)
    torso = rt.m.body("torso_link").id
    nose = rt.d.xpos[torso] + rt.d.xmat[torso].reshape(3, 3) @ np.array([0.1, 0, 0.35])
    lm = np.zeros((33, 3))
    for s, (SH, EL, WR, PK, IX, HP, KN, AN, HE, FT) in (
            ("l", (L_SHOULDER, L_ELBOW, L_WRIST, L_PINKY, L_INDEX, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT)),
            ("r", (R_SHOULDER, R_ELBOW, R_WRIST, R_PINKY, R_INDEX, R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT))):
        for idx, k in ((SH, "shoulder"), (EL, "elbow"), (WR, "wrist"), (PK, "hand"), (IX, "hand"),
                       (HP, "hip"), (KN, "knee"), (AN, "ankle"), (HE, "heel"), (FT, "toe")):
            lm[idx] = pts[f"{s}_{k}"]
    lm[NOSE] = nose
    lm -= 0.5 * (lm[L_HIP] + lm[R_HIP])
    yaw = rng.uniform(-np.pi, np.pi) if yaw is None else yaw
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    robot_to_cam = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])     # x right, y down, z away from camera
    return scale * (lm @ Rz.T @ robot_to_cam.T) + rng.normal(0, noise, lm.shape)


def selftest(xml, n=100, seed=0):
    """Round trip: random robot pose -> fake MediaPipe landmarks -> IK -> compare with the original pose."""
    model = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(model)
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [model.joint(j).name for j in hinge]
    qadr = model.jnt_qposadr[hinge]
    lo, hi = model.jnt_range[hinge].T

    for label, scale, noise in (("perfect landmarks, robot-sized person", 1.0, 0.0),
                                ("1.3x bigger person, 1 cm landmark noise", 1.3, 0.01)):
        rng = np.random.default_rng(seed)
        rt = Retargeter(model, seed=seed)
        fit, perr, jerr, t0 = [], [], [], time.perf_counter()
        while len(perr) < n:
            q = model.qpos0.copy()
            q[qadr] = rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo))   # middle 60% of each range
            d.qpos[:] = q
            mujoco.mj_forward(model, d)
            if any(model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2] for c in d.contact[:d.ncon]):
                continue                                                     # skip self-colliding poses
            truth = rt.robot_points(q)
            lm = fake_mediapipe(rt, q, rng, scale=scale, noise=noise)
            assert nose_forward(lm) > 0, "fake landmarks came out mirrored"
            qs, info = rt.solve(lm, warm_start=False)                       # single images: no warm start
            got = rt.robot_points(qs)
            fit.append(info["mean_error"])
            perr.append([np.linalg.norm(got[k] - truth[k]) for k in info["point_error"]])
            jerr.append(np.abs(qs[qadr] - q[qadr]))
        perr, jerr = np.array(perr) * 100, np.degrees(np.array(jerr))
        print(f"\n{label}: {n} random non-colliding poses, {1000 * (time.perf_counter() - t0) / n:.0f} ms/pose")
        print(f"  IK fit to its targets       : mean {np.mean(fit) * 100:.2f} cm")
        print(f"  keypoints vs original pose  : mean {perr.mean():.2f} cm, 95th pct {np.percentile(perr, 95):.2f} cm")
        print(f"  joint angles vs original    : median {np.median(jerr):.1f} deg")
        print("    " + "  ".join(f"{nm.replace('_joint', '')} {e:.0f}" for nm, e in zip(names, np.median(jerr, axis=0))))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video")
    src.add_argument("--webcam", type=int)
    src.add_argument("--image")
    src.add_argument("--landmarks", help=".npy of shape (T, 33, 3) or (T, 33, 4) with visibility")
    src.add_argument("--selftest", action="store_true")
    p.add_argument("--model", help="MediaPipe pose_landmarker_{lite,full,heavy}.task")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--out", help="save qpos / landmarks / errors to this .npz")
    p.add_argument("--view", action="store_true", help="show the robot live in the MuJoCo viewer")
    p.add_argument("--smooth", type=float, default=0.5,
                   help="landmark smoothing across frames, EMA factor in [0, 1); 0 = off, higher = smoother but laggier")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()
    if args.selftest:
        selftest(args.xml)
    elif args.video or args.webcam is not None or args.image or args.landmarks:
        run(args)
    else:
        p.error("give --selftest, --video, --webcam, --image or --landmarks")


if __name__ == "__main__":
    main()
