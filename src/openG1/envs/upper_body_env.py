"""Open G1 upper body (torso fixed, head, 2 x 7-DoF servo arms): two-handed reaching.

Each episode places a green target for the left hand and a red one for the right hand in front of the
robot. Actions in [-1, 1] are servo position targets, scaled onto each actuator's ctrlrange.
"""
from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from ..meshes import check_meshes

UPPER_XML = str(Path(__file__).resolve().parent.parent / "assets" / "open_g1_upper.xml")
CAMERA = {"distance": 1.6, "lookat": np.array((0.15, 0.0, 1.0)), "elevation": -10.0, "azimuth": 150.0}


class OpenG1UpperBodyReachEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        xml_file: str = UPPER_XML,
        frame_skip: int = 10,                         # 50 Hz control
        target_box_left=((0.18, 0.40), (0.05, 0.35), (0.85, 1.25)),   # world x, y, z ranges (m)
        target_box_right=((0.18, 0.40), (-0.35, -0.05), (0.85, 1.25)),
        success_radius: float = 0.03,
        ctrl_cost_weight: float = 1e-3,
        action_rate_weight: float = 1e-2,
        reset_noise_scale: float = 0.05,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, target_box_left, target_box_right, success_radius,
                                ctrl_cost_weight, action_rate_weight, reset_noise_scale, **kwargs)
        self._boxes = (np.array(target_box_left), np.array(target_box_right))
        self._success_radius = success_radius
        self._ctrl_cost_weight = ctrl_cost_weight
        self._action_rate_weight = action_rate_weight
        self._reset_noise_scale = reset_noise_scale
        self._targets = np.zeros((2, 3))
        self._last_action = None

        if xml_file == UPPER_XML:
            check_meshes(xml_file)
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None,
                           default_camera_config=CAMERA, **kwargs)
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        self._hand_sites = [self.model.site("left_hand").id, self.model.site("right_hand").id]
        self._mocap = [self.model.body("target_left").mocapid[0], self.model.body("target_right").mocapid[0]]
        lo, hi = self.model.actuator_ctrlrange.T
        self._ctrl_lo, self._ctrl_hi = lo.copy(), hi.copy()
        self.action_space = Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self._last_action = np.zeros(self.model.nu)
        self.observation_space = Box(-np.inf, np.inf, shape=(self._get_obs().shape[0],), dtype=np.float64)

    # ------------------------------------------------------------------ helpers
    def _hands(self):
        return np.stack([self.data.site_xpos[i] for i in self._hand_sites])

    def _get_obs(self):
        hands = self._hands()
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat, hands.ravel(),
                               self._targets.ravel(), (self._targets - hands).ravel()]).astype(np.float64)

    def _sample_targets(self):
        for k, box in enumerate(self._boxes):
            self._targets[k] = self.np_random.uniform(box[:, 0], box[:, 1])
            self.data.mocap_pos[self._mocap[k]] = self._targets[k]

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ctrl = self._ctrl_lo + (action + 1.0) * 0.5 * (self._ctrl_hi - self._ctrl_lo)
        self.do_simulation(ctrl, self.frame_skip)

        dist = np.linalg.norm(self._targets - self._hands(), axis=1)
        ctrl_cost = self._ctrl_cost_weight * float(np.sum(action ** 2))
        rate_cost = self._action_rate_weight * float(np.sum((action - self._last_action) ** 2))
        self._last_action = action
        reward = -float(dist.sum()) - ctrl_cost - rate_cost
        success = bool(np.all(dist < self._success_radius))
        info = {"dist_left": float(dist[0]), "dist_right": float(dist[1]), "is_success": success}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, False, False, info

    def reset_model(self):
        n = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-n, n, size=self.model.nq)
        qpos = np.clip(qpos, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        self.set_state(qpos, self.init_qvel.copy())
        self._sample_targets()
        self._last_action = np.zeros(self.model.nu)
        return self._get_obs()
