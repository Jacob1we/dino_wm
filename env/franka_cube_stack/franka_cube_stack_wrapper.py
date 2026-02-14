"""
FrankaCubeStackWrapper — Gym-Env fuer DINO WM Planning (wie FlexEnvWrapper).

Implementiert die identische Schnittstelle wie env/deformable_env/FlexEnvWrapper.py,
damit plan.py die Franka-Umgebung GENAU so nutzen kann wie deformable_env:

    env = gym.make("franka_cube_stack")
    obs, state = env.prepare(seed, init_state)
    obses, states = env.rollout(seed, init_state, actions)
    metrics = env.eval_state(goal_state, cur_state)

Action-Format: 6D EE-Positionen [x_start, y_start, z_start, x_end, y_end, z_end]
  - Fuer jede Action wird der EE via RMPFlow IK zu [x_end, y_end, z_end] bewegt
  - x_start/y_start/z_start sind informativ (aktuelle EE-Position beim Training)

State-Format: 14D EEF States [7 Gelenkwinkel + 7 Gelenkgeschwindigkeiten]

Proprio-Format: 3D EE-Position [x, y, z]

WICHTIG: Isaac Sim muss VOR dem Import dieses Moduls gestartet werden!
"""

import os
import sys
import numpy as np
import gym
import torch


def aggregate_dct(dcts):
    """Aggregiert Liste von Dicts zu einem Dict mit gestackten Arrays."""
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct


class FrankaCubeStackWrapper(gym.Env):
    """
    Gym-Wrapper fuer Franka Cube Stacking — wie FlexEnvWrapper.

    Fuehrt 6D EE-Actions via RMPFlow IK aus und gibt RGB-Bilder zurueck.
    Identische Schnittstelle wie deformable_env fuer nahtlose plan.py Integration.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.action_dim = 6   # [x_start, y_start, z_start, x_end, y_end, z_end]
        self.proprio_start_idx = 0
        self.proprio_end_idx = 3
        self.success_threshold = 0.05  # 5cm XY-Distanz

        # Isaac Sim Komponenten (lazy init)
        self._sim_initialized = False
        self._world = None
        self._franka = None
        self._gripper = None
        self._cubes = []
        self._cameras = []
        self._rmpflow = None
        self._articulation_controller = None
        self._base_pos = np.array([0.0, 0.0, 0.0])
        self._target_position = None

        # Config
        self._config = kwargs.get("config", None)
        self._n_cubes = 1
        self._cube_side = 0.05

        # IK-Parameter
        self._ik_max_steps = 20
        self._ik_threshold = 0.005

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    def _ensure_sim_initialized(self):
        """Startet Isaac Sim und erstellt die Szene (einmalig)."""
        if self._sim_initialized:
            return

        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom, UsdShade, UsdLux, Gf, Sdf

        # Franka-Imports
        fcs_base = os.path.expanduser(
            "~/Desktop/isaacsim/00_Franka_Cube_Stack/Franka_Cube_Stacking"
        )
        if os.path.exists(fcs_base) and fcs_base not in sys.path:
            sys.path.insert(0, fcs_base)

        from Franka_Env_JW.rmpflow_controller_jw import RMPFlowController_JW
        from Franka_Env_JW.franka_jw import Franka

        cfg = self._load_config()
        self._n_cubes = cfg.get("n_cubes", 1)
        self._cube_side = cfg.get("cube_size", 0.05)

        # World
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        # Franka
        self._franka = self._world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )
        self._gripper = self._franka.gripper

        # Wuerfel
        vis_rand = cfg.get("visual_randomization", {})
        fixed_color = vis_rand.get("fixed_cube_color", [0.2, 0.4, 0.9])
        for i in range(self._n_cubes):
            cube = self._world.scene.add(DynamicCuboid(
                prim_path=f"/World/Cube_{i}",
                name=f"cube_{i}",
                size=self._cube_side,
                color=np.array(fixed_color),
                position=np.array([0.4, 0.0, self._cube_side / 2]),
            ))
            self._cubes.append(cube)

        # Bodenplatte, Licht (visuell identisch zum Training)
        stage = self._world.stage
        self._setup_plane(stage, cfg)
        self._setup_light(stage, cfg)

        # Kameras
        self._cameras = []
        cam_res = tuple(cfg.get("cam_resolution", [224, 224]))
        cam_freq = cfg.get("cam_frequency", 20)
        cameras_cfg = cfg.get("cameras", [
            {"name": "front_right", "position": [1.6, -2.0, 1.27],
             "euler": [66.0, 0.0, 32.05]}
        ])
        for idx, cam_cfg in enumerate(cameras_cfg):
            name = cam_cfg.get("name", f"camera_{idx}")
            cam_xform = f"/World/CameraXform_{name}"
            cam_prim = f"{cam_xform}/Camera"
            UsdGeom.Xform.Define(stage, cam_xform)
            UsdGeom.Camera.Define(stage, cam_prim)
            xform_api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cam_xform))
            xform_api.SetTranslate(Gf.Vec3d(*cam_cfg["position"]))
            xform_api.SetRotate(Gf.Vec3f(*cam_cfg["euler"]))
            camera = Camera(prim_path=cam_prim, frequency=cam_freq, resolution=cam_res)
            self._cameras.append(camera)

        # Reset + RMPFlow
        self._world.reset()
        self._rmpflow = RMPFlowController_JW(
            name="rmpflow_wrapper",
            robot_articulation=self._franka,
            physics_dt=1.0 / 60.0,
        )
        self._articulation_controller = self._franka.get_articulation_controller()

        # Warm-up
        for _ in range(20):
            self._world.step(render=True)
        for cam in self._cameras:
            cam.initialize()
        self._gripper.open()
        for _ in range(10):
            self._world.step(render=True)

        self._base_pos, _ = self._franka.get_local_pose()
        self._base_pos = np.array(self._base_pos)
        self._sim_initialized = True
        print(f"[FrankaCubeStackWrapper] Initialisiert (n_cubes={self._n_cubes})")

    def _load_config(self):
        """Laedt config.yaml (optional, mit Defaults)."""
        import yaml
        if self._config is not None:
            return self._config

        cfg_path = os.path.expanduser(
            "~/Desktop/isaacsim/00_Franka_Cube_Stack/Franka_Cube_Stacking/config.yaml"
        )
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f)
            cfg = {
                "cube_size": raw.get("cubes", {}).get("side", 0.05),
                "n_cubes": raw.get("cubes", {}).get("count", 1),
                "yaw_range": raw.get("cubes", {}).get("yaw_range", [-45, 45]),
                "cam_resolution": raw.get("camera", {}).get("resolution", [224, 224]),
                "cam_frequency": raw.get("camera", {}).get("frequency", 20),
                "cameras": [],
                "visual_randomization": raw.get("visual_randomization", {}),
                "scene": raw.get("scene", {}),
                "robot_min_reach": raw.get("robot", {}).get("min_reach", 0.3),
                "robot_max_reach": raw.get("robot", {}).get("max_reach", 0.75),
                "scene_width": raw.get("scene", {}).get("width", 0.60),
                "scene_length": raw.get("scene", {}).get("length", 0.75),
            }
            for cam in raw.get("camera", {}).get("cameras", [])[:4]:
                cfg["cameras"].append({
                    "name": cam.get("name", "cam"),
                    "position": cam["position"],
                    "euler": cam["euler"],
                })
            if not cfg["cameras"]:
                cfg["cameras"] = [{"name": "front_right",
                                   "position": [1.6, -2.0, 1.27],
                                   "euler": [66.0, 0.0, 32.05]}]
            return cfg
        else:
            return {
                "cube_size": 0.05, "n_cubes": 1, "yaw_range": [-45, 45],
                "cam_resolution": [224, 224], "cam_frequency": 20,
                "cameras": [{"name": "front_right",
                             "position": [1.6, -2.0, 1.27],
                             "euler": [66.0, 0.0, 32.05]}],
                "visual_randomization": {},
                "scene": {"width": 0.60, "length": 0.75, "plane_lift": 0.001},
                "robot_min_reach": 0.3, "robot_max_reach": 0.75,
                "scene_width": 0.60, "scene_length": 0.75,
            }

    def _setup_plane(self, stage, cfg):
        from pxr import UsdGeom, UsdShade, Gf, Sdf
        scene_cfg = cfg.get("scene", {})
        sw = scene_cfg.get("width", 0.60)
        sl = scene_cfg.get("length", 0.75)
        pl = scene_cfg.get("plane_lift", 0.001)
        vis = cfg.get("visual_randomization", {})
        rgba = vis.get("fixed_plane_rgba", [0.82, 0.82, 0.82, 1.0])
        mesh = UsdGeom.Mesh.Define(stage, "/World/Plane")
        p0 = Gf.Vec3d(0, -sw/2, pl)
        p1 = Gf.Vec3d(0, sw/2, pl)
        p2 = Gf.Vec3d(sl, sw/2, pl)
        p3 = Gf.Vec3d(sl, -sw/2, pl)
        mesh.CreatePointsAttr([p0, p1, p2, p3])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        up = Gf.Vec3f(0, 0, 1)
        mesh.CreateNormalsAttr([up]*4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)
        mat = UsdShade.Material.Define(stage, "/World/Plane/Mat")
        shader = UsdShade.Shader.Define(stage, "/World/Plane/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*rgba[:3]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(rgba[3])
        surf_out = mat.CreateSurfaceOutput()
        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        surf_out.ConnectToSource(shader_out)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

    def _setup_light(self, stage, cfg):
        from pxr import UsdGeom, UsdLux, Gf
        vis = cfg.get("visual_randomization", {})
        lc = vis.get("fixed_light", {
            "position": [0.4, 0.0, 2.0],
            "intensity": 6500.0,
            "radius": 0.5,
        })
        lp = lc.get("position", [0.4, 0.0, 2.0])
        UsdGeom.Xform.Define(stage, "/World/light_xform")
        UsdGeom.XformCommonAPI(
            stage.GetPrimAtPath("/World/light_xform")
        ).SetTranslate(Gf.Vec3d(*lp))
        light = UsdLux.SphereLight.Define(stage, "/World/light_xform/light")
        light.GetIntensityAttr().Set(float(lc.get("intensity", 6500.0)))
        light.GetRadiusAttr().Set(float(lc.get("radius", 0.5)))
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # =========================================================================
    # DOMAIN RANDOMIZATION (identisch zu fcs_main_parallel.py)
    # =========================================================================

    def _domain_randomization(self, seed):
        """Seed-basierte deterministische Wuerfel-/Target-Platzierung."""
        cfg = self._load_config()
        rng = np.random.default_rng(seed)
        cube_side = self._cube_side
        min_dist = cube_side * 3
        min_reach = cfg.get("robot_min_reach", 0.3)
        max_reach = cfg.get("robot_max_reach", 0.75)
        scene_w = cfg.get("scene_width", 0.60)
        scene_l = cfg.get("scene_length", 0.75)
        w = 0.6 * cube_side  # z-Offset (wie fcs_main)

        points = []
        for _ in range(self._n_cubes + 1):
            placed = False
            for _ in range(100):
                u = rng.uniform(0, scene_l)
                v = rng.uniform(-scene_w / 2, scene_w / 2)
                pt = self._base_pos + np.array([u, v, w])
                dist_base = np.linalg.norm(pt[:2] - self._base_pos[:2])
                if dist_base < min_reach or dist_base > max_reach:
                    continue
                if all(np.linalg.norm(pt[:2] - p[:2]) >= min_dist for p in points):
                    points.append(pt)
                    placed = True
                    break
            if not placed:
                points.append(
                    self._base_pos + np.array([0.4, 0.0, 1.1 * cube_side])
                )

        yaw_range = cfg.get("yaw_range", [-45, 45])
        orientations = []
        for n in range(self._n_cubes):
            cube_rng = np.random.default_rng(seed + n)
            yaw_deg = float(cube_rng.uniform(*yaw_range))
            yaw_rad = np.radians(yaw_deg)
            quat = np.array([np.cos(yaw_rad/2), 0, 0, np.sin(yaw_rad/2)])
            orientations.append(quat)

        for i, cube in enumerate(self._cubes):
            cube.set_world_pose(position=points[i], orientation=orientations[i])

        self._target_position = points[self._n_cubes]

    # =========================================================================
    # GYM ENV INTERFACE (wie FlexEnvWrapper)
    # =========================================================================

    def seed(self, seed):
        np.random.seed(seed)

    def eval_state(self, goal_state, cur_state):
        """Bewertet Zielerreichung: EE-Positions-Distanz."""
        goal_ee = goal_state[:3] if len(goal_state) >= 3 else goal_state
        cur_ee = cur_state[:3] if len(cur_state) >= 3 else cur_state
        distance = float(np.linalg.norm(goal_ee - cur_ee))
        success = distance < self.success_threshold
        return {"success": success, "distance": distance}

    def update_env(self, env_info):
        """No-op (wie FlexEnvWrapper)."""
        pass

    def sample_random_init_goal_states(self, seed):
        """Sampelt Init-/Goal-States."""
        self._ensure_sim_initialized()
        self.seed(seed)
        self._rmpflow.reset()
        self._world.reset()
        self._gripper.open()
        for _ in range(10):
            self._world.step(render=True)

        self._domain_randomization(seed)
        for _ in range(20):
            self._world.step(render=True)

        js = self._franka.get_joints_state()
        init_state = np.zeros(14, dtype=np.float32)
        init_state[:7] = np.array(js.positions[:7])
        if js.velocities is not None:
            init_state[7:14] = np.array(js.velocities[:7])
        goal_state = init_state.copy()
        return init_state, goal_state

    def prepare(self, seed, init_state):
        """
        Reset mit kontrolliertem init_state — wie FlexEnvWrapper.prepare().

        Returns:
            obs: {"visual": (H,W,3) BGR, "proprio": (3,)}
            state: 14D EEF States
        """
        self._ensure_sim_initialized()
        self.seed(seed)

        self._rmpflow.reset()
        self._world.reset()
        self._gripper.open()
        for _ in range(10):
            self._world.step(render=True)

        self._domain_randomization(seed)
        for _ in range(30):
            self._world.step(render=True)

        return self._get_obs(), self._get_state()

    def step(self, action):
        """
        Einzelne 6D Action ausfuehren via RMPFlow IK.

        Action: [x_start, y_start, z_start, x_end, y_end, z_end]
        Bewegt EE zu [x_end, y_end, z_end].
        """
        action = np.array(action, dtype=np.float64)
        target_ee = action[3:6] if len(action) >= 6 else action[:3]
        target_ori = np.array([0.0, 1.0, 0.0, 0.0])

        for _ in range(self._ik_max_steps):
            ik_action = self._rmpflow.forward(
                target_end_effector_position=target_ee,
                target_end_effector_orientation=target_ori,
            )
            self._articulation_controller.apply_action(ik_action)
            self._world.step(render=True)
            ee_pos, _ = self._franka.end_effector.get_world_pose()
            if np.linalg.norm(ee_pos - target_ee) < self._ik_threshold:
                break

        obs = self._get_obs()
        state = self._get_state()
        return obs, 0.0, False, {"state": state, "pos_agent": state[:3]}

    def step_multiple(self, actions):
        """
        Aktionssequenz ausfuehren — wie FlexEnvWrapper.step_multiple().

        Args:
            actions: (T, 6) Actions
        Returns:
            obses, rewards, dones, infos
        """
        obses, infos = [], []
        for action in actions:
            obs, _, _, info = self.step(action)
            obses.append(obs)
            infos.append(info)

        return aggregate_dct(obses), 0, False, aggregate_dct(infos)

    def rollout(self, seed, init_state, actions):
        """
        Kompletter Rollout — wie FlexEnvWrapper.rollout().

        Returns:
            obses: {"visual": (T+1,H,W,3), "proprio": (T+1,3)}
            states: (T+1, 14)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)

        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states

    # =========================================================================
    # INTERNE HILFSMETHODEN
    # =========================================================================

    def _get_obs(self):
        """Observation-Dict: visual (BGR!) + proprio (EE pos)."""
        rgb = self._get_rgb(0)
        bgr = rgb[:, :, ::-1].copy()  # RGB -> BGR (Training-Konvention)
        ee_pos, _ = self._franka.end_effector.get_world_pose()
        return {
            "visual": bgr,
            "proprio": np.array(ee_pos, dtype=np.float32)[:3],
        }

    def _get_state(self):
        """14D EEF State."""
        js = self._franka.get_joints_state()
        state = np.zeros(14, dtype=np.float32)
        state[:7] = np.array(js.positions[:7], dtype=np.float32)
        if js.velocities is not None:
            state[7:14] = np.array(js.velocities[:7], dtype=np.float32)
        return state

    def _get_rgb(self, camera_idx=0):
        """RGB-Bild (H, W, 3) uint8."""
        if camera_idx >= len(self._cameras):
            camera_idx = 0
        rgba = self._cameras[camera_idx].get_rgba()
        if rgba is None or rgba.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        rgb = np.array(rgba)[:, :, :3].copy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        return rgb
