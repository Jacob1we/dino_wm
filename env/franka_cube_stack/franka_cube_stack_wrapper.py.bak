"""
FrankaCubeStackWrapper - Minimaler Wrapper nach FlexEnvWrapper Vorbild.

Referenz: env/deformable_env/FlexEnvWrapper.py (~170 Zeilen)
"""

import os
import sys
import numpy as np
import gym
import torch

# Hilfsfunktion aus FlexEnvWrapper
def aggregate_dct(dcts):
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
    Minimaler Wrapper für Franka Cube Stacking - wie FlexEnvWrapper.
    
    WICHTIG: Isaac Sim muss VOR dem Import dieses Moduls gestartet werden!
    """
    
    def __init__(self, **kwargs):
        """Initialisiert den Wrapper (Isaac Sim wird lazy gestartet)."""
        super().__init__()
        
        # Dimensionen (wie FlexEnvWrapper)
        self.action_dim = 6  # EE positions: [x_start, y_start, z_start, x_end, y_end, z_end]
        self.proprio_start_idx = 0
        self.proprio_end_idx = 3
        self.success_threshold = 0.05  # 5cm
        
        # Isaac Sim Komponenten (lazy init)
        self._sim_initialized = False
        self._world = None
        self._env = None
        self._controller = None
        self._camera = None
        
        print(f"[FrankaCubeStackWrapper] Initialisiert (lazy)")
    
    def _ensure_sim_initialized(self):
        """Lazy initialization von Isaac Sim."""
        if self._sim_initialized:
            return
            
        # Isaac Sim imports (müssen nach SimulationApp Start sein!)
        from omni.isaac.core import World
        import isaacsim.core.utils.stage as stage_utils
        
        # Franka Cube Stack Pfad finden
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fcs_paths = [
            os.path.expanduser("~/Desktop/isaacsim/00_Franka_Cube_Stack"),
            os.path.join(script_dir, "../../../Franka_Cube_Stacking"),
            "Franka_Cube_Stacking",
        ]
        
        for fcs_path in fcs_paths:
            if os.path.exists(fcs_path):
                if fcs_path not in sys.path:
                    sys.path.insert(0, fcs_path)
                break
        
        from fcs_main_parallel import Franka_Cube_Stack, get_rgb
        self._get_rgb = get_rgb
        
        # World erstellen
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()
        self._stage = stage_utils.get_current_stage()
        
        # Environment erstellen
        self._env = Franka_Cube_Stack(
            robot_name="Franka",
            env_idx=0,
            offset=np.array([0.0, 0.0, 0.0]),
        )
        self._env.setup_world(world=self._world, stage=self._stage)
        self._world.reset()
        
        # Controller und Kamera
        self._controller = self._env.setup_post_load()
        self._camera = self._env.add_scene_cam()
        self._camera.initialize()
        
        # Warm-up
        for _ in range(10):
            self._world.step(render=True)
        
        self._sim_initialized = True
        print("[FrankaCubeStackWrapper] Isaac Sim initialisiert")
    
    def seed(self, seed):
        """Setzt Random Seed."""
        np.random.seed(seed)
    
    def eval_state(self, goal_state, cur_state):
        """
        Bewertet ob Ziel erreicht - wie FlexEnvWrapper.eval_state()
        
        Args:
            goal_state: Ziel EE-Position (3,) oder State (14,)
            cur_state: Aktuelle EE-Position (3,) oder State (14,)
        """
        goal_ee = goal_state[:3] if goal_state.shape[0] > 3 else goal_state
        cur_ee = cur_state[:3] if cur_state.shape[0] > 3 else cur_state
        
        distance = np.linalg.norm(goal_ee - cur_ee)
        success = distance < self.success_threshold
        
        print(f"Distance: {distance:.4f}m, Success: {success}")
        return {"success": success, "distance": distance}
    
    def update_env(self, env_info):
        """No-op wie FlexEnvWrapper."""
        pass
    
    def sample_random_init_goal_states(self, seed):
        """Sampelt zufällige Init/Goal States."""
        self.seed(seed)
        self._ensure_sim_initialized()
        
        # Reset und State holen
        self._controller.reset()
        cube_pos, target_pos = self._env.domain_randomization(seed)
        self._world.reset()
        
        for _ in range(5):
            self._world.step(render=True)
        
        # Init State = aktuelle EE Position
        ee_pos, _ = self._env.franka.end_effector.get_world_pose()
        init_state = np.array(ee_pos, dtype=np.float32)[:3]
        
        # Goal State = über dem Ziel
        goal_state = np.array([target_pos[0], target_pos[1], 0.15], dtype=np.float32)
        
        return init_state, goal_state
    
    def prepare(self, seed, init_state):
        """
        Reset mit kontrolliertem init_state - wie FlexEnvWrapper.prepare()
        
        Returns:
            obs: {"visual": (H,W,3), "proprio": (3,)}
            state: EE-Position (3,) oder full state
        """
        self.seed(seed)
        self._ensure_sim_initialized()
        
        self._controller.reset()
        self._env.domain_randomization(seed)
        self._world.reset()
        
        for _ in range(5):
            self._world.step(render=True)
        
        # Observation holen
        rgb = self._get_rgb(self._camera, env_idx=0)
        if rgb is None:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        
        ee_pos, _ = self._env.franka.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32)[:3]
        
        obs = {
            "visual": rgb,
            "proprio": ee_pos,
        }
        
        # State = EE Position (wie proprio)
        state = ee_pos
        
        return obs, state
    
    def step_multiple(self, actions):
        """
        Führt Aktionssequenz aus - wie FlexEnvWrapper.step_multiple()
        
        Args:
            actions: (T, action_dim) Actions
            
        Returns:
            obses: {"visual": (T,H,W,C), "proprio": (T,3)}
            rewards: 0
            dones: False
            infos: {"state": (T,3), "pos_agent": (T,3)}
        """
        obses = []
        infos = []
        
        for action in actions:
            # Action ausführen (Controller-Step)
            all_obs = self._world.get_observations()
            controller_action = self._controller.forward(observations=all_obs)
            self._env.franka.get_articulation_controller().apply_action(controller_action)
            self._world.step(render=True)
            
            # Observation holen
            rgb = self._get_rgb(self._camera, env_idx=0)
            if rgb is None:
                rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            
            ee_pos, _ = self._env.franka.end_effector.get_world_pose()
            ee_pos = np.array(ee_pos, dtype=np.float32)[:3]
            
            obs = {
                "visual": rgb,
                "proprio": ee_pos,
            }
            info = {"pos_agent": ee_pos, "state": ee_pos}
            
            obses.append(obs)
            infos.append(info)
        
        obses = aggregate_dct(obses)
        infos = aggregate_dct(infos)
        
        return obses, 0, False, infos
    
    def rollout(self, seed, init_state, actions):
        """
        Kompletter Rollout - wie FlexEnvWrapper.rollout()
        
        Args:
            seed: Random seed
            init_state: Initial state
            actions: (T, action_dim) Actions
            
        Returns:
            obses: {"visual": (T+1,H,W,C), "proprio": (T+1,3)}
            states: (T+1, state_dim)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        
        # Prepend initial observation (wie FlexEnvWrapper)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        
        return obses, states


# Für gym.make() Kompatibilität
def make_franka_env(**kwargs):
    return FrankaCubeStackWrapper(**kwargs)
