"""
FrankaCubeStackWrapper - Environment Wrapper für DINO World Model Planning

Dieser Wrapper implementiert die gleiche Schnittstelle wie FlexEnvWrapper
(siehe env/deformable_env/FlexEnvWrapper.py) und ermöglicht die Integration
des DINO World Models mit dem Isaac Sim Franka Cube Stacking Szenario.

=============================================================================
SCHNITTSTELLEN-ÜBERSICHT (aus plan.py und planning/evaluator.py)
=============================================================================

Die plan.py verwendet folgende Environment-Methoden:

1. prepare(seed, init_state) → obs, state
   - Referenz: plan.py Zeile ~220, evaluator.py Zeile ~110
   - Setzt das Environment in einen definierten Anfangszustand
   
2. step_multiple(actions) → obses, rewards, dones, infos
   - Referenz: FlexEnvWrapper.py Zeile ~115
   - Führt eine Sequenz von Aktionen aus
   
3. rollout(seed, init_state, actions) → obses, states
   - Referenz: evaluator.py Zeile ~113-116, serial_vector_env.py Zeile ~85
   - Kombiniert prepare() + step_multiple() für kompletten Rollout
   
4. eval_state(goal_state, cur_state) → metrics
   - Referenz: evaluator.py Zeile ~150, serial_vector_env.py Zeile ~32
   - Bewertet ob Zielzustand erreicht wurde
   
5. update_env(env_info) → None
   - Referenz: plan.py Zeile ~230
   - Aktualisiert Environment-Konfiguration aus Dataset
   
6. sample_random_init_goal_states(seed) → init_state, goal_state
   - Referenz: plan.py Zeile ~215 (nur für goal_source="random_state")
   - Samplet zufällige Anfangs- und Zielzustände

=============================================================================
DATENFORMATE
=============================================================================

obs (Observation Dictionary):
    {
        "visual": np.array shape (H, W, 3) oder (T, H, W, 3) - RGB Bilder
        "proprio": np.array shape (proprio_dim,) - Propriozeption (EE-Position)
    }

state:
    np.array shape (state_dim,) - Vollständiger Roboter-Zustand
    Für Franka: [ee_pos(3), ee_quat(4), gripper(1), joints(7), joint_vel(7)] = 22 dim

action:
    np.array shape (action_dim,) - Roboter-Kommando
    Für Franka: [joint_cmd(7), gripper_cmd(2)] = 9 dim
    ACHTUNG: Bei frameskip>1 werden Aktionen konkateniert!
    Mit frameskip=5: action shape = (45,)

=============================================================================
"""

import os
import gym
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional, Any


class FrankaCubeStackWrapper(gym.Env):
    """
    Environment Wrapper für Franka Cube Stacking mit Isaac Sim.
    
    Diese Klasse kann in zwei Modi verwendet werden:
    
    1. OFFLINE-MODUS (Standard):
       - Verwendet nur das trainierte World Model für Planung
       - Keine Live-Verbindung zu Isaac Sim nötig
       - Evaluiert Aktionen nur im Latent-Space
       
    2. ONLINE-MODUS (mit Isaac Sim):
       - Verbindet sich mit laufender Isaac Sim Instanz
       - Führt geplante Aktionen real aus
       - Bekommt echtes visuelles Feedback
    
    Attributes:
        action_dim (int): Dimension des Action-Vektors (9 für Franka)
        state_dim (int): Dimension des State-Vektors (22 für Franka)
        proprio_dim (int): Dimension der Propriozeption (3 = EE-Position)
        img_size (Tuple[int, int]): Bildgröße (H, W)
        
    Referenzen:
        - FlexEnvWrapper: env/deformable_env/FlexEnvWrapper.py
        - SerialVectorEnv: env/serial_vector_env.py
        - PlanWorkspace: plan.py Zeile ~110
        - PlanEvaluator: planning/evaluator.py
    """
    
    # =========================================================================
    # Konstanten für Franka Cube Stacking
    # =========================================================================
    
    # Action-Dimensionen (aus DINO_WM_TRAINING_DOCUMENTATION.md)
    # Action = [joint_cmd(7), gripper_cmd(2)]
    ACTION_DIM = 9
    
    # State-Dimensionen
    # State = [ee_pos(3), ee_quat(4), gripper(1), joints(7), joint_vel(7)]
    STATE_DIM = 22
    
    # Proprio-Indizes (EE-Position für World Model)
    PROPRIO_START_IDX = 0
    PROPRIO_END_IDX = 3  # EE-Position (x, y, z)
    
    # Erfolgs-Schwellenwert für Cube Stacking
    SUCCESS_THRESHOLD = 0.05  # 5cm Abstand zum Ziel
    
    def __init__(
        self,
        isaac_sim_interface: Optional[Any] = None,
        img_size: Tuple[int, int] = (224, 224),
        offline_mode: bool = True,
    ):
        """
        Initialisiert den FrankaCubeStackWrapper.
        
        Args:
            isaac_sim_interface: Optional - Interface zu Isaac Sim
                                 (None für Offline-Planung)
            img_size: Bildgröße für Beobachtungen (H, W)
            offline_mode: True für reine World Model Planung,
                         False für Live-Ausführung in Isaac Sim
                         
        Referenz: FlexEnvWrapper.__init__ (Zeile ~40)
        """
        super().__init__()
        
        self.isaac_sim = isaac_sim_interface
        self.img_size = img_size
        self.offline_mode = offline_mode
        
        # Dimensionen setzen (verwendet von plan.py)
        self.action_dim = self.ACTION_DIM
        self.state_dim = self.STATE_DIM
        self.proprio_start_idx = self.PROPRIO_START_IDX
        self.proprio_end_idx = self.PROPRIO_END_IDX
        self.success_threshold = self.SUCCESS_THRESHOLD
        
        # Gym Spaces definieren
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "visual": gym.spaces.Box(
                low=0, high=255,
                shape=(img_size[0], img_size[1], 3),
                dtype=np.uint8
            ),
            "proprio": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.PROPRIO_END_IDX - self.PROPRIO_START_IDX,),
                dtype=np.float32
            )
        })
        
        # Interner Zustand
        self._current_state = None
        self._current_obs = None
        
        print(f"[FrankaCubeStackWrapper] Initialisiert")
        print(f"  - Modus: {'Offline (World Model only)' if offline_mode else 'Online (Isaac Sim)'}")
        print(f"  - Action Dim: {self.action_dim}")
        print(f"  - State Dim: {self.state_dim}")
        print(f"  - Image Size: {img_size}")
    
    # =========================================================================
    # HAUPTSCHNITTSTELLEN für plan.py und evaluator.py
    # =========================================================================
    
    def prepare(
        self,
        seed: int,
        init_state: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Setzt das Environment in einen definierten Anfangszustand.
        
        Diese Methode wird aufgerufen von:
        - evaluator.py: eval_actions() Zeile ~110
        - FlexEnvWrapper: rollout() Zeile ~162
        
        Args:
            seed: Random Seed für Reproduzierbarkeit
            init_state: Initialzustand des Roboters
                       Shape: (state_dim,) = (22,) für Franka
                       Format: [ee_pos(3), ee_quat(4), gripper(1), 
                               joints(7), joint_vel(7)]
        
        Returns:
            obs: Dictionary mit Beobachtungen
                 {"visual": (H, W, 3), "proprio": (3,)}
            state: Aktueller State (identisch mit init_state nach Reset)
            
        Referenz: FlexEnvWrapper.prepare() Zeile ~118
        """
        np.random.seed(seed)
        
        if self.offline_mode:
            # Offline: Dummy-Observation erstellen
            obs = self._create_dummy_observation(init_state)
            state = init_state.copy()
        else:
            # Online: Isaac Sim zurücksetzen
            obs, state = self._reset_isaac_sim(init_state)
        
        self._current_state = state
        self._current_obs = obs
        
        return obs, state
    
    def step_multiple(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, np.ndarray]]:
        """
        Führt eine Sequenz von Aktionen aus.
        
        Diese Methode wird aufgerufen von:
        - FlexEnvWrapper: rollout() Zeile ~163
        - Intern von rollout()
        
        Args:
            actions: Aktionssequenz
                    Shape: (T, action_dim) = (T, 9) für Franka
                    Jede Aktion: [joint_cmd(7), gripper_cmd(2)]
        
        Returns:
            obses: Dictionary mit Beobachtungssequenz
                   {"visual": (T, H, W, 3), "proprio": (T, 3)}
            rewards: Kumulierte Belohnung (float)
            dones: Episode beendet? (bool)
            infos: Zusätzliche Informationen
                   {"state": (T, state_dim), "pos_agent": (T, ...)}
                   
        Referenz: FlexEnvWrapper.step_multiple() Zeile ~130
        """
        T = actions.shape[0]
        
        obses_list = []
        states_list = []
        
        for t in range(T):
            action = actions[t]
            
            if self.offline_mode:
                # Offline: Simuliere Zustandsänderung (vereinfacht)
                obs, state = self._simulate_step(action)
            else:
                # Online: Führe in Isaac Sim aus
                obs, state = self._execute_isaac_sim_step(action)
            
            obses_list.append(obs)
            states_list.append(state)
            
            self._current_state = state
            self._current_obs = obs
        
        # Aggregiere zu Arrays
        obses = self._aggregate_observations(obses_list)
        states = np.stack(states_list)
        
        # Dummy rewards/dones (World Model braucht diese nicht)
        rewards = 0.0
        dones = False
        infos = {
            "state": states,
            "pos_agent": states[:, :3]  # EE-Position
        }
        
        return obses, rewards, dones, infos
    
    def rollout(
        self,
        seed: int,
        init_state: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Führt einen kompletten Rollout durch (prepare + step_multiple).
        
        Diese Methode wird aufgerufen von:
        - evaluator.py: eval_actions() Zeile ~113-116
        - serial_vector_env.py: rollout() Zeile ~85
        - plan.py: PlanWorkspace.prepare_targets() Zeile ~248
        
        WICHTIG: Dies ist die Hauptschnittstelle für die Evaluation!
        Der Planner generiert Aktionen, diese werden hier ausgeführt,
        und die resultierenden Beobachtungen werden mit dem Ziel verglichen.
        
        Args:
            seed: Random Seed
            init_state: Anfangszustand (state_dim,)
            actions: Aktionssequenz (T, action_dim)
                    ACHTUNG: Diese sind bereits denormalisiert!
                    (Siehe evaluator.py Zeile ~112-113)
        
        Returns:
            obses: Dictionary mit Beobachtungen
                   {"visual": (T+1, H, W, 3), "proprio": (T+1, 3)}
                   HINWEIS: T+1 weil Initial-Observation inkludiert!
            states: State-Sequenz (T+1, state_dim)
            
        Referenz: FlexEnvWrapper.rollout() Zeile ~158
        """
        # 1. Environment vorbereiten
        obs_0, state_0 = self.prepare(seed, init_state)
        
        # 2. Aktionen ausführen
        obses, rewards, dones, infos = self.step_multiple(actions)
        
        # 3. Initial-Observation hinzufügen (T+1 Observations)
        for key in obses.keys():
            obses[key] = np.concatenate([
                np.expand_dims(obs_0[key], axis=0),
                obses[key]
            ], axis=0)
        
        # 4. States zusammenfügen
        states = np.concatenate([
            np.expand_dims(state_0, axis=0),
            infos["state"]
        ], axis=0)
        
        return obses, states
    
    def eval_state(
        self,
        goal_state: np.ndarray,
        cur_state: np.ndarray
    ) -> Dict[str, Any]:
        """
        Bewertet ob der aktuelle Zustand dem Zielzustand entspricht.
        
        Diese Methode wird aufgerufen von:
        - evaluator.py: _compute_rollout_metrics() Zeile ~150
        - serial_vector_env.py: eval_state() Zeile ~32
        
        Für Cube Stacking: Prüft ob der Würfel korrekt gestapelt ist
        basierend auf der End-Effector Position oder Cube-Position.
        
        Args:
            goal_state: Zielzustand (state_dim,) oder (n_particles, 4)
            cur_state: Aktueller Zustand (gleiches Format)
        
        Returns:
            metrics: Dictionary mit Evaluationsmetriken
                    {
                        "success": bool - Ziel erreicht?
                        "distance": float - Abstand zum Ziel
                        "ee_distance": float - EE-Position Abstand
                    }
                    
        Referenz: FlexEnvWrapper.eval_state() Zeile ~50
        """
        # Extrahiere EE-Positionen (erste 3 Dimensionen)
        goal_ee = goal_state[:3] if goal_state.ndim == 1 else goal_state[:, :3].mean(axis=0)
        cur_ee = cur_state[:3] if cur_state.ndim == 1 else cur_state[:, :3].mean(axis=0)
        
        # Berechne Distanz
        ee_distance = np.linalg.norm(goal_ee - cur_ee)
        
        # Erfolg wenn innerhalb Schwellenwert
        success = ee_distance < self.success_threshold
        
        metrics = {
            "success": success,
            "distance": ee_distance,
            "ee_distance": ee_distance,
        }
        
        print(f"[eval_state] EE Distance: {ee_distance:.4f}m, "
              f"Success: {success} (threshold: {self.success_threshold}m)")
        
        return metrics
    
    def update_env(self, env_info: Any) -> None:
        """
        Aktualisiert die Environment-Konfiguration.
        
        Diese Methode wird aufgerufen von:
        - plan.py: PlanWorkspace.prepare_targets() Zeile ~230
        
        Für deformable_env wird hier z.B. die Objektkonfiguration
        aus dem Dataset übernommen. Für Franka ist dies meist ein No-Op.
        
        Args:
            env_info: Environment-spezifische Konfiguration aus Dataset
                     (property_params.pkl oder ähnlich)
                     
        Referenz: FlexEnvWrapper.update_env() Zeile ~62
        """
        # Für Franka Cube Stack: Könnte Cube-Positionen etc. setzen
        # Aktuell: No-Op (Konfiguration kommt aus Dataset)
        pass
    
    def sample_random_init_goal_states(
        self,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sampelt zufällige Anfangs- und Zielzustände.
        
        Diese Methode wird aufgerufen von:
        - plan.py: PlanWorkspace.prepare_targets() Zeile ~215
          (nur wenn goal_source="random_state")
        - serial_vector_env.py: sample_random_init_goal_states() Zeile ~21
        
        Für Cube Stacking: Generiert zufällige Würfel-Positionen
        und entsprechende Roboter-Konfigurationen.
        
        Args:
            seed: Random Seed
        
        Returns:
            init_state: Zufälliger Anfangszustand (state_dim,)
            goal_state: Zufälliger Zielzustand (state_dim,)
            
        Referenz: FlexEnvWrapper.sample_random_init_goal_states() Zeile ~64
        """
        np.random.seed(seed)
        
        # Beispiel: Zufällige EE-Positionen im Arbeitsbereich
        workspace_bounds = {
            "x": (0.3, 0.7),   # Vor dem Roboter
            "y": (-0.3, 0.3), # Links/Rechts
            "z": (0.1, 0.5),  # Höhe
        }
        
        # Zufällige Init-Position
        init_ee = np.array([
            np.random.uniform(*workspace_bounds["x"]),
            np.random.uniform(*workspace_bounds["y"]),
            np.random.uniform(*workspace_bounds["z"]),
        ])
        
        # Zufällige Goal-Position (gestapelter Würfel = höher)
        goal_ee = np.array([
            np.random.uniform(*workspace_bounds["x"]),
            np.random.uniform(*workspace_bounds["y"]),
            np.random.uniform(0.2, 0.4),  # Höher für gestapelten Würfel
        ])
        
        # Erstelle vollständige States (Rest mit Nullen auffüllen)
        init_state = np.zeros(self.state_dim, dtype=np.float32)
        init_state[:3] = init_ee
        
        goal_state = np.zeros(self.state_dim, dtype=np.float32)
        goal_state[:3] = goal_ee
        
        return init_state, goal_state
    
    # =========================================================================
    # HILFSMETHODEN
    # =========================================================================
    
    def _create_dummy_observation(
        self,
        state: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Erstellt eine Dummy-Observation für Offline-Modus.
        
        Im Offline-Modus haben wir keine echten Bilder.
        Diese Methode erstellt Platzhalter.
        
        Args:
            state: Aktueller State
            
        Returns:
            obs: Dummy-Observation Dictionary
        """
        obs = {
            "visual": np.zeros(
                (self.img_size[0], self.img_size[1], 3),
                dtype=np.uint8
            ),
            "proprio": state[self.PROPRIO_START_IDX:self.PROPRIO_END_IDX].astype(np.float32)
        }
        return obs
    
    def _simulate_step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Simuliert einen Schritt im Offline-Modus.
        
        HINWEIS: Dies ist eine vereinfachte Placeholder-Simulation!
        Im Offline-Modus passiert die eigentliche Vorhersage im World Model,
        nicht hier. Diese Methode gibt nur Dummy-Werte zurück.
        
        Args:
            action: Auszuführende Aktion (beliebige Dimension)
            
        Returns:
            obs: Dummy-Observation
            state: Aktueller State (unverändert)
        """
        if self._current_state is None:
            state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            state = self._current_state.copy()
            
            # Im Offline-Modus: Verwende Action um EE-Position zu aktualisieren
            # Action Format kann variieren (6D EE-pos oder 9D joints)
            action = np.atleast_1d(action).flatten()
            
            if action.shape[0] >= 6:
                # EE-Position Format: [x_start, y_start, z_start, x_end, y_end, z_end]
                # Verwende end-position als neue EE-Position
                state[:3] = action[3:6] if action.shape[0] >= 6 else action[:3]
        
        obs = self._create_dummy_observation(state)
        return obs, state
    
    def _aggregate_observations(
        self,
        obs_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregiert eine Liste von Observations zu Arrays.
        
        Args:
            obs_list: Liste von Observation-Dictionaries
            
        Returns:
            Aggregiertes Dictionary mit gestackten Arrays
            
        Referenz: FlexEnvWrapper.aggregate_dct() Zeile ~25
        """
        result = {}
        for key in obs_list[0].keys():
            result[key] = np.stack([obs[key] for obs in obs_list])
        return result
    
    # =========================================================================
    # ISAAC SIM INTEGRATION (für Online-Modus)
    # =========================================================================
    
    def _reset_isaac_sim(
        self,
        init_state: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Setzt Isaac Sim zurück und gibt Observation zurück.
        
        Verwendet das IsaacSimInterface aus Franka_Cube_Stacking/isaac_sim_interface.py
        
        Args:
            init_state: Gewünschter Anfangszustand (22D)
            
        Returns:
            obs: Observation aus Isaac Sim {"visual": (H,W,3), "proprio": (3,)}
            state: Aktueller State (22,)
        """
        if self.isaac_sim is None:
            raise RuntimeError(
                "Isaac Sim Interface nicht verbunden! "
                "Erstelle das Interface und übergebe es bei Initialisierung:\n"
                "  from Franka_Cube_Stacking.isaac_sim_interface import IsaacSimInterface\n"
                "  interface = IsaacSimInterface()\n"
                "  interface.setup()\n"
                "  wrapper = FrankaCubeStackWrapper(isaac_sim_interface=interface, offline_mode=False)"
            )
        
        # Verwende Interface reset() mit aktuellem Seed
        seed = np.random.randint(0, 10000)
        obs, state = self.isaac_sim.reset(seed=seed, init_state=init_state)
        
        return obs, state
    
    def _execute_isaac_sim_step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Führt eine Aktion in Isaac Sim aus.
        
        Verwendet das IsaacSimInterface aus Franka_Cube_Stacking/isaac_sim_interface.py
        
        Args:
            action: Auszuführende Aktion [joint_cmd(7), gripper_cmd(2)]
            
        Returns:
            obs: Neue Observation {"visual": (H,W,3), "proprio": (3,)}
            state: Neuer State (22,)
        """
        if self.isaac_sim is None:
            raise RuntimeError(
                "Isaac Sim Interface nicht verbunden!"
            )
        
        # Verwende Interface step()
        obs, state, done = self.isaac_sim.step(action)
        
        return obs, state
    
    # =========================================================================
    # GYM INTERFACE (Standard Gym Methoden)
    # =========================================================================
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Standard Gym reset() - ruft prepare() mit zufälligem State auf."""
        if seed is not None:
            np.random.seed(seed)
        init_state = np.zeros(self.state_dim, dtype=np.float32)
        obs, _ = self.prepare(seed or 0, init_state)
        return obs
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Standard Gym step() - führt einzelne Aktion aus."""
        obses, rewards, dones, infos = self.step_multiple(action.reshape(1, -1))
        obs = {k: v[0] for k, v in obses.items()}
        return obs, rewards, dones, infos
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Gibt aktuelles Bild zurück."""
        if self._current_obs is not None:
            return self._current_obs["visual"]
        return None
    
    def close(self) -> None:
        """Schließt Environment und Verbindungen."""
        if self.isaac_sim is not None:
            # TODO: Isaac Sim Verbindung trennen
            pass


# =============================================================================
# HILFSFUNKTIONEN für Integration
# =============================================================================

def create_franka_env_for_planning(
    n_envs: int = 1,
    offline_mode: bool = True,
    img_size: Tuple[int, int] = (224, 224),
    isaac_sim_interface: Optional[Any] = None,
) -> "SerialVectorEnv":
    """
    Erstellt eine SerialVectorEnv mit FrankaCubeStackWrapper Instanzen.
    
    Diese Funktion wird in plan.py verwendet um das Environment zu erstellen.
    
    Referenz: plan.py Zeile ~465-475 (für deformable_env)
    
    Args:
        n_envs: Anzahl paralleler Environments
        offline_mode: True für World Model only, False für Isaac Sim
        img_size: Bildgröße (224, 224 für DINO WM)
        isaac_sim_interface: Optional - Interface für Online-Modus
                            Nur relevant wenn offline_mode=False
        
    Returns:
        SerialVectorEnv mit n_envs FrankaCubeStackWrapper
        
    Beispiel (Offline):
        >>> from env.franka_cube_stack import create_franka_env_for_planning
        >>> env = create_franka_env_for_planning(n_envs=5)
        >>> obs, state = env.prepare(seeds, init_states)
        
    Beispiel (Online mit Isaac Sim):
        >>> from Franka_Cube_Stacking.isaac_sim_interface import IsaacSimInterface
        >>> interface = IsaacSimInterface(headless=True)
        >>> interface.setup()
        >>> env = create_franka_env_for_planning(
        ...     n_envs=1,  # Online nur mit 1 Env!
        ...     offline_mode=False,
        ...     isaac_sim_interface=interface
        ... )
    """
    from env.serial_vector_env import SerialVectorEnv
    
    if not offline_mode and n_envs > 1:
        print("[WARNING] Online-Modus unterstützt nur n_envs=1!")
        print("         Isaac Sim kann nicht parallel mit mehreren Environments laufen.")
        n_envs = 1
    
    envs = [
        FrankaCubeStackWrapper(
            isaac_sim_interface=isaac_sim_interface if not offline_mode else None,
            offline_mode=offline_mode,
            img_size=img_size,
        )
        for _ in range(n_envs)
    ]
    
    return SerialVectorEnv(envs)


def create_franka_env_online(
    isaac_sim_interface: Any,
    img_size: Tuple[int, int] = (224, 224),
) -> "FrankaCubeStackWrapper":
    """
    Erstellt einen einzelnen FrankaCubeStackWrapper für Online-Planung.
    
    Convenience-Funktion für den häufigsten Use-Case: Ein einzelnes
    Environment mit Isaac Sim Verbindung.
    
    Args:
        isaac_sim_interface: IsaacSimInterface Instanz (muss setup() aufgerufen haben!)
        img_size: Bildgröße (224, 224 für DINO WM)
        
    Returns:
        FrankaCubeStackWrapper im Online-Modus
        
    Beispiel:
        >>> from Franka_Cube_Stacking.isaac_sim_interface import IsaacSimInterface
        >>> from env.franka_cube_stack import create_franka_env_online
        >>> 
        >>> # Interface erstellen und starten
        >>> interface = IsaacSimInterface(headless=False)
        >>> interface.setup()
        >>> 
        >>> # Environment erstellen
        >>> env = create_franka_env_online(interface)
        >>> 
        >>> # Rollout durchführen
        >>> obs, state = env.prepare(seed=42, init_state=np.zeros(22))
        >>> actions = np.zeros((10, 9))  # 10 Dummy-Actions
        >>> obses, states = env.rollout(seed=42, init_state=state, actions=actions)
    """
    if isaac_sim_interface is None:
        raise ValueError(
            "isaac_sim_interface darf nicht None sein!\n"
            "Erstelle zuerst ein Interface:\n"
            "  from Franka_Cube_Stacking.isaac_sim_interface import IsaacSimInterface\n"
            "  interface = IsaacSimInterface()\n"
            "  interface.setup()"
        )
    
    if not isaac_sim_interface.is_initialized:
        raise RuntimeError(
            "IsaacSimInterface ist nicht initialisiert!\n"
            "Rufe interface.setup() auf bevor du es verwendest."
        )
    
    return FrankaCubeStackWrapper(
        isaac_sim_interface=isaac_sim_interface,
        offline_mode=False,
        img_size=img_size,
    )
