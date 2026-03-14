# Franka Cube Stacking → DINO World Model Pipeline

**Checkliste für die Integration von Isaac Sim Datensätzen in das DINO WM Training**

---

## ✅ Phase 1: Datensatz-Generierung (Isaac Sim)

### 1.1 Data Logger implementieren
- [x] `FrankaDataLogger` Klasse erstellen (`data_logger.py`)
- [x] State-Extraktion: `get_franka_state()` → 22-dim Vektor
- [x] Action-Extraktion: `get_franka_action()` → 9-dim Vektor

### 1.2 Datensammlung-Script
- [x] `fcs_main_parallel.py` für parallele Episoden-Generierung
- [x] Validierung: `validate_stacking()` prüft Stacking-Erfolg
- [x] Domain Randomization: Würfel-Positionen, Licht, Materialien

### 1.3 Datensatz-Struktur generieren
- [x] `states.pth` — Shape: `(N, T_max, 22)` float32
- [x] `actions.pth` — Shape: `(N, T_max, 9)` float32
- [x] `metadata.pkl` — `{'episode_lengths': [...], ...}`
- [x] Episode-Ordner `000000/obses.pth` — `(T, 256, 256, 3)` uint8

**Aktueller Datensatz:** `isaacsim/00_my_envs/Franka_Cube_Stacking_JW/dataset/franka_cube_stack_ds/`
- 36 Episoden generiert
- 20 Episoden mit vollständigen Metadaten (max 935 Timesteps)

---

## ✅ Phase 2: DINO WM Dataset-Integration

### 2.1 Dataset-Loader erstellen
> **Commit:** `def3474` — *"franka cube stack dataset und yaml config"*

- [x] `datasets/franka_cube_stack_dset.py` erstellen
- [x] `FrankaCubeStackDataset(TrajDataset)` Klasse implementieren
- [x] `load_franka_cube_stack_slice_train_val()` Funktion für Train/Val Split

### 2.2 Hydra Config erstellen
> **Commit:** `def3474` — *"franka cube stack dataset und yaml config"*

- [x] `conf/env/franka_cube_stack.yaml` anlegen
- [x] Dataset-Target: `datasets.franka_cube_stack_dset.load_franka_cube_stack_slice_train_val`
- [x] `data_path` konfigurierbar via `${oc.env:DATASET_DIR,./data}`

### 2.3 Kompatibilität mit TrajDataset
> **Referenz:** `f61b0ca` — *"cleanup imports; cleanup config"*

- [x] `get_seq_length(idx)` implementiert
- [x] `get_frames(idx, frames)` für Frame-Zugriff
- [x] `proprio_dim`, `action_dim`, `state_dim` Attribute

---

## ✅ Phase 3: Pfad-Konfiguration

### 3.1 Datensatz-Pfad festlegen
- [x] Option B: `data_path` in `franka_cube_stack.yaml` direkt anpassen
  - Pfad: `/media/tsp_jw/.../franka_cube_stack_ds`

### 3.2 Pfad validieren (2026-01-06)
- [x] `states.pth` ladbar — Shape: `(100, 935, 22)` ✓
- [x] `actions.pth` ladbar — Shape: `(100, 935, 9)` ✓
- [x] Episode-Ordner existieren — `000000/obses.pth` ✓
- [x] Metadata-Konsistenz — 100 Episoden, state_dim=22, action_dim=9 ✓

---

## ✅ Phase 4: Training starten

### 4.1 Konfiguration prüfen
- [x] `conf/train.yaml` Defaults verstehen (basierend auf `0a9492f`)
- [x] `img_size: 224` — Resize von 256×256 auf 224×224
- [x] `frameskip=5`, `num_hist=3`, `num_pred=1`

### 4.2 Training ausführen (2026-01-07)
- [x] Conda Environment aktivieren (`dino_wm`)
- [x] Training gestartet: `python train.py env=franka_cube_stack frameskip=5 num_hist=3`
- [x] WandB Logging: https://wandb.ai/jacob-weyer-rwth-aachen-university/dino_wm

### 4.3 Checkpoint speichern
> **Referenz:** `0a9492f` — *"add checkpoints"*

- [x] Checkpoints unter: `outputs/2026-01-07/09-33-48/`
- [ ] `model_latest.pth` nach Training vorhanden

### 4.4 GPU Thermal Management (RTX A4000)
> **Datum:** 2026-01-13 — *"Training Freeze durch Thermal Throttling"*

- [x] Problem identifiziert: GPU erreicht 83°C → Thermal Throttling
- [x] Power Limit reduziert: `sudo nvidia-smi -pl 100` (von 140W)
- [x] GreenWithEnvy (GWE) installiert für Lüftersteuerung
- [x] Custom Fan Profile "JW" erstellt (100% ab 70-75°C)
- [x] GWE Autostart aktiviert

### 4.5 Training Freeze / Deadlock Problem
> **Datum:** 2026-01-13 — *"Wiederholte Training-Freezes unabhängig von Temperatur"*

**Symptome:**
- [x] Training freezt nach zufälliger Zeit (nicht temperaturabhängig)
- [x] GPU-Temperatur bei Freeze: ~48°C (normal)
- [x] Zombie-Prozess `[python] <defunct>` entsteht
- [x] Mehrere Subprozesse bleiben hängen
- [x] Keine Fehlermeldung, kein OOM, kein Crash

**Diagnose:**
```bash
ps aux | grep train.py
# Zeigt: Hauptprozess + Zombie + mehrere DataLoader-Worker
```

**Ausgeschlossene Ursachen:**
- [x] GPU Thermal Throttling → NEIN (Temp bei Freeze: 48°C)
- [x] DataLoader Multiprocessing (`num_workers > 0`) → NEIN (freezt auch mit `num_workers=0`)
- [x] `preload_images=True` bereits aktiv → Hilft NICHT
- [x] Datensatz-Größe → NEIN (freezt mit 10 und 100 Episoden)
- [x] Zeit-basiert → NEIN (Freeze nach 15min, 48min, variabel)

**WandB Monitoring Analyse (25 Runs):**
- RAM ist KONSTANT (~4-5GB) - **KEIN Speicherleck**
- GPU-Temperatur variiert (50-85°C) bei Crash - **NICHT temperaturabhängig**
- Crashes zeitlich verstreut (15min - 60min+) - **KEIN zeitliches Muster**
- GPU Power fällt plötzlich von ~80W auf ~20W ab

**Noch offene Fragen:**
- Warum stoppt GPU plötzlich?
- Ist es ein CUDA/Driver-Problem?
- Hängt es mit WandB Network Traffic zusammen?

**Lösungsversuche:**
- [x] `num_workers: 0` setzen → Freeze bleibt
- [x] `preload_images: True` → Freeze bleibt
- [ ] `WANDB_MODE=offline` (WandB deaktivieren) - **PRIORITÄT 1**
- [ ] `training.num_reconstruct_samples: 0` (keine Visualisierungen)
- [ ] RAM-Monitoring während Training: `watch -n 5 free -h`
- [ ] gc.collect() nach jeder Epoche einfügen

**Nächster Debug-Schritt (RAM-Leak Test):**
```bash
# Terminal 1: RAM überwachen
watch -n 5 'free -h && ps aux --sort=-%mem | head -5'

# Terminal 2: Training OHNE WandB starten
WANDB_MODE=disabled python train.py env=franka_cube_stack training.num_reconstruct_samples=0
```

### 4.6 Segmentation Fault Problem (2026-01-14)
> **Symptome:** Training crasht mit "Segmentation fault (core dumped)" nach 6-7 Iterationen

**Behobene Probleme:**
- [x] NaN-Werte in Actions (18/935 H5-Dateien pro Episode)
- [x] Fix in `franka_cube_stack_dset.py`: `np.nan_to_num(action, nan=0.0)`
- [x] Fix in `data_logger.py`: NaN-Prüfung vor Speichern

**Aktueller Status:**
- [x] Alte Datensätze (4D delta_pose): Training läuft nach NaN-Fix ✓
- [ ] Neue Datensätze (6D ee_pos): Segfault bleibt → weitere Diagnose nötig

**Debug-Schritte für Segfault:**
- [ ] GPU-Speicher prüfen: `nvidia-smi`
- [ ] Bilddaten validieren:
  ```bash
  python -c "
  import torch
  o = torch.load('fcs_datasets/2026_01_14_1639_fcs_dset/000000/obses.pth')
  print(f'Shape: {o.shape}, NaN: {torch.isnan(o).any()}, Min/Max: {o.min()}/{o.max()}')
  "
  ```
- [ ] GDB Backtrace für genaue Crash-Position:
  ```bash
  gdb -ex run -ex bt --args python train.py --config-name train.yaml env=franka_cube_stack
  ```
- [ ] CUDA Synchronous Execution:
  ```bash
  CUDA_LAUNCH_BLOCKING=1 python train.py env=franka_cube_stack
  ```
- [ ] Kleinere Batch-Size testen:
  ```bash
  python train.py env=franka_cube_stack batch_size=2
  ```

---

## 🔄 Phase 5: Online-Planung (Isaac Sim Interface)

> **Ziel:** Live-Ausführung von geplanten Aktionen in Isaac Sim

### 5.1 Isaac Sim Interface implementieren

**Datei:** `Franka_Cube_Stacking/isaac_sim_interface.py`

- [ ] `IsaacSimInterface` Klasse erstellen
- [ ] Verbindung zu laufender Isaac Sim Instanz
- [ ] State-Extraktion (22D) analog zu Data Logger
- [ ] Action-Ausführung über bestehenden Controller

**Methoden-Übersicht:**

| Methode | Beschreibung | Referenz |
|---------|--------------|----------|
| `__init__(config_path)` | Initialisiert Isaac Sim Szene | `fcs_main_parallel.py` |
| `reset(init_state)` | Setzt Roboter in Anfangszustand | `domain_randomization()` |
| `step(action)` | Führt Action aus, gibt obs+state zurück | `controller.forward()` |
| `get_observation()` | Extrahiert RGB-Bild (224×224) | `get_rgb()` |
| `get_state()` | Extrahiert 22D State-Vektor | `extract_cube_states()` |
| `close()` | Beendet Simulation | `simulation_app.close()` |

### 5.2 State-Extraktion (22 Dimensionen)

```python
def get_franka_state(franka) -> np.ndarray:
    """
    Extrahiert den vollständigen Roboter-Zustand.
    
    Returns:
        np.ndarray: Shape (22,)
            [0:3]   EE Position (x, y, z)
            [3:7]   EE Quaternion (w, x, y, z)
            [7]     Gripper Opening (0-1)
            [8:15]  Joint Positions (7 DOF)
            [15:22] Joint Velocities (7 DOF)
    """
    # End-Effektor Pose
    ee_pos, ee_quat = franka.end_effector.get_world_pose()
    
    # Gripper-Öffnung (normalisiert)
    gripper_pos = franka.gripper.get_joint_positions()
    gripper_opening = gripper_pos[0] / 0.04  # Max 4cm
    
    # Joint States
    joint_positions = franka.get_joint_positions()[:7]
    joint_velocities = franka.get_joint_velocities()[:7]
    
    return np.concatenate([
        ee_pos.flatten(),           # [0:3]
        ee_quat.flatten(),          # [3:7]
        [gripper_opening],          # [7]
        joint_positions.flatten(),  # [8:15]
        joint_velocities.flatten()  # [15:22]
    ]).astype(np.float32)
```

### 5.3 Observation-Extraktion (224×224 RGB)

```python
def get_observation(camera, franka) -> dict:
    """
    Extrahiert Observation für World Model.
    
    Returns:
        dict: {
            "visual": np.ndarray (224, 224, 3) uint8,
            "proprio": np.ndarray (3,) float32 - EE Position
        }
    """
    # RGB-Bild (bereits 224×224 aus Kamera-Config)
    rgba = camera.get_rgba()
    rgb = rgba[:, :, :3].astype(np.uint8)
    
    # Proprio = EE Position
    ee_pos, _ = franka.end_effector.get_world_pose()
    proprio = ee_pos.flatten()[:3].astype(np.float32)
    
    return {
        "visual": rgb,
        "proprio": proprio
    }
```

### 5.4 Action-Ausführung

**Action-Format für DINO WM Planning:**
- Mit `frameskip=5`: Action Shape = `(45,)` → 5 × 9D Actions
- Ohne frameskip: Action Shape = `(9,)` → 1 × 9D Action

```python
def apply_action(controller, action: np.ndarray, frameskip: int = 5):
    """
    Führt Action-Sequenz aus.
    
    Args:
        controller: StackingController_JW Instanz
        action: np.ndarray Shape (frameskip * 9,) oder (9,)
        frameskip: Anzahl Frames pro Action
    """
    action_dim = 9  # [joint_cmd(7), gripper_cmd(2)]
    
    if action.shape[0] == frameskip * action_dim:
        # Konkatenierte Actions aufteilen
        actions = action.reshape(frameskip, action_dim)
    else:
        actions = action.reshape(1, action_dim)
    
    for act in actions:
        joint_cmd = act[:7]   # Joint Commands
        gripper_cmd = act[7:] # Gripper Commands
        
        # Controller-Schritt ausführen
        controller.forward(
            joint_positions=joint_cmd,
            gripper_action=gripper_cmd
        )
        world.step(render=True)
```

### 5.5 Interface-Klasse Implementierung

**Datei:** `Franka_Cube_Stacking/isaac_sim_interface.py`

```python
"""
Isaac Sim Interface für DINO World Model Online-Planung.

Verbindet FrankaCubeStackWrapper mit laufender Isaac Sim Instanz.
Basiert auf fcs_main_parallel.py Architektur.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

class IsaacSimInterface:
    """
    Interface zwischen DINO WM Planner und Isaac Sim.
    
    Verwendung:
        # In separatem Terminal: Isaac Sim starten
        # python fcs_main_parallel.py --mode=online
        
        # Im DINO WM Planner:
        from isaac_sim_interface import IsaacSimInterface
        
        interface = IsaacSimInterface(config_path="config.yaml")
        obs, state = interface.reset()
        
        # Planning Loop
        for action in planned_actions:
            obs, state, done = interface.step(action)
    
    Attributes:
        img_size (tuple): Bildgröße (224, 224)
        state_dim (int): 22
        action_dim (int): 9
        frameskip (int): 5 (muss mit Training übereinstimmen!)
    """
    
    IMG_SIZE = (224, 224)
    STATE_DIM = 22
    ACTION_DIM = 9
    PROPRIO_DIM = 3
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        headless: bool = False,
        frameskip: int = 5
    ):
        """
        Initialisiert Isaac Sim Interface.
        
        Args:
            config_path: Pfad zur config.yaml
            headless: True für GUI-lose Ausführung
            frameskip: Muss mit Training-Config übereinstimmen!
        """
        self.config_path = Path(config_path)
        self.headless = headless
        self.frameskip = frameskip
        
        # Isaac Sim Komponenten (werden in setup() initialisiert)
        self.simulation_app = None
        self.world = None
        self.env = None
        self.controller = None
        self.camera = None
        
        self._is_initialized = False
    
    def setup(self) -> None:
        """
        Initialisiert Isaac Sim Szene.
        
        WICHTIG: Muss VOR reset() aufgerufen werden!
        """
        if self._is_initialized:
            return
            
        # Isaac Sim starten
        import isaacsim
        from isaacsim import SimulationApp
        
        self.simulation_app = SimulationApp({"headless": self.headless})
        
        # Komponenten importieren (nach SimulationApp Start!)
        from omni.isaac.core import World
        from Franka_Env_JW import Stacking_JW, StackingController_JW
        
        # World erstellen
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # Environment erstellen (Single-Env für Online-Modus)
        from fcs_main_parallel import Franka_Cube_Stack
        self.env = Franka_Cube_Stack(env_idx=0)
        self.env.setup_world(self.world)
        
        self.world.reset()
        
        # Controller initialisieren
        self.controller = self.env.setup_post_load()
        
        # Kamera hinzufügen
        self.camera = self.env.add_scene_cam()
        self.camera.initialize()
        
        self._is_initialized = True
    
    def reset(
        self,
        init_state: Optional[np.ndarray] = None,
        seed: int = 42
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Setzt Environment zurück.
        
        Args:
            init_state: Optional - Spezifischer Anfangszustand (22D)
            seed: Random-Seed für Domain Randomization
        
        Returns:
            obs: {"visual": (224,224,3), "proprio": (3,)}
            state: np.ndarray (22,)
        """
        if not self._is_initialized:
            self.setup()
        
        # Domain Randomization
        self.env.domain_randomization(seed)
        self.world.reset()
        
        # Optional: Spezifischen State setzen
        if init_state is not None:
            self._set_robot_state(init_state)
        
        # Warm-up Steps für Kamera
        for _ in range(10):
            self.world.step(render=True)
        
        obs = self._get_observation()
        state = self._get_state()
        
        return obs, state
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool]:
        """
        Führt Action aus.
        
        Args:
            action: np.ndarray Shape (frameskip * 9,) oder (9,)
        
        Returns:
            obs: {"visual": (224,224,3), "proprio": (3,)}
            state: np.ndarray (22,)
            done: bool - Episode beendet
        """
        self._apply_action(action)
        
        obs = self._get_observation()
        state = self._get_state()
        done = False  # TODO: Termination-Bedingung
        
        return obs, state, done
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Extrahiert aktuelle Observation."""
        from fcs_main_parallel import get_rgb
        
        rgb = get_rgb(self.camera)
        if rgb is None:
            rgb = np.zeros((*self.IMG_SIZE, 3), dtype=np.uint8)
        
        ee_pos, _ = self.env.franka.end_effector.get_world_pose()
        proprio = np.atleast_1d(ee_pos).flatten()[:3].astype(np.float32)
        
        return {
            "visual": rgb,
            "proprio": proprio
        }
    
    def _get_state(self) -> np.ndarray:
        """Extrahiert 22D State-Vektor."""
        franka = self.env.franka
        
        # EE Pose
        ee_pos, ee_quat = franka.end_effector.get_world_pose()
        ee_pos = np.atleast_1d(ee_pos).flatten()[:3]
        ee_quat = np.atleast_1d(ee_quat).flatten()[:4]
        
        # Gripper (normalisiert auf 0-1)
        gripper_pos = franka.gripper.get_joint_positions()
        gripper_opening = float(gripper_pos[0]) / 0.04
        
        # Joints
        joint_pos = franka.get_joint_positions()[:7]
        joint_vel = franka.get_joint_velocities()[:7]
        
        return np.concatenate([
            ee_pos, ee_quat, [gripper_opening],
            joint_pos, joint_vel
        ]).astype(np.float32)
    
    def _set_robot_state(self, state: np.ndarray) -> None:
        """Setzt Roboter in spezifischen State."""
        # Joint Positions aus State extrahieren
        joint_positions = state[8:15]
        self.env.franka.set_joint_positions(joint_positions)
        
        # Gripper setzen
        gripper_opening = state[7] * 0.04  # Denormalisieren
        self.env.franka.gripper.set_joint_positions(
            np.array([gripper_opening, gripper_opening])
        )
    
    def _apply_action(self, action: np.ndarray) -> None:
        """Führt Action(s) aus."""
        action = np.atleast_1d(action).flatten()
        
        # Aufteilen falls konkateniert
        if action.shape[0] == self.frameskip * self.ACTION_DIM:
            actions = action.reshape(self.frameskip, self.ACTION_DIM)
        else:
            actions = action.reshape(-1, self.ACTION_DIM)
        
        for act in actions:
            # TODO: Action-Interpretation anpassen
            # Aktuell: Direktes Joint-Kommando
            self.world.step(render=True)
    
    def close(self) -> None:
        """Beendet Isaac Sim."""
        if self.simulation_app is not None:
            self.simulation_app.close()
            self._is_initialized = False
```

### 5.6 Integration in FrankaCubeStackWrapper

**Datei:** `dino_wm/env/franka_cube_stack/franka_cube_stack_wrapper.py`

Die Bildgröße im Wrapper von 256×256 auf 224×224 anpassen:

```python
def __init__(
    self,
    isaac_sim_interface: Optional[Any] = None,
    img_size: Tuple[int, int] = (224, 224),  # Angepasst!
    offline_mode: bool = True,
):
```

### 5.7 Checkliste Online-Modus

- [x] `isaac_sim_interface.py` in `Franka_Cube_Stacking/` erstellt
- [x] State-Extraktion implementiert: `_get_state()` → 22D ✓
- [x] Observation-Extraktion implementiert: `_get_observation()` → 224×224 RGB ✓
- [x] Action-Ausführung implementiert: `_apply_single_action()` 
- [x] `FrankaCubeStackWrapper` mit Interface verbunden
- [x] `create_franka_env_online()` Hilfsfunktion erstellt
- [ ] End-to-End Test: Dataset-Episode replizieren

### 5.8 Testprotokoll

```bash
# 1. Isaac Sim Interface standalone testen
cd Franka_Cube_Stacking
python -c "
from isaac_sim_interface import IsaacSimInterface
interface = IsaacSimInterface(headless=True)
obs, state = interface.reset(seed=42)
print(f'Obs visual: {obs[\"visual\"].shape}')  # (224, 224, 3)
print(f'Obs proprio: {obs[\"proprio\"].shape}')  # (3,)
print(f'State: {state.shape}')  # (22,)
interface.close()
"

# 2. Mit DINO WM Wrapper testen
cd dino_wm
python -c "
from env.franka_cube_stack import FrankaCubeStackWrapper
# Online-Modus erfordert Isaac Sim Interface
# wrapper = FrankaCubeStackWrapper(isaac_sim_interface=..., offline_mode=False)
"

# 3. Planning mit Online-Ausführung
python plan.py env=franka_cube_stack \
    goal_source=dataset \
    planner=cem \
    online_mode=true
```

---

## ⚠️ Hardware-Voraussetzungen

### GPU Kühlung für RTX A4000

Die RTX A4000 ist eine Blower-Karte mit konservativer VBIOS-Lüfterkurve. Für stabiles Training:

| Einstellung | Empfohlen | Tool |
|-------------|-----------|------|
| Power Limit | 100-110W | `nvidia-smi -pl` |
| Lüfter @ 70°C | 80-100% | GreenWithEnvy |
| Lüfter @ 75°C | 100% | GreenWithEnvy |
| Max Temp | <80°C | Monitoring |

**Installation GreenWithEnvy:**
```bash
sudo apt install flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install flathub com.leinardi.gwe
```

---

## Schnellreferenz

### Datenformat-Mapping

| Isaac Sim (Data Logger) | DINO WM (Dataset-Loader) |
|------------------------|--------------------------|
| `states.pth` (N, T, 22) | `self.states` |
| `actions.pth` (N, T, 9) | `self.actions` |
| `metadata.pkl['episode_lengths']` | `self.seq_lengths` |
| `000xxx/obses.pth` (T, 224, 224, C) | `obs['visual']` (T, C, 224, 224) |

### State-Vektor (22 Dimensionen)
```
[0:3]   EE Position (x, y, z)
[3:7]   EE Quaternion (w, x, y, z)
[7]     Gripper Opening (0-1)
[8:15]  Joint Positions (7 DOF)
[15:22] Joint Velocities (7 DOF)
```

### Action-Vektor (9 Dimensionen)
```
[0:7]   Joint Commands (Position/Velocity)
[7:9]   Gripper Commands
```

---

## Git-Historie (relevante Commits)

| Commit | Beschreibung | Dateien |
|--------|--------------|---------|
| `def3474` | franka cube stack dataset und yaml config | `franka_cube_stack_dset.py`, `franka_cube_stack.yaml` |
| `bae45b1` | Übernehme .gitignore aus yml-conf | `.gitignore` |
| `0a9492f` | add checkpoints | `README.md`, `plan_*.yaml` |
| `f61b0ca` | cleanup imports; cleanup config | Diverse Dataset-Loader |

---

*Erstellt: 2026-01-06*
*Letzte Aktualisierung: 2026-02-01*
*Workspace: `/home/tsp_jw/Desktop/dino_wm/`*

