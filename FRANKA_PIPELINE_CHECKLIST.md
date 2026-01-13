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
| `000xxx/obses.pth` (T, H, W, C) | `obs['visual']` (T, C, H, W) |

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
*Letzte Aktualisierung: 2026-01-13*
*Workspace: `/home/tsp_jw/Desktop/dino_wm/`*

