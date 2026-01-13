# 🧠 DINO World Model - Vollständige Training-Dokumentation

> Eine detaillierte, chronologische Dokumentation des gesamten Trainingsprozesses für das DINO World Model mit dem Franka Cube Stacking Datensatz.

---

## 📑 Inhaltsverzeichnis

1. [Überblick und Konzept](#1-überblick-und-konzept)
2. [Datensatz-Struktur](#2-datensatz-struktur)
3. [Konfiguration und Parameter](#3-konfiguration-und-parameter)
4. [Training-Pipeline (Chronologisch)](#4-training-pipeline-chronologisch)
5. [Modell-Architektur](#5-modell-architektur)
6. [Loss-Funktionen](#6-loss-funktionen)
7. [Training starten](#7-training-starten)
8. [Glossar](#8-glossar)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Überblick und Konzept

### Was ist das DINO World Model?

Das **DINO World Model** ist ein visuelles Weltmodell, das lernt, zukünftige visuelle Zustände eines Roboters vorherzusagen. Es kombiniert:

- **DINO v2 Encoder**: Vortrainiertes Vision-Modell von Meta zur Bildrepräsentation
- **ViT Predictor**: Vision Transformer zur Vorhersage im Latent-Space
- **VQ-VAE Decoder**: Rekonstruktion von Bildern aus dem Latent-Space

### Konzept der Vorhersage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WORLD MODEL KONZEPT                                   │
│                                                                             │
│    Gegeben:  [Bild_t-2, Bild_t-1, Bild_t] + [Aktionen]                     │
│    Ziel:     Vorhersage von Bild_t+1 im Latent-Space                       │
│                                                                             │
│    Das Modell lernt die DYNAMIK der Welt:                                  │
│    "Wenn ich diese Bilder sehe und diese Aktion ausführe,                  │
│     wie wird die Welt danach aussehen?"                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Datensatz-Struktur

### 2.1 Dein Datensatz: `2026_01_13_1152_fcs_dset`

```
2026_01_13_1152_fcs_dset/
├── states.pth          # Roboter-Zustände: (10, 932, 22)
├── actions.pth         # Aktionen: (10, 932, 9)
├── metadata.pkl        # Metadaten
├── seq_lengths.pkl     # Sequenzlängen
├── cameras/            # Kamera-Konfiguration
└── 000000/ ... 000009/ # 10 Episoden
    ├── obses.pth       # RGB-Bilder: (932, 256, 256, 3)
    ├── images/         # PNG-Bilder (optional)
    └── property_params.pkl
```

### 2.2 Datensatz-Dimensionen

| Komponente | Form | Beschreibung |
|------------|------|--------------|
| **States** | `(10, 932, 22)` | 10 Episoden, 932 Timesteps, 22 State-Dimensionen |
| **Actions** | `(10, 932, 9)` | 10 Episoden, 932 Timesteps, 9 Action-Dimensionen |
| **Images** | `(932, 256, 256, 3)` | 932 RGB-Bilder pro Episode |

### 2.3 State-Vektor Aufbau (22 Dimensionen)

State = [ee_pos(3), ee_quat(4), gripper(1), joints(7), joint_vel(7)]
         ├──────────────────┘
         └── Proprio: Nur EE-Position (erste 3 Dimensionen) wird als
             "Proprioceptive Input" für das Modell verwendet

### 2.4 Action-Vektor Aufbau (9 Dimensionen)
Der Action-Vektor enthält die Roboter-Kommandos und setzt sich wie folgt zusammen:
Action = [joint_cmd(7), gripper_cmd(2)]
          ├─────────┘    ├──────────┘
          │              └── Gripper-Fingerposition (links, rechts)
          └── 7 Joint-Positionen (Soll-Werte für Gelenke 0-6)

Index	Dimension	Beschreibung	Typischer Wertebereich
0-6	joint_cmd[0:7]	Joint-Positionen (Radiant)	ca. -3.0 bis +3.0
7-8	gripper_cmd[0:2]	Gripper-Finger (links/rechts)	0.0 (geschlossen) bis 0.04 (offen)

Beispiel-Action aus deinem Datensatz:
[-0.095, -0.521, 0.047, -2.841, 0.031, 2.886, 0.842, 0.0, 0.0]
  ├──────────────────────────────────────────────────────┘  └────┘
  │                                                         Gripper
  └── 7 Joint-Sollpositionen                               (geschlossen)

Hinweis: Bei frameskip > 1 werden mehrere aufeinanderfolgende Actions konkateniert:
Mit frameskip=5: Effektive Action-Dimension = 9 × 5 = 45
Format: [action_t, action_t+1, action_t+2, action_t+3, action_t+4]
---

## 3. Konfiguration und Parameter

### 3.1 Haupt-Konfigurationsdatei: `conf/train.yaml`

```yaml
# KRITISCHE PARAMETER
frameskip: 5       # Temporales Subsampling
num_hist: 3        # Anzahl Kontext-Frames
num_pred: 1        # Anzahl Vorhersage-Frames (nur 1 unterstützt)
img_size: 224      # Bildgröße für Encoder

### Temporales Subsampling (frameskip)
Temporales Subsampling bedeutet, dass nur jeder n-te Frame aus der Originalsequenz verwendet wird, anstatt alle Frames.

# TRAINING
training:
  epochs: 100
  batch_size: 12
  seed: 0
  save_every_x_epoch: 1
  encoder_lr: 1e-6      # DINO Encoder (meist eingefroren)
  decoder_lr: 3e-4      # VQ-VAE Decoder
  predictor_lr: 5e-4    # ViT Predictor
  action_encoder_lr: 5e-4

# EMBEDDING DIMENSIONEN
action_emb_dim: 10      # Action Embedding Dimension
proprio_emb_dim: 10     # Proprio Embedding Dimension
concat_dim: 1           # Wie Embeddings kombiniert werden (0 oder 1)

# MODELL-KOMPONENTEN
model:
  train_encoder: False   # DINO wird NICHT trainiert (vortrainiert)
  train_predictor: True  # Predictor wird trainiert
  train_decoder: True    # Decoder wird trainiert
```

### 3.2 Parameter-Erklärung: `frameskip`

**Frameskip** definiert das temporale Subsampling der Daten:

Temporales Subsampling bedeutet, dass nur jeder n-te Frame aus der Originalsequenz verwendet wird, anstatt alle Frames.

Original-Aufnahme (30 FPS, 932 Frames):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │16 │17 │18 │19 │...
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Mit frameskip=5 (jeder 5. Frame):
┌───┐           ┌───┐           ┌───┐           ┌───┐
│ 0 │           │ 5 │           │10 │           │15 │  ...
└───┘           └───┘           └───┘           └───┘
  ↓               ↓               ↓               ↓
Frame 0 ──────► Frame 1 ──────► Frame 2 ──────► Frame 3  (für das Modell verwendet)

# Warum Subsampling?
Vorteil	                Erklärung
Größere Bewegung	      Zwischen Frame 0 und Frame 5 passiert mehr als zwischen Frame 0 und Frame 1 → leichter zu lernen
Weniger Redundanz	      Aufeinanderfolgende Frames sind oft fast identisch
Effektivere Aktionen	  5 Aktionen werden zu einer kombiniert → reichhaltigere Information
Längere Zeitspannen	    Mit gleicher Anzahl Frames kann mehr Zeit abgedeckt werden

**Auswirkungen:**
- **Größere visuelle Differenzen** zwischen Frames → einfacher zu lernen
- **Mehr Bewegung pro Schritt** → Modell muss größere Dynamik erfassen
- **Aktionen werden konkateniert**: 5 Aktionen → 1 kombinierte Aktion
  - `action_dim_effektiv = action_dim × frameskip = 9 × 5 = 45`

### 3.3 Parameter-Erklärung: `num_hist`
Kontext-Frames (num_hist)
Kontext-Frames sind die Anzahl der vergangenen Bilder, die dem Modell als Input gegeben werden, um die Zukunft vorherzusagen.

Beispiel mit num_hist=3, num_pred=1, frameskip=5:
Input:      [Frame0, Frame5, Frame10] + [Actions 0-14]
                    ↓ Predictor
Output:     Vorhersage für Frame15

Zeitlinie:      t=0     t=5     t=10    t=15
                 │       │        │       │
                 ▼       ▼        ▼       ▼
              ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
              │Bild │ │Bild │ │Bild │ │Bild │
              │  0  │ │  1  │ │  2  │ │  3  │
              └─────┘ └─────┘ └─────┘ └─────┘
                 │       │        │       │
                 └───────┴────────┘       │
                         │                │
                    KONTEXT (3)      VORHERSAGE (1)
                    (num_hist)        (num_pred)
                         │                │
                         ▼                ▼
              ┌──────────────────┐  ┌──────────────┐
              │ INPUT für Modell │  │ ZIEL/TARGET  │
              │ [Bild0,Bild1,    │  │   [Bild3]    │
              │  Bild2,Actions]  │  │              │
              └──────────────────┘  └──────────────┘

# Warum mehrere Kontext-Frames?
Grund	              Erklärung
Geschwindigkeit	    Aus 2+ Frames kann Bewegungsrichtung inferiert werden
Beschleunigung	    Aus 3+ Frames kann Beschleunigung erkannt werden
Verdeckungen	      Objekte, die in einem Frame verdeckt sind, können in anderen sichtbar sein
Ambiguität	        Ein einzelnes Bild kann mehrdeutig sein (steht still? bewegt sich?)

**Warum wichtig:**
- Mehr Historie = besseres Verständnis der Dynamik
- Geschwindigkeit/Beschleunigung können inferiert werden
- Trade-off: Mehr Speicher, aber bessere Vorhersagen


### Zusammenspiel beider Parameter frameskip und num_hist
Gesamte Sequenzlänge pro Sample = (num_hist + num_pred) × frameskip
                                = (3 + 1) × 5 = 20 Original-Frames

Aus deinen 932 Frames pro Episode:
├── Sample 1: Frames [0, 5, 10, 15]  → Input: [0,5,10], Target: [15]
├── Sample 2: Frames [1, 6, 11, 16]  → Input: [1,6,11], Target: [16]
├── Sample 3: Frames [2, 7, 12, 17]  → Input: [2,7,12], Target: [17]
...
└── Sample 913: Frames [912, 917, 922, 927]

= 913 Trainingssamples pro Episode

### 3.4 Action & Proprio Embedding Prozess

Die `action_emb_dim: 10` und `proprio_emb_dim: 10` entsprechen **nicht** den Rohdimensionen deiner Daten (Action: 9, Proprio: 3). Stattdessen werden die Rohdaten durch einen **lernbaren Encoder** in diese Embedding-Dimensionen transformiert.

#### Schritt 1: Frameskip-Konkatenation (nur für Actions)

Bevor die Aktionen eingebettet werden, werden sie durch den `frameskip` konkateniert:

```
Deine Original-Aktionen:     9 Dimensionen pro Frame

Mit frameskip=5:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│Action t │Action t+1│Action t+2│Action t+3│Action t+4│
│  (9)    │   (9)   │   (9)   │   (9)   │   (9)   │
└─────────┴─────────┴─────────┴─────────┴─────────┘
                         │
                         ▼ Konkatenation
              ┌─────────────────────────┐
              │   Kombinierte Action    │
              │      (9 × 5 = 45)       │
              └─────────────────────────┘
```

#### Schritt 2: Embedding durch Conv1d

Der `ProprioceptiveEmbedding`-Encoder transformiert die Rohdaten in kompakte Embeddings:

```
ACTION ENCODER:
───────────────
Input:  (Batch, Time, 45)   ← 45 = action_dim × frameskip = 9 × 5
              │
              ▼
        Conv1d(45 → 10)     ← Lernbare Projektion
              │
              ▼
Output: (Batch, Time, 10)   ← action_emb_dim


PROPRIO ENCODER:
────────────────
Input:  (Batch, Time, 3)    ← EE-Position (x, y, z)
              │
              ▼
        Conv1d(3 → 10)      ← Lernbare Projektion
              │
              ▼
Output: (Batch, Time, 10)   ← proprio_emb_dim
```

#### Warum diese Transformation?

| Aspekt | Erklärung |
|--------|-----------|
| **Dimensionsreduktion** | 45 → 10 komprimiert die Action-Information |
| **Lernbare Features** | Netzwerk lernt, welche Action-Kombinationen wichtig sind |
| **Kompatibilität** | Kleinere Embedding-Dimension passt besser zu DINO (384 dim) |
| **Regularisierung** | Verhindert Overfitting auf hochdimensionale Inputs |

#### Finale Embedding-Zusammensetzung (concat_dim=1)

Pro Patch im Latent-Space werden alle Embeddings konkateniert:

```
┌──────────────────────────────────────────────────────────────┐
│ DINO Visual (384) │ Proprio Emb (10) │ Action Emb (10) │     │
├───────────────────┼──────────────────┼─────────────────┤     │
│       384         │        10        │       10        │= 404│
└──────────────────────────────────────────────────────────────┘
```

#### Code-Referenz

```python
# Aus models/proprio.py - ProprioceptiveEmbedding:
self.patch_embed = nn.Conv1d(
    in_chans,      # 45 für Actions (9×5), 3 für Proprio
    emb_dim,       # 10 (action_emb_dim / proprio_emb_dim)
    kernel_size=1,
    stride=1
)
```

**Zusammenfassung des Datenflusses:**
```
Actions:  (B, T, 9) ──frameskip──► (B, T, 45) ──Conv1d──► (B, T, 10)
Proprio:  (B, T, 3) ─────────────────────────► Conv1d──► (B, T, 10)
```

### 3.5 Umgebungs-Konfiguration: `conf/env/franka_cube_stack.yaml`

```yaml
name: franka_cube_stack
dataset:
  _target_: "datasets.franka_cube_stack_dset.load_franka_cube_stack_slice_train_val"
  data_path: /pfad/zu/deinem/datensatz
  n_rollout: null        # null = alle Rollouts laden
  normalize_action: true # Aktionen werden z-normalisiert
  split_ratio: 0.9       # 90% Train, 10% Validation
  transform:
    _target_: "datasets.img_transforms.default_transform"
    img_size: 224        # Resize auf 224x224

num_workers: 4           # Dataloader Workers
decoder_path: null       # Optional: Vortrainierter Decoder
```

---

## 4. Training-Pipeline (Chronologisch)

### Phase 1: Initialisierung

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 1: Konfiguration laden                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Alle Hyperparameter und Pfade aus YAML-Dateien einlesen.            │
│  Hydra ermöglicht hierarchische Konfiguration und Command-Line-Overrides.   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  python train.py env=franka_cube_stack                                      │
│                                                                             │
│  → Hydra lädt: conf/train.yaml + conf/env/franka_cube_stack.yaml           │
│  → Parameter werden zusammengeführt                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 2: Trainer-Objekt erstellen                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Zentrale Klasse, die Training, Validation und Logging koordiniert.  │
│  Accelerator abstrahiert GPU/Multi-GPU und integriert Weights & Biases.     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  class Trainer:                                                             │
│      def __init__(self, cfg):                                               │
│          self.cfg = cfg                                                     │
│          self.accelerator = Accelerator(log_with="wandb")                   │
│          self.device = self.accelerator.device  # GPU                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Datensatz laden und vorbereiten

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 3: FrankaCubeStackDataset laden                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Rohdaten (Bilder, States, Actions) von der Festplatte laden.        │
│  Bilder werden in RAM gecacht für schnellen Zugriff während des Trainings.  │
│  Z-Normalisierung stabilisiert das Training durch einheitliche Wertebereiche│
│  ─────────────────────────────────────────────────────────────────────────  │
│  class FrankaCubeStackDataset:                                              │
│      def __init__(self, data_path, ...):                                    │
│          # 1. States und Actions laden                                      │
│          self.states = torch.load("states.pth")    # (10, 932, 22)         │
│          self.actions = torch.load("actions.pth")  # (10, 932, 9)          │
│                                                                             │
│          # 2. Episoden-Längen aus metadata.pkl                              │
│          self.seq_lengths = [932, 932, ..., 932]   # 10 × 932              │
│                                                                             │
│          # 3. Proprio extrahieren (EE-Position)                             │
│          self.proprios = self.states[..., :3]      # (10, 932, 3)          │
│                                                                             │
│          # 4. Z-Normalisierung (wenn normalize_action=True)                 │
│          self.actions = (self.actions - mean) / std                         │
│          self.proprios = (self.proprios - mean) / std                       │
│                                                                             │
│          # 5. Alle Bilder in RAM laden (preload_images=True)                │
│          self.images_cache = [                                              │
│              torch.load("000000/obses.pth"),  # (932, 256, 256, 3)         │
│              torch.load("000001/obses.pth"),                                │
│              ...                                                            │
│          ]                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 4: Train/Validation Split                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Episoden in Training und Validation aufteilen zur Overfitting-Kontrolle.│
│  Validation-Daten werden NIE zum Training verwendet, nur zur Evaluation.    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Mit split_ratio=0.9 und 10 Episoden:                                       │
│  - Training: 9 Episoden (zufällig ausgewählt)                               │
│  - Validation: 1 Episode                                                    │
│                                                                             │
│  split_traj_datasets(dataset, train_fraction=0.9)                           │
│  → train_set = TrajSubset(dataset, [0,1,2,3,4,5,6,7,8])  # Beispiel        │
│  → val_set = TrajSubset(dataset, [9])                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 5: TrajSlicerDataset erstellen                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Lange Episoden in kurze, überlappende Trainings-Samples schneiden.  │
│  Frameskip und num_hist bestimmen die Länge und Auflösung jedes Samples.    │
│  Shuffling der Slices verhindert, dass das Modell Sequenz-Reihenfolge lernt.│
│  ─────────────────────────────────────────────────────────────────────────  │
│  Parameter:                                                                 │
│  - num_frames = num_hist + num_pred = 3 + 1 = 4                            │
│  - frameskip = 5                                                            │
│                                                                             │
│  Für jede Episode (T=932 Frames):                                           │
│  - Benötigte Frames pro Sample: 4 × 5 = 20                                 │
│  - Mögliche Start-Positionen: 932 - 20 + 1 = 913                           │
│                                                                             │
│  Slices pro Episode:                                                        │
│  [                                                                          │
│    (episode_idx, 0, 20),    # Frames 0,5,10,15                              │
│    (episode_idx, 1, 21),    # Frames 1,6,11,16                              │
│    (episode_idx, 2, 22),    # Frames 2,7,12,17                              │
│    ...                                                                      │
│    (episode_idx, 912, 932), # Frames 912,917,922,927                        │
│  ]                                                                          │
│                                                                             │
│  GESAMT: 9 Episoden × 913 Slices = ~8.217 Training-Samples                 │
│          1 Episode × 913 Slices = ~913 Validation-Samples                   │
│                                                                             │
│  → Slices werden zufällig gemischt (shuffle)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Modelle initialisieren

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 6: DINO v2 Encoder laden                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Vortrainiertes Vision-Modell extrahiert semantische Bild-Features.  │
│  DINO wurde auf Millionen Bildern trainiert und ist hier EINGEFROREN.       │
│  Output: 256 Patch-Tokens à 384 Dimensionen pro Bild.                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  self.encoder = DinoV2Encoder(                                              │
│      name="dinov2_vits14",          # ViT-Small, Patch 14                   │
│      feature_key="x_norm_patchtokens"                                       │
│  )                                                                          │
│                                                                             │
│  # Lädt vortrainiertes Modell von Facebook                                  │
│  torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")                 │
│                                                                             │
│  Eigenschaften:                                                             │
│  - emb_dim: 384 (Feature-Dimension)                                         │
│  - patch_size: 14 (jeder 14×14 Pixel-Block = 1 Token)                      │
│  - latent_ndim: 2 (Patches sind 2D angeordnet)                              │
│                                                                             │
│  Für 224×224 Bilder:                                                        │
│  - num_patches = (224/14)² = 16² = 256 Patches                              │
│                                                                             │
│  WICHTIG: train_encoder=False → Parameter sind eingefroren!                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 7: Action & Proprio Encoder laden                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Roboter-Aktionen und Propriozeption in kompakte Embeddings wandeln. │
│  Conv1d lernt, welche Action-Kombinationen für Vorhersagen relevant sind.   │
│  Diese Encoder werden MIT trainiert (im Gegensatz zu DINO).                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  # Action Encoder                                                           │
│  self.action_encoder = ProprioceptiveEmbedding(                             │
│      in_chans=45,      # action_dim × frameskip = 9 × 5                    │
│      emb_dim=10        # action_emb_dim aus config                          │
│  )                                                                          │
│  # Verwendet 1D Convolution: Conv1d(45 → 10)                                │
│                                                                             │
│  # Proprio Encoder                                                          │
│  self.proprio_encoder = ProprioceptiveEmbedding(                            │
│      in_chans=3,       # proprio_dim (EE-Position x,y,z)                    │
│      emb_dim=10        # proprio_emb_dim aus config                         │
│  )                                                                          │
│  # Verwendet 1D Convolution: Conv1d(3 → 10)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 8: ViT Predictor laden                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Kernkomponente - lernt zukünftige Zustände im Latent-Space vorherzusagen.│
│  Kausale Maske verhindert, dass das Modell in die Zukunft "schaut".         │
│  6 Transformer-Blöcke mit 16 Attention-Heads für komplexe Dynamik-Modellierung.│
│  ─────────────────────────────────────────────────────────────────────────  │
│  self.predictor = ViTPredictor(                                             │
│      num_patches=198,  # 196 visual + 2 (proprio + action bei concat_dim=0) │
│                        # oder 196 bei concat_dim=1                          │
│      num_frames=3,     # num_hist                                           │
│      dim=404,          # 384 (DINO) + 10 (action) + 10 (proprio)           │
│      depth=6,          # 6 Transformer-Blöcke                               │
│      heads=16,         # 16 Attention-Heads                                 │
│      mlp_dim=2048,     # Feed-Forward Dimension                             │
│      dropout=0.1                                                            │
│  )                                                                          │
│                                                                             │
│  # Verwendet KAUSALE ATTENTION MASK                                         │
│  # → Kann nur vergangene Frames sehen, nicht zukünftige                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 9: VQ-VAE Decoder laden                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Rekonstruiert Bilder aus dem Latent-Space zur Visualisierung.       │
│  Upsampling von 16×16 auf 224×224 durch transponierte Convolutions.         │
│  Quantisierung ist deaktiviert für kontinuierlichen, glatten Latent-Space.  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  self.decoder = VQVAE(                                                      │
│      channel=384,       # Entspricht DINO emb_dim                           │
│      n_embed=2048,      # Codebook-Größe (nicht verwendet wenn quantize=F)  │
│      n_res_block=4,     # Residual Blocks                                   │
│      n_res_channel=128,                                                     │
│      quantize=False     # KEINE Quantisierung (kontinuierlicher Latent)    │
│  )                                                                          │
│                                                                             │
│  # Architektur:                                                             │
│  # Latent (14×14×384) → Upsample (4×) → 56×56 → Upsample (4×) → 224×224×3  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 10: VWorldModel zusammensetzen                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Alle Komponenten zu einem einheitlichen World Model verbinden.      │
│  Definiert den Datenfluss: Encode → Concatenate → Predict → Decode.         │
│  concat_dim=1 bedeutet, dass Embeddings entlang der Feature-Dimension kombiniert werden.│
│  ─────────────────────────────────────────────────────────────────────────  │
│  self.model = VWorldModel(                                                  │
│      encoder=self.encoder,                                                  │
│      proprio_encoder=self.proprio_encoder,                                  │
│      action_encoder=self.action_encoder,                                    │
│      predictor=self.predictor,                                              │
│      decoder=self.decoder,                                                  │
│      num_hist=3,                                                            │
│      num_pred=1,                                                            │
│      concat_dim=1  # Embeddings werden entlang Feature-Dimension konkateniert│
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Training-Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 11: Optimizer initialisieren                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Separate Optimizer für jede Komponente mit unterschiedlichen Lernraten.│
│  AdamW für Predictor (mit Weight Decay), Adam für Encoder/Decoder.          │
│  Niedrige Encoder-LR (1e-6) da DINO-Weights meist eingefroren bleiben.      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  self.encoder_optimizer = Adam(encoder.parameters(), lr=1e-6)               │
│  self.predictor_optimizer = AdamW(predictor.parameters(), lr=5e-4)          │
│  self.decoder_optimizer = Adam(decoder.parameters(), lr=3e-4)               │
│  self.action_encoder_optimizer = AdamW(                                     │
│      [action_encoder.params, proprio_encoder.params], lr=5e-4               │
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 12: Training-Epoch                                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Hauptschleife - iteriert über alle Batches und aktualisiert Gewichte.│
│  Forward Pass berechnet Vorhersagen, Backward Pass berechnet Gradienten.    │
│  Nur trainierbare Komponenten (Predictor, Decoder, Action-Encoder) werden aktualisiert.│
│  ─────────────────────────────────────────────────────────────────────────  │
│  for epoch in range(1, 101):  # 100 Epochen                                │
│      for batch in dataloader:                                               │
│          obs, act, state = batch                                            │
│          │                                                                  │
│          │  obs['visual']: (B, 4, 3, 224, 224) - 4 Bilder                  │
│          │  obs['proprio']: (B, 4, 3) - 4 EE-Positionen                    │
│          │  act: (B, 4, 45) - 4 × (9×5) konkatenierte Aktionen             │
│          │  state: (B, 4, 22) - 4 vollständige States                      │
│          ▼                                                                  │
│      ┌─────────────────────────────────────────────────────────────────┐   │
│      │  FORWARD PASS (siehe nächstes Diagramm)                         │   │
│      │  z_pred, visual_pred, visual_recon, loss = model(obs, act)      │   │
│      └─────────────────────────────────────────────────────────────────┘   │
│          │                                                                  │
│          ▼                                                                  │
│      ┌─────────────────────────────────────────────────────────────────┐   │
│      │  BACKWARD PASS                                                   │   │
│      │  1. encoder_optimizer.zero_grad()                               │   │
│      │  2. predictor_optimizer.zero_grad()                             │   │
│      │  3. decoder_optimizer.zero_grad()                               │   │
│      │  4. action_encoder_optimizer.zero_grad()                        │   │
│      │                                                                  │   │
│      │  accelerator.backward(loss)  # Gradient berechnen               │   │
│      │                                                                  │   │
│      │  # NUR trainierbare Komponenten updaten:                        │   │
│      │  predictor_optimizer.step()      # ✓ train_predictor=True       │   │
│      │  decoder_optimizer.step()         # ✓ train_decoder=True        │   │
│      │  action_encoder_optimizer.step()  # ✓ immer                     │   │
│      │  # encoder_optimizer.step()      # ✗ train_encoder=False       │   │
│      └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Forward Pass im Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 13: Forward Pass - Encoding                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Rohdaten (Bilder, Proprio, Actions) in einheitliche Embeddings wandeln.│
│  DINO transformiert Bilder in 256 semantische Patch-Tokens.                 │
│  Action/Proprio-Encoder komprimieren Sensordaten in 10-dimensionale Vektoren.│
│  ─────────────────────────────────────────────────────────────────────────  │
│  INPUT:                                                                     │
│  ───────                                                                    │
│  obs['visual']: (B, 4, 3, 224, 224)   # 4 RGB Bilder                       │
│  obs['proprio']: (B, 4, 3)            # 4 EE-Positionen (x,y,z)            │
│  act: (B, 4, 45)                      # 4 konkatenierte Aktionen           │
│                                                                             │
│                                                                             │
│  VISUAL ENCODING (DINO v2):                                                 │
│  ──────────────────────────                                                 │
│  1. Bilder reshapen: (B, 4, 3, 224, 224) → (B×4, 3, 224, 224)              │
│  2. Optional resize auf encoder_image_size (für DINO patch alignment)       │
│  3. DINO forward: (B×4, 3, 224, 224) → (B×4, 256, 384)                     │
│                   [batch×time, num_patches, emb_dim]                        │
│  4. Reshape zurück: (B×4, 256, 384) → (B, 4, 256, 384)                     │
│                                                                             │
│  z_visual: (B, 4, 256, 384)                                                 │
│            ↑    ↑    ↑                                                      │
│         batch time patches                                                  │
│                                                                             │
│                                                                             │
│  PROPRIO ENCODING:                                                          │
│  ─────────────────                                                          │
│  proprio_encoder(obs['proprio'])                                            │
│  (B, 4, 3) → Conv1d → (B, 4, 10)                                           │
│                                                                             │
│  z_proprio: (B, 4, 10)                                                      │
│                                                                             │
│                                                                             │
│  ACTION ENCODING:                                                           │
│  ────────────────                                                           │
│  action_encoder(act)                                                        │
│  (B, 4, 45) → Conv1d → (B, 4, 10)                                          │
│                                                                             │
│  z_action: (B, 4, 10)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 14: Forward Pass - Concatenation (concat_dim=1)                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Visuelle, propriozeptive und Action-Information zu einem Vektor vereinen.│
│  Tiling repliziert Proprio/Action auf alle 256 Patches für einheitliche Dim.│
│  Ergebnis: Jeder Patch enthält visuelle + Roboter-Information (404 dim).    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Mit concat_dim=1: Embeddings werden entlang Feature-Dimension konkateniert │
│                                                                             │
│  z_visual:  (B, 4, 256, 384)  # 256 Patches × 384 dim                      │
│  z_proprio: (B, 4, 10)        # → tile auf (B, 4, 256, 10)                 │
│  z_action:  (B, 4, 10)        # → tile auf (B, 4, 256, 10)                 │
│                                                                             │
│  Konkatenation:                                                             │
│  z = concat([z_visual, z_proprio_tiled, z_action_tiled], dim=-1)           │
│                                                                             │
│  z: (B, 4, 256, 384+10+10) = (B, 4, 256, 404)                              │
│                                                                             │
│  Visualisierung eines Patches:                                              │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ DINO Features (384) │ Proprio (10) │ Action (10) │ = 404 dim │          │
│  └──────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 15: Forward Pass - Prediction                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Transformer sagt basierend auf Historie den nächsten Zustand vorher.│
│  Kausale Maske: Frame 2 kann Frame 0,1,2 sehen, aber nicht Frame 3.         │
│  Target ist um 1 Zeitschritt verschoben - Modell lernt "was kommt als nächstes".│
│  ─────────────────────────────────────────────────────────────────────────  │
│  Source (Input für Predictor):                                              │
│  z_src = z[:, :num_hist]   = z[:, :3]   # Erste 3 Zeitschritte             │
│  z_src: (B, 3, 256, 404)                                                    │
│                                                                             │
│  Target (Ground Truth):                                                     │
│  z_tgt = z[:, num_pred:]   = z[:, 1:]   # Letzte 3 Zeitschritte            │
│  z_tgt: (B, 3, 256, 404)   # Zeitlich um 1 verschoben                      │
│                                                                             │
│  ViT Predictor:                                                             │
│  ─────────────                                                              │
│  1. Reshape: (B, 3, 256, 404) → (B, 768, 404)  # 3×256 = 768 Tokens        │
│  2. Position Embedding addieren                                             │
│  3. 6× Transformer Blocks mit KAUSALER MASKE                               │
│  4. Output: (B, 768, 404) → (B, 3, 256, 404)                               │
│                                                                             │
│  z_pred: (B, 3, 256, 404)                                                   │
│                                                                             │
│  Kausale Maske Visualisierung:                                              │
│  ┌─────────────────────────────────────────┐                               │
│  │     Frame 0   Frame 1   Frame 2         │                               │
│  │  F0   ✓         ✗         ✗             │  ← Kann nur sich selbst sehen │
│  │  F1   ✓         ✓         ✗             │  ← Kann F0 und sich sehen     │
│  │  F2   ✓         ✓         ✓             │  ← Kann alle sehen            │
│  └─────────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 16: Forward Pass - Decoding                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Latent-Space Vorhersagen zurück in Pixel-Bilder wandeln.            │
│  Ermöglicht visuelle Inspektion der Vorhersagequalität.                     │
│  Decoder trainiert auf Rekonstruktion, nicht auf Vorhersage (detached).     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Separate Embeddings:                                                       │
│  z_pred: (B, 3, 256, 404)                                                   │
│      ↓                                                                      │
│  z_visual_pred: (B, 3, 256, 384)  # Nur DINO Features                      │
│  z_proprio_pred: (B, 3, 10)        # Proprio (nicht decodiert)             │
│  z_action_pred: (B, 3, 10)         # Action (nicht decodiert)              │
│                                                                             │
│  VQ-VAE Decoder:                                                            │
│  ───────────────                                                            │
│  1. Reshape: (B, 3, 256, 384) → (B×3, 16, 16, 384)  # √256 = 16           │
│  2. Permute: (B×3, 16, 16, 384) → (B×3, 384, 16, 16)                       │
│  3. Upsample 4×: (B×3, 384, 16, 16) → (B×3, 384, 64, 64)                   │
│  4. Decode: (B×3, 384, 64, 64) → (B×3, 3, 224, 224)                        │
│  5. Reshape: (B×3, 3, 224, 224) → (B, 3, 3, 224, 224)                      │
│                                                                             │
│  visual_pred: (B, 3, 3, 224, 224)  # Vorhersagte Bilder                    │
│                                                                             │
│  Zusätzlich: Rekonstruktion der Originalen                                  │
│  visual_recon: (B, 4, 3, 224, 224)  # Alle 4 Frames rekonstruiert          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Loss-Berechnung

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 17: Loss-Berechnung                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Fehler zwischen Vorhersage und Ground Truth quantifizieren.         │
│  z_loss trainiert den Predictor im kompakten Latent-Space (robust).         │
│  decoder_loss trainiert den Decoder für gute Bildrekonstruktion.            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LATENT SPACE LOSS (z_loss)                                          │   │
│  │  ─────────────────────────                                           │   │
│  │  z_pred: (B, 3, 256, 404) - Vorhersage                              │   │
│  │  z_tgt:  (B, 3, 256, 404) - Ground Truth (detached)                 │   │
│  │                                                                      │   │
│  │  z_visual_loss = MSE(z_pred[..., :384], z_tgt[..., :384])           │   │
│  │  z_proprio_loss = MSE(z_pred[..., 384:394], z_tgt[..., 384:394])    │   │
│  │  z_loss = MSE(z_pred[..., :394], z_tgt[..., :394])  # ohne Action   │   │
│  │                                                                      │   │
│  │  Gewichtung: 1.0                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    +                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DECODER LOSS (Reconstruction)                                       │   │
│  │  ─────────────────────────────                                       │   │
│  │  visual_recon: (B, 4, 3, 224, 224) - Rekonstruierte Bilder          │   │
│  │  obs['visual']: (B, 4, 3, 224, 224) - Original Bilder               │   │
│  │                                                                      │   │
│  │  recon_loss = MSE(visual_recon, obs['visual'])                      │   │
│  │  vq_loss = 0 (da quantize=False)                                    │   │
│  │  decoder_loss = recon_loss + 0.25 × vq_loss                         │   │
│  │                                                                      │   │
│  │  Gewichtung: 1.0                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    +                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DECODER PREDICTION LOSS (Optional)                                  │   │
│  │  ──────────────────────────────────                                  │   │
│  │  visual_pred: (B, 3, 3, 224, 224) - Vorhersagte Bilder              │   │
│  │  obs['visual'][:, 1:]: (B, 3, 3, 224, 224) - Ground Truth           │   │
│  │                                                                      │   │
│  │  pred_recon_loss = MSE(visual_pred, obs['visual'][:, 1:])           │   │
│  │  (Dieser Loss wird geloggt aber nicht zum Training verwendet)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    =                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TOTAL LOSS                                                          │   │
│  │  ──────────                                                          │   │
│  │  loss = z_loss + decoder_loss                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 7: Validation und Logging

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 18: Validation                                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Modell auf ungesehenen Daten evaluieren zur Overfitting-Erkennung.  │
│  Open-Loop Rollout testet autoregressive Vorhersage über mehrere Schritte.  │
│  model.eval() deaktiviert Dropout und Batch-Normalisierung für Konsistenz.  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  model.eval()  # Keine Gradientenberechnung                                │
│                                                                             │
│  1. Standard Validation (wie Training, aber ohne Optimizer-Steps)          │
│  2. Open-Loop Rollout:                                                      │
│     - Nimm erste num_hist Frames                                           │
│     - Sage iterativ zukünftige Frames vorher                               │
│     - Vergleiche mit Ground Truth                                          │
│                                                                             │
│  Rollout-Visualisierung:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  t=0   t=1   t=2   t=3   t=4   t=5   ...                            │   │
│  │                                                                      │   │
│  │  GT:   [F0]  [F1]  [F2]  [F3]  [F4]  [F5]  ...                      │   │
│  │                                                                      │   │
│  │  Pred: [F0]  [F1]  [F2]  [P3]  [P4]  [P5]  ...                      │   │
│  │         ↑     ↑     ↑     ↑                                         │   │
│  │       Input Input Input Vorhersage (autoregressiv)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 19: Logging zu Weights & Biases                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Training-Fortschritt visualisieren und Experimente vergleichen.     │
│  Loss-Kurven zeigen Konvergenz, Bild-Metriken (PSNR/SSIM) zeigen Qualität.  │
│  Visualisierungen helfen, Fehlerquellen schnell zu identifizieren.          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Geloggte Metriken:                                                         │
│  - train_loss, val_loss                                                     │
│  - train_z_loss, val_z_loss                                                 │
│  - train_z_visual_loss, val_z_visual_loss                                   │
│  - train_z_proprio_loss, val_z_proprio_loss                                 │
│  - train_decoder_recon_loss, val_decoder_recon_loss                         │
│  - z_visual_err_rollout, z_proprio_err_rollout                              │
│  - Image Quality Metrics (PSNR, SSIM, etc.)                                 │
│                                                                             │
│  Visualisierungen:                                                          │
│  - Rekonstruierte Bilder vs. Ground Truth                                   │
│  - Vorhersage-Sequenzen                                                     │
│  - Rollout-Plots                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 20: Checkpoint speichern                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Zweck: Modell-Zustand sichern für späteres Laden oder Inferenz.            │
│  Speichert alle Weights + Optimizer-States für nahtlose Fortsetzung.        │
│  model_latest.pth wird bei jedem Save überschrieben, model_N.pth bleibt.    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  if epoch % save_every_x_epoch == 0:                                        │
│      torch.save({                                                           │
│          'epoch': epoch,                                                    │
│          'encoder': encoder.state_dict(),         # DINO Weights           │
│          'predictor': predictor.state_dict(),     # ViT Weights            │
│          'decoder': decoder.state_dict(),         # VQ-VAE Weights         │
│          'action_encoder': action_encoder.state_dict(),                     │
│          'proprio_encoder': proprio_encoder.state_dict(),                   │
│          'encoder_optimizer': ...,                                          │
│          'predictor_optimizer': ...,                                        │
│          'decoder_optimizer': ...,                                          │
│      }, f"checkpoints/model_{epoch}.pth")                                   │
│                                                                             │
│  Gespeichert in: outputs/DATUM/ZEIT/checkpoints/                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Modell-Architektur

### 5.1 DINO v2 Encoder

**Was ist DINO?**
DINO (Self-**DI**stillation with **NO** labels) ist ein selbstüberwachtes Vision-Modell von Meta/Facebook, das ohne Labels trainiert wurde.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DINO v2 ViT-Small/14                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: RGB Bild (3, 224, 224)                                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Patch Embedding                                                     │   │
│  │  - Bild in 14×14 Patches aufteilen                                  │   │
│  │  - 224/14 = 16 Patches pro Seite                                    │   │
│  │  - 16 × 16 = 256 Patches total                                      │   │
│  │  - Jeder Patch: 14×14×3 = 588 Pixel → Linear → 384 dim              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Transformer Encoder (12 Blöcke)                                     │   │
│  │  - Self-Attention über alle 256 Patches                             │   │
│  │  - Lernt räumliche Beziehungen                                      │   │
│  │  - Vortrainiert auf ImageNet (ohne Labels!)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  Output: Patch Tokens (256, 384)                                            │
│          [num_patches, emb_dim]                                             │
│                                                                             │
│  Eigenschaften:                                                             │
│  - Vortrainiert: Parameter werden NICHT verändert                          │
│  - Semantic Features: Lernt bedeutungsvolle visuelle Repräsentationen      │
│  - Patch-basiert: Erhält räumliche Information                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Warum DINO?**
- Starke, generalisierbare visuelle Features
- Funktioniert gut auf Robotik-Domäne (obwohl auf natürliche Bilder trainiert)
- Keine zusätzlichen Labels nötig

### 5.2 Action & Proprio Encoder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Proprioceptive Embedding (MLP)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Action Encoder:                                                            │
│  ───────────────                                                            │
│  Input: (B, T, 45)     # 45 = 9 actions × 5 frameskip                      │
│           │                                                                 │
│           ▼                                                                 │
│  Conv1d(45 → 10, kernel=1, stride=1)                                       │
│           │                                                                 │
│           ▼                                                                 │
│  Output: (B, T, 10)    # Komprimierte Action-Repräsentation                │
│                                                                             │
│                                                                             │
│  Proprio Encoder:                                                           │
│  ────────────────                                                           │
│  Input: (B, T, 3)      # EE-Position (x, y, z)                             │
│           │                                                                 │
│           ▼                                                                 │
│  Conv1d(3 → 10, kernel=1, stride=1)                                        │
│           │                                                                 │
│           ▼                                                                 │
│  Output: (B, T, 10)    # Komprimierte Proprio-Repräsentation               │
│                                                                             │
│  Zweck:                                                                     │
│  - Dimensionsreduktion für effiziente Verarbeitung                         │
│  - Lernt relevante Features aus rohen Sensor-Daten                         │
│  - Wird MIT trainiert (im Gegensatz zu DINO)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 ViT Predictor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Vision Transformer Predictor                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: z (B, 3, 256, 404)  # 3 Frames × 256 Patches × 404 dim             │
│           │                                                                 │
│           ▼                                                                 │
│  Reshape: (B, 768, 404)     # 3×256 = 768 Tokens                           │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Positional Embedding                                                │   │
│  │  - Lernbares Embedding: (1, 768, 404)                               │   │
│  │  - Addiert zu Input                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  6× Transformer Block                                                │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Multi-Head Attention (16 Heads)                              │  │   │
│  │  │  - KAUSAL: Kann nur vergangene Tokens sehen                   │  │   │
│  │  │  - Query, Key, Value: Linear(404 → 64×16)                     │  │   │
│  │  │  - Output: Linear(64×16 → 404)                                │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │            │ + Residual                                              │   │
│  │            ▼                                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Feed-Forward Network                                         │  │   │
│  │  │  Linear(404 → 2048) → GELU → Linear(2048 → 404)              │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │            │ + Residual                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  LayerNorm                                                                  │
│           │                                                                 │
│           ▼                                                                 │
│  Reshape: (B, 768, 404) → (B, 3, 256, 404)                                 │
│           │                                                                 │
│           ▼                                                                 │
│  Output: z_pred (B, 3, 256, 404)                                            │
│                                                                             │
│  Kausale Maske (Visualisierung für 3 Frames × 2 Patches):                  │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │        P0_F0  P1_F0  P0_F1  P1_F1  P0_F2  P1_F2    │                   │
│  │ P0_F0    1      1      0      0      0      0      │                   │
│  │ P1_F0    1      1      0      0      0      0      │                   │
│  │ P0_F1    1      1      1      1      0      0      │                   │
│  │ P1_F1    1      1      1      1      0      0      │                   │
│  │ P0_F2    1      1      1      1      1      1      │                   │
│  │ P1_F2    1      1      1      1      1      1      │                   │
│  └─────────────────────────────────────────────────────┘                   │
│  (1 = kann sehen, 0 = maskiert)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 VQ-VAE Decoder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VQ-VAE Decoder                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: z_visual (B, T, 256, 384)   # Nur visuelle Features                │
│           │                                                                 │
│           ▼                                                                 │
│  Reshape: (B×T, 16, 16, 384)        # √256 = 16                            │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Quantize (DEAKTIVIERT: quantize=False)                             │   │
│  │  - Bei aktiviert: Diskretisierung in Codebook                       │   │
│  │  - Hier: Kontinuierlicher Latent-Space                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  Permute: (B×T, 384, 16, 16)        # Channel-first für Conv              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Upsample Block (4×)                                                 │   │
│  │  Conv2d(384, 384) + 4× ResBlock + ConvTranspose2d(stride=2) ×2      │   │
│  │  (16, 16) → (32, 32) → (64, 64)                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  (B×T, 384, 64, 64)                                                         │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Decode Block (4×)                                                   │   │
│  │  Conv2d(384, 384) + 4× ResBlock + ConvTranspose2d(stride=2) ×2      │   │
│  │  (64, 64) → (128, 128) → (256, 256) → Resize → (224, 224)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  Output: (B×T, 3, 224, 224)         # RGB Bilder                           │
│           │                                                                 │
│           ▼                                                                 │
│  Reshape: (B, T, 3, 224, 224)                                               │
│                                                                             │
│  ResBlock Architektur:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input ─────────────────────────────────────┐                       │   │
│  │    │                                        │                       │   │
│  │    ▼                                        │                       │   │
│  │  ReLU → Conv3×3 → ReLU → Conv1×1 ──────────(+)──→ Output           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Gesamtarchitektur: VWorldModel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Visual World Model                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                           INPUT                                     │    │
│  │  obs['visual']: (B, 4, 3, 224, 224)                                │    │
│  │  obs['proprio']: (B, 4, 3)                                         │    │
│  │  act: (B, 4, 45)                                                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                    │              │              │                          │
│                    ▼              ▼              ▼                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   DINO Encoder   │  │  Proprio Encoder │  │  Action Encoder  │          │
│  │   (FROZEN)       │  │  (TRAINABLE)     │  │  (TRAINABLE)     │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│          │                     │                      │                     │
│          ▼                     ▼                      ▼                     │
│    (B, 4, 256, 384)      (B, 4, 10)            (B, 4, 10)                   │
│          │                     │                      │                     │
│          └─────────────────────┼──────────────────────┘                     │
│                                │                                            │
│                                ▼                                            │
│                    ┌──────────────────────┐                                │
│                    │     CONCATENATE      │                                │
│                    │   (concat_dim=1)     │                                │
│                    └──────────────────────┘                                │
│                                │                                            │
│                                ▼                                            │
│                         (B, 4, 256, 404)                                    │
│                                │                                            │
│              ┌─────────────────┴─────────────────┐                          │
│              │                                   │                          │
│              ▼                                   ▼                          │
│       z_src[:, :3]                        z_tgt[:, 1:]                      │
│     (Historie)                           (Target)                           │
│              │                                   │                          │
│              ▼                                   │                          │
│  ┌──────────────────────┐                       │                          │
│  │    ViT Predictor     │                       │                          │
│  │    (TRAINABLE)       │                       │                          │
│  └──────────────────────┘                       │                          │
│              │                                   │                          │
│              ▼                                   │                          │
│        z_pred                                    │                          │
│     (B, 3, 256, 404)                            │                          │
│              │                                   │                          │
│              └───────────────┬───────────────────┘                          │
│                              │                                              │
│                              ▼                                              │
│                    ┌──────────────────┐                                    │
│                    │   LATENT LOSS    │                                    │
│                    │  MSE(z_pred,     │                                    │
│                    │      z_tgt)      │                                    │
│                    └──────────────────┘                                    │
│                              │                                              │
│              ┌───────────────┴───────────────┐                              │
│              │                               │                              │
│              ▼                               ▼                              │
│  ┌──────────────────────┐        ┌──────────────────────┐                  │
│  │   VQ-VAE Decoder     │        │   VQ-VAE Decoder     │                  │
│  │   (TRAINABLE)        │        │   (TRAINABLE)        │                  │
│  │   auf z_pred         │        │   auf z (alle)       │                  │
│  └──────────────────────┘        └──────────────────────┘                  │
│              │                               │                              │
│              ▼                               ▼                              │
│      visual_pred                     visual_recon                          │
│    (B, 3, 3, 224, 224)            (B, 4, 3, 224, 224)                      │
│              │                               │                              │
│              │                               ▼                              │
│              │                    ┌──────────────────┐                      │
│              │                    │  DECODER LOSS    │                      │
│              │                    │  MSE(recon, gt)  │                      │
│              │                    └──────────────────┘                      │
│              │                               │                              │
│              └───────────────────────────────┘                              │
│                              │                                              │
│                              ▼                                              │
│                    ┌──────────────────┐                                    │
│                    │    TOTAL LOSS    │                                    │
│                    │ z_loss + decoder │                                    │
│                    └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Loss-Funktionen

### 6.1 Übersicht aller Losses

| Loss Name | Formel | Gewichtung | Zweck |
|-----------|--------|------------|-------|
| `z_loss` | MSE(z_pred, z_tgt) | 1.0 | Hauptloss für Predictor |
| `z_visual_loss` | MSE(z_pred_visual, z_tgt_visual) | (geloggt) | Nur visuelle Features |
| `z_proprio_loss` | MSE(z_pred_proprio, z_tgt_proprio) | (geloggt) | Nur Proprio-Features |
| `decoder_recon_loss` | MSE(visual_recon, obs_visual) | 1.0 | Rekonstruktionsqualität |
| `decoder_vq_loss` | Commitment Loss | 0.25 | VQ Regularisierung (=0 wenn quantize=False) |
| `decoder_loss` | recon + 0.25×vq | 1.0 | Decoder-Training |

### 6.2 Warum diese Kombination?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ZWEI-STUFEN TRAINING-STRATEGIE                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. LATENT LOSS (z_loss):                                                   │
│     - Trainiert den Predictor im kompakten Latent-Space                    │
│     - Weniger anfällig für Pixel-Level Noise                               │
│     - Fokussiert auf semantische Vorhersage                                │
│                                                                             │
│  2. DECODER LOSS (decoder_loss):                                            │
│     - Trainiert den Decoder zur Bildrekonstruktion                         │
│     - Stellt sicher, dass Latent-Space interpretierbar bleibt             │
│     - Ermöglicht Visualisierung der Vorhersagen                            │
│                                                                             │
│  WICHTIG: Decoder wird auf z.detach() trainiert                            │
│           → Decoder-Gradients fließen NICHT zum Predictor                  │
│           → Verhindert, dass Decoder den Predictor "betrügt"               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Training starten

### 7.1 Basis-Kommando

```bash
cd /media/tsp_jw/fc8bca1b-cab8-4522-81d0-06172d2beae8/dino_wm2

# Standard-Training
python train.py env=franka_cube_stack

# Mit expliziten Parametern
python train.py env=franka_cube_stack \
    frameskip=5 \
    num_hist=3 \
    training.epochs=100 \
    training.batch_size=8
```

### 7.2 Empfohlene Parameter für deinen Datensatz

Da du nur 10 Episoden hast, hier optimierte Einstellungen:

```bash
python train.py env=franka_cube_stack \
    frameskip=3 \                    # Feinere Dynamik
    num_hist=3 \                     # Standard Kontext
    training.epochs=200 \            # Mehr Epochen (kleiner Datensatz)
    training.batch_size=8 \          # Kleinere Batch-Size
    training.predictor_lr=3e-4 \     # Etwas niedriger
    training.decoder_lr=2e-4 \       # Etwas niedriger
    debug=True                       # Wandb Debug-Projekt
```

### 7.3 Erwartete Ausgabe

```
outputs/
└── 2026-01-13/
    └── 15-30-45/                    # Zeitstempel
        ├── checkpoints/
        │   ├── model_latest.pth
        │   ├── model_1.pth
        │   ├── model_2.pth
        │   └── ...
        ├── train/
        │   └── train_e00001_b0.png  # Visualisierungen
        ├── valid/
        │   └── valid_e00001_b0.png
        ├── rollout_plots/
        │   └── e1_rollout/
        └── hydra.yaml               # Gespeicherte Konfiguration
```

### 7.4 Monitoring mit Weights & Biases

Training wird automatisch zu W&B geloggt:
- Projekt: `dino_wm_debug` (wenn `debug=True`) oder `dino_wm`
- Metriken: Loss-Kurven, Image Metrics, Visualisierungen

---

## 8. Glossar

| Begriff | Erklärung |
|---------|-----------|
| **DINO** | Self-Distillation with No Labels - vortrainiertes Vision-Modell |
| **ViT** | Vision Transformer - Transformer-Architektur für Bilder |
| **VQ-VAE** | Vector Quantized Variational Autoencoder - generatives Modell |
| **Patch** | Bildausschnitt (14×14 Pixel bei DINO) |
| **Embedding** | Kompakte Vektorrepräsentation |
| **Latent Space** | Komprimierter Repräsentationsraum |
| **Frameskip** | Temporales Subsampling - jeder n-te Frame wird verwendet |
| **num_hist** | Anzahl Kontext-Frames als Input |
| **num_pred** | Anzahl vorherzusagender zukünftiger Frames |
| **Causal Mask** | Verhindert, dass Modell zukünftige Frames sieht |
| **MSE** | Mean Squared Error - quadratischer Fehler |
| **Proprio** | Proprioceptive Daten - Roboter-Eigenwahrnehmung (z.B. Gelenkwinkel) |
| **Accelerator** | HuggingFace Tool für verteiltes Training |

---

## 9. Troubleshooting

### 9.1 Training Freeze / Deadlock (kein Temperatur-Problem)

**Symptome:**
- Training stoppt ohne Fehlermeldung
- GPU-Temperatur ist normal (~48°C)
- `ps aux | grep train.py` zeigt Zombie-Prozess `[python] <defunct>`
- Mehrere DataLoader-Worker-Prozesse hängen

**Ursache:** PyTorch DataLoader Multiprocessing Deadlock
- `num_workers > 0` kann bei `torch.load()` in Subprozessen deadlocken
- Bekanntes PyTorch-Issue bei großen Tensoren

**Lösung 1: num_workers auf 0 setzen**
```yaml
# conf/env/franka_cube_stack.yaml
num_workers: 0  # Deaktiviert Multiprocessing - langsamer aber stabil
```

**Lösung 2: preload_images aktivieren (bereits Standard)**
```python
# In FrankaCubeStackDataset - lädt alle Bilder beim Init in RAM
preload_images=True  # Verhindert torch.load() in Worker-Prozessen
```

**Lösung 3: Debugging aktivieren**
```bash
CUDA_LAUNCH_BLOCKING=1 python train.py env=franka_cube_stack
```

### 9.2 GPU Out of Memory (OOM)

**Symptome:**
- `RuntimeError: CUDA out of memory`
- Training crasht beim ersten Batch

**Lösung:**
```bash
# Batch-Size reduzieren
python train.py env=franka_cube_stack training.batch_size=8

# Oder num_hist reduzieren
python train.py env=franka_cube_stack num_hist=2
```

### 9.3 GPU Thermal Throttling

**Symptome:**
- Training wird langsamer über Zeit
- GPU-Temperatur >80°C
- `nvidia-smi` zeigt reduzierte Taktrate

**Lösung:**
```bash
# Power Limit reduzieren
sudo nvidia-smi -pl 100

# Lüftersteuerung mit GreenWithEnvy
flatpak run com.leinardi.gwe
```

---

## Anhang: Datensatz-Statistiken

Für deinen Datensatz `2026_01_13_1152_fcs_dset`:

| Metrik | Wert |
|--------|------|
| Episoden | 10 |
| Frames pro Episode | 932 |
| Gesamtframes | 9.320 |
| State-Dimension | 22 |
| Action-Dimension | 9 |
| Bildgröße | 256×256 → resize auf 224×224 |
| Training-Samples (frameskip=5, num_hist=3) | ~8.217 |
| Validation-Samples | ~913 |
| Speicherbedarf (Bilder) | ~10 × 932 × 256 × 256 × 3 ≈ 1.8 GB |

---

*Dokumentation erstellt am 13.01.2026*

