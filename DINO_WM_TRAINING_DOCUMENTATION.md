# 🧠 DINO World Model - Vollständige Training-Dokumentation

> Eine detaillierte, chronologische Dokumentation des gesamten Trainingsprozesses für das DINO World Model mit dem Franka Cube Stacking Datensatz.

---

## 📑 Inhaltsverzeichnis

1. [Überblick und Konzept](#1-überblick-und-konzept)
2. [Datensatz-Struktur](#2-datensatz-struktur)
3. [Konfiguration und Parameter](#3-konfiguration-und-parameter)
4. [Training-Pipeline (Chronologisch)](#4-training-pipeline-chronologisch)
5. [Modell-Architektur](#5-modell-architektur)
6. [Proprioceptive Encoder — Vollständiger Trainingsablauf](#6-proprioceptive-encoder--vollständiger-trainingsablauf)
7. [Loss-Funktionen](#7-loss-funktionen)
8. [W&B Metriken und Monitoring](#8-wb-metriken-und-monitoring)
9. [Training starten](#9-training-starten)
10. [Glossar](#10-glossar)
11. [🚨 KRITISCH: Action-Observation Temporale Alignment-Analyse (20.02.2026)](#-kritisch-action-observation-temporale-alignment-analyse-20022026)

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

### 2.1 Aktueller Datensatz (Primitiv-basiert)

**Aktuell:** `NEps1000_RobOpac0_NPrim20_NCams4_NCube1` (985 Episoden, 20 Primitive/Timesteps)

```
NEps1000_RobOpac0_NPrim20_NCams4_NCube1/
├── states.pth          # Würfelpositionen: (985, 20, N_cubes*4)
├── actions.pth         # Aktionen: (985, 20, 8)  ← 8D mit Gripper
├── metadata.pkl        # Metadaten
├── seq_lengths.pkl     # Sequenzlängen pro Episode
├── cameras/            # Kamera-Konfiguration
└── 000000/ ... 000984/ # 985 Episoden
    ├── obses.pth       # RGB-Bilder: (20, 256, 256, 3) uint8
    ├── 00.h5 ... 19.h5 # Pro Primitiv eine H5-Datei
    │   ├── action        # (8,) 8D-Action [start_pos, g_start, end_pos, g_end]
    │   ├── eef_states    # (1, 1, 14) → 14D EEF-Zustand (Start+End)
    │   ├── positions     # (1, N_cubes, 4) Würfelpositionen (homogen)
    │   ├── observations/ # color + depth Bilder
    │   └── info/         # Metadaten (phase, n_steps, movement_distance, ...)
    └── property_params.pkl
```

**Wichtig:** Jeder Timestep = 1 Bewegungsprimitiv (nicht 1 Simulations-Frame!)  
Ein Primitiv fasst mehrere Simulations-Schritte zu einer diskreten Bewegungseinheit zusammen.

### 2.2 Datensatz-Dimensionen

| Komponente | Form | Beschreibung |
|------------|------|--------------|
| **Actions** | `(985, 20, 8)` | 985 Episoden, 20 Primitive, **8D** Action (mit Gripper) |
| **EEF States** | `(1, 1, 14)` pro H5 | 14D End-Effector-Zustand (Start + End des Primitivs) |
| **Images** | `(20, 256, 256, 3)` pro Episode | 20 RGB-Bilder (1 pro Primitiv, nach Bewegung) |
| **Proprio** | `(T, 3)` extrahiert | Nur EE-Position (x,y,z) = `eef[:, :3]` |

### 2.3 EEF-States Aufbau (14 Dimensionen) — Proprio-Quelle

Die `eef_states` speichern den End-Effector-Zustand am **Ende** (current) und **Anfang** (previous) 
jedes Primitivs. Das Format folgt der Referenz aus `robot_env.py` (Rope/Deformable Datensatz):

```
eef_states (14D) = [pos_end(3), pos_start(3), quat_end(4), quat_start(4)]
                    ├─────────┘ ├───────────┘ ├──────────┘ ├────────────┘
                    │           │             │            └── Orientierung am Primitiv-START
                    │           │             └── Orientierung am Primitiv-ENDE (aktuell)
                    │           └── EE-Position am Primitiv-START (vorherig)
                    └── EE-Position am Primitiv-ENDE (aktuell)

Index  Dim           Beschreibung                      Typ
─────  ───           ────────────                      ───
0-2    pos_end       EE-Position NACH Bewegung (x,y,z) float64, Meter (lokal)
3-5    pos_start     EE-Position VOR Bewegung (x,y,z)  float64, Meter (lokal)
6-9    quat_end      EE-Quaternion NACH Bewegung        float64, [qx,qy,qz,qw]
10-13  quat_start    EE-Quaternion VOR Bewegung          float64, [qx,qy,qz,qw]
```

**Proprio-Extraktion:** Nur `eef[:, :3]` (= `pos_end`, aktuelle EE-Position) wird als 
Proprioceptive Input für das Modell verwendet → **proprio_dim = 3**.

**Referenz-Vergleich:**
| | Franka (unser Datensatz) | Rope/Deformable (Referenz) |
|---|---|---|
| eef_states Format | `[pos_end, pos_start, quat_end, quat_start]` | `[pos_cur, pos_prev, quat_cur, quat_prev]` |
| Proprio verwendet | `eef[:, :3]` = pos_end (3D) | `np.zeros(1)` = Dummy (1D, nicht genutzt) |
| Semantik [0:3] | Aktuelle EE-Position | Aktuelle Partikelposition |
| Semantik [3:6] | Vorherige EE-Position | Vorherige Partikelposition |

### 2.4 Action-Vektor Aufbau (8 Dimensionen)

Der Action-Vektor beschreibt eine Bewegungsprimitiv als Start→End-Transition des End-Effectors:

```
Action (8D) = [x_start, y_start, z_start, g_start, x_end, y_end, z_end, g_end]
               ├──────────────────────────────────┘ ├──────────────────────────┘
               │  Primitiv-START (vorher)            │  Primitiv-ENDE (nachher)
               └── Wo war der EE?                    └── Wohin hat er sich bewegt?

Index  Dim        Beschreibung                          Wertebereich
─────  ───        ────────────                          ────────────
0      x_start    Start-Position X (vor/zurück)         ~0.2 - 0.7 m
1      y_start    Start-Position Y (links/rechts)       ~-0.3 - 0.3 m
2      z_start    Start-Position Z (Höhe)               ~0.05 - 0.4 m
3      g_start    Gripper-State am Start                 0.0 (zu) / 0.04 (auf)
4      x_end      End-Position X                         ~0.2 - 0.7 m
5      y_end      End-Position Y                         ~-0.3 - 0.3 m
6      z_end      End-Position Z                         ~0.05 - 0.4 m
7      g_end      Gripper-State am Ende                  0.0 (zu) / 0.04 (auf)
```

**Beispiel-Action** (APPROACH-Primitiv):
```
[0.475, -0.018, 0.320, 0.040, 0.475, -0.018, 0.160, 0.040]
 ├───────────────────────────────────┘ ├──────────────────────┘
 Start: (0.475, -0.018, 0.320), Gripper offen
                                       End: (0.475, -0.018, 0.160), Gripper offen
 → Abwärtsbewegung: Δz = -0.16m (Annäherung an Würfel)
```

**Ohne Gripper-Tracking** (ältere Datensätze): Action ist 6D = `[x_start, y_start, z_start, x_end, y_end, z_end]`

**Hinweis zur zeitlichen Ordnung (Action vs. EEF States):**
- Action: `[start → end]` = zeitlich vorwärts (Bewegungsbefehl: von wo nach wo)
- EEF States: `[current, previous]` = aktuell zuerst (Zustandsbeschreibung: wo bin ich, wo war ich)
- Diese unterschiedliche Konvention ist **kein Problem**, weil sie verschiedene Zwecke erfüllen:
  Action = Bewegungsrichtung, EEF States = Zustandsinformation. Das Modell lernt die Semantik.
  Das `proprio` nutzt ohnehin nur `eef[:, :3]` = pos_end = aktuelle Position.

**Kein Frameskip bei Primitiv-Datensätzen:**
Da jeder Timestep bereits ein ganzes Bewegungsprimitiv repräsentiert (nicht ein einzelner
Simulations-Frame), wird `frameskip=1` verwendet. Frameskip-Konkatenation entfällt.
Effektive Action-Dimension = 8 (nicht 8 × frameskip)
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
  encoder_lr: 1e-6      # DINO Encoder (eingefroren)
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

### 3.4 Hyperparameter-Abhängigkeiten: Grenzen und Formeln

Die Parameter `frameskip`, `num_hist`, `num_pred`, `batch_size` und die Episodenlänge `T` stehen in direktem Zusammenhang. Falsche Kombinationen führen zu **0 Training-Samples** oder einem **Freeze bei der Validation**.

#### Zentrale Formeln

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FORMEL 1: Benötigte Frames pro Sample                                      │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  benötigte_frames = (num_hist + num_pred) × frameskip                       │
│                                                                             │
│  Beispiel: (6 + 1) × 2 = 14                                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FORMEL 2: Training funktioniert (Slices > 0)                               │
│  ────────────────────────────────────────────                               │
│                                                                             │
│  (num_hist + num_pred) × frameskip  ≤  T                                    │
│                                                                             │
│  ⟹  num_hist  ≤  ⌊T / frameskip⌋ - num_pred                               │
│  ⟹  num_hist  ≤  ⌊T / frameskip⌋ - 1                                      │
│                                                                             │
│  Wenn diese Bedingung NICHT erfüllt ist:                                    │
│  → 0 Slices, kein Training möglich                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FORMEL 3: Slices pro Episode (= Trainingssamples pro Episode)              │
│  ─────────────────────────────────────────────────────────────              │
│                                                                             │
│  slices = T - (num_hist + num_pred) × frameskip + 1                         │
│                                                                             │
│  Beispiel (T=22, num_hist=6, frameskip=2):                                  │
│  slices = 22 - (6+1)×2 + 1 = 22 - 14 + 1 = 9                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FORMEL 4: Steps pro Epoch (= tqdm-Balken Länge)                            │
│  ───────────────────────────────────────────────                            │
│                                                                             │
│  train_samples = Σ max(0, T_i - (num_hist+num_pred) × frameskip + 1)       │
│                  über alle Train-Episoden                                   │
│                                                                             │
│  ≈ nutzbare_train_episoden × slices_pro_episode                             │
│                                                                             │
│  steps_pro_epoch = ⌈ train_samples / batch_size ⌉                          │
│                                                                             │
│  Hinweis: num_workers hat KEINEN Einfluss auf die Anzahl der Steps.         │
│  Workers beschleunigen nur das Laden der Daten.                             │
│                                                                             │
│  Beispiel (499 Ep., T=22, num_hist=6, frameskip=2, batch_size=4):           │
│  train_episoden ≈ 419 (von 449, da ~30 Ep. zu kurz)                        │
│  train_samples  = 419 × 9 = 3771                                           │
│  steps          = ⌈3771 / 4⌉ = 943                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FORMEL 5: Validation-Rollout friert NICHT ein                              │
│  ────────────────────────────────────────────                               │
│                                                                             │
│  Die openloop_rollout-Funktion setzt:                                       │
│      min_horizon = 2 + num_hist                                             │
│                                                                             │
│  und prüft (strikt größer!):                                                │
│      max_horizon = ⌊(T - 1) / frameskip⌋  >  min_horizon                   │
│                                                                             │
│  ⟹  ⌊(T-1) / frameskip⌋  >  2 + num_hist                                  │
│  ⟹  num_hist  <  ⌊(T-1) / frameskip⌋ - 2                                  │
│                                                                             │
│  Wenn diese Bedingung NICHT erfüllt ist:                                    │
│  → Endlos-Schleife! Training hängt nach dem Train-Balken.                   │
│                                                                             │
│  ACHTUNG: Diese Grenze ist STRENGER als die Training-Grenze!                │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Übersichtstabelle: Maximales num_hist nach T und frameskip

**T = 25** (z.B. `primLogger_NEps1000_ActInt2`)

| frameskip | Max num_hist (Training) | Max num_hist (Rollout ohne Freeze) | Slices bei max Rollout |
|-----------|------------------------|------------------------------------|------------------------|
| 1 | 24 | 21 | 4 |
| 2 | 11 | 9 | 3 |
| **3** | **7** | **5** | **8** |
| 4 | 5 | 3 | 6 |
| 5 | 4 | 2 | 6 |

**T = 22** (z.B. `NEps500_RobOpac10`)

| frameskip | Max num_hist (Training) | Max num_hist (Rollout ohne Freeze) | Slices bei max Rollout |
|-----------|------------------------|------------------------------------|------------------------|
| 1 | 21 | 18 | 4 |
| **2** | **10** | **7** | **8** |
| 3 | 6 | 4 | 7 |
| 4 | 4 | 2 | 5 |
| 5 | 3 | 1 | 3 |

**T = 21** (z.B. `NEps500_RobOpac10` ältere Version)

| frameskip | Max num_hist (Training) | Max num_hist (Rollout ohne Freeze) | Slices bei max Rollout |
|-----------|------------------------|------------------------------------|------------------------|
| 1 | 20 | 17 | 4 |
| **2** | **9** | **6** | **9** |
| 3 | 6 | 3 | 7 |
| 4 | 4 | 2 | 2 |
| 5 | 3 | 1 | 2 |

#### Empfohlene Konfigurationen

```
┌────────────────────────────────────────────────────────────────────────┐
│  Ziel: Maximale Historie bei stabilem Training + Validation            │
│                                                                        │
│  T=25, frameskip=3:  num_hist=5  → 8 Slices/Ep   ✅ Empfohlen         │
│  T=25, frameskip=2:  num_hist=6  → 12 Slices/Ep  ✅ Empfohlen         │
│  T=22, frameskip=2:  num_hist=6  → 9 Slices/Ep   ✅ Empfohlen         │
│  T=22, frameskip=3:  num_hist=4  → 7 Slices/Ep   ✅ OK                │
│                                                                        │
│  ⚠️  Nicht verwenden (Rollout-Freeze):                                 │
│  T=25, frameskip=3, num_hist=6  → Training OK, Rollout hängt!         │
│  T=22, frameskip=3, num_hist=5  → Training OK, Rollout hängt!         │
│  T=25, frameskip=4, num_hist=6  → Training scheitert (0 Slices)       │
└────────────────────────────────────────────────────────────────────────┘
```

#### Diagnostik: Warum friert mein Training ein?

```
Symptom: tqdm-Balken "Epoch X Train: 100%" fertig, danach keine Ausgabe mehr

Ursache: openloop_rollout() in val() sucht endlos nach einer
         Trajektorie die lang genug ist → while-Schleife terminiert nie

Prüfung:
  1. Berechne: min_horizon = 2 + num_hist
  2. Berechne: max_horizon = ⌊(T - 1) / frameskip⌋
  3. Wenn max_horizon ≤ min_horizon → FREEZE!

Lösung:
  → num_hist reduzieren, oder
  → frameskip reduzieren, oder
  → openloop_rollout in train.py absichern (max_attempts + Fallback)
```

### 3.5 Action & Proprio Embedding Prozess

Die `action_emb_dim: 10` und `proprio_emb_dim: 10` entsprechen **nicht** den Rohdimensionen der Daten (Action: 8, Proprio: 3). Stattdessen werden die Rohdaten durch einen **lernbaren Encoder** in diese Embedding-Dimensionen transformiert.

#### Schritt 1: Kein Frameskip bei Primitiv-Datensätzen

Bei Primitiv-basierten Datensätzen repräsentiert jeder Timestep bereits ein ganzes Bewegungsprimitiv 
(mehrere Simulations-Schritte zusammengefasst). Daher wird `frameskip=1` verwendet:

```
Primitiv-basiert (frameskip=1):
┌──────────────────────────────────────┐
│  Action pro Primitiv: 8 Dimensionen  │
│  [x_s, y_s, z_s, g_s, x_e, y_e, z_e, g_e]  │
│  Keine Konkatenation nötig!          │
└──────────────────────────────────────┘

Effektive Action-Dimension = 8 × 1 = 8 (nicht vergrößert durch frameskip)
```

#### Schritt 2: Embedding durch Conv1d

Der `ProprioceptiveEmbedding`-Encoder transformiert die Rohdaten in kompakte Embeddings:

```
ACTION ENCODER:
───────────────
Input:  (Batch, Time, 8)    ← 8D: [start_pos(3), g_start(1), end_pos(3), g_end(1)]
              │
              ▼
        Conv1d(8 → 10)      ← Lernbare Projektion (kernel_size=1)
              │
              ▼
Output: (Batch, Time, 10)   ← action_emb_dim


PROPRIO ENCODER:
────────────────
Input:  (Batch, Time, 3)    ← 3D: EE-Position [x, y, z] (= eef[:, :3] = pos_end)
              │
              ▼
        Conv1d(3 → 10)      ← Lernbare Projektion (kernel_size=1)
              │
              ▼
Output: (Batch, Time, 10)   ← proprio_emb_dim
```

**Hinweis:** `Conv1d(kernel_size=1, stride=1)` ist äquivalent zu einer punktweisen linearen 
Transformation (Fully-Connected Layer pro Zeitschritt). Es werden KEINE temporalen Faltungen 
über benachbarte Zeitschritte durchgeführt.

#### Warum diese Transformation?

| Aspekt | Erklärung |
|--------|-----------|
| **Dimensionsanpassung** | 8D Action / 3D Proprio → einheitlich 10D Embedding |
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
    in_chans,      # 8 für Actions (8D Primitiv), 3 für Proprio
    emb_dim,       # 10 (action_emb_dim / proprio_emb_dim)
    kernel_size=1,
    stride=1
)

# Aus train.py - Dynamische Dimensionen aus Datensatz:
proprio_encoder = ProprioceptiveEmbedding(
    in_chans=datasets["train"].proprio_dim,  # 3 (auto-detektiert)
    emb_dim=cfg.proprio_emb_dim              # 10 (aus Config)
)
action_encoder = ProprioceptiveEmbedding(
    in_chans=datasets["train"].action_dim,   # 8 (auto-detektiert aus H5)
    emb_dim=cfg.action_emb_dim               # 10 (aus Config)
)
```

**Zusammenfassung des Datenflusses:**
```
Actions:  (B, T, 8)  ─────────────────► (B, T, 8)  ──Conv1d──► (B, T, 10)
                      (kein frameskip)
Proprio:  (B, T, 3)  ─────────────────► (B, T, 3)  ──Conv1d──► (B, T, 10)
                      eef[:, :3]
Proprio:  (B, T, 3) ─────────────────────────► Conv1d──► (B, T, 10)
```

### 3.6 Umgebungs-Konfiguration: `conf/env/franka_cube_stack.yaml`

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

### 3.7 VRAM-Analyse und Validierungs-Lastspitze

Die GPU-Speicherbelegung ist die **harte Grenze** für die Hyperparameter-Wahl. Auf der NVIDIA A5000 (24 564 MiB) bestimmt der VRAM maßgeblich, wie hoch `num_hist` bei gegebenem `batch_size` und `frameskip` gewählt werden kann.

#### 3.7.1 VRAM-Modell: Drei Kostenklassen

Der VRAM-Verbrauch zerfällt in drei Kategorien:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  VRAM-Zerlegung                                                             │
│  ═══════════════                                                            │
│                                                                             │
│  1. FESTE KOSTEN (~559 MiB, konfigurationsunabhängig)                       │
│     ├─ Frozen Encoder (DINOv2 ViT-S/14):  21M × 4 Bytes   ≈  80 MiB       │
│     ├─ Trainable Weights (fp16):          31M × 2 Bytes   ≈  59 MiB       │
│     ├─ AdamW Optimizer States:            31M × 12 Bytes  ≈ 355 MiB       │
│     │  (fp32 master copy + momentum + variance)                            │
│     └─ Gradients (fp16):                  31M × 2 Bytes   ≈  59 MiB       │
│                                                                             │
│  2. AKTIVIERUNGEN (~13 908 MiB bei bs=4, nh=6, fs=2) ← HAUPTTREIBER       │
│     ├─ DINOv2 Encoder:      linear in batch_size × (num_hist + 1)          │
│     │  (12 Layers × Attention + FF pro Frame)                              │
│     ├─ ViT Predictor:       QUADRATISCH in num_hist × 196                  │
│     │  Attention-Matrix: O((num_hist × 196)²) ← KRITISCH                  │
│     │  (6 Layers, 16 Heads, seq_len = num_hist × 196)                     │
│     ├─ VQVAE Decoder:       linear in batch_size × (num_hist + 1) × 2     │
│     │  (2× Forward: Prediction + Reconstruction)                          │
│     └─ Misc: Loss-Buffers, einops-Temporärtensoren, Tiling                │
│                                                                             │
│  3. CUDA OVERHEAD (~2000 MiB, Basis-Kosten)                                │
│     ├─ PyTorch CUDA Context                                                │
│     ├─ cuDNN Workspace                                                     │
│     └─ CUDA Memory Allocator Reservierung                                  │
│                                                                             │
│  GESAMT = Feste Kosten + Aktivierungen + CUDA Overhead                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Warum Attention quadratisch skaliert:**

$$\text{VRAM}_{\text{Attention}} \propto B \times H_{\text{heads}} \times (\text{num\_hist} \times 196)^2 \times D \times 2$$

| num_hist | seq_len (×196) | Attention-Speicher (relativ) |
|----------|---------------|------------------------------|
| 1        | 196           | 1×                           |
| 3        | 588           | 9×                           |
| 6        | 1176          | 36×                          |
| 10       | 1960          | 100×                         |

#### 3.7.2 Empirische Kalibrierung

Theoretische VRAM-Formeln unterschätzen systematisch, da sie folgende Faktoren nicht erfassen:
- PyTorch Autograd-Graph (speichert Computation Graph für Backward)
- CUDA Memory Allocator Fragmentierung
- Temporäre Tensoren bei `einops.rearrange`, `torch.cat`, `repeat`
- cuDNN Workspace für optimierte Convolution-Kernel

**Kalibrierung an realen Messdaten:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  REFERENZ-MESSPUNKT (A5000, Epoch 1)                                        │
│  ───────────────────────────────────                                        │
│  Konfiguration: batch_size=4, num_hist=6, frameskip=2                       │
│  Gemessen:      16 467 MiB  (67.0% von 24 564 MiB)                         │
│                                                                             │
│  Zerlegung:                                                                 │
│    Feste Kosten:     559 MiB                                                │
│    CUDA Overhead:   2 000 MiB                                               │
│    Activations:    13 908 MiB  (= 16467 - 559 - 2000)                       │
│                                                                             │
│  Theoretische Activations: ~2 524 MiB (viel zu niedrig!)                    │
│  → Kalibrierungsfaktor: 13 908 / 2 524 = 5.51×                             │
│                                                                             │
│  Kreuzvalidierung:                                                          │
│  ┌───────────┬────────┬──────────────────────┬──────────┬──────────────┐     │
│  │ batch_size│num_hist│ Geschätzt (MiB)      │ Gemessen │ Quelle       │     │
│  ├───────────┼────────┼──────────────────────┼──────────┼──────────────┤     │
│  │     4     │   6    │ 16 467 (= Referenz)  │ 16 467   │ Epoch 1 Log  │     │
│  │     8     │   3-4  │ ~14 598 (59.4%)      │ ~60%     │ train.yaml   │     │
│  │    16     │   3-4  │ >24 564 (OOM)        │ OOM!     │ train.yaml   │     │
│  │    32     │   3-4  │ >24 564 (OOM)        │ OOM!     │ train.yaml   │     │
│  └───────────┴────────┴──────────────────────┴──────────┴──────────────┘     │
│                                                                             │
│  ✅ Schätzung "59.4%" passt zum Kommentar "~60%" in train.yaml              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 3.7.3 Validierungs-Lastspitze (Val Peak)

**Kritischer Befund:** Die Validierungsphase verbraucht **mehr VRAM** als das Training!

Drei Ursachen im Code (`train.py`, Methode `val()`):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PROBLEM 1: openloop_rollout() vor dem Val-Loop                             │
│  ──────────────────────────────────────────────                             │
│  → model.rollout() baut z-Tensor iterativ auf via torch.cat                │
│  → decode_obs() erzeugt Bilder auf der GPU                                 │
│  → CUDA-Allocator hält freigegebene Blöcke als fragmentierten Cache        │
│                                                                             │
│  PROBLEM 2: Val Forward Pass OHNE torch.no_grad()                          │
│  ──────────────────────────────────────────────────                         │
│  for batch in valid_dataloader:                                             │
│      model(obs, act)      ← baut vollen Computation Graph!                │
│      encode_obs(obs)      ← ZUSÄTZLICHER Encoder-Pass (für Plots)          │
│                                                                             │
│  → Identische Activation-Kosten wie Training Forward Pass                  │
│  → Autograd-Graph wird gebaut, obwohl backward() nie aufgerufen wird       │
│  → Verschwendet VRAM durch gespeicherte Zwischenergebnisse                 │
│                                                                             │
│  PROBLEM 3: Kein torch.cuda.empty_cache() zwischen Rollout und Val-Loop    │
│  ──────────────────────────────────────────────────────────────────         │
│  → Fragmentierte Blöcke vom Rollout + neue Val-Allokationen                │
│  → CUDA-Allocator findet keine zusammenhängenden Blöcke                    │
│  → ~12% zusätzlicher Overhead durch Fragmentierung                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Formel: Validierungs-Lastspitze**

$$\text{VRAM}_{\text{Val Peak}} = \bigl(\underbrace{F}_{\text{Feste Kosten}} + \underbrace{C}_{\text{CUDA}} + \underbrace{A_{\text{val}}}_{\substack{\text{Val Activations}\\\text{(= Train Fwd)}}} + \underbrace{R}_{\text{Rollout-Residuen}} + \underbrace{P}_{\text{Extra Plot}}\bigr) \times \underbrace{1.12}_{\text{Fragmentierung}}$$

**Beispielrechnung für aktuelle Konfiguration (bs=4, nh=6, fs=2, T=22):**

| Komponente | MiB | Erklärung |
|------------|-----|-----------|
| Training VRAM | 16 467 | Gemessener Wert |
| + Rollout-Residuen | ~73 | z-Tensor + Decode-Bilder + Fragmentierung |
| + Extra Plot-Decode | ~151 | encode_obs + eval_images beim 1. Batch |
| × Fragmentierung (1.12) | | CUDA-Allocator Overhead |
| **= Val Peak** | **~19 239** | **78.3% von 24 564 MiB** |
| Overhead vs. Training | **+16.8%** | |

#### 3.7.4 Maximales num_hist nach VRAM (inkl. Val Peak)

**Szenario: T=22, frameskip=2, batch_size=4 (aktuelle Konfiguration)**

| num_hist | Train VRAM | Train % | Val Peak | Val Peak % | Status |
|----------|-----------|---------|----------|------------|--------|
| 3 | 6 654 MiB | 27.1% | 7 783 MiB | 31.7% | ✅ Sicher |
| 4 | 8 699 MiB | 35.4% | 10 182 MiB | 41.5% | ✅ Sicher |
| 5 | 12 101 MiB | 49.3% | 14 157 MiB | 57.6% | ✅ Sicher |
| 6 | 16 467 MiB | 67.0% | 19 239 MiB | 78.3% | ✅ Aktuell |
| **7** | **21 797 MiB** | **88.7%** | **25 479 MiB** | **103.7%** | **⚠️ Val OOM!** |
| 8 | 28 092 MiB | 114.4% | – | – | ❌ Train OOM |

> **Ergebnis:** Bei `batch_size=4, frameskip=2` ist `num_hist=6` das Maximum,
> das sowohl Training als auch Validierung ohne OOM übersteht.
> `num_hist=7` würde im Training noch passen (88.7%), aber die
> Validierung sprengt den VRAM (103.7%)!

#### 3.7.5 Optimale Konfigurationen (Solver-Ergebnisse)

Der Optimierungssolver (`hyperparameter_analysis.py`) maximiert `num_hist` unter
der harten Grenze `Val Peak ≤ 90% × 24 564 MiB`:

**Szenarien (T=22, E=500):**

| Rang | Config (bs/nh/fs) | Train VRAM | Val Peak | Val Peak % | Slices/Ep | Score |
|------|-------------------|-----------|----------|------------|-----------|-------|
| 1 | bs=4, nh=7, fs=2 | 21 797 | ~25 479 | ~103.7% | 8 | ⚠️ Val OOM |
| **2** | **bs=4, nh=6, fs=2** | **16 467** | **19 239** | **78.3%** | **9** | **Gewählt ✅** |
| 3 | bs=2, nh=7, fs=2 | ~11 266 | ~13 182 | ~53.7% | 8 | Machbar |
| 4 | bs=1, nh=8, fs=2 | ~7 621 | ~8 917 | ~36.3% | 7 | Machbar |

> **Begründung für bs=4, nh=6, fs=2:**
> - Maximales `num_hist` bei `batch_size ≥ 4` (stabile Gradientenschätzung)
> - 9 Slices/Episode → gute Dateneffizienz
> - Val Peak bei 78.3% → ausreichend Headroom
> - Korrespondiert mit Paper-Empfehlung: Zhou et al. nutzen `batch_size=32`
>   auf A6000 (48 GB), skaliert linear: 32 × (24564/49152) ≈ 16 → unser bs=4
>   mit höherem nh kompensiert durch kleineren bs

#### 3.7.6 Vergleich mit Paper (Zhou et al. 2025)

| Parameter | Paper (PushT/PointMaze) | Unsere Konfig. (Franka) | Begründung |
|-----------|------------------------|------------------------|------------|
| GPU | A6000 (48 GB) | A5000 (24.5 GB) | ~50% VRAM |
| batch_size | 32 | 4 | VRAM-limitiert |
| num_hist | 1–3 | 6 | Maximiert (Priorität 1) |
| frameskip | 1–5 | 2 | Franka: langsame Dynamik |
| Epochen | 100 | 100 | Identisch |
| num_pred | 1 | 1 | Identisch |

#### 3.7.7 Generierte Analyse-Plots

Alle Plots befinden sich in `hyperparameter_analysis/` und wurden mit
`hyperparameter_analysis.py` erzeugt (PDF + PNG):

| # | Datei | Inhalt |
|---|-------|--------|
| 01 | `01_feasibility_heatmap_T{22,25}.pdf` | Machbarkeitskarte: num_hist × batch_size (grün/gelb/rot) |
| 02 | `02_vram_vs_batch_numhist_T22.pdf` | VRAM-Kurven: Train + Val Peak vs. batch_size pro num_hist |
| 03 | `03_samples_efficiency_T{22,25}.pdf` | Slices/Ep + Steps/Ep über num_hist × frameskip |
| 04 | `04_optimal_frontier_T{22,25}.pdf` | Pareto-Front: Score vs. Val Peak pro Konfiguration |
| 05 | `05_vram_breakdown_T22.pdf` | Gestapeltes Balkendiagramm: Weights, Optimizer, Activations, Val-Overhead |
| 06 | `06_attention_scaling.pdf` | Quadratische Attention-Skalierung über seq_len |
| 07 | `07_paper_comparison.pdf` | Unsere Konfig vs. Paper-Referenz (skaliert auf A5000) |
| 08 | `08_sweep_table_T{22,25}.pdf` | Vollständige Sweep-Tabelle mit Status-Codes |
| 09 | `09_validation_peak_T{22,25}.pdf` | Validierungs-Lastspitze: Training vs. Val Peak + Zerlegung |

#### 3.7.8 Potentielle Code-Verbesserungen (train.py)

Die folgenden Änderungen würden die Validierungs-Lastspitze um ~12–15% senken:

```python
# FIX 1: torch.no_grad() um den Validation Loop
# Aktuell (train.py, val()):
for i, batch in enumerate(valid_dataloader):
    out = self.model(obs, act)         # ← baut Computation Graph!

# Verbesserung:
with torch.no_grad():                  # ← spart ~gleiche Activations wie Forward
    for i, batch in enumerate(valid_dataloader):
        out = self.model(obs, act)     # ← kein Graph, nur Inference

# FIX 2: torch.cuda.empty_cache() zwischen Rollout und Val Loop
# Nach openloop_rollout und vor dem Val Loop einfügen:
torch.cuda.empty_cache()              # ← räumt CUDA-Allocator auf
```

> **Hinweis:** Diese Fixes wurden NICHT angewendet, um die Reproduzierbarkeit
> gegenüber dem Originalcode (Zhou et al. 2025) zu wahren. Die VRAM-Analyse
> berücksichtigt diese Overhead-Quellen in der Parameterwahl.

---

## 4. Training-Pipeline (Chronologisch)

### Phase 1: Initialisierung

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT 1: Konfiguration laden                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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
│                                                                             │
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

## 6. Proprioceptive Encoder — Vollständiger Trainingsablauf

> **Verifiziert am 14.02.2026** — Alle Code-Pfade, Variablennamen und Tensor-Dimensionen wurden anhand des
> Quellcodes nachvollzogen. Referenzmodell: `outputs/2026-02-09/17-59-59` (500 Episoden, frameskip=2, num_hist=4).

### 6.1 Überblick: Was wird trainiert und warum?

Der **Proprioceptive Encoder** (`proprio_encoder`) ist eine lernbare Projektion, die die rohe
EE-Position (End-Effector Position, 3D) in einen kompakten Embedding-Vektor (10D) transformiert.
Er wird **gemeinsam** mit dem Action Encoder, dem ViT Predictor und dem VQ-VAE Decoder trainiert.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            TRAINIERBARE vs. EINGEFRORENE KOMPONENTEN                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Komponente              Trainiert?    Optimizer                   LR       │
│  ─────────────────────   ──────────    ──────────────────────── ─────────── │
│  DINO v2 Encoder         ✗ FROZEN     (encoder_optimizer)        1e-6 (*)  │
│  Proprio Encoder         ✓ TRAINIERT  action_encoder_optimizer   5e-4      │
│  Action Encoder          ✓ TRAINIERT  action_encoder_optimizer   5e-4      │
│  ViT Predictor           ✓ TRAINIERT  predictor_optimizer        2e-4      │
│  VQ-VAE Decoder          ✓ TRAINIERT  decoder_optimizer          1e-4      │
│                                                                             │
│  (*) encoder_optimizer existiert, aber .step() wird NIE aufgerufen          │
│      weil train_encoder=False → alle Parameter haben requires_grad=False   │
│                                                                             │
│  WICHTIG: Proprio Encoder und Action Encoder teilen sich denselben          │
│           Optimizer (action_encoder_optimizer) und dieselbe Learning Rate!  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Datensatz → Proprio-Extraktion (Schritt für Schritt)

#### 6.2.1 Rohdaten im Datensatz

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATENSATZ-QUELLE (Primitiv-basiert)                                        │
│  Pfad: fcs_datasets/NEps1000_RobOpac0_NPrim20_NCams4_NCube1/              │
│                                                                             │
│  Pro Episode (z.B. 000042/):                                                │
│  ├── obses.pth         # (T, H, W, C) = (20, 256, 256, 3) BGR uint8       │
│  ├── 00.h5             # Primitiv 0 (= Timestep 0)                        │
│  │   ├── action        # (8,) 8D-Action                                   │
│  │   │                   [x_s, y_s, z_s, g_s, x_e, y_e, z_e, g_e]         │
│  │   ├── eef_states    # (1, 1, 14) → 14D EEF-Zustand                     │
│  │   │                   [pos_end(3), pos_start(3), quat_end(4),           │
│  │   │                    quat_start(4)]                                    │
│  │   ├── positions     # (1, N_cubes, 4) Würfelpositionen (homogen)        │
│  │   └── info/         # n_steps, movement_distance, phase,                │
│  │                       primitive_name, primitive_type                      │
│  ├── 01.h5 ... 19.h5                                                       │
│  └── property_params.pkl                                                    │
│                                                                             │
│  985 Episoden × 20 Primitive = 19.700 Datenpunkte                          │
│                                                                             │
│  DETAIL: eef_states[0, 0, :] = 14D Vektor:                                │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ [0:3]  pos_end   = EE-Position NACH Bewegung (aktuell)          │       │
│  │ [3:6]  pos_start = EE-Position VOR Bewegung (vorherig)          │       │
│  │ [6:10] quat_end  = EE-Quaternion NACH Bewegung                  │       │
│  │ [10:14]quat_start= EE-Quaternion VOR Bewegung                   │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  DETAIL: action[:] = 8D Vektor:                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ [0:3]  start_pos = EE-Position am Primitiv-Start                │       │
│  │ [3]    g_start   = Gripper-State am Start (0.0/0.04)            │       │
│  │ [4:7]  end_pos   = EE-Position am Primitiv-Ende                 │       │
│  │ [7]    g_end     = Gripper-State am Ende (0.0/0.04)             │       │
│  └──────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Laden in `FrankaCubeStackDataset.__init__()` (franka_cube_stack_dset.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT A: H5-Dateien lesen (pro Episode, pro Primitiv/Timestep)           │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (franka_cube_stack_dset.py, __init__):                                │
│  ──────────────────────────────────────────                                 │
│  for t in range(episode_length):        # t = 0..19 (20 Primitive)         │
│      with h5py.File(f"{t:02d}.h5") as f:                                   │
│          action = f["action"][:]        # → numpy (8,) 8D-Action           │
│          eef = f["eef_states"][:]       # → numpy (1, 1, 14)               │
│          eef_states.append(eef.flatten())  # → numpy (14,)                 │
│                                                                             │
│  Variablen nach dem Loop:                                                   │
│  self.all_actions[i]    : numpy (20, 8)   # 20 Primitive × 8D Actions     │
│  self.all_eef_states[i] : numpy (20, 14)  # 20 Primitive × 14D EEF        │
│                                                                             │
│  Konvertierung zu Tensoren:                                                 │
│  self.actions_tensors[i] = torch.from_numpy(actions).float()  # (20, 8)    │
│  self.eef_tensors[i]    = torch.from_numpy(eef).float()       # (20, 14)   │
│                                                                             │
│  Automatische Dimensions-Erkennung:                                         │
│  self.action_dim = actions.shape[-1]  # → 8 (auto-detektiert aus H5)       │
│  self.proprio_dim = 3                 # → fest: nur eef[:, :3] = pos_end   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT B: Z-Score-Normalisierungs-Statistiken berechnen                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (franka_cube_stack_dset.py, __init__, normalize_action=True):         │
│  ──────────────────────────────────────────────────────────────────         │
│                                                                             │
│  # Alle EEF-Daten aller Episoden zusammenfassen:                            │
│  all_eef_flat = torch.cat(self.eef_tensors, dim=0)  # (19700, 14)         │
│                                                                             │
│  # Proprio-Statistiken: NUR erste 3 Dimensionen (= pos_end = aktuelle Pos) │
│  self.proprio_mean = all_eef_flat[:, :3].mean(dim=0)  # (3,)              │
│  self.proprio_std  = all_eef_flat[:, :3].std(dim=0) + 1e-6  # (3,)        │
│                                                                             │
│  Typische Werte (985 Episoden):                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  proprio_mean ≈ [0.476, 0.017, 0.161]   (Meter, Weltkoord.)    │        │
│  │  proprio_std  ≈ [0.124, 0.161, 0.072]   (Streuung in Meter)    │        │
│  │                                                                  │        │
│  │  Interpretation:                                                 │        │
│  │  - x ≈ 0.476m ± 0.124m (vor/zurück)                             │        │
│  │  - y ≈ 0.017m ± 0.161m (links/rechts, zentriert)                │        │
│  │  - z ≈ 0.161m ± 0.072m (Höhe über Tisch)                        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  WICHTIG: Diese Statistiken werden bei Inferenz/Planning benötigt!         │
│  Der Planner muss die gleiche Normalisierung verwenden.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.3 Proprio-Extraktion in `get_frames()` (franka_cube_stack_dset.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT C: Proprio für einen Batch-Eintrag extrahieren                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (franka_cube_stack_dset.py, get_frames):                              │
│  ─────────────────────────────────────────────                              │
│  def get_frames(self, idx, frames):                                         │
│      eef = self.eef_tensors[idx][frames]        # (T_slice, 14)            │
│      proprio = (eef[:, :3] - self.proprio_mean) / self.proprio_std          │
│      #           ↑                                                          │
│      #  Nur erste 3 Dims: pos_end = EE-Position NACH Bewegung [x, y, z]    │
│      #  = Aktuelle Position des End-Effectors                               │
│      #                                                                      │
│      obs = {"visual": image, "proprio": proprio}                            │
│      return obs, act, state, {}                                             │
│                                                                             │
│  Tensor-Dimensionen (Beispiel: frameskip=1, num_hist=4, num_pred=1):        │
│  ──────────────────────────────────────────────────────────────────         │
│  Input frames (nach TrajSlicerDataset):                                     │
│    frames = [start, start+1, start+2, start+3, start+4]  # 5 Frames       │
│              ↑ frameskip=1 bei Primitiv-Datensätzen (jeder Schritt)         │
│                                                                             │
│  eef:     (5, 14)   ← 5 selektierte Zeitschritte, 14D EEF                 │
│  eef[:, :3]: (5, 3) ← NUR pos_end [x, y, z] (aktuelle EE-Position)        │
│  proprio: (5, 3)    ← z-normalisiert                                       │
│                                                                             │
│  Normalisierung (Element-weise):                                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  proprio[t] = (eef[t, :3] - proprio_mean) / proprio_std        │        │
│  │                                                                  │        │
│  │  eef[t, :3] = pos_end = EE-Position am Ende des Primitivs t     │        │
│  │  Beispiel: eef = [0.45, 0.02, 0.16]                              │        │
│  │  proprio  = ([0.45, 0.02, 0.16] - [0.476, 0.017, 0.161])        │        │
│  │             / [0.124, 0.161, 0.072]                              │        │
│  │           = [-0.21, 0.019, -0.014]   ← ~N(0,1) verteilt         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.4 Frameskip-Anwendung in `TrajSlicerDataset` (traj_dset.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT D: Frameskip und Slicing                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  WICHTIG für Primitiv-Datensätze: frameskip = 1                             │
│  → Jeder Timestep ist bereits ein Bewegungsprimitiv!                        │
│  → Kein temporales Subsampling nötig.                                       │
│                                                                             │
│  Code (traj_dset.py, TrajSlicerDataset.__getitem__):                        │
│  ──────────────────────────────────────────────────                         │
│  def __getitem__(self, idx):                                                │
│      i, start, end = self.slices[idx]    # z.B. (42, 3, 8)                │
│      obs, act, state, _ = self.dataset[i]  # Volle Episode laden          │
│      for k, v in obs.items():                                               │
│          obs[k] = v[start:end:self.frameskip]  # frameskip=1: alle         │
│      state = state[start:end:self.frameskip]                                │
│      act = act[start:end]                                                   │
│      act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)         │
│      return obs, act, state                                                 │
│                                                                             │
│  Beispiel (frameskip=1, num_frames=5, start=3, end=8):                      │
│  ─────────────────────────────────────────────────────                      │
│                                                                             │
│  Primitiv-Sequenz der Episode:                                              │
│  Index:  0  1  2 [3] [4] [5] [6] [7]  8  9 ...                            │
│                  ↑    ↑    ↑    ↑    ↑                                     │
│  Frames:        F0   F1   F2   F3   F4     (alle, frameskip=1)             │
│                                                                             │
│  obs['proprio']: v[3:8:1] = v[[3, 4, 5, 6, 7]]  → (5, 3)                 │
│  obs['visual']:  v[3:8:1] = v[[3, 4, 5, 6, 7]]  → (5, 3, 224, 224)       │
│                                                                             │
│  act: v[3:8] = 5 Actions → rearrange zu (5, 8)                             │
│       ↑ n=num_frames=5, f=frameskip=1, d=action_dim=8                      │
│       ↑ (5×1, 8) → (5, 1×8=8)  ← Keine Konkatenation!                    │
│                                                                             │
│  KRITISCH: Proprio wird mit demselben Frameskip subsampled wie Visual!      │
│  → Proprio und Visual sind zeitlich perfekt synchron.                       │
│  → Bei frameskip=1: Actions werden 1:1 durchgereicht (8D bleibt 8D)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Batch-Zusammenstellung — Tensoren beim Dataloader-Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATALOADER OUTPUT (1 Batch)                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Konfiguration: B=8, num_hist=4, num_pred=1, frameskip=1, action_dim=8     │
│                 proprio_dim=3, img_size=224                                 │
│                                                                             │
│  obs, act, state = next(dataloader)                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Variable           │ Shape                  │ Beschreibung         │   │
│  ├─────────────────────┼────────────────────────┼──────────────────────┤   │
│  │ obs['visual']       │ (8, 5, 3, 224, 224)    │ 5 RGB Bilder         │   │
│  │ obs['proprio']      │ (8, 5, 3)              │ 5 EE-Positionen      │   │
│  │ act                 │ (8, 5, 8)              │ 5 × 8D Actions       │   │
│  │ state               │ (8, 5, 14)             │ 5 EEF-Zustände       │   │
│  └─────────────────────┴────────────────────────┴──────────────────────┘   │
│                                                                             │
│  Wobei:                                                                     │
│  - 5 = num_hist + num_pred = 4 + 1 = 5 Zeitschritte                       │
│  - 8 = action_dim (8D Primitiv-Action, kein frameskip)                     │
│  - 3 = proprio_dim (EE x, y, z = eef[:, :3] = pos_end)                    │
│  - 14 = eef_dim (voller EEF-Zustand mit Start+End)                         │
│                                                                             │
│  obs['proprio'] Beispiel-Werte (z-normalisiert):                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Batch 0, Frame 0: [-0.21,  0.02, -0.01]  ← ~N(0,1)              │   │
│  │  Batch 0, Frame 1: [-0.18,  0.05,  0.12]                          │   │
│  │  Batch 0, Frame 2: [-0.15,  0.09,  0.25]  ← Roboter bewegt sich  │   │
│  │  Batch 0, Frame 3: [-0.10,  0.11,  0.38]                          │   │
│  │  Batch 0, Frame 4: [-0.05,  0.13,  0.50]  ← Target-Frame          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Proprio Encoder — Architektur und Forward Pass

#### 6.4.1 Instanziierung in `train.py` (init_models)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT E: Proprio Encoder Instanziierung                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (train.py, init_models):                                              │
│  ─────────────────────────────                                              │
│  self.proprio_encoder = hydra.utils.instantiate(                            │
│      self.cfg.proprio_encoder,    # → ProprioceptiveEmbedding              │
│      in_chans=self.datasets["train"].proprio_dim,  # = 3                   │
│      emb_dim=self.cfg.proprio_emb_dim,             # = 10                  │
│  )                                                                          │
│                                                                             │
│  Hydra-Konfiguration (conf/proprio_encoder/proprio.yaml):                   │
│  ─────────────────────────────────────────────────────────                  │
│  _target_: models.proprio.ProprioceptiveEmbedding                           │
│  num_frames: 2          # ← Nicht relevant (nur für pos_embed, unused)     │
│  tubelet_size: 1        # ← kernel_size = stride = 1                       │
│  use_3d_pos: False      # ← Kein 3D Positional Embedding                  │
│                                                                             │
│  Resultierende Instanz:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ProprioceptiveEmbedding(                                           │   │
│  │    in_chans = 3          # Input: EE-Position (x, y, z)            │   │
│  │    emb_dim  = 10         # Output: Proprio-Embedding               │   │
│  │    (patch_embed): Conv1d(                                           │   │
│  │      in_channels  = 3,   # 3 → Proprio-Dimensionen                 │   │
│  │      out_channels = 10,  # 10 → proprio_emb_dim                    │   │
│  │      kernel_size  = 1,   # Punkt-weise Projektion                   │   │
│  │      stride       = 1    # Kein Downsampling                        │   │
│  │    )                                                                │   │
│  │  )                                                                  │   │
│  │                                                                     │   │
│  │  Trainierbare Parameter:                                            │   │
│  │  - patch_embed.weight: (10, 3, 1) = 30 Parameter                   │   │
│  │  - patch_embed.bias:   (10,)      = 10 Parameter                   │   │
│  │  ─────────────────────────────────────────────                      │   │
│  │  GESAMT: 40 trainierbare Parameter                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.4.2 Forward Pass des Proprio Encoders (models/proprio.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT F: ProprioceptiveEmbedding.forward(x)                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (models/proprio.py):                                                  │
│  ──────────────────────────                                                 │
│  def forward(self, x):                                                      │
│      # x: (B, T, D) = (8, 5, 3)                                           │
│      x = x.permute(0, 2, 1)   # → (B, D, T) = (8, 3, 5)                  │
│      x = self.patch_embed(x)  # Conv1d: (8, 3, 5) → (8, 10, 5)           │
│      x = x.permute(0, 2, 1)   # → (B, T, emb_dim) = (8, 5, 10)           │
│      return x                                                               │
│                                                                             │
│  Tensor-Fluss im Detail:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  INPUT x:          (B=8, T=5, D=3)                                   │   │
│  │  ┌──────────────────────────────────────┐                           │   │
│  │  │  Batch 0: [[x₀,y₀,z₀],              │  ← 5 EE-Positionen       │   │
│  │  │            [x₁,y₁,z₁],              │     (z-normalisiert)     │   │
│  │  │            [x₂,y₂,z₂],              │                          │   │
│  │  │            [x₃,y₃,z₃],              │                          │   │
│  │  │            [x₄,y₄,z₄]]              │                          │   │
│  │  └──────────────────────────────────────┘                           │   │
│  │                     │                                                │   │
│  │                     ▼ permute(0, 2, 1)                               │   │
│  │                                                                      │   │
│  │  PERMUTED:          (B=8, D=3, T=5)                                  │   │
│  │  ┌──────────────────────────────────────┐                           │   │
│  │  │  Batch 0: [[x₀,x₁,x₂,x₃,x₄],      │  ← Channels-first       │   │
│  │  │            [y₀,y₁,y₂,y₃,y₄],      │     für Conv1d            │   │
│  │  │            [z₀,z₁,z₂,z₃,z₄]]      │                          │   │
│  │  └──────────────────────────────────────┘                           │   │
│  │                     │                                                │   │
│  │                     ▼ Conv1d(3→10, k=1, s=1)                        │   │
│  │                                                                      │   │
│  │  CONV OUTPUT:       (B=8, emb=10, T=5)                               │   │
│  │  ┌──────────────────────────────────────┐                           │   │
│  │  │  Pro Zeitschritt t:                  │                           │   │
│  │  │  emb_t = W × [x_t, y_t, z_t] + b   │  ← Lineare Projektion    │   │
│  │  │          ↑                           │     W: (10, 3)           │   │
│  │  │    10×3 Matrix                       │     b: (10,)             │   │
│  │  └──────────────────────────────────────┘                           │   │
│  │                     │                                                │   │
│  │                     ▼ permute(0, 2, 1)                               │   │
│  │                                                                      │   │
│  │  OUTPUT:            (B=8, T=5, emb_dim=10)                           │   │
│  │  ┌──────────────────────────────────────┐                           │   │
│  │  │  z_proprio: 10D Embedding pro Frame  │  ← Verwendbar für        │   │
│  │  │  [e₀, e₁, e₂, ..., e₉]              │     Concat mit DINO      │   │
│  │  └──────────────────────────────────────┘                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Mathematisch: Conv1d(k=1, s=1) ≡ nn.Linear(3, 10)                        │
│  → Punkt-weise lineare Transformation, identisch für jeden Zeitschritt     │
│  → Keine Aktivierungsfunktion (rein linear!)                               │
│  → Jeder Zeitschritt wird unabhängig transformiert                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Embedding-Fusion: Proprio + Visual + Action (encode-Methode)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT G: VWorldModel.encode(obs, act) — Fusion aller Modalitäten        │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (visual_world_model.py, encode):                                      │
│  ─────────────────────────────────────                                      │
│  def encode(self, obs, act):                                                │
│      z_dct = self.encode_obs(obs)    # → {"visual": ..., "proprio": ...}   │
│      act_emb = self.encode_act(act)  # → (B, T, action_emb_dim)           │
│      # concat_dim=1 → Fusion entlang Feature-Dimension                    │
│                                                                             │
│  Aufrufe im Detail:                                                         │
│  ─────────────────                                                          │
│                                                                             │
│  1) encode_obs(obs):                                                        │
│     ├── DINO Encoder:                                                       │
│     │   obs['visual']: (8, 5, 3, 224, 224)                                 │
│     │     → rearrange: (40, 3, 224, 224)                                   │
│     │     → DINO forward: (40, 256, 384)                                   │
│     │     → rearrange: (8, 5, 256, 384)                                    │
│     │   visual_embs: (B=8, T=5, P=256, D=384)                              │
│     │                                                                       │
│     └── Proprio Encoder:                                                    │
│         obs['proprio']: (8, 5, 3)                                           │
│           → proprio_encoder.forward: (8, 5, 10)                            │
│         proprio_emb: (B=8, T=5, emb_dim=10)                                │
│                                                                             │
│  2) encode_act(act):                                                        │
│     act: (8, 5, 12)                                                         │
│       → action_encoder.forward: (8, 5, 10)                                 │
│     act_emb: (B=8, T=5, emb_dim=10)                                        │
│                                                                             │
│  3) Fusion (concat_dim=1):                                                  │
│     ─────────────────────                                                   │
│     # Proprio tiling: (B,T,10) → unsqueeze → (B,T,1,10) → tile →          │
│     #                  (B,T,256,10) → repeat(num_proprio_repeat=1) →       │
│     #                  (B,T,256,10)                                         │
│     proprio_tiled = repeat(proprio_emb.unsqueeze(2),                        │
│                            "b t 1 a -> b t f a", f=256)                    │
│     proprio_repeated = proprio_tiled.repeat(1, 1, 1, 1)   # ×1            │
│                                                                             │
│     # Action tiling: identisch                                              │
│     act_tiled = repeat(act_emb.unsqueeze(2),                                │
│                        "b t 1 a -> b t f a", f=256)                        │
│     act_repeated = act_tiled.repeat(1, 1, 1, 1)   # ×1                    │
│                                                                             │
│     # Concatenation entlang letzer Dimension:                               │
│     z = torch.cat([visual_embs, proprio_repeated, act_repeated], dim=3)    │
│                                                                             │
│  Resultat:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  z: (B=8, T=5, P=256, D=404)                                       │   │
│  │                                                                      │   │
│  │  Aufbau der 404 Dimensionen pro Patch:                               │   │
│  │  ┌──────────────────┬──────────────┬──────────────┐                 │   │
│  │  │  DINO Visual     │  Proprio Emb │  Action Emb  │                 │   │
│  │  │  z[..., :384]    │ z[...,384:394]│ z[...,394:404]│                │   │
│  │  │  384 dim         │  10 dim      │  10 dim      │                 │   │
│  │  │  (FROZEN)        │ (TRAINIERT)  │ (TRAINIERT)  │                 │   │
│  │  └──────────────────┴──────────────┴──────────────┘                 │   │
│  │                                                                      │   │
│  │  JEDER der 256 Patches enthält dieselben Proprio/Action-Werte       │   │
│  │  (getiled über alle Patches)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 Prediction und Loss-Berechnung für Proprio

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT H: Forward Pass mit Source/Target-Split und Loss                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (visual_world_model.py, forward):                                     │
│  ──────────────────────────────────────                                     │
│                                                                             │
│  z = self.encode(obs, act)                                                  │
│  z: (B=8, T=5, P=256, D=404)                                               │
│                                                                             │
│  # Source/Target Aufteilung:                                                │
│  z_src = z[:, :num_hist]     = z[:, :4]     # (8, 4, 256, 404)            │
│  z_tgt = z[:, num_pred:]     = z[:, 1:]     # (8, 4, 256, 404)            │
│                                                                             │
│  Zeitliche Zuordnung (num_hist=4, num_pred=1):                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Zeitschritt:    t=0    t=1    t=2    t=3    t=4                   │     │
│  │                                                                    │     │
│  │  z_src:         [F0]   [F1]   [F2]   [F3]                         │     │
│  │                                         ↓ Predictor                │     │
│  │  z_pred:        [P1]   [P2]   [P3]   [P4]                         │     │
│  │                                                                    │     │
│  │  z_tgt:         [F1]   [F2]   [F3]   [F4]   ← Ground Truth       │     │
│  │                                                                    │     │
│  │  Vergleich: z_pred[i] soll z_tgt[i] vorhersagen                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ViT Predictor:                                                             │
│  z_pred = self.predict(z_src)   # (8, 4, 256, 404)                         │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  ║  LOSS-BERECHNUNG (concat_dim=1)                                     ║    │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  Variablen-Mapping (self.proprio_dim=10, self.action_dim=10):              │
│                                                                             │
│  z-Vektor Layout pro Patch (404 dim):                                       │
│  ┌─────────────────┬─────────────┬─────────────┐                           │
│  │    Visual (384)  │ Proprio (10)│ Action (10) │                           │
│  │  Indices: [0:384]│ [384:394]   │ [394:404]   │                           │
│  └─────────────────┴─────────────┴─────────────┘                           │
│                                                                             │
│  Code (visual_world_model.py, forward, concat_dim=1):                       │
│  ─────────────────────────────────────────────────────                      │
│                                                                             │
│  # 1) z_visual_loss: NUR visuelle Features (384 dim)                       │
│  z_visual_loss = MSE(                                                       │
│      z_pred[:, :, :, :-(10+10)],       # z_pred[..., :384]                 │
│      z_tgt[:, :, :, :-(10+10)].detach()                                    │
│  )                                                                          │
│  # Shape: MSE über (8, 4, 256, 384) vs (8, 4, 256, 384)                   │
│  # → Skalar                                                                │
│                                                                             │
│  # 2) z_proprio_loss: NUR Proprio-Embedding (10 dim)                       │
│  z_proprio_loss = MSE(                                                      │
│      z_pred[:, :, :, -(10+10):-10],    # z_pred[..., 384:394]             │
│      z_tgt[:, :, :, -(10+10):-10].detach()                                 │
│  )                                                                          │
│  # Shape: MSE über (8, 4, 256, 10) vs (8, 4, 256, 10)                     │
│  # → Skalar                                                                │
│  # HINWEIS: Alle 256 Patches haben identische Proprio-Werte (getiled)      │
│  # → MSE wird über alle Patches gemittelt, aber da identisch = kein Fehler │
│                                                                             │
│  # 3) z_loss: Visual + Proprio ZUSAMMEN (394 dim, OHNE Action)             │
│  z_loss = MSE(                                                              │
│      z_pred[:, :, :, :-10],            # z_pred[..., :394]                 │
│      z_tgt[:, :, :, :-10].detach()                                         │
│  )                                                                          │
│  # Shape: MSE über (8, 4, 256, 394) vs (8, 4, 256, 394)                   │
│  # → Skalar                                                                │
│  # ↑ DAS IST DER HAUPTLOSS, der zum Training-Loss addiert wird!           │
│                                                                             │
│  loss = loss + z_loss   ← Proprio ist TEIL des Haupt-Losses!               │
│                                                                             │
│  WICHTIG:                                                                   │
│  ─────────                                                                  │
│  • z_loss enthält IMPLIZIT den Proprio-Loss (da 394 = 384 + 10)            │
│  • z_visual_loss und z_proprio_loss werden NUR geloggt, nicht addiert      │
│  • Action-Embedding (letzte 10 dim) wird NICHT in den Loss einbezogen      │
│  • z_tgt wird mit .detach() abgetrennt → DINO-Encoder bekommt keinen       │
│    Gradient (ist ohnehin eingefroren, aber detach ist zusätzliche Sicherh.)│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.7 Gradient-Fluss und Optimizer-Update

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT I: Backward Pass und Gradient-Fluss zum Proprio Encoder            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (train.py, train):                                                    │
│  ──────────────────────                                                     │
│  # 1. Zero-Grad für alle Optimizer                                         │
│  self.encoder_optimizer.zero_grad()                                         │
│  self.decoder_optimizer.zero_grad()                                         │
│  self.predictor_optimizer.zero_grad()                                       │
│  self.action_encoder_optimizer.zero_grad()  ← Setzt Gradienten auf 0      │
│  #  ↑ Dieser Optimizer enthält BEIDE: action_encoder UND proprio_encoder   │
│                                                                             │
│  # 2. Backward Pass                                                        │
│  self.accelerator.backward(loss)                                            │
│  #  loss = z_loss + decoder_loss                                           │
│  #       = MSE(z_pred[..., :394], z_tgt[..., :394])    ← enthält Proprio  │
│  #       + MSE(visual_recon, obs_visual) + 0.25 × vq_loss                  │
│                                                                             │
│  # 3. Optimizer-Steps (NUR trainierbare Komponenten)                       │
│  # self.encoder_optimizer.step()  ← NICHT aufgerufen (train_encoder=False) │
│  self.decoder_optimizer.step()             # ✓ train_decoder=True          │
│  self.predictor_optimizer.step()           # ✓ train_predictor=True        │
│  self.action_encoder_optimizer.step()      # ✓ IMMER aufgerufen           │
│  # ↑ Updated sowohl action_encoder.parameters() ALS AUCH                   │
│  #   proprio_encoder.parameters()!                                          │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  ║  GRADIENT-FLUSS ZUM PROPRIO ENCODER (Rückwärtspfad)                 ║    │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  z_loss = MSE(z_pred[..., :394], z_tgt[..., :394].detach())                │
│     │                                                                       │
│     │  ∂z_loss / ∂z_pred                                                   │
│     ▼                                                                       │
│  z_pred = self.predict(z_src)                                               │
│     │                                                                       │
│     │  ∂z_pred / ∂z_src  (durch ViT Predictor)                             │
│     ▼                                                                       │
│  z_src = z[:, :4]                                                           │
│     │                                                                       │
│     │  ∂z_src / ∂z  (Identity, nur Slicing)                                │
│     ▼                                                                       │
│  z = cat([visual_embs, proprio_tiled, act_tiled], dim=-1)                   │
│     │                                                                       │
│     │  ∂z / ∂proprio_tiled  (Identity, nur Concat-Rückpropagation)         │
│     ▼                                                                       │
│  proprio_tiled = repeat(proprio_emb.unsqueeze(2), ..., f=256)              │
│     │                                                                       │
│     │  ∂proprio_tiled / ∂proprio_emb  (Summierung über 256 Patches)        │
│     ▼                                                                       │
│  proprio_emb = self.proprio_encoder(obs['proprio'])                        │
│     │                                                                       │
│     │  ∂proprio_emb / ∂W_proprio  (Conv1d Gradient)                        │
│     ▼                                                                       │
│  W_proprio = self.proprio_encoder.patch_embed.weight  # (10, 3, 1)        │
│  b_proprio = self.proprio_encoder.patch_embed.bias    # (10,)             │
│                                                                             │
│  → action_encoder_optimizer.step() aktualisiert W und b!                   │
│                                                                             │
│  GRADIENT-VERSTÄRKUNG DURCH TILING:                                        │
│  Da proprio_emb auf 256 Patches getiled wird, wird der Gradient             │
│  über alle 256 Patches summiert:                                            │
│  ∂L/∂proprio_emb = Σ(p=0..255) ∂L/∂proprio_tiled[p]                       │
│  → Faktor ~256× stärkerer Gradient als ohne Tiling                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.7.1 Optimizer-Konfiguration (train.py, init_optimizers)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  OPTIMIZER FÜR PROPRIO ENCODER                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (train.py, init_optimizers):                                          │
│  ─────────────────────────────────                                          │
│  self.action_encoder_optimizer = torch.optim.AdamW(                         │
│      itertools.chain(                                                       │
│          self.action_encoder.parameters(),   # Conv1d(12→10): 130 Params   │
│          self.proprio_encoder.parameters()   # Conv1d(3→10):  40 Params    │
│      ),                                                                     │
│      lr=self.cfg.training.action_encoder_lr  # = 5e-4 = 0.0005            │
│  )                                                                          │
│                                                                             │
│  Parameter-Übersicht:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Modell-Komponente     │ Parameter          │ Shape      │ Anzahl  │   │
│  ├───────────────────────┼────────────────────┼────────────┼─────────┤   │
│  │  action_encoder       │ patch_embed.weight │ (10, 12, 1)│    120  │   │
│  │  action_encoder       │ patch_embed.bias   │ (10,)      │     10  │   │
│  │  proprio_encoder      │ patch_embed.weight │ (10,  3, 1)│     30  │   │
│  │  proprio_encoder      │ patch_embed.bias   │ (10,)      │     10  │   │
│  ├───────────────────────┼────────────────────┼────────────┼─────────┤   │
│  │  GESAMT               │                    │            │    170  │   │
│  └───────────────────────┴────────────────────┴────────────┴─────────┘   │
│                                                                             │
│  AdamW-Eigenschaften:                                                       │
│  - Learning Rate: 5e-4                                                      │
│  - Weight Decay: Standard (0.01)                                            │
│  - Betas: Standard (0.9, 0.999)                                            │
│  - Eps: Standard (1e-8)                                                     │
│                                                                             │
│  WICHTIG: Beide Encoder teilen NICHT die Gewichte, nur den Optimizer!      │
│  → Jeder hat eigene W und b, aber dieselbe Learning Rate.                  │
│  → AdamW verwaltet separate Momentum- und Varianz-Statistiken              │
│     für jeden Parameter.                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.8 Separate-Embedding: Proprio aus z extrahieren (separate_emb)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT J: Proprio aus dem kombinierten z-Tensor extrahieren               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (visual_world_model.py, separate_emb, concat_dim=1):                  │
│  ──────────────────────────────────────────────────────────                 │
│  def separate_emb(self, z):                                                 │
│      # z: (B, T, P=256, D=404)                                             │
│      # self.proprio_dim = 10 (proprio_emb_dim × num_proprio_repeat = 10×1) │
│      # self.action_dim  = 10 (action_emb_dim × num_action_repeat = 10×1)  │
│                                                                             │
│      z_visual  = z[..., :-(10+10)]              # z[..., :384]   → (B,T,256,384) │
│      z_proprio = z[..., -(10+10):-10]           # z[..., 384:394]→ (B,T,256,10)  │
│      z_act     = z[..., -10:]                   # z[..., 394:404]→ (B,T,256,10)  │
│                                                                             │
│      # Rückgängigmachung des Tilings:                                      │
│      z_proprio = z_proprio[:, :, 0, :10 // 1]  # → (B, T, 10)             │
│      z_act     = z_act[:, :, 0, :10 // 1]      # → (B, T, 10)             │
│      # ↑ Nimmt nur Patch 0, da alle 256 Patches identisch sind            │
│      # ↑ :10//1 = :10 (Division durch num_proprio_repeat=1)               │
│                                                                             │
│      z_obs = {"visual": z_visual, "proprio": z_proprio}                    │
│      return z_obs, z_act                                                    │
│                                                                             │
│  Output-Dimensionen:                                                        │
│  ┌───────────────────┬────────────────────────┐                            │
│  │ z_obs["visual"]   │ (B, T, 256, 384)       │                            │
│  │ z_obs["proprio"]  │ (B, T, 10)             │ ← Proprio Embedding       │
│  │ z_act             │ (B, T, 10)             │                            │
│  └───────────────────┴────────────────────────┘                            │
│                                                                             │
│  Verwendung:                                                                │
│  - decode_obs() nutzt z_obs["visual"] für VQ-VAE Decoder                   │
│  - z_obs["proprio"] wird NICHT decodiert (kein Proprio-Decoder!)            │
│  - z_obs["proprio"] wird bei Planning für Rollout-Auswertung genutzt       │
│    (Proprio-Anteil der Objective Function)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.9 Rollout: Proprio im autoregressiven Vorhersage-Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT K: Autoregressive Vorhersage mit Proprio (rollout-Methode)         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (visual_world_model.py, rollout):                                     │
│  ──────────────────────────────────────                                     │
│  def rollout(self, obs_0, act):                                             │
│      # obs_0['visual']:  (1, num_hist, 3, 224, 224)  = (1, 4, 3, 224, 224)│
│      # obs_0['proprio']: (1, num_hist, 3)             = (1, 4, 3)          │
│      # act:              (1, num_hist+H, action_dim)  = (1, 4+H, 12)      │
│                                                                             │
│      num_obs_init = obs_0['visual'].shape[1]   # = 4                       │
│      act_0 = act[:, :4]      # Initiale Actions: (1, 4, 12)               │
│      action = act[:, 4:]     # Zukünftige Actions: (1, H, 12)             │
│                                                                             │
│      z = self.encode(obs_0, act_0)   # (1, 4, 256, 404)                   │
│      # ↑ Enthält Proprio der initialen 4 Frames (aus obs_0)                │
│                                                                             │
│      # Autoregressive Schleife:                                             │
│      t = 0                                                                  │
│      while t < H:                                                           │
│          z_pred = self.predict(z[:, -4:])  # Letzte 4 Frames              │
│          z_new = z_pred[:, -1:]            # Nur letzter pred Frame        │
│          z_new = self.replace_actions_from_z(z_new, action[:, t:t+1])      │
│          z = torch.cat([z, z_new], dim=1)  # Anhängen                      │
│          t += 1                                                             │
│                                                                             │
│  Was passiert mit Proprio im Rollout?                                       │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  1. INITIAL (t=0): z enthält echte Proprio-Embeddings aus obs_0             │
│     z[..., 384:394] = proprio_encoder(obs_0['proprio'])                    │
│                                                                             │
│  2. VORHERSAGE (t>0): z_pred enthält VORHERGESAGTE Proprio-Embeddings      │
│     z_pred = predict(z_src)                                                 │
│     z_pred[..., 384:394] = ViT-Vorhersage für Proprio-Embedding           │
│     ↑ Der ViT Predictor sagt ALLE 404 Dimensionen vorher,                 │
│       einschließlich der 10 Proprio-Dimensionen!                            │
│                                                                             │
│  3. ACTION REPLACEMENT: replace_actions_from_z() ersetzt NUR die            │
│     Action-Dimensionen (394:404), NICHT die Proprio-Dimensionen!           │
│     z_new[..., 384:394] = vorhergesagtes Proprio (unverändert)             │
│     z_new[..., 394:404] = neues Action-Embedding (ersetzt)                 │
│                                                                             │
│  Zeitlicher Verlauf von Proprio im z-Tensor:                                │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Frame:  F0    F1    F2    F3    F4    F5    F6    ...            │     │
│  │                                                                    │     │
│  │  Proprio: REAL  REAL  REAL  REAL  PRED  PRED  PRED  ...          │     │
│  │          └──── aus obs_0 ────┘  └─── vom Predictor ──────┘        │     │
│  │                                                                    │     │
│  │  Visual:  REAL  REAL  REAL  REAL  PRED  PRED  PRED  ...          │     │
│  │  Action:  REAL  REAL  REAL  REAL  NEW   NEW   NEW   ...          │     │
│  │                                  └ replace_actions_from_z() ┘     │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  Am Ende: z_obses, z = self.separate_emb(z)                                │
│  z_obses["proprio"]: (1, 4+H+1, 10) ← Alle Proprio-Embeddings            │
│  z_obses["visual"]:  (1, 4+H+1, 256, 384) ← Alle Visual-Embeddings       │
│                                                                             │
│  BEDEUTUNG: Das Training des Proprio Encoders beeinflusst direkt           │
│  die Qualität der Proprio-Vorhersage im Rollout!                           │
│  → Schlecht trainierter Proprio Encoder = schlechte Proprio-Vorhersage     │
│  → Guter Proprio Encoder = Predictor kann EE-Trajektorie korrekt          │
│    vorhersagen                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.10 Checkpoint: Proprio Encoder speichern und laden

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHRITT L: Checkpoint-Speicherung                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Code (train.py, __init__):                                                 │
│  ──────────────────────────                                                 │
│  self._keys_to_save = ["epoch"]                                             │
│  # ... encoder, predictor, decoder (bedingt) ...                            │
│  self._keys_to_save += ["action_encoder", "proprio_encoder"]                │
│  # ↑ IMMER gespeichert, unabhängig von train_encoder/train_predictor!      │
│                                                                             │
│  Code (train.py, save_ckpt):                                                │
│  ────────────────────────────                                               │
│  ckpt = {}                                                                  │
│  for k in self._keys_to_save:                                               │
│      ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])              │
│                                                                             │
│  Checkpoint-Inhalt (model_50.pth):                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Key                  │ Typ                        │ Inhalt         │   │
│  ├───────────────────────┼────────────────────────────┼────────────────┤   │
│  │  "epoch"              │ int                        │ 50             │   │
│  │  "encoder"            │ DinoV2Encoder              │ DINO Weights   │   │
│  │  "encoder_optimizer"  │ Adam state_dict            │ Opt. States    │   │
│  │  "predictor"          │ ViTPredictor               │ ViT Weights    │   │
│  │  "predictor_optimizer"│ AdamW state_dict           │ Opt. States    │   │
│  │  "decoder"            │ VQVAE                      │ Decoder Wts    │   │
│  │  "decoder_optimizer"  │ Adam state_dict            │ Opt. States    │   │
│  │  "action_encoder"     │ ProprioceptiveEmbedding    │ Conv1d(12→10)  │   │
│  │  "proprio_encoder"    │ ProprioceptiveEmbedding    │ Conv1d(3→10)   │   │
│  └───────────────────────┴────────────────────────────┴────────────────┘   │
│                                                                             │
│  HINWEIS: action_encoder_optimizer wird NICHT als separater Key             │
│  gespeichert, da er in _keys_to_save nicht enthalten ist!                  │
│  → Bei Checkpoint-Resumption wird der Optimizer NEU initialisiert.          │
│  → Proprio/Action Encoder GEWICHTE werden geladen, aber Optimizer-State    │
│    (Momentum, Varianz) geht verloren.                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.11 Gesamtflowchart: Proprio Encoder Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PROPRIO ENCODER — VOLLSTÄNDIGER TRAININGS-FLOWCHART       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  1. DATENSATZ LADEN                                                  │  │
│  │  H5: eef_states (1,1,14) → flatten → (14,) → [:3] = EE pos (3D)    │  │
│  │  500 Ep × 25 Frames → all_eef_flat: (12500, 14)                     │  │
│  │  proprio_mean = all_eef_flat[:, :3].mean(0) → (3,) ≈ [0.48,0.02,0.16] │
│  │  proprio_std  = all_eef_flat[:, :3].std(0)  → (3,) ≈ [0.12,0.16,0.07] │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  2. BATCH-VORBEREITUNG (pro Training-Iteration)                      │  │
│  │  TrajSlicerDataset: frameskip=2, num_frames=5                        │  │
│  │  get_frames() → proprio = (eef[:, :3] - mean) / std → (5, 3)       │  │
│  │  Dataloader collate → obs['proprio']: (B=8, T=5, D=3)              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  3. PROPRIO ENCODING                                                 │  │
│  │  proprio_encoder.forward(obs['proprio'])                             │  │
│  │  (B=8, T=5, D=3)                                                    │  │
│  │    → permute(0,2,1) → (8, 3, 5)                                    │  │
│  │    → Conv1d(3→10, k=1) → (8, 10, 5)                                │  │
│  │    → permute(0,2,1) → (8, 5, 10)                                   │  │
│  │  proprio_emb: (B=8, T=5, emb=10)                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  4. PARALLEL: VISUAL + ACTION ENCODING                               │  │
│  │  visual_embs = DINO(obs['visual']) → (B=8, T=5, P=256, D=384)      │  │
│  │  act_emb = action_encoder(act)     → (B=8, T=5, emb=10)            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  5. FUSION (concat_dim=1)                                            │  │
│  │  proprio_tiled: (8,5,10) → tile auf 256 Patches → (8,5,256,10)     │  │
│  │  act_tiled:     (8,5,10) → tile auf 256 Patches → (8,5,256,10)     │  │
│  │  z = cat([visual_embs, proprio_tiled, act_tiled], dim=-1)            │  │
│  │  z: (B=8, T=5, P=256, D=404)                                        │  │
│  │       └── 384 visual ── 10 proprio ── 10 action ──┘                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  6. SRC/TGT SPLIT                                                    │  │
│  │  z_src = z[:, :4]   → (8, 4, 256, 404)  ← Input für Predictor      │  │
│  │  z_tgt = z[:, 1:]   → (8, 4, 256, 404)  ← Ground Truth             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  7. PREDICTION (ViT Predictor)                                       │  │
│  │  z_src: (8, 4, 256, 404) → reshape → (8, 1024, 404)                │  │
│  │    → 6× Transformer Blocks (kausale Maske, 16 Heads)                │  │
│  │    → reshape → z_pred: (8, 4, 256, 404)                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  8. LOSS-BERECHNUNG                                                  │  │
│  │  z_visual_loss  = MSE(z_pred[...,:384],   z_tgt[...,:384].detach()) │  │
│  │  z_proprio_loss = MSE(z_pred[...,384:394],z_tgt[...,384:394].detach())│ │
│  │  z_loss         = MSE(z_pred[...,:394],   z_tgt[...,:394].detach()) │  │
│  │                   ↑ Visual + Proprio, OHNE Action                    │  │
│  │  total_loss = z_loss + decoder_loss                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  9. BACKWARD + OPTIMIZER STEP                                        │  │
│  │  accelerator.backward(total_loss)                                    │  │
│  │  Gradient fließt: loss → z_pred → ViT → z_src → z → proprio_tiled  │  │
│  │    → proprio_emb → Conv1d.weight/bias (∂L/∂W, ∂L/∂b)               │  │
│  │                                                                      │  │
│  │  action_encoder_optimizer.step()                                     │  │
│  │    → AdamW-Update für:                                               │  │
│  │      • action_encoder.patch_embed.weight  (10, 12, 1) → 120 Params  │  │
│  │      • action_encoder.patch_embed.bias    (10,)       →  10 Params  │  │
│  │      • proprio_encoder.patch_embed.weight (10,  3, 1) →  30 Params  │  │
│  │      • proprio_encoder.patch_embed.bias   (10,)       →  10 Params  │  │
│  │    lr = 5e-4, Gesamt: 170 Parameter                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  10. CHECKPOINT (nach jeder Epoch)                                   │  │
│  │  torch.save({"proprio_encoder": ProprioceptiveEmbedding, ...})       │  │
│  │  Gespeichert: Conv1d(3→10) Weights + Bias = 40 Parameter            │  │
│  │  Pfad: outputs/DATUM/ZEIT/checkpoints/model_{epoch}.pth              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  W&B Logging:                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  train_z_proprio_loss  │ val_z_proprio_loss  │ z_proprio_err_rollout │  │
│  │  (geloggt pro Epoch)   │ (geloggt pro Epoch) │ (Rollout-Fehler)     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.12 Zusammenfassung der Tensor-Dimensionen (Referenzmodell 500 Ep)

| Variable | Shape | Datei | Beschreibung |
|----------|-------|-------|-------------|
| `eef_states` (raw) | `(1, 1, 14)` | H5-Datei | Roher EEF-Zustand pro Timestep |
| `eef.flatten()` | `(14,)` | franka_cube_stack_dset.py | Geflachter EEF |
| `self.eef_tensors[i]` | `(25, 14)` | franka_cube_stack_dset.py | EEF pro Episode |
| `all_eef_flat` | `(12500, 14)` | franka_cube_stack_dset.py | Alle EEF concateniert |
| `self.proprio_mean` | `(3,)` | franka_cube_stack_dset.py | Mean der EE-Position |
| `self.proprio_std` | `(3,)` | franka_cube_stack_dset.py | Std der EE-Position |
| `eef[:, :3]` | `(T, 3)` | get_frames() | EE-Position [x,y,z] |
| `proprio` (normalisiert) | `(T, 3)` | get_frames() | Z-normalisiert ~N(0,1) |
| `obs['proprio']` (Batch) | `(B, T, 3)` = `(8, 5, 3)` | Dataloader | Proprio pro Batch |
| `proprio_emb` | `(B, T, 10)` = `(8, 5, 10)` | encode_obs() | Nach Conv1d |
| `proprio_tiled` | `(B, T, P, 10)` = `(8, 5, 256, 10)` | encode() | Auf Patches getiled |
| `z` (fusioniert) | `(B, T, P, D)` = `(8, 5, 256, 404)` | encode() | Visual+Proprio+Action |
| `z_src` | `(8, 4, 256, 404)` | forward() | Input für Predictor |
| `z_tgt` | `(8, 4, 256, 404)` | forward() | Ground Truth |
| `z_pred` | `(8, 4, 256, 404)` | forward() | Vorhersage |
| `z_pred[..., 384:394]` | `(8, 4, 256, 10)` | forward() | Vorhergesagtes Proprio-Emb |
| `z_tgt[..., 384:394]` | `(8, 4, 256, 10)` | forward() | Ground Truth Proprio-Emb |
| `z_proprio_loss` | Skalar | forward() | MSE(pred, tgt) für Proprio |
| `z_loss` | Skalar | forward() | MSE(pred, tgt) für Visual+Proprio |
| `W_proprio` | `(10, 3, 1)` | proprio_encoder | Conv1d Gewichte (30 Param) |
| `b_proprio` | `(10,)` | proprio_encoder | Conv1d Bias (10 Param) |

### 6.13 Konfigurationsparameter-Referenz

| Parameter | Config-Pfad | Wert (500 Ep) | Bedeutung |
|-----------|-------------|---------------|-----------|
| `proprio_emb_dim` | `conf/train.yaml` | `10` | Output-Dimension des Proprio Encoders |
| `num_proprio_repeat` | `conf/train.yaml` | `1` | Wiederholungsfaktor für Tiling (1 = keine Wiederholung) |
| `proprio_dim` | `conf/env/franka_cube_stack.yaml` | `3` | Input-Dimension (EE x,y,z) |
| `action_encoder_lr` | `conf/train.yaml` | `5e-4` | Learning Rate für Proprio+Action Optimizer |
| `normalize_action` | `conf/train.yaml` | `true` | Z-Normalisierung von Proprio und Actions |
| `concat_dim` | `conf/train.yaml` | `1` | Fusion entlang Feature-Dimension |
| `frameskip` | `conf/train.yaml` | `2` | Temporal Subsampling |
| `num_hist` | `conf/train.yaml` | `4` | Anzahl Kontext-Frames |
| `train_predictor` | `conf/train.yaml → model` | `true` | Aktiviert Predictor + Action/Proprio Optimizer |

---

## 7. Loss-Funktionen

### 7.1 Übersicht aller Losses

| Loss Name | Formel | Gewichtung | Zweck |
|-----------|--------|------------|-------|
| `z_loss` | MSE(z_pred, z_tgt) | 1.0 | Hauptloss für Predictor |
| `z_visual_loss` | MSE(z_pred_visual, z_tgt_visual) | (geloggt) | Nur visuelle Features |
| `z_proprio_loss` | MSE(z_pred_proprio, z_tgt_proprio) | (geloggt) | Nur Proprio-Features |
| `decoder_recon_loss` | MSE(visual_recon, obs_visual) | 1.0 | Rekonstruktionsqualität |
| `decoder_vq_loss` | Commitment Loss | 0.25 | VQ Regularisierung (=0 wenn quantize=False) |
| `decoder_loss` | recon + 0.25×vq | 1.0 | Decoder-Training |

### 7.2 Warum diese Kombination?

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

## 8. W&B Metriken und Monitoring

### 8.1 Übersicht aller geplotteten Metriken

Das Training loggt automatisch zahlreiche Metriken zu Weights & Biases. Hier eine vollständige Übersicht:

#### 8.1.1 Hauptverluste (Loss)

| Metrik | Definition | Ziel |
|--------|------------|------|
| `train_loss` / `val_loss` | Gesamtverlust (kombiniert alle Komponenten) | ↓ niedrig |
| `train_z_loss` / `val_z_loss` | Verlust im latenten Raum (z-Space) - Hauptmetrik für Predictor | ↓ niedrig |
| `train_z_visual_loss` / `val_z_visual_loss` | Visueller Encoder-Verlust im latenten Raum (nur 384 DINO-Features) | ↓ niedrig |
| `train_z_proprio_loss` / `val_z_proprio_loss` | Propriozeptiver Verlust im latenten Raum (10 proprio-dim) | ↓ niedrig |

#### 8.1.2 Decoder-Verluste

| Metrik | Definition | Ziel |
|--------|------------|------|
| `decoder_loss_reconstructed` | Rekonstruktionsverlust (Bild → Encoder → Decoder → Bild) | ↓ niedrig |
| `decoder_loss_pred` | Verlust für vorhergesagte Frames (durch Predictor) | ↓ niedrig |
| `decoder_recon_loss_*` | Reiner Rekonstruktionsverlust ohne VQ-Komponente | ↓ niedrig |
| `decoder_vq_loss_*` | Vector-Quantization Verlust (= 0, wenn `quantize: false`) | ↓ niedrig |

#### 8.1.3 Bildqualitätsmetriken

Diese Metriken messen die Qualität der rekonstruierten/vorhergesagten Bilder:

| Metrik | Definition | Optimal |
|--------|------------|---------|
| `img_mse_reconstructed` / `img_mse_pred` | Mean Squared Error der Pixel | ↓ niedrig (< 0.01 gut) |
| `img_l1_reconstructed` / `img_l1_pred` | L1 Norm (mittlerer absoluter Fehler) | ↓ niedrig |
| `img_l2_reconstructed` / `img_l2_pred` | L2 Norm (euklidischer Abstand) | ↓ niedrig |
| `img_ssim_reconstructed` / `img_ssim_pred` | Structural Similarity Index (Struktur-Ähnlichkeit) | ↑ hoch (max 1.0, > 0.9 gut) |
| `img_psnr_reconstructed` / `img_psnr_pred` | Peak Signal-to-Noise Ratio (dB) | ↑ hoch (> 30 gut, > 40 exzellent) |
| `img_lpips_reconstructed` / `img_lpips_pred` | Learned Perceptual Image Patch Similarity | ↓ niedrig (< 0.1 gut) |

**Hinweis:** 
- `*_reconstructed`: Decoder rekonstruiert den Input direkt (keine Vorhersage)
- `*_pred`: Decoder rekonstruiert die Vorhersage des Predictors

#### 8.1.4 Rollout-Fehler (Latent Space)

Diese Metriken bewerten die Vorhersagequalität über mehrere Zeitschritte:

| Metrik | Definition |
|--------|------------|
| `z_visual_err_pred` | Vorhersagefehler im visuellen latenten Raum (1-Schritt) |
| `z_visual_err_rollout` | Akkumulierter Fehler über mehrere Vorhersage-Schritte |
| `z_visual_err_rollout_1framestart` | Rollout-Fehler, beginnend vom ersten Frame |
| `z_visual_err_full` | Gesamter visueller Rollout-Fehler über alle Frames |
| `z_visual_err_next1` | Fehler für den nächsten einzelnen Frame |
| `z_proprio_err_pred` | Vorhersagefehler für Propriozeption (1-Schritt) |
| `z_proprio_err_rollout` | Akkumulierter Proprio-Fehler über mehrere Schritte |
| `z_proprio_err_rollout_1framestart` | Proprio-Rollout-Fehler, beginnend vom ersten Frame |
| `z_proprio_err_full` | Gesamter Propriozeption-Rollout-Fehler |
| `z_proprio_err_next1` | Proprio-Fehler für den nächsten Frame |

### 8.2 Interpretation der Metriken

#### Gute Trainingskurven zeigen:
```
train_loss         ────────────────────────────────────────  
                  ╲                                          Konvergenz
                   ╲___________________________________  ←   (flach)
                                                             
val_loss          ────────────────────────────────────────
                  ╲
                   ╲___________________________________  ←   Ähnlich zu train
                                                             
train_img_ssim    ────────────────────────────────────────
                              _________________________ 
                             ╱                          ←   Anstieg zu ~0.9+
                   _________╱                                
```

#### Typische Probleme:

**1. Overfitting:**
```
train_loss        val_loss
   │                 │
   ╲                 ╲
    ╲_______          ╲____╱‾‾‾‾‾  ← val steigt wieder an!
                                     (train fällt weiter)
```

**2. Underfitting:**
```
train_loss = val_loss
     │
     │_______________  ← Beide stagnieren auf hohem Niveau
```

**3. Instabilität:**
```
train_loss
     │╱╲  ╱╲  ╱╲
     │  ╲╱  ╲╱  ╲╱  ← Starke Schwankungen
```

### 8.3 Overfitting-Diagnose und Lösungsansätze

#### 8.3.1 Typische Overfitting-Indikatoren

Overfitting tritt auf, wenn das Modell die Trainingsdaten "auswendig lernt" statt zu generalisieren:

| Symptom | Betroffene Metriken |
|---------|---------------------|
| Val-Loss steigt nach anfänglichem Abfall | `val_loss`, `val_z_loss` |
| Train-Metriken verbessern sich weiter | `train_loss` fällt weiter |
| Steigende Image-Fehler auf Validation | `val_img_mse_*`, `val_img_l2_*` steigen |
| Sinkende Image-Qualität auf Validation | `val_img_psnr_*`, `val_img_ssim_*` fallen |
| Akkumulierende Rollout-Fehler | `val_z_visual_err_full`, `val_z_proprio_err_full` steigen |

#### 8.3.2 Besonders anfällige Metriken

Basierend auf Experimenten mit kleinen Datensätzen (20 Episoden):

1. **`val_z_proprio_loss`** - Steigt oft ab Epoch 40-60
2. **`val_z_visual_err_full`** - Akkumulierter Fehler wächst kontinuierlich
3. **`val_img_mse_reconstructed`** - Verschlechtert sich nach Epoch 50
4. **`val_decoder_loss_reconstructed`** - Steigt langsam an

#### 8.3.3 Lösungsansätze gegen Overfitting

| Ansatz | Konfiguration | Empfehlung |
|--------|---------------|------------|
| **Learning Rate reduzieren** | `decoder_lr: 1e-4` (von 3e-4)<br>`predictor_lr: 2e-4` (von 5e-4) | ✓ Erste Maßnahme |
| **Weniger Epochen** | `training.epochs: 50` (von 100) | ✓ Bei kleinen Datensätzen |
| **Mehr Dropout** | `predictor.dropout: 0.2` (von 0.1) | ✓ Regularisierung |
| **Early Stopping** | Manuell bei Anstieg von val_loss | ✓ Bester Checkpoint wählen |
| **Learning Rate Scheduler** | CosineAnnealingLR oder ReduceLROnPlateau | ⚠️ Nicht implementiert |
| **Weight Decay** | In AdamW Optimizer | ⚠️ Erfordert Code-Änderung |
| **Mehr Trainingsdaten** | Zusätzliche Episoden sammeln | ⚠️ Aufwändig |
| **Data Augmentation** | Bild-Transformationen | ⚠️ Erfordert Code-Änderung |

#### 8.3.4 Dropout erklärt

**Was ist Dropout?**

Dropout ist eine Regularisierungstechnik, die während des Trainings zufällig einen Prozentsatz der Neuronen "ausschaltet" (auf 0 setzt):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DROPOUT MECHANISMUS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OHNE Dropout (Inferenz):       MIT Dropout (Training, p=0.2):             │
│                                                                             │
│    ●───●───●───●───●              ●───○───●───●───○                        │
│    │   │   │   │   │              │       │   │                            │
│    ●───●───●───●───●              ●───●───○───●───●                        │
│    │   │   │   │   │              │   │       │   │                        │
│    ●───●───●───●───●              ○───●───●───●───●                        │
│                                                                             │
│    Alle Neuronen aktiv           20% zufällig deaktiviert (○)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Warum hilft Dropout gegen Overfitting?**

| Effekt | Erklärung |
|--------|-----------|
| **Verhindert Co-Adaptation** | Neuronen können sich nicht auf andere Neuronen "verlassen" |
| **Ensemble-Effekt** | Trainiert implizit viele verschiedene Sub-Netzwerke |
| **Robustere Features** | Jedes Neuron muss unabhängig nützlich sein |
| **Noise Injection** | Fügt Rauschen hinzu, das Generalisierung fördert |

**Dropout im ViT Predictor:**

Im DINO World Model wird Dropout an zwei Stellen im Predictor verwendet:

```yaml
# conf/predictor/vit.yaml
predictor:
  dropout: 0.1      # Dropout nach Attention & Feed-Forward Layers
  emb_dropout: 0    # Dropout nach Embedding Layer (aktuell 0)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ViT Predictor - Dropout Positionen                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Embedding                                                            │
│       │                                                                     │
│       ▼                                                                     │
│  [Embedding Dropout] ← emb_dropout (Standard: 0)                           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────┐                                  │
│  │  Transformer Block (×6)              │                                  │
│  │  ┌────────────────────────────────┐  │                                  │
│  │  │  Multi-Head Attention          │  │                                  │
│  │  │         │                      │  │                                  │
│  │  │    [Dropout] ← dropout (0.1)   │  │                                  │
│  │  │         │                      │  │                                  │
│  │  │  Feed-Forward Network          │  │                                  │
│  │  │         │                      │  │                                  │
│  │  │    [Dropout] ← dropout (0.1)   │  │                                  │
│  │  └────────────────────────────────┘  │                                  │
│  └──────────────────────────────────────┘                                  │
│       │                                                                     │
│       ▼                                                                     │
│  Output                                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Empfohlene Dropout-Werte:**

| Datensatz-Größe | Empfohlenes Dropout | Begründung |
|-----------------|---------------------|------------|
| < 20 Episoden | 0.3 - 0.4 | Starke Regularisierung nötig |
| 20-50 Episoden | 0.2 - 0.3 | Moderate Regularisierung |
| 50-100 Episoden | 0.1 - 0.2 | Leichte Regularisierung |
| > 100 Episoden | 0.0 - 0.1 | Wenig/keine Regularisierung |

**Wichtig:** Dropout ist nur während des **Trainings** aktiv. Bei Inferenz (`model.eval()`) werden alle Neuronen verwendet, aber die Gewichte werden skaliert.

#### 8.3.5 Empfohlene Konfiguration für kleine Datensätze (< 50 Episoden)

```yaml
# conf/train.yaml Anpassungen
training:
  epochs: 50          # Reduziert von 100
  decoder_lr: 1e-4    # Reduziert von 3e-4
  predictor_lr: 2e-4  # Reduziert von 5e-4

predictor:
  dropout: 0.2        # Erhöht von 0.1
```

#### 8.3.6 Optimales Checkpoint-Auswahl

Bei Overfitting **NICHT** das letzte Checkpoint verwenden! Stattdessen:

1. W&B Dashboard öffnen
2. Epoch mit niedrigstem `val_loss` identifizieren (oft Epoch 40-60)
3. Entsprechendes Checkpoint laden: `checkpoints/model_XX.pth`

```python
# Beispiel: Bestes Modell laden
best_epoch = 45  # Aus W&B abgelesen
checkpoint_path = f"outputs/DATUM/ZEIT/checkpoints/model_{best_epoch}.pth"
```

---

## 9. Training starten

### 9.1 Basis-Kommando

```bash
cd /path/to/dino_wm

# Standard-Training
python train.py env=franka_cube_stack

# Mit expliziten Parametern
python train.py env=franka_cube_stack \
    frameskip=5 \
    num_hist=3 \
    training.epochs=100 \
    training.batch_size=8
```

### 9.2 Empfohlene Parameter für deinen Datensatz

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

### 9.3 Erwartete Ausgabe

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

### 9.4 Monitoring mit Weights & Biases

Training wird automatisch zu W&B geloggt:
- Projekt: `dino_wm_debug` (wenn `debug=True`) oder `dino_wm`
- Metriken: Loss-Kurven, Image Metrics, Visualisierungen

---

## 9.5 Klarstellung: Pixel-Space vs. Meter-Space — Kein Problem für das Training

> **Analyse vom 09.02.2026** — Die DINO-WM-Architektur ist vollständig einheitsagnostisch.

### Hintergrund der Fragestellung

Beim Vergleich des Franka Cube Stacking Datensatzes mit den Referenz-Datensätzen (Rope, Push-T, Wall) fiel auf, dass diese **unterschiedliche Koordinatensysteme** verwenden. Die Befürchtung: Kann das DINO World Model mit Meter-Koordinaten trainiert werden, wenn es mit Pixel-/Sim-Koordinaten entwickelt wurde?

### Analyse der Action-Räume aller Datensätze

| Datensatz | Action-Dimensionen | Koordinatensystem | Roh-Wertebereich |
|-----------|-------------------|-------------------|------------------|
| **Rope** (Referenz) | 4D: `[x_start, z_start, x_end, z_end]` | FleX-Simulator-Einheiten | ca. ±4 |
| **Push-T** (Referenz) | 2D: `[dx, dy]` | Pixel-Space (÷100) | ca. ±0.2 |
| **Wall** (Referenz) | 2D: `[a1, a2]` | Eigener Sim-Space | ca. ±0.5 |
| **Franka** (unserer) | 6D: `[x_s, y_s, z_s, x_e, y_e, z_e]` | Meter (Isaac Sim, lokal) | ca. 0.0–0.8 |

**Zentrale Erkenntnis:** Schon die Referenz-Datensätze sind untereinander **nicht einheitlich** — Rope nutzt Sim-Einheiten (±4), Push-T nutzt skalierte Pixel (±0.2), Wall nutzt wieder andere Sim-Einheiten (±0.5). Die Architektur wurde **bewusst** so designed, dass das Koordinatensystem keine Rolle spielt.

### Warum die Einheit irrelevant ist — Der Datenfluss

```
Schritt 1: Z-Score-Normalisierung (im Dataset-Loader)
──────────────────────────────────────────────────────
Rohdaten (beliebige Einheit)  →  normalized = (raw - mean) / std  →  ~N(0, 1)

  Rope:   [-3.2, 1.1, -2.8, 0.5] → norm. ≈ [-0.8, 0.3, -0.7, 0.1]
  Franka: [0.45, 0.02, 0.35, 0.51, -0.01, 0.38] → norm. ≈ [-0.2, 0.1, 0.9, 0.3, -0.1, 0.6]
  → Für das Modell sehen BEIDE wie ~N(0,1)-verteilte Vektoren aus!

Schritt 2: Action Encoder (lernbar)
──────────────────────────────────────────────────────
normalized_action (action_dim) → nn.Conv1d → action_embedding (10D)
  → Lineare Projektion, lernt beliebige Skalierung
  → Keine hardcodierten Annahmen über Einheiten

Schritt 3: Predictor (ViT)
──────────────────────────────────────────────────────
[visual_patches, proprio_emb, action_emb] → ViT Predictor → predicted_patches
  → Action-Embedding ist nur Conditioning-Signal
  → Loss wird NUR auf visuellen Patches berechnet
  → Action-Skala hat keinen Einfluss auf den Gradienten
```

### Voraussetzungen (beide erfüllt ✅)

1. **`action_dim` korrekt konfiguriert:** In `conf/env/franka_cube_stack.yaml` ist `action_dim` passend zum Datensatz-Format gesetzt (6 für `ee_pos`-Format, 4 für `delta_pose`).

2. **`action_mean`/`action_std` korrekt berechnet:** Der `FrankaCubeStackDataset`-Loader berechnet Z-Score-Statistiken on-the-fly aus allen Episoden. Seit dem Grid-Offset-Fix (Commit `a9af071`) enthalten die Daten korrekte lokale Meter-Werte → Mean/Std sind realistisch.

### Was NICHT nötig ist

- ❌ Konvertierung Meter → Pixel
- ❌ Anpassung der Action-Skala an Referenz-Datensätze
- ❌ Sonderbehandlung im Modell oder Preprocessor
- ❌ Änderung der Loss-Funktion

---

## 10. Glossar

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

---

## 🚨 KRITISCH: Action-Observation Temporale Alignment-Analyse (20.02.2026)

### Problemstellung

Es wurde eine **fundamentale Inkompatibilität** zwischen dem FCS-Datensatz (Franka Cube Stacking) und der Referenz-Konvention des DINO-WM Papers (Rope/Deformable Environments) bei der zeitlichen Zuordnung von Actions und Observations identifiziert.

**Kernfrage:** Beschreibt `action[t]` den Übergang von `obs[t]` zu `obs[t+1]` (vorwärtsblickend) oder den Übergang, der zu `obs[t]` führte (rückwärtsblickend)?

### Analyse der Referenz-Konvention (Rope / Deformable Environment)

#### Datenfluss bei der Generierung

In `FlexEnvWrapper.rollout()` ([env/deformable_env/FlexEnvWrapper.py](env/deformable_env/FlexEnvWrapper.py#L156)):
```python
def rollout(self, seed, init_state, actions):
    obs, state_dct = self.prepare(seed, init_state)  # obs_initial (VOR jeder Action)
    obses, rewards, dones, infos = self.step_multiple(actions)  # T Action-Ergebnisse
    for k in obses.keys():
        obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])  # obs_initial VORANSTELLEN
    # Ergebnis: T+1 Beobachtungen für T Actions
```

- `obs_initial` = Zustand **VOR** der ersten Action
- `step_multiple()` liefert T Bilder — jeweils am **ENDE** jeder Action
- `rollout()` stellt `obs_initial` voran → **T+1 Beobachtungen für T Actions**

#### Konvention in den .pth-Dateien

In `DeformDataset.get_frames()` ([datasets/deformable_env_dset.py](datasets/deformable_env_dset.py#L95)):
```python
image = torch.load(obs_dir / "obses.pth")
act = self.actions[idx, frames]
image = image[frames]  # Gleicher Index!
```

In `plan.py` `sample_traj_segment_from_dset()` ([plan.py](plan.py#L263)):
```python
obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
act = act[offset : offset + self.frameskip * self.goal_H]
# traj_len = frameskip * goal_H + 1 → obs hat EINE MEHR Einträge als act!
```

**Beweis:** Das Planning nimmt `traj_len = frameskip * goal_H + 1` Observations aber nur `frameskip * goal_H` Actions. Die eine Extra-Observation ist die **initiale Beobachtung VOR der ersten Action**.

#### Rope-Konvention zusammengefasst

```
Zeitachse:  obs[0]  →(act[0])→  obs[1]  →(act[1])→  obs[2]  → ...
              ↑                    ↑                    ↑
          INITIAL              Ergebnis              Ergebnis
        (vor Action)          von act[0]            von act[1]
```

- `obs[t]` = Zustand **VOR** Ausführung von `act[t]`
- `act[t]` beschreibt die Transition `obs[t] → obs[t+1]` (vorwärtsblickend)
- `act[t]` wird **VON `obs[t]` aus** ausgeführt

### Analyse der FCS-Konvention (Franka Cube Stacking)

#### Datenfluss im primitive_data_logger.py

In `_save_primitive_h5()` ([isaacsim/00_Franka_Cube_Stack/...primitive_data_logger.py](../isaacsim/00_Franka_Cube_Stack/Franka_Cube_Stacking/primitive_data_logger.py#L722)):
```python
# Bilder am ENDE des Primitivs (nach der Bewegung)
rgb = end_data["rgb_images"]
# EEF-Position am ENDE des Primitivs
ee_pos = end_data["ee_pos"] - env_offset
```

In `end_episode()`:
```python
# obses.pth wird aus ep["imgs_list"] erstellt — NUR End-of-Primitive Bilder
obses = torch.stack(ep["imgs_list"]).squeeze(1)
torch.save(obses, obses_path)
```

#### Verifizierung mit echten Daten

```
=== Episode 0 (20 Primitive) ===
  obses.pth: 20 Bilder — GLEICHE Anzahl wie Actions
  Timing-Check: action[t].start_pos vs eef_states[t-1][:3]
  t=1: start=[0.4894,0.0899,0.4166] vs eef[0]=[0.4853,0.0852,0.4180] => OK (d<0.01)
  t=2: start=[0.5238,0.1816,0.3655] vs eef[1]=[0.5207,0.1711,0.3736] => OK (d≈0.014)
```

**Bestätigt:** `action[t].start_pos ≈ eef_states[t-1]` — Action t startet dort, wo Action t-1 endete. Also beschreibt `action[t]` den Übergang von `obs[t-1]` nach `obs[t]`.

#### FCS-Konvention zusammengefasst

```
Zeitachse:  ???  →(act[0])→  obs[0]  →(act[1])→  obs[1]  →(act[2])→  obs[2]
              ↑                ↑                    ↑                    ↑
          INITIAL           Ergebnis             Ergebnis             Ergebnis
       (NICHT GESPEICHERT)  von act[0]           von act[1]           von act[2]
```

- `obs[t]` = Zustand **NACH** Ausführung von `act[t]` (Ergebnis)
- `act[t]` beschreibt die Transition `obs[t-1] → obs[t]` (rückwärtsblickend)
- `act[t]` hat `obs[t]` **PRODUZIERT**
- Es gibt **KEIN** initiales Bild vor der ersten Action

### Der Off-by-One Fehler

#### Im TrajSlicerDataset (Training)

Mit `frameskip=2`, `num_frames=7`:
```python
obs_window = [obs[start], obs[start+2], obs[start+4], ..., obs[start+12]]  # 7 Bilder
act_window = [(act[start],act[start+1]), (act[start+2],act[start+3]), ...]  # 7 Gruppen
```

**In Rope-Konvention (korrekt):**
- `act_group[0] = (act[start], act[start+1])`
- `act[start]` transitiert `obs[start] → obs[start+1]`
- `act[start+1]` transitiert `obs[start+1] → obs[start+2]`
- Kombiniert: `obs[start] → obs[start+2]` = `obs_window[0] → obs_window[1]` ✓

**In FCS-Konvention (FEHLERHAFT):**
- `act_group[0] = (act[start], act[start+1])`
- `act[start]` transitiert `obs[start-1] → obs[start]` ← **RÜCKWÄRTS** (aus dem Window hinaus!)
- `act[start+1]` transitiert `obs[start] → obs[start+1]` ← Nur EIN Schritt vorwärts
- Kombiniert: `obs[start-1] → obs[start+1]`, NICHT `obs[start] → obs[start+2]` ❌
- **Das Modell erhält Actions, die NICHT zur beobachteten Transition passen!**

#### Auswirkung auf das Training

Das VWorldModel lernt in `forward()` ([models/visual_world_model.py](models/visual_world_model.py#L192)):
```python
z_src = z[:, :num_hist]     # Encode(obs[0..5], act[0..5])
z_tgt = z[:, num_pred:]     # Encode(obs[1..6], act[1..6])
z_pred = predict(z_src)     # Vorhersage
loss = criterion(z_pred, z_tgt)  # Soll z_tgt matchen
```

- In **Rope**: `z_src[0] = (obs[0], act[0])` wobei `act[0]` von `obs[0]` wegführt → Modell lernt: "gegeben Zustand + ausgehende Action → vorhersage nächster Zustand"
- In **FCS**: `z_src[0] = (obs[0], act[0])` wobei `act[0]` zu `obs[0]` **hinführte** → Modell lernt: "gegeben Ergebnis + Action die es produzierte → vorhersage nächstes Ergebnis"

Das Modell lernt eine **semantisch verschobene Korrelation**. Die Actions beschreiben nicht die Transition zwischen den beobachteten Zuständen, sondern eine um 1 verschobene Transition.

#### Auswirkung auf das Planning (CEM)

Beim Planning (`VWorldModel.rollout()`):
1. CEM schlägt Actions vor als "was soll der Roboter **als nächstes tun**" (vorwärtsblickend)
2. Das Modell erwartet aber Actions als "was hat den **aktuellen Zustand produziert**" (rückwärtsblickend)
3. → **Semantischer Mismatch** zwischen CEM und Modell

Dies könnte eine **Hauptursache** für die CEM-Divergenz sein (neben den fehlenden Action Bounds).

### Implementierter Fix: START-Bild statt END-Bild im Data Logger (21.02.2026)

**Entscheidung:** Der Fix wurde im `primitive_data_logger.py` implementiert (nicht im Dataloader), weil:
1. Daten sind an der Quelle korrekt — jeder Loader/jedes Tool bekommt die richtige Semantik
2. Kein Datenverlust (weiterhin T obs + T act pro Episode, statt T-1 beim Loader-Shift)
3. Kein Workaround in jedem neuen Loader nötig
4. Debugging einfacher — Rohdaten auf Disk haben die richtige Semantik

**Konkrete Änderung in `_save_primitive_h5()`:**

```python
# VORHER (falsch): Bild am ENDE des Primitivs
rgb = end_data["rgb_images"]     # obs[t] = Zustand NACH act[t] ❌

# NACHHER (korrekt): Bild am START des Primitivs
rgb = obs_data["rgb_images"]     # obs[t] = Zustand VOR act[t] ✓
# obs_data = start_data (übergeben von _finalize_primitive_fixed/phase)
```

Alle drei Aufrufstellen (`_finalize_primitive_fixed`, `_finalize_primitive_phase`, `_segment_into_fixed_primitives`) übergeben jetzt `start_data` statt `end_data` als Beobachtungsdaten. Die Action bleibt unverändert (`[start_pos → end_pos]`).

**Resultierende Konvention (identisch mit Rope):**
```
obs[0]  →(act[0])→  obs[1]  →(act[1])→  obs[2]  → ...
  ↑                    ↑                    ↑
START Prim 0       START Prim 1          START Prim 2
(VOR Bewegung)     (= ENDE Prim 0)      (= ENDE Prim 1)
```

### Gleicher Fix im MinDataLogger (21.02.2026)

Der `min_data_logger.py` hatte den **gleichen backward-looking Bug**: Jede H5-Datei enthielt das Bild vom aktuellen Zustand (`image[t]`) zusammen mit `action = [prev_pos, curr_pos]` — d.h. das Bild zeigte den Zustand NACH der Action.

**Fix: Buffer-Ansatz für Forward-Looking Alignment:**
```python
# VORHER (falsch, backward-looking):
# H5(t): image=image[t], action=[pos(t-1), pos(t)] → Bild zeigt Zustand NACH Action ❌

# NACHHER (korrekt, forward-looking):
# Step 0: buffer {image0, pos0}    → kein H5 (kein Forward-Action bekannt)
# Step 1: save H5 (image0, [pos0→pos1]) → buffer {image1, pos1}
# Step 2: save H5 (image1, [pos1→pos2]) → buffer {image2, pos2}
# end():  save H5 (image2, [pos2→pos2]) → Dummy-Action (letzter Obs) ✓
```

Neue Hilfsmethoden:
- `_save_step_h5()`: Zentrale H5-Speicherlogik (obs_data + action)
- `_flush_buffer_final()`: Speichert letzten Buffer mit Dummy-Action in `end_episode()`
- `_save_last_frame_if_needed()`: Angepasst auf `buffered_at_frame`-Tracking

### Anpassungen im Anwendungscode (21.02.2026)

**`fcs_main_parallel.py`:** Keine Code-Änderungen nötig (log_step()-API unverändert).
Dokumentation aktualisiert:
- `collect_timestep_data()`: Docstring dokumentiert Erfassungsreihenfolge (VOR Action-Ausführung)
- `save_successful_episode()`: Docstring referenziert Rope-Konvention beider Logger
- Hauptschleife: Kommentar betont WICHTIG: Erfassung VOR action-Ausführung

**`planning_client.py`:** Keine Code-Änderungen nötig (PlanningLogger = separater, simpler Logger).
Dokumentation aktualisiert:
- `PlanningLogger`: Docstring dokumentiert temporale Konvention
- `log_step_if_active()`: Docstring dokumentiert Aufruf-Semantik (VOR nächster Action)

#### Konsequenz

⚠️ **Der Datensatz muss NEU GENERIERT werden. Das aktuell trainierte Modell (260218/11-58) wurde mit der falschen Konvention trainiert und muss nach der Neugenerierung NEU TRAINIERT werden.**

### Überprüfungs-Checkliste

| Prüfpunkt | Rope (Referenz) | FCS (alt, fehlerhaft) | FCS (nach Fix) |
|-----------|-----------------|----------------------|-----------------|
| `obs[t]` zeigt Zustand... | VOR `act[t]` | NACH `act[t]` ❌ | VOR `act[t]` ✓ |
| `act[t]` beschreibt... | `obs[t]→obs[t+1]` | `obs[t-1]→obs[t]` ❌ | `obs[t]→obs[t+1]` ✓ |
| Bild-Zeitpunkt | START (vor Bewegung) | ENDE (nach Bewegung) ❌ | START (vor Bewegung) ✓ |
| Actions pro Episode | T | T | T |
| Obs pro Episode | T | T | T |
| act_group passt zu obs_window | ✓ | ❌ (verschoben) | ✓ |
| Datenverlust | — | — | Keiner ✓ |

