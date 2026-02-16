# DINO WM Planning Pipeline — End-to-End Datenfluss

## Referenz-Konfiguration (500-Episoden-Modell)

```
Modell:           2026-02-09/17-59-59
Encoder:          DINOv2 ViT-S/14 (frozen)
Predictor:        ViT (depth=6, heads=16, mlp_dim=2048)
Decoder:          VQVAE (channel=384)
img_size:         224 × 224
patch_size:       14
num_patches:      (224/14)² = 16 × 16 = 256
encoder_emb_dim:  384  (ViT-S)
num_hist:         4
num_pred:         1
frameskip:        2
concat_dim:       1  (Dimension-Concatenation)

action_dim:       6  (base)  →  6 × 2 = 12  (mit frameskip)
proprio_dim:      3  (base)  →  3 × 1 = 3   (num_proprio_repeat=1)
action_emb_dim:   10  →  10 × 1 = 10  (num_action_repeat=1)
proprio_emb_dim:  10  →  10 × 1 = 10  (num_proprio_repeat=1)

GPU:              NVIDIA RTX A5000 (24 GB)
```

---

## ÜBERSICHT: 7 Stufen der Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ISAAC SIM (Client)                          │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ 1. Bild  │    │ 2. EEF-Pos   │    │ 7. IK-Ausführung         │   │
│  │ erfassen │    │ auslesen     │    │ move_ee_to(target)       │   │
│  └────┬─────┘    └──────┬───────┘    └────────────▲─────────────┘   │
│       │                 │                          │                 │
│       └────────┬────────┘                          │                 │
│                ▼                                   │                 │
│     ┌──────────────────┐               ┌───────────┴──────────┐     │
│     │ TCP Socket SEND  │               │ TCP Socket RECV      │     │
│     │ {image, ee_pos}  │               │ {actions: [[6D],...]}│     │
│     └────────┬─────────┘               └───────────▲──────────┘     │
└──────────────┼─────────────────────────────────────┼────────────────┘
               │              Port 5555              │
               ▼                                     │
┌──────────────┼─────────────────────────────────────┼────────────────┐
│              │         DINO WM (Server)            │                │
│     ┌────────▼─────────┐               ┌───────────┴──────────┐     │
│     │ 3. Preprocessing │               │ 6. Denormalisierung  │     │
│     │ img_to_obs()     │               │ + Rearrange          │     │
│     │ transform_obs()  │               │ → Sub-Actions        │     │
│     └────────┬─────────┘               └───────────▲──────────┘     │
│              ▼                                     │                 │
│     ┌──────────────────┐               ┌───────────┴──────────┐     │
│     │ 4. WM Encoding   │               │ 5. CEM Optimierung   │     │
│     │ DINO + Proprio   │──────────────▶│ 300 Samples ×        │     │
│     │ + Action Embed   │               │ 30 Iterationen       │     │
│     └──────────────────┘               └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## STUFE 1: Bild erfassen (Isaac Sim)

**Datei:** `planning_eval.py` → `FrankaCubeStackWrapper.get_obs_for_planner()`

```
Kamera (Isaac Sim)
    │
    ▼
RGBA (224, 224, 4) uint8       ← Isaac Sim gibt RGBA zurück
    │  [:, :, :3]              ← Alpha-Kanal abschneiden
    ▼
RGB  (224, 224, 3) uint8       ← Standard-RGB
    │  [:, :, ::-1]            ← Kanal-Reihenfolge umkehren
    ▼
BGR  (224, 224, 3) uint8       ← DINO WM wurde auf BGR trainiert!
```

**Warum BGR?**
Die Trainingsdaten wurden mit `color_imgs[..., ::-1]` gespeichert
(PrimitiveDataLogger), was RGB→BGR konvertiert. Das Training des
DINOv2-Encoders und ViT-Predictors fand auf BGR-Bildern statt.
Die Inferenz muss dasselbe Format verwenden.

**Ausgabe:** `img` — numpy array `(224, 224, 3)` uint8, BGR

---

## STUFE 2: EEF-Position auslesen (Isaac Sim)

**Datei:** `planning_eval.py` → `FrankaCubeStackWrapper.get_ee_pose()`

```
franka.end_effector.get_world_pose()
    │
    ▼
(position, orientation)
    │
    ▼
position = (x, y, z) float64    ← Weltkoordinaten in Metern
    │  .astype(np.float32)[:3]
    ▼
ee_pos = [x, y, z] float32      ← 3D EEF-Position
```

**Typische Werte (Franka Workspace):**
```
x ∈ [0.3, 0.7]   ← Vorwärts/Rückwärts (vom Roboter-Sockel)
y ∈ [-0.3, 0.3]   ← Links/Rechts
z ∈ [0.05, 0.5]   ← Höhe über dem Tisch
```

**Normalisierungs-Statistiken (aus 500 Episoden):**
```
proprio_mean = [0.476, 0.017, 0.161]
proprio_std  = [0.124, 0.161, 0.072]
```

**Ausgabe:** `ee_pos` — numpy array `(3,)` float32

---

## STUFE 2b: Socket-Nachricht senden

**Datei:** `planning_eval.py` / `planning_client.py`

```
Nachricht (pickle dict):
{
    "cmd":    "plan",                    ← String
    "image":  np.ndarray (224,224,3),    ← uint8 BGR
    "ee_pos": np.ndarray (3,),           ← float32 [x, y, z]
}
    │
    │  pickle.dumps(msg)
    ▼
bytes                                    ← Serialisiert
    │
    │  sendall(len.to_bytes(8,'big') + data)
    ▼
TCP Socket (localhost:5555)              ← 8-Byte Länge + Payload
```

**Drei Kommandos:**

| Kommando | Wann | image | ee_pos |
|----------|------|-------|--------|
| `set_goal` | Einmalig vor dem Planen | Goal-Bild (BGR) | EEF am Goal-Zustand |
| `plan` | Online-MPC: jeder Step | Aktuelles Bild (BGR) | Aktuelle EEF-Pos |
| `plan_all` | Offline: einmalig | Start-Bild (BGR) | Start EEF-Pos |

---

## STUFE 3: Preprocessing (Server)

**Datei:** `planning_server.py` → `img_to_obs()` + `Preprocessor.transform_obs()`

### 3a. img_to_obs() — Rohdaten zu Obs-Dict

```
img (224, 224, 3) uint8 BGR
ee_pos (3,) float32
    │
    │  np.newaxis, np.newaxis → Batch + Time Dimensionen
    │  ee_pos.reshape(1, 1, 3)
    ▼
obs_dict = {
    "visual":  (1, 1, 224, 224, 3) float32    ← [batch, time, H, W, C]
    "proprio": (1, 1, 3) float32               ← [batch, time, proprio_dim]
}
```

### 3b. Preprocessor.transform_obs() — Normalisierung

```
obs["visual"] = (1, 1, 224, 224, 3) float32
    │
    │  rearrange("b t h w c -> b t c h w")     ← Channel-first für PyTorch
    ▼
(1, 1, 3, 224, 224) float32
    │
    │  / 255.0                                  ← Pixel zu [0, 1]
    ▼
(1, 1, 3, 224, 224) float32 ∈ [0, 1]
    │
    │  transforms.Resize(224)                   ← No-op (bereits 224)
    │  transforms.CenterCrop(224)               ← No-op (bereits 224)
    │  transforms.Normalize([0.5]*3, [0.5]*3)   ← (x - 0.5) / 0.5
    ▼
(1, 1, 3, 224, 224) float32 ∈ [-1, 1]          ← Fertig für DINOv2

obs["proprio"] = (1, 1, 3) float32
    │
    │  (proprio - proprio_mean) / proprio_std    ← z-Score
    │  Beispiel: ([0.47, 0.02, 0.16] - [0.476, 0.017, 0.161]) / [0.124, 0.161, 0.072]
    │          ≈ [-0.05, 0.02, -0.01]
    ▼
(1, 1, 3) float32 ∈ ca. [-3, 3]                ← Normalisierte Proprio
```

**KRITISCH:** Mit proprio=[0,0,0] (der alte Bug) ergab sich:
```
(0 - [0.476, 0.017, 0.161]) / [0.124, 0.161, 0.072] ≈ [-3.84, -0.11, -2.24]
→ Extrem out-of-distribution → Unsinnige Vorhersagen!
```

**Ausgabe:**
```
trans_obs_0 = {
    "visual":  (1, 1, 3, 224, 224) float32 ∈ [-1, 1]    ← auf GPU
    "proprio": (1, 1, 3) float32 ∈ ca. [-3, 3]           ← auf GPU
}
trans_obs_g = {  ← Gleiche Struktur für Goal }
```

---

## STUFE 4: World Model Encoding

**Datei:** `models/visual_world_model.py` → `encode_obs()` + `encode()`

### 4a. Visual Encoding — DINOv2

```
visual = (1, 1, 3, 224, 224) float32
    │
    │  rearrange("b t c h w -> (b*t) c h w")   ← Flatten batch+time
    ▼
(1, 3, 224, 224) float32
    │
    │  encoder_transform()                      ← DINOv2-eigene Normalisierung
    │  DinoV2Encoder.forward()
    │    └─ dinov2_vits14.forward_features()
    │    └─ ["x_norm_patchtokens"]              ← 256 Patch-Tokens
    ▼
(1, 256, 384) float32                           ← (B*T, num_patches, emb_dim)
    │
    │  rearrange("(b t) p d -> b t p d", b=1)
    ▼
visual_embs = (1, 1, 256, 384) float32         ← [batch, time, patches, dim]
```

**Patch-Berechnung:**
```
Bild: 224 × 224 Pixel
Patch-Size: 14 × 14
Patches pro Dimension: 224 / 14 = 16
Gesamte Patches: 16 × 16 = 256
Embedding pro Patch: 384 (ViT-S)
```

### 4b. Proprio Encoding — ProprioceptiveEmbedding

```
proprio = (1, 1, 3) float32                   ← [batch, time, proprio_dim]
    │
    │  ProprioceptiveEmbedding.forward()
    │    ├─ permute(0, 2, 1)                   ← (1, 3, 1) für Conv1d
    │    ├─ Conv1d(in=3, out=384, kernel=2, stride=2)
    │    │  Konfiguration: num_frames=2, tubelet_size=1
    │    │  ACHTUNG: Conv1d mit kernel_size=tubelet_size
    │    │  Input T=1, Output T' = ⌊(1-1)/1⌋ + 1 = 1
    │    └─ permute(0, 2, 1)
    ▼
proprio_emb = (1, 1, 384) float32             ← [batch, time', emb_dim]
```

Hinweis: Der ProprioceptiveEmbedding-Encoder hat `num_frames=2` in der Config,
aber die Conv1d-Kernel-Size ist `tubelet_size=1`. Die tatsächliche temporale
Verarbeitung hängt von der Input-Länge ab. Bei T=1 Input kommt T'=1 Output.

### 4c. Action Encoding — ProprioceptiveEmbedding

```
act = (300, H, 12) float32                    ← [samples, horizon, full_action_dim]
    │                                             12 = base_dim(6) × frameskip(2)
    │  ProprioceptiveEmbedding.forward()
    │    ├─ permute → Conv1d(in=12, out=384, kernel=1)
    ▼
act_emb = (300, H, 384) float32              ← [samples, horizon, emb_dim]
```

Hinweis: Für die Initialisierung wird nur `act_0 = act[:, :num_obs_init]`
verwendet (die erste Aktion für den Anfangsframe).

### 4d. Concat (concat_dim=1) — Dimension-Verkettung

```
visual_embs  = (B, T, 256, 384) float32      ← Visuelle Patch-Embeddings
    │
    │  proprio_tiled:
    │    proprio_emb (B, T, 1, 384)
    │    → repeat über alle 256 Patches: (B, T, 256, 384)
    │    → repeat × num_proprio_repeat(=1): (B, T, 256, 10)
    │    Hinweis: proprio_dim = 10 × 1 = 10
    │
    │  act_tiled:
    │    act_emb (B, T, 1, 384)
    │    → repeat über alle 256 Patches: (B, T, 256, 384)
    │    → repeat × num_action_repeat(=1): (B, T, 256, 10)
    │    Hinweis: action_dim = 10 × 1 = 10
    │
    │  torch.cat([visual, proprio_tiled, act_tiled], dim=3)
    ▼
z = (B, T, 256, 404) float32
     │         │    │
     │         │    └── 384 (visual) + 10 (proprio) + 10 (action)
     │         └─────── 256 Patches
     └───────────────── T Zeitschritte (1 beim Init)
```

**WICHTIG:** Jeder der 256 Patches bekommt die GLEICHEN Proprio- und
Action-Werte angehängt (Tiling). Das ist redundant, aber der ViT-Predictor
kann so über die Dimension-Achse auf alle Informationen zugreifen.

---

## STUFE 5: CEM-Optimierung

**Datei:** `planning/cem.py` → `plan()`

### 5a. Initialisierung

```
mu    = (1, H, 12) float32 ∈ [0]              ← Mittelwert der Verteilung
sigma = (1, H, 12) float32 ∈ [var_scale]      ← Standardabweichung
         │  │   │
         │  │   └── full_action_dim = 6 × 2 = 12
         │  └────── Horizon H (z.B. 5 offline, 2 online)
         └───────── n_evals = 1 (Server hat B=1)
```

**Aktions-Raum (normalisiert):**
Der CEM optimiert im z-normalisierten Aktionsraum.
Jede der H Horizon-Aktionen hat 12 Dimensionen:
```
action[h] = [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂]
             └─── frameskip 0 ───┘  └─── frameskip 1 ───┘
             [xs₀, ys₀, zs₀,       [xs₁, ys₁, zs₁,
              xe₀, ye₀, ze₀]        xe₁, ye₁, ze₁]
```

Bedeutung (nach Denormalisierung):
```
xs₀, ys₀, zs₀ = EEF-Start-Position Sub-Action 0 [Meter]
xe₀, ye₀, ze₀ = EEF-End-Position Sub-Action 0   [Meter] ← IK-Ziel!
xs₁, ys₁, zs₁ = EEF-Start-Position Sub-Action 1 [Meter]
xe₁, ye₁, ze₁ = EEF-End-Position Sub-Action 1   [Meter] ← IK-Ziel!
```

### 5b. CEM-Iterationsschleife

```
Für jede Iteration i = 1..opt_steps:
    │
    │  1. Samples generieren
    │     action = randn(300, H, 12) × sigma + mu
    │     action[0] = mu                     ← Erste Sample = aktueller Mittelwert
    │     → (300, H, 12) float32             ← Normalisierte Aktionen
    │
    │  2. Obs für alle Samples expandieren
    │     cur_trans_obs_0["visual"]  = repeat(obs_0, "1 ... -> 300 ...")
    │     cur_trans_obs_0["proprio"] = repeat(obs_0, "1 ... -> 300 ...")
    │     → visual:  (300, 1, 3, 224, 224)
    │     → proprio: (300, 1, 3)
    │
    │  3. World Model Rollout (mit Chunking!)
    │     ┌─────────────────────────────────────────────────┐
    │     │ ChunkedRolloutWrapper:                          │
    │     │   Chunk 1: Samples   0..24  → wm.rollout()     │
    │     │   Chunk 2: Samples  25..49  → wm.rollout()     │
    │     │   ...                                           │
    │     │   Chunk 12: Samples 275..299 → wm.rollout()    │
    │     │   → torch.cat(alle Chunks)                      │
    │     └─────────────────────────────────────────────────┘
    │     z_obses = {"visual": (300, H+2, 256, 384),
    │                "proprio": (300, H+2, 10)}
    │
    │  4. Objective berechnen
    │     loss = MSE(z_obses["visual"][:, -1:], z_obs_g["visual"])
    │          + 0.5 × MSE(z_obses["proprio"][:, -1:], z_obs_g["proprio"])
    │     → (300,) float32                   ← Ein Loss pro Sample
    │
    │  5. Eliten auswählen
    │     topk_idx = argsort(loss)[:topk]    ← z.B. top 30
    │     topk_action = action[topk_idx]     ← (30, H, 12)
    │
    │  6. Verteilung updaten
    │     mu    = topk_action.mean(dim=0)    ← (H, 12) neuer Mittelwert
    │     sigma = topk_action.std(dim=0)     ← (H, 12) neue Std
    │
    ▼
Ergebnis: mu = (1, H, 12) float32           ← Optimierte normalisierte Aktionen
```

### 5c. World Model Rollout (Detail)

```
wm.rollout(obs_0, act):
    │
    │  obs_0 = {"visual": (B, 1, 3, 224, 224), "proprio": (B, 1, 3)}
    │  act   = (B, H, 12)
    │
    │  num_obs_init = 1
    │  act_0  = act[:, :1]       ← (B, 1, 12)  Erste Aktion
    │  action = act[:, 1:]       ← (B, H-1, 12) Rest
    │
    │  z = encode(obs_0, act_0)  ← (B, 1, 256, 404)  Initial-Zustand
    │
    │  Auto-regressiver Predict-Loop:
    │  t=0: z_pred = predict(z[:, -4:])           ← max 4 Frames (num_hist)
    │       z_new = z_pred[:, -1:]                 ← Nächster Frame
    │       z_new = replace_actions(z_new, action[:, 0:1])
    │       z = cat([z, z_new])                    ← (B, 2, 256, 404)
    │  t=1: z_pred = predict(z[:, -4:])
    │       z_new = replace_actions(z_pred[:, -1:], action[:, 1:2])
    │       z = cat([z, z_new])                    ← (B, 3, 256, 404)
    │  ...
    │  t=H-2: → z = (B, H, 256, 404)
    │
    │  Finale Prediction (ohne Action):
    │  z_pred = predict(z[:, -4:])
    │  z_new = z_pred[:, -1:]
    │  z = cat([z, z_new])                         ← (B, H+1, 256, 404)
    │
    │  separate_emb(z):
    │    z_visual  = z[..., :384]                  ← (B, H+1, 256, 384)
    │    z_proprio = z[..., 384:394]               ← (B, H+1, 256, 10)
    │                 → [:, :, 0, :10]             ← (B, H+1, 10) Entile
    │    z_action  = z[..., 394:404]               ← (B, H+1, 256, 10)
    │                 → [:, :, 0, :10]             ← (B, H+1, 10) Entile
    ▼
z_obses = {"visual": (B, H+1, 256, 384), "proprio": (B, H+1, 10)}
```

### 5d. Objective Funktion (mode="last")

```
z_obs_pred = z_obses aus Rollout:
    visual:  (300, H+1, 256, 384)
    proprio: (300, H+1, 10)

z_obs_tgt = Goal-Encoding (einmalig berechnet):
    visual:  (300, 1, 256, 384)    ← Expanded Goal
    proprio: (300, 1, 10)

loss_visual  = MSE(pred["visual"][:, -1:],  tgt["visual"]).mean(dims 1..4)
             → (300,) float32

loss_proprio = MSE(pred["proprio"][:, -1:], tgt["proprio"]).mean(dims 1..2)
             → (300,) float32

loss = loss_visual + 0.5 × loss_proprio
     → (300,) float32               ← Ein Skalar pro Trajectory-Sample
```

---

## STUFE 6: Denormalisierung + Rearrange (Server)

**Datei:** `planning_server.py`

### 6a. CEM-Output extrahieren

```
actions = mu                              ← CEM Ergebnis
    │
    │  Für "plan" (Online): Nur erste Horizon-Aktion
    │  actions[0, 0:1]                    ← (1, 12) normalisiert
    │
    │  Für "plan_all" (Offline): Alle Horizon-Aktionen
    │  actions[0]                         ← (H, 12) normalisiert
    ▼
```

### 6b. Denormalisierung

```
actions_norm = (T, 12) float32            ← T=1 (online) oder T=H (offline)
    │
    │  Preprocessor.denormalize_actions():
    │  denorm = actions × action_std + action_mean
    │
    │  action_mean (12D) = [μ₁..μ₆, μ₁..μ₆]   ← repeat(base_mean, 2)
    │  action_std  (12D) = [σ₁..σ₆, σ₁..σ₆]   ← repeat(base_std, 2)
    │
    │  Base-Statistiken (aus 500 Episoden):
    │  mean = [0.476, 0.017, 0.161, 0.475, 0.017, 0.161]
    │  std  = [0.124, 0.161, 0.072, 0.125, 0.161, 0.072]
    │          ├─ x_start ─┤─ y_s ─┤─ z_s ─┤─ x_end ─┤─ y_e ─┤─ z_e ─┤
    ▼
denorm = (T, 12) float32                  ← Metrische Werte [Meter]
```

### 6c. Rearrange — Frameskip aufteilen

```
denorm = (T, 12) float32
    │
    │  rearrange("t (f d) -> (t f) d", f=2)
    │
    │  Jede 12D Horizon-Aktion wird in 2 × 6D Sub-Actions gesplittet:
    │  [a₁..a₆ | a₇..a₁₂] → [[a₁..a₆], [a₇..a₁₂]]
    ▼
sub_actions = (T×2, 6) float32
    │
    │  Jede Sub-Action hat das Format:
    │  [x_start, y_start, z_start, x_end, y_end, z_end]
    │   ├──── EEF vor der Bewegung ────┤├── EEF nach der Bewegung ──┤
    │
    │  .tolist()
    ▼
JSON-serialisierbare Liste von Listen
```

**Beispiel für H=5, offline:**
```
CEM Output:      (1, 5, 12)   ← 5 Horizon-Steps × 12D
denorm:          (5, 12)      ← Denormalisiert
sub_actions:     (10, 6)      ← 5 × 2 = 10 einzelne Bewegungen
```

**Beispiel für H=2, online (MPC):**
```
CEM Output:      (1, 2, 12)   ← 2 Horizon-Steps × 12D
denorm[0, 0:1]:  (1, 12)      ← Nur erster Horizon-Step
sub_actions:     (2, 6)        ← 2 Sub-Aktionen
```

---

## STUFE 6b: Socket-Antwort senden

```
response = {
    "status":    "ok",                           ← String
    "actions":   [[6D], [6D], ...],              ← Liste von 6D Listen
    "n_actions": 2 (online) oder 10 (offline),   ← Anzahl Sub-Actions
    "plan_time": 12.3,                           ← Nur bei plan_all [Sekunden]
}
    │
    │  pickle.dumps(response)
    │  sendall(len + data)
    ▼
TCP Socket → Client
```

---

## STUFE 7: IK-Ausführung (Isaac Sim)

**Datei:** `planning_eval.py` / `planning_client.py`

### 7a. Action-Interpretation

```
action = [x_start, y_start, z_start, x_end, y_end, z_end]
          │                           │
          │ IGNORIERT                  │ VERWENDET
          │ (Kontext/Referenz)         │ = IK-Zielposition
          ▼                           ▼
                                target_ee = action[3:6]
                                         = [x_end, y_end, z_end]
```

**Warum nur `action[3:6]`?**
Das Aktionsformat ist `ee_pos`: Start- und End-Position des EEF pro
Zeitschritt. Für die Ausführung ist nur die Zielposition relevant.
Die Start-Position ist implizit die aktuelle EEF-Position.

### 7b. IK via RMPFlow

```
target_ee = [x_end, y_end, z_end] float32    ← Zielposition [Meter]
    │
    │  env.move_ee_to(target_ee, max_steps=20, threshold=0.005)
    │
    │  Konvergenz-Schleife:
    │  for step in range(max_steps):
    │      joint_action = rmpflow.forward(
    │          target_end_effector_position=target_ee,
    │          target_end_effector_orientation=EE_DEFAULT_ORIENT
    │      )
    │      articulation_controller.apply_action(joint_action)
    │      world.step(render=True)
    │      ee_pos = franka.end_effector.get_world_pose()[0]
    │      if |ee_pos - target_ee| < threshold: BREAK
    ▼
Roboter hat sich zu target_ee bewegt (oder max_steps erreicht)
```

**Parameter:**
```
EE_DEFAULT_ORIENT = [1, 0, 0, 0]    ← Quaternion (w,x,y,z) = Nach unten
max_steps = 20                       ← Maximale Sim-Steps pro Sub-Action
threshold = 0.005                    ← 5mm Konvergenz-Schwelle
```

---

## ZUSAMMENFASSUNG: Datentypen-Tabelle

| Stufe | Variable | Shape | Dtype | Wertebereich | Einheit |
|-------|----------|-------|-------|-------------|---------|
| 1 | `img` (BGR) | (224, 224, 3) | uint8 | [0, 255] | Pixel |
| 2 | `ee_pos` | (3,) | float32 | [0.3..0.7, -0.3..0.3, 0.05..0.5] | Meter |
| 3a | `obs["visual"]` | (1, 1, 224, 224, 3) | float32 | [0, 255] | Pixel |
| 3a | `obs["proprio"]` | (1, 1, 3) | float32 | [0.3..0.7, ...] | Meter |
| 3b | `trans_obs["visual"]` | (1, 1, 3, 224, 224) | float32 | [-1, 1] | norm. |
| 3b | `trans_obs["proprio"]` | (1, 1, 3) | float32 | [-3, 3] | z-Score |
| 4a | `visual_embs` | (B, T, 256, 384) | float32 | — | Latent |
| 4b | `proprio_emb` | (B, T, 384) | float32 | — | Latent |
| 4d | `z` (concat) | (B, T, 256, 404) | float32 | — | Latent |
| 5a | `mu` (CEM init) | (1, H, 12) | float32 | [0] | z-norm. |
| 5a | `sigma` (CEM init) | (1, H, 12) | float32 | [var_scale] | z-norm. |
| 5b | `action` (Samples) | (300, H, 12) | float32 | [-3, 3] | z-norm. |
| 5b | `loss` | (300,) | float32 | [0, ∞) | MSE |
| 5b | `mu` (CEM result) | (1, H, 12) | float32 | [-2, 2] | z-norm. |
| 6b | `denorm` | (T, 12) | float32 | [0.1..0.7] | Meter |
| 6c | `sub_actions` | (T×2, 6) | float32 | [0.1..0.7] | Meter |
| 7a | `target_ee` | (3,) | float32 | [0.3..0.7, ...] | Meter |

---

## VARIABLEN-REFERENZ

### Modell-Konfiguration

| Variable | Wert | Bedeutung |
|----------|------|-----------|
| `img_size` | 224 | Bildgröße in Pixel |
| `patch_size` | 14 | DINOv2 Patch-Größe |
| `num_patches` | 256 | 16×16 Patches pro Bild |
| `encoder_emb_dim` | 384 | DINOv2 ViT-S Embedding-Dimension |
| `num_hist` | 4 | Frames im Predictor-Fenster |
| `num_pred` | 1 | Vorherzusagende Frames |
| `frameskip` | 2 | Sub-Actions pro Horizon-Step |
| `concat_dim` | 1 | 1 = Dimension-Concat, 0 = Token-Concat |

### Dimensionen

| Variable | Wert | Berechnung |
|----------|------|------------|
| `base_action_dim` | 6 | ee_pos Format: [xs, ys, zs, xe, ye, ze] |
| `full_action_dim` | 12 | base_action_dim × frameskip = 6 × 2 |
| `proprio_dim` | 3 | EEF Position: [x, y, z] |
| `action_emb_dim` | 10 | Nach ProprioceptiveEmbedding |
| `proprio_emb_dim` | 10 | Nach ProprioceptiveEmbedding |
| `num_action_repeat` | 1 | Wiederholungen im Tiling |
| `num_proprio_repeat` | 1 | Wiederholungen im Tiling |
| `total_emb_dim` | 404 | 384 + 10 + 10 (visual + proprio + action) |

### CEM-Parameter

| Variable | Online (MPC) | Offline |
|----------|-------------|---------|
| `horizon` | 2 | 5 |
| `num_samples` | 64 | 300 |
| `opt_steps` | 5 | 30 |
| `topk` | 10 | 30 |
| `var_scale` | 1.0 | 1.0 |
| `Sub-Actions gesamt` | 2×2 = 4 (nur 2 exec.) | 5×2 = 10 |

### Dataset-Statistiken (500 Episoden, ee_pos Format)

```
action_mean = [0.476, 0.017, 0.161, 0.475, 0.017, 0.161]
action_std  = [0.124, 0.161, 0.072, 0.125, 0.161, 0.072]
               x_s    y_s    z_s    x_e    y_e    z_e

proprio_mean = [0.476, 0.017, 0.161]
proprio_std  = [0.124, 0.161, 0.072]
                 x      y      z

state_mean = [14D Vektor]    ← EEF State: [pos(3), pos(3), quat(4), quat(4)]
state_std  = [14D Vektor]
```

---

## SEQUENZDIAGRAMM: Online-MPC

```
 Isaac Sim                           Planning Server
    │                                      │
    │  1. set_goal(goal_img, goal_ee)      │
    │ ─────────────────────────────────── ▶ │
    │                                      │  encode_obs(goal) → z_obs_g
    │  ◀ ─────────────────────────────── {"status":"ok"}
    │                                      │
    │  ┌── MPC Loop (max_steps mal) ──┐    │
    │  │                              │    │
    │  │  2. plan(cur_img, cur_ee)    │    │
    │  │ ─────────────────────────── ▶ │    │
    │  │                              │    │  transform_obs(cur) → trans_obs_0
    │  │                              │    │  CEM: 300 × 30 × rollout
    │  │                              │    │  denorm + rearrange → 2 Sub-Actions
    │  │  ◀ ─────────────────────── actions│
    │  │                              │    │
    │  │  3. Sub-Action 0: move_ee_to │    │
    │  │  4. Sub-Action 1: move_ee_to │    │
    │  │                              │    │
    │  └──────────────────────────────┘    │
    │                                      │
    │  5. quit()                           │
    │ ─────────────────────────────────── ▶ │
    ▼                                      ▼
```

---

## SEQUENZDIAGRAMM: Offline (plan_all)

```
 Isaac Sim                           Planning Server
    │                                      │
    │  1. set_goal(goal_img, goal_ee)      │
    │ ─────────────────────────────────── ▶ │
    │                                      │  encode_obs(goal) → z_obs_g
    │  ◀ ─────────────────────────────── {"status":"ok"}
    │                                      │
    │  2. plan_all(start_img, start_ee)    │
    │ ─────────────────────────────────── ▶ │
    │                                      │  transform_obs → trans_obs_0
    │                                      │  CEM: 300 × 30 × rollout
    │                                      │  denorm + rearrange → 10 Sub-Actions
    │  ◀ ─────────────────────────────── 10 actions
    │                                      │
    │  3. Step 0:  move_ee_to(action[0][3:6])
    │  4. Step 1:  move_ee_to(action[1][3:6])
    │     ...
    │  12. Step 9: move_ee_to(action[9][3:6])
    │                                      │
    │  13. quit()                          │
    │ ─────────────────────────────────── ▶ │
    ▼                                      ▼
```

---

## BUGFIXES IM PLANNING SERVER (16.02.2026)

> Systematische Analyse der Code-Pfade `plan.py` vs. `planning_server.py`.
> Fünf Bugs identifiziert und behoben. Details in DINO_WM_PLANNING_DOCUMENTATION.md §12.

### Übersicht der Fixes

```
┌─────────────────────────────────────────────────────────────────────┐
│                PLANNING SERVER BUGFIXES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FIX 1: model.eval() nach load_model()                             │
│  ─────────────────────────────────────                              │
│  Vorher: Model bleibt im train()-Modus                             │
│          → Dropout/Stochastik aktiv bei Inferenz                   │
│  Nachher: model.eval() → deterministische Predictions              │
│                                                                     │
│  FIX 2: Warm-Start Null-Bias behoben                               │
│  ─────────────────────────────────                                  │
│  Vorher: Letzte Action im Warm-Start = [0,0,...,0]                 │
│          → CEM-Init biased Richtung Dataset-Mittelwert             │
│  Nachher: Letzte bekannte Action wird wiederholt                   │
│          → Physikalisch sinnvolle Trägheitsannahme                 │
│                                                                     │
│  FIX 3: CUDA Cache-Fragmentierung behoben                          │
│  ──────────────────────────────────────                              │
│  Vorher: empty_cache() INNERHALB der Chunk-Schleife               │
│          → VRAM-Fragmentierung, paradoxerweise mehr OOM            │
│  Nachher: empty_cache() nur einmal NACH der Schleife               │
│                                                                     │
│  FIX 4: Evaluator=None → Bewusst akzeptiert                       │
│  ──────────────────────────────────────────                          │
│  Server hat keine Env → kein Early-Stop via Evaluator möglich      │
│  Client-seitige MPC-Loop übernimmt diese Rolle                     │
│                                                                     │
│  FIX 5: ChunkedRolloutWrapper Robustheit                           │
│  ──────────────────────────────────────                              │
│  Vorher: __getattr__ Fallback maskiert Fehler                      │
│  Nachher: Explizite to()/state_dict() Forwarding-Methoden          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Warm-Start: Warum Null-Bias ein Problem ist

```
MPC-Loop mit Null-Bias (ALT):
─────────────────────────────
Step 0: Plan = [a₀, a₁, a₂, a₃, a₄]     ← CEM von Null
Step 1: Init = [a₁, a₂, a₃, a₄, 0⃗]      ← Shift + NULLEN
                                  ↑
                        Im z-normalisierten Raum
                        = "bewege zum Dataset-Mittelwert"
Step 2: Init = [a₂, a₃, a₄, 0⃗, 0⃗]      ← 2 Null-Actions
Step 3: Init = [a₃, a₄, 0⃗, 0⃗, 0⃗]       ← Dominiert von Nullen!
→ Roboter driftet zum Workspace-Zentrum statt zum Ziel

MPC-Loop mit Last-Action-Repeat (NEU):
──────────────────────────────────────
Step 0: Plan = [a₀, a₁, a₂, a₃, a₄]
Step 1: Init = [a₁, a₂, a₃, a₄, a₄]     ← Shift + REPEAT
Step 2: Init = [a₂, a₃, a₄, a₄, a₄]     ← Fortsetzung der Tendenz
→ CEM startet mit physikalisch sinnvoller Schätzung
```

### CEM-Ablauf: Was plan.py tut vs. was man erwarten könnte

```
┌─────────────────────────────────────────────────────────────────────┐
│           HÄUFIGES MISSVERSTÄNDNIS (KORRIGIERT)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FALSCH: "n_evals parallele Envs wählen die beste Aktion"         │
│                                                                     │
│  RICHTIG: n_evals = verschiedene Szenarien (Init/Goal-Paare)       │
│           300 Samples = Kandidaten PRO Szenario (nur im WM!)       │
│           topk = Eliten-Selektion PRO Szenario                     │
│           Env-Evaluation = nur Monitoring + Early-Stop              │
│                                                                     │
│  Der CEM optimiert im LATENT SPACE des World Models.               │
│  Die echte Env wird nur zum Validieren benutzt, NICHT              │
│  zum Auswählen — die Auswahl passiert über die Objective           │
│  Function (MSE im Embedding-Space).                                 │
│                                                                     │
│  Im Planning Server: n_evals=1 (ein Roboter, ein Szenario)        │
│  → Kein Unterschied in der CEM-Qualität selbst                    │
│  → Unterschied nur: kein Early-Stop und kein Monitoring            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Verbleibende Architektur-Unterschiede (kein Bugfix möglich)

```
plan.py                                  planning_server.py
────────                                 ─────────────────
PlanEvaluator (Env-Rollout,              evaluator=None
Videos, Metriken, Early-Stop)            → Client übernimmt Eval

MPCPlanner (automatisches                Socket-Loop ersetzt MPC:
Replanning + Action-Maskierung)          plan → execute → neues Bild → plan

n_evals=5 (Batch über Szenarien)         n_evals=1 (ein Roboter)

prepare_targets() aus Dataset            Goals via Socket (set_goal)

WandB Logging + logs.json                LoggingRun auf stdout
```
