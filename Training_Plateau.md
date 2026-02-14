# Training-Plateau-Analyse: DINO-WM auf Franka Cube Stacking

> **Datum:** 15. Februar 2026  
> **Kontext:** Drei aufeinanderfolgende Trainingsläufe mit steigender Datensatzgröße (200 → 500 → 1000 Episoden) zeigen keine signifikante Verbesserung der Vorhersagequalität. Diese Analyse untersucht die Ursachen und gibt Handlungsempfehlungen.

---

## Inhaltsverzeichnis

1. [Zusammenfassung der Runs](#1-zusammenfassung-der-runs)
2. [Paper-Referenzwerte](#2-paper-referenzwerte)
3. [Vergleich: Deine Werte vs. Paper](#3-vergleich-deine-werte-vs-paper)
4. [Identifizierte Probleme](#4-identifizierte-probleme)
   - 4.1 [Konfundierte Experimente](#41-konfundierte-experimente--drei-variablen-gleichzeitig-geändert)
   - 4.2 [Batch-Size zu klein](#42-batch-size-zu-klein-8-vs-32)
   - 4.3 [Unvollständige Epochen](#43-keine-der-runs-hat-100-epochen-erreicht)
   - 4.4 [Abgesenkte Learning Rates](#44-learning-rates-abgesenkt)
   - 4.5 [Bug: val_z_*_err_next1](#45-bug-val_z_err_next1-bei--2)
   - 4.6 [Inhärente Task-Komplexität](#46-inhärente-task-komplexität-franka-vs-paper-environments)
5. [Ist val_z_visual_loss ≈ 0.3 normal?](#5-ist-val_z_visual_loss--03-normal)
6. [Handlungsempfehlungen](#6-handlungsempfehlungen)
7. [Quellen](#7-quellen)

---

## 1. Zusammenfassung der Runs

Drei Trainingsläufe, identifiziert über die W&B-Legenden:

| Eigenschaft | Gelb (200ep) | Rot (500ep) | Hellblau (1000ep) |
|---|---|---|---|
| **W&B Run** | `!00_AI10_RO10_nCa4_nCu1_nS23_LRde1e-4_LRpr2e-4_hf5_h3_DO01_` | `2026-02-09/17-59-59_franka_cube_stack_f2_h4_p1` | `2026-02-14/21-30-33_franka_cube_stack_f2_h5_p1` |
| **Episoden** | 200 | 500 | 1000 |
| **num_hist (H)** | 3 | 4 | 5 |
| **frameskip** | 5 (aus Name) → 2 (unklar) | 2 | 2 |
| **Epochen (erreicht)** | ~30 | ~35 | ~50 |
| **batch_size** | 8 | 8 | 8 |
| **predictor_lr** | 2e-4 | 2e-4 | 2e-4 |
| **decoder_lr** | 1e-4 | 1e-4 | 1e-4 |

### Konvergierte Metriken (ca.-Werte aus Plots, Epoch ~50 für Hellblau)

| Metrik | Gelb (200ep) | Rot (500ep) | Hellblau (1000ep) | Trend |
|---|---|---|---|---|
| **val_z_visual_loss** | ~0.35 | ~0.30 | ~0.30 | ≈ Plateau |
| **val_z_visual_err_pred** | ~0.35 | ~0.30 | ~0.28 | Minimal ↓ |
| **val_z_visual_err_rollout** | ~0.35 | ~0.30 | ~0.30 | ≈ Plateau |
| **val_z_proprio_loss** | ~0.04 | ~0.02 | ~0.03 | Schwankend |
| **val_img_ssim_pred** | ~0.80 | ~0.82 | ~0.88 | Leicht ↑ |
| **val_img_lpips_pred** | ~0.15 | ~0.10 | ~0.10 | ≈ Plateau |
| **val_img_psnr_pred** | ~18 | ~20 | ~21 | Leicht ↑ |
| **val_img_mse_pred** | ~0.04 | ~0.02 | ~0.02 | ≈ Plateau |

**Kernbeobachtung:** Der `val_z_visual_loss` (MSE im DINOv2 Latent Space) konvergiert bei allen Runs auf ca. **0.28–0.35** — eine Verdreifachung der Datenmenge bringt bestenfalls marginale Verbesserung.

---

## 2. Paper-Referenzwerte

### 2.1 Prediction Quality (Table 4 & 9, Appendix A.7)

Das Paper berichtet **LPIPS** und **SSIM** auf den vorhergesagten Bildern (Predictor → Decoder → Bild vs. Ground Truth):

| Environment | LPIPS_pred ↓ | SSIM_pred ↑ | Baseline (bestes) LPIPS |
|---|---|---|---|
| **PushT** (2D Schieben) | **0.007** | **0.985** | 0.039 (DINO CLS) |
| **Wall** (2D Navigation) | **0.0016** | **0.997** | 0.002 (ResNet) |
| **Rope** (XArm, deformabel) | **0.009** | **0.985** | 0.023 (R3M) |
| **Granular** (XArm, Partikel) | **0.035** | **0.940** | 0.080 (R3M/ResNet) |

> **Quelle:** Zhou et al. (2025), Table 4 (Seite 8) und Table 9 (Appendix A.7, Seite 17)

### 2.2 Scaling-Law-Ablation (Table 5, Appendix A.4.1)

Das Paper untersucht auf PushT den Einfluss der Datensatzgröße:

| Dataset Size | Success Rate ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| n=200 | 0.08 | 0.949 | 0.056 |
| n=1000 | 0.48 | 0.973 | 0.013 |
| n=5000 | 0.72 | 0.981 | 0.007 |
| n=10000 | 0.88 | 0.984 | 0.006 |
| n=18500 | 0.92 | 0.987 | 0.005 |

> **Wichtig:** Im Paper verbessert sich LPIPS um **4.3x** von 200→1000 Episoden (0.056→0.013). Bei deinen Runs ist **quasi kein Unterschied** sichtbar. Das ist **nicht normal** — es deutet auf ein Problem in der Trainingskonfiguration oder inhärente Task-Schwierigkeit hin.

> **Quelle:** Zhou et al. (2025), Table 5 (Appendix A.4.1, Seite 14)

### 2.3 Trainings-Hyperparameter laut Paper (Table 11 & 12)

**Shared Hyperparameters (alle Environments):**

| Parameter | Paper-Wert (Table 12) | Dein Wert | Abweichung? |
|---|---|---|---|
| Image Size | 224 | 224 | ✅ |
| Optimizer | AdamW | AdamW | ✅ |
| Epochs | **100** | **30–50** | ❌ **Unvollständig** |
| Batch Size | **32** | **8** | ❌ **4x kleiner** |
| Decoder LR | **3e-4** | **1e-4** | ❌ **3x kleiner** |
| Predictor LR | **5e-5** (Paper) | **2e-4** | ⚠️ **4x größer** |
| Action Encoder LR | 5e-4 | 5e-4 | ✅ |
| Action Emb Dim | 10 | 10 | ✅ |
| Encoder | DINOv2 ViT-S/14 (frozen) | DINOv2 ViT-S/14 (frozen) | ✅ |
| Predictor | ViT (depth=6, heads=16, mlp=2048) | ViT (depth=6, heads=16, mlp=2048) | ✅ |

**Environment-spezifische Hyperparameter (Table 11, vergleichbare Tasks):**

| Environment | H (num_hist) | Frameskip | Dataset Size | Traj. Length |
|---|---|---|---|---|
| **Rope** (6D Action, XArm) | 1 | 1 | 1000 | 5 |
| **Granular** (XArm) | 1 | 1 | 1000 | 5 |
| **PushT** (2D) | 3 | 5 | 18500 | 100–300 |
| **PointMaze** | 3 | 5 | 2000 | 100 |
| **Dein Franka CS** | **3→4→5** | **2** | **200→500→1000** | **~466** (932/2) |

> **Quelle:** Zhou et al. (2025), Table 11 & 12 (Appendix A.9, Seite 17–18)

### 2.4 Diskrepanz: Paper vs. GitHub-Code

**Kritischer Fund:** Der GitHub-Code hat `predictor_lr: 5e-4`, aber das Paper (Table 12) listet `5e-5`. Das ist ein **10-facher Unterschied**. Unklar, welcher Wert tatsächlich für die Paper-Ergebnisse verwendet wurde.

| Parameter | Paper Table 12 | GitHub `conf/train.yaml` | Dein Wert |
|---|---|---|---|
| `predictor_lr` | **5e-5** | **5e-4** | **2e-4** |
| `decoder_lr` | 3e-4 | 3e-4 | 1e-4 |

> **Quelle:** GitHub `gaoyuezhou/dino_wm`, Commit `0a9492f`, `conf/train.yaml`

---

## 3. Vergleich: Deine Werte vs. Paper

### 3.1 Bildqualitäts-Metriken (Predicted)

| Metrik | Paper Granular | Paper Rope | **Dein Franka (1000ep)** | Differenz |
|---|---|---|---|---|
| SSIM_pred ↑ | 0.940 | 0.985 | **~0.88** | **-6% bis -11%** |
| LPIPS_pred ↓ | 0.035 | 0.009 | **~0.10** | **3x–11x schlechter** |
| PSNR_pred ↑ | — | — | **~21 dB** | Kein Vergleich |

### 3.2 Bildqualitäts-Metriken (Reconstructed)

| Metrik | **Dein Franka (1000ep)** | Interpretation |
|---|---|---|
| SSIM_reconstructed | ~0.92 | Decoder funktioniert grundsätzlich |
| LPIPS_reconstructed | ~0.07 | Deutlich besser als predicted |
| PSNR_reconstructed | ~28 dB | Solide Rekonstruktion |

**Schlussfolgerung:** Der Decoder funktioniert gut (Reconstructed-Metriken sind akzeptabel). Das Problem liegt primär im **Predictor** — die vorhergesagten Latents weichen signifikant von den echten ab (`z_visual_loss ≈ 0.3`), was sich dann in schlechten `_pred`-Metriken niederschlägt.

---

## 4. Identifizierte Probleme

### 4.1 Konfundierte Experimente — Drei Variablen gleichzeitig geändert

**Problem:** Zwischen den Runs wurden sowohl die Datensatzgröße ALS AUCH `num_hist` geändert:

```
Run 1 (Gelb):     200ep + h3
Run 2 (Rot):      500ep + h4
Run 3 (Hellblau): 1000ep + h5
```

**Warum das problematisch ist:**

`num_hist` verändert direkt die Tokenlänge im Predictor-Input:

```
Tokens = num_hist × num_patches = H × 256

h=3:  3 × 256 =  768 Tokens
h=4:  4 × 256 = 1024 Tokens
h=5:  5 × 256 = 1280 Tokens
```

Mehr Tokens bedeutet:
1. **Schwierigere Attention-Aufgabe** für den ViT (Complexity $O(n^2)$)
2. **Mehr Parameterraum** zu lernen (Temporal-Relationen über längere Sequenzen)
3. **Höherer GPU-Speicherverbrauch** (erklart eventuell auch den kleinen Batch-Size)

Der potentielle Gewinn durch mehr Trainingsdaten (1000 vs. 200 Episoden) wird durch die erhöhte Modellkomplexität (h5 vs. h3) **aufgefressen oder überkompensiert**.

**Paper-Referenz:** Die Causal-Mask-Ablation (Table 6, Appendix A.4.2) zeigt, dass `h=3` mit Causal Mask auf PushT eine Success Rate von **0.92** erreicht — mehr History hilft nur marginal, wenn überhaupt. Für die Deformable Tasks (Rope, Granular) wurde sogar nur `h=1` verwendet.

### 4.2 Batch-Size zu klein (8 vs. 32)

**Konfiguration:**
```yaml
# Dein train.yaml
training:
  batch_size: 8  # GPU nur bei ~60% bei 8; 16 → OOM; 32 → OOM
```

**Paper-Standard:** `batch_size: 32` (Table 12)

**Auswirkungen eines 4x kleineren Batch-Size:**

| Aspekt | batch=32 | batch=8 | Effekt |
|---|---|---|---|
| Gradient-Rauschen | Niedrig | **Hoch** | Instabile Updates |
| Gradient-Steps/Epoch | N/32 | N/8 = **4x mehr** | Mehr, aber noisiger Steps |
| Effektive LR | LR | effektiv **~2x–4x LR** | Overshoot möglich |
| GPU-Effizienz | Optimal | Suboptimal | Langsameres Training |
| Generalisierung | Gut bei großem BS | Oft schlechter | Höherer val_loss |

**Empfehlung:** Gradient Accumulation implementieren:
```python
# Pseudo-Code für effektive batch_size=32 mit GPU batch_size=8
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Die `Accelerator`-Klasse von HuggingFace unterstützt das nativ:
```python
accelerator = Accelerator(gradient_accumulation_steps=4)
```

### 4.3 Keine der Runs hat 100 Epochen erreicht

**Aus den Plots (x-Achse "Step" = Epochen):**

| Run | Erreichte Epochen | Paper-Standard | Fehlende % |
|---|---|---|---|
| Gelb (200ep) | ~30 | 100 | **70%** |
| Rot (500ep) | ~35 | 100 | **65%** |
| Hellblau (1000ep) | ~50 | 100 | **50%** |

**Warum das kritisch ist:**

Aus den Plots ist erkennbar, dass insbesondere:
- `val_z_visual_loss` bei Epoch ~50 noch nicht vollständig flach ist (leichter Abwärtstrend)
- `val_img_ssim_pred` und `val_img_psnr_pred` noch steigen
- `val_img_lpips_pred` noch fällt

Das Modell ist **nicht voll konvergiert**. Bei größeren Datensätzen (1000ep) sind sogar **mehr** Epochen nötig als bei kleinen, weil:
1. Jede Epoche hat 4x mehr Batches als bei 200ep (bei gleicher batch_size)
2. Aber die Vielfalt der Daten braucht mehr Durchgänge um gelernt zu werden
3. Die Gradient-Steps pro Datenpunkt sind proportional, aber der Feature-Raum ist größer

### 4.4 Learning Rates abgesenkt

#### Predictor LR: Widersprüchliche Quellen

| Quelle | Predictor LR |
|---|---|
| Paper Table 12 | **5e-5** |
| GitHub `conf/train.yaml` | **5e-4** (10x höher!) |
| Dein `train.yaml` | **2e-4** |

Dein Wert liegt **zwischen** den beiden Quellen. Aber:
- Falls das Paper recht hat (5e-5), ist dein Wert **4x zu hoch** → Training könnte über das Minimum hinausschießen
- Falls der Code recht hat (5e-4), ist dein Wert **2.5x zu niedrig** → Training konvergiert langsamer

**Empfehlung:** Beide testen. Einen Run mit 5e-5 und einen mit 5e-4.

#### Decoder LR: 3x zu niedrig

| Parameter | Paper/Code | Dein Wert |
|---|---|---|
| `decoder_lr` | **3e-4** | **1e-4** |

Der Decoder-Loss beeinflusst zwar **nicht** die Planungsfähigkeit (er ist vom Predictor entkoppelt), aber er beeinflusst direkt die Bildqualitäts-Metriken (SSIM, LPIPS, PSNR). Ein langsamer Decoder erklärt teilweise die niedrigeren SSIM- und höheren LPIPS-Werte.

### 4.5 Bug: val_z_*_err_next1 bei -2

**Beobachtung:** In den Plots zeigen `val_z_visual_err_next1` und `val_z_proprio_err_next1` einen konstanten Wert von **-2** über alle Epochen.

**Ursache:** Bekannter Bug in `train.py`, Zeile ~449:

```python
slices = {
    "full": (None, None),
    "pred": (-self.model.num_pred, None),
    "next1": (-self.model.num_pred, -self.model.num_pred + 1),  # ← BUG
}
```

Mit `num_pred=1`:
```python
"next1": (-1, -1 + 1) = (-1, 0)
```

Das erzeugt ein **leeres Slice** `tensor[:, -1:0]`, da `-1 < 0` (Ende vor Anfang). Der resultierende Tensor hat Shape `(b, 0, ...)`, und der MSE-Loss gibt dann einen sinnlosen Default-Wert zurück, der als -2 geloggt wird.

**Auswirkung:** Keine auf das Training — die `next1`-Metrik ist nur ein Monitoring-Artefakt und fließt nicht in den Gradienten ein. Aber sie erzeugt irreführende Plots.

**Fix (optional):**
```python
"next1": (-self.model.num_pred, None if self.model.num_pred == 1 else -self.model.num_pred + 1),
```

### 4.6 Inhärente Task-Komplexität: Franka vs. Paper-Environments

Dies ist möglicherweise die **wichtigste Erkenntnis**: Franka Cube Stacking ist fundamental anders und schwerer als alle Environments im Paper.

#### Detaillierter Vergleich

| Eigenschaft | PushT | Wall | Rope/Granular | **Franka Cube Stacking** |
|---|---|---|---|---|
| **Szene** | 2D Top-Down | 2D Top-Down | 3D isometrisch | **3D perspektivisch** |
| **Agent DOF** | 2 (x, y) | 2 (x, y) | 6 (XArm) | **7 (Panda) + Gripper** |
| **Action-Dim** | 2 | 2 | 4 | **6** (EE-Positionen) |
| **Action Space** | Kontinuierlich, klein | Kontinuierlich, klein | 4D XArm | **6D im 3D-Raum** |
| **Dynamik** | Kontakt-Schieben | Freie Bewegung | Seil-Deformation | **Rigid-Body + Greifen + Stapeln** |
| **Visuell** | Einfache Shapes | Einfache 2D | Seil + Tisch | **Roboter + Würfel + Tisch + Schatten** |
| **Traj. Länge** | 100–300 | 50 | **5** | **~466** (932 / frameskip=2) |
| **Dataset Size** | 18.500 | 1.920 | 1.000 | **200–1000** |
| **Visuelle Veränderung** | Objekt bewegt sich | Agent bewegt sich | Seil verformt sich | **Minimale Pixel-Änderungen** (Roboterarm + kleiner Würfel) |

#### Kritischer Punkt: Visuelles Signal-Rausch-Verhältnis

Bei Franka Cube Stacking ändern sich zwischen zwei Frames nur wenige Pixel (der Roboterarm bewegt sich minimal, der kleine Würfel verschiebt sich um Millimeter). Der Großteil des Bildes (Tisch, Hintergrund, Roboter-Basis) bleibt **identisch**. Das bedeutet:

- Die DINOv2-Patch-Embeddings unterscheiden sich zwischen aufeinanderfolgenden Frames **minimal**
- Das MSE-Ziel ist daher schon bei einer "naiven" Vorhersage (vorheriges Frame kopieren) niedrig
- Ein `frameskip=2` macht kaum Unterschied bei ~466 Frames → die visuelle Differenz pro Skip ist winzig
- Ein `frameskip=5` (wie im Paper für PushT/Maze) würde mehr visuelle Differenz erzeugen

**Vergleich der effektiven "visuellen Veränderungsrate":**

```
PushT:     100 Frames / fs=5 = 20 effektive Steps, große Objekt-Verschiebung pro Step
Rope:      5 Frames / fs=1 = 5 effektive Steps, starke Deformation pro Step
Franka CS: 932 Frames / fs=2 = 466 Steps, MINIMALE Änderung pro Step
```

---

## 5. Ist val_z_visual_loss ≈ 0.3 normal?

### Kurzantwort: Nein — aber die absolute Zahl ist schwer zu bewerten

Das Paper berichtet **keine** rohen `z_visual_loss`-Werte (MSE im DINOv2 Embedding Space). Es berichtet nur die nachgelagerten Metriken (LPIPS, SSIM, Success Rate). Daher fehlt ein direkter Vergleichswert.

### Was der Wert bedeutet

Der `z_visual_loss` ist:

$$\mathcal{L}_{pred} = \|p_\theta(\text{enc}(o_{t-H:t}), \phi(a_{t-H:t})) - \text{enc}(o_{t+1})\|^2$$

Mit:
- `enc(o)` hat Shape `(batch, 256, 384)` — 256 Patches mit 384 Dimensionen
- MSE wird über alle Patches und Dimensionen gemittelt
- Wert 0.3 bedeutet: Die durchschnittliche quadrierte Abweichung pro Element ist 0.3

### Was auf ein Problem hindeutet

1. **Alle drei Runs konvergieren zum gleichen Wert (~0.3)** → Modell-Kapazitäts-Plateau
2. **Mehr Daten helfen nicht** → Mögliche Ursachen:
   - Konfundierte num_hist-Änderung (siehe 4.1)
   - Das Modell hat genug Kapazität für die Trainings-Distribution, aber der Loss wird durch andere Faktoren dominiert (z.B. die letzten ~2% der Frames, die schwer vorherzusagen sind)
   - Der Action-Encoder mappt die 6D-Actions nicht informativ genug
3. **Die Lücke zwischen Reconstructed und Predicted ist groß:**
   - `val_img_ssim_reconstructed ≈ 0.92` vs. `val_img_ssim_pred ≈ 0.88`
   - `val_img_lpips_reconstructed ≈ 0.07` vs. `val_img_lpips_pred ≈ 0.10`
   - Das zeigt, dass der Decoder prinzipiell funktioniert, aber die Predictor-Outputs zu verrauscht sind

### GitHub-Issues als Indiz

Issue #24 ("Difficulty Reproducing Results for the 'Rope' Environment", Oktober 2025) berichtet ein **ähnliches Problem** für das Rope-Environment: Die Predictions driften schnell von der Realität ab. Der Issue ist **unbeantwortet** (Open, keine Replies vom Autor).

> **Quelle:** https://github.com/gaoyuezhou/dino_wm/issues/24

---

## 6. Handlungsempfehlungen (Verfeinert: Low Hanging Fruits zuerst)

> **Leitprinzip:** Minimaler Aufwand → größter Hebel. Zuerst die Konvergenz-Blockaden beseitigen, dann die Task-Lernbarkeit erhöhen, zuletzt die wissenschaftliche Ablation durchführen.

### Priorität 1: Das "Stabilisierungs-Setup" — BS & LR fixen (Höchste Priorität)

**Diagnose:** Die aktuelle Konfiguration (hohe LR + kleine Batch-Size) **verhindert Konvergenz**. Das erklärt, warum der Loss bei ~0.3 stagniert — das Modell springt um das Optimum herum, statt es zu erreichen.

#### Aktion 1a: Gradient Accumulation auf 4 setzen (effektiv BS=32)

Über die `Accelerator`-Klasse von HuggingFace:

```python
# In train.py, Trainer.__init__:
self.accelerator = Accelerator(
    log_with="wandb",
    gradient_accumulation_steps=4,  # 4 × 8 = 32 effektiv
)
```

**Aufwand:** ~5 Minuten Code-Änderung.

**Effekt:** 4x stabilere Gradienten. Die effektive Batch-Size erreicht den Paper-Standard (32), ohne den GPU-Speicher zu überschreiten. Die bisherige Konfiguration (BS=8) erzeugt 4x so viel Gradient-Rauschen, was bei kleinen visuellen Deltas (Franka-Szene) besonders schädlich ist.

#### Aktion 1b: Predictor LR auf 5e-5 senken (Paper-Wert, Table 12)

```yaml
# In train.yaml:
training:
  predictor_lr: 5e-5  # Paper Table 12, aktuell 2e-4 (4x zu hoch!)
```

**Aufwand:** Konfig-Änderung.

**Begründung:** Die Kombination aus hoher LR (2e-4) und kleiner BS (8) ergibt eine **effektive LR die ~16x über dem Paper-Setup liegt** (4x durch BS × 4x durch LR). Mit BS=32 (via Accumulation) und LR=5e-5 stimmt das Verhältnis wieder.

**Erwartetes Ergebnis:** Der `val_z_visual_loss` sollte **deutlich unter 0.3 fallen** — idealerweise Richtung 0.1 oder darunter.

---

### Priorität 2: Signal-to-Noise Verhältnis verbessern — Frameskip erhöhen

**Diagnose (aus 4.6):** Bei `frameskip=2` ist das visuelle Delta $\Delta z = \text{enc}(o_{t+2}) - \text{enc}(o_t)$ zwischen aufeinanderfolgenden Frames fast Null. Das Modell lernt effektiv die **Identität** (kopiere den vorherigen Frame), nicht die Dynamik.

#### Aktion: Frameskip auf 5 setzen

```yaml
# In train.yaml:
frameskip: 5  # Paper-Standard für alle Nicht-Deformable-Tasks
```

**Vergleich der visuellen Veränderungsraten:**

```
frameskip=2: 932/2 = 466 Steps/Episode, Δ_visual ≈ winzig   → Modell lernt Identität
frameskip=5: 932/5 = 186 Steps/Episode, Δ_visual ≈ sichtbar → Modell lernt Dynamik
```

**Warum das hilft:**
1. Erzwingt sichtbare Bewegungsvorhersage → stärkeres Gradienten-Signal
2. Die MSE hat einen "Boden", der durch das Rauschen der fast-identischen Embeddings bestimmt wird. Frameskip=5 hebt die Decke an.
3. Das Paper verwendet `frameskip=5` für PushT (2D, ähnliche Dynamik-Komplexität), PointMaze, Reacher und Wall. Nur die Deformable-Tasks (Rope, Granular) nutzen `frameskip=1`, aber dort sind die Trajektorien nur **5 Frames lang** — also sind alle Frames visuell verschieden.

**Aufwand:** Konfig-Änderung, kein neuer Datensatz nötig (Frameskip wird beim Slicing angewendet).

---

### Priorität 3: Decoder LR auf 3e-4 zurücksetzen

```yaml
# In train.yaml:
training:
  decoder_lr: 3e-4  # Paper-Wert & GitHub-Default, aktuell 1e-4
```

**Aufwand:** Konfig-Änderung.

**Begründung:** Der Decoder ist vom Predictor entkoppelt (Gradients fließen nur durch den Decoder, nicht durch den Predictor). Daher ist das eine "Gratis"-Verbesserung der Bildqualitäts-Metriken (SSIM, LPIPS, PSNR), ohne das Predictor-Training zu beeinflussen. Da dein aktueller `decoder_lr` (1e-4) 3x unter dem Paper-Wert liegt, erklärt das einen Teil der schlechteren Bildmetriken.

---

### Priorität 4: Saubere Daten-Ablation (Erst nach Fix 1 & 2)

> **Voraussetzung:** `val_z_visual_loss` fällt unter ~0.1 mit den Fixes aus Prio 1 & 2. Erst dann macht ein Datensatz-Vergleich wissenschaftlich Sinn.

**Aktion:** Clean Run mit konservativen, fixen Parametern:

```bash
python train.py \
  env=franka_cube_stack \
  frameskip=5 \
  num_hist=3 \                    # ← Konservativ, Paper-Standard für PushT
  training.epochs=100 \           # ← Volle Paper-Epochen
  training.batch_size=8 \         # ← BS=8, aber mit Gradient Accumulation = 4
  training.predictor_lr=5e-5 \    # ← Paper-Wert
  training.decoder_lr=3e-4        # ← Paper-Wert
```

Dann **zwei** identische Runs durchführen:
- Run A: `FRANKA_DATA_PATH=.../NEps500_...`
- Run B: `FRANKA_DATA_PATH=.../NEps1000_...`

**Vergleich:** Zeigt, ob mehr Daten bei korrekter Konfiguration tatsächlich helfen.

---

### Zusammenfassung des verfeinerten Plans

| Prio | Maßnahme | Warum? | Aufwand |
|---|---|---|---|
| **1** | Gradient Acc. (BS=32) + `predictor_lr=5e-5` | Behebt das **Konvergenz-Problem** (Ursache für Loss 0.3). Effektive LR war ~16x zu hoch. | 5 min Code + Konfig |
| **2** | `frameskip=5` | Erhöht das visuelle Delta, macht den Task **"lernbarer"**. Modell soll Dynamik lernen, nicht Identität. | Konfig-Änderung |
| **3** | `decoder_lr=3e-4` | "Gratis" Verbesserung der Bildmetriken (SSIM, LPIPS, PSNR). | Konfig-Änderung |
| **4** | Clean Run (1000ep, h=3, 100 Epochen) | Nachweis des Scaling-Laws — **erst sinnvoll nach Fix 1 & 2**. | 1 Run (~8h) |

### Fazit

Das bisherige Setup war **"zu laut"** (Batch Size 8 statt 32) und **"zu aggressiv"** (LR 2e-4 statt 5e-5), um die feinen Bewegungen des Franka-Arms im DINOv2-Latent-Space zu lernen. Zusätzlich war das **visuelle Signal zu schwach** (frameskip=2), sodass das Modell kaum Unterschied zwischen Input und Target hatte.

Mit den Anpassungen aus Prio 1 & 2 sollte der `val_z_visual_loss` **signifikant unter 0.3 fallen**. Erst danach macht die Daten-Ablation (Prio 4) wissenschaftlich Sinn.

---

## 7. Quellen

| Referenz | Beschreibung |
|---|---|
| Zhou et al. (2025) | "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning", arXiv:2411.04983v2 |
| Table 4 (S. 8) | LPIPS-Vergleich aller World Models |
| Table 5 (S. 14) | Scaling-Law-Ablation auf PushT (200–18500 Trajektorien) |
| Table 9 (S. 17) | LPIPS und SSIM über alle Environments |
| Table 11 (S. 17) | Environment-spezifische Hyperparameter |
| Table 12 (S. 18) | Shared Training-Hyperparameter |
| GitHub Repo | https://github.com/gaoyuezhou/dino_wm (Commit `0a9492f`) |
| GitHub Issue #24 | "Difficulty Reproducing Results for the 'Rope' Environment" (unbeantwortet) |
| DINO WM Metriken Doku | `dino_wm/DINO_WM_METRIKEN.md` |
| Training Doku | `dino_wm/DINO_WM_TRAINING_DOCUMENTATION.md` |
