# CEM Loss Analyse — Modell 260218/11-58 vs 2026-02-09/17-59-59

**Datum:** 2026-02-19  
**Modell:** `260218/11-58` (num_hist=6, action_dim=8, 1000 Episoden, EEFfix+TrackGrip)  
**Vergleich:** `2026-02-09/17-59-59` (num_hist=4, action_dim=6, 500 Episoden)

## Beobachtung

CEM-Loss bei ~1,08 statt ~0,34 bei ähnlichen Bedingungen.
Geplante Zielpositionen wirkten unsinnig (z.B. `target_ee=[0.843, 0.480, 0.003]`).

## Ursachen

### 1. 🔴 Kritischer Index-Bug in Action-Extraktion (BEHOBEN)

**Die Hauptursache für unsinnige Robot-Bewegungen.**

Das Action-Format hat sich von 6D auf 8D geändert:

| Format | Indizes |
|--------|---------|
| **Alt (6D):** `[x_s, y_s, z_s, x_e, y_e, z_e]` | `action[3:6]` = `[x_e, y_e, z_e]` ✓ |
| **Neu (8D):** `[x_s, y_s, z_s, g_s, x_e, y_e, z_e, g_e]` | `action[3:6]` = `[g_s, x_e, y_e]` ✗ |

Beim Wechsel auf 8D-Actions (mit Gripper) wurde der Index-Offset nicht angepasst.
Der Roboter erhielt `[gripper_state, x_end, y_end]` als Zielposition statt `[x_end, y_end, z_end]`.

**Konsequenzen:**
- **Roboter fährt zu falschen Positionen** — `g_start` (0 oder 1) wird als x-Koordinate interpretiert
- **z-Koordinate wird ignoriert** — der Roboter bewegt sich nur in einer Ebene
- **MPC-Feedback-Loop ist gebrochen** — falsche Position → falsches Bild → CEM optimiert vom falschen Zustand → Loss konvergiert nicht

**Betroffene Dateien (alle gefixt):**
- `planning_server.py:570` — Display: `sa[3], sa[4], sa[5]` → `sa[4], sa[5], sa[6]`
- `planning_client.py:533` — **Ausführung + Display**: `action[3:6]` → `action[4:7]`
- `planning_client.py:577` — **Ausführung + Display**: `action[3:6]` → `action[4:7]`
- `planning_eval.py:421` — **Ausführung**: `action[3:6]` → `action[4:7]`
- `planning_eval.py:439` — **Ausführung**: `action[3:6]` → `action[4:7]`

**Hinweis:** `00_Archiv/planning_eval_pre_refactor.py` enthält den gleichen Bug, wurde aber
als Archiv-Datei nicht gefixt.

**Rückwärtskompatibilität:** Der Fix nutzt `action[4:7] if len(action) >= 7 else action[3:6]`,
sodass alte 6D-Actions weiterhin korrekt verarbeitet werden.

### 2. 🟡 Größerer CEM-Suchraum (80D vs 60D)

| Parameter | Vorher (0.34) | Jetzt (~1.08) |
|-----------|---------------|---------------|
| action_dim | 6 (ohne Gripper) | **8** (mit Gripper) |
| frameskip | 2 | 2 |
| full_action_dim | 12 | **16** |
| horizon | 5 | 5 |
| **Suchraum** | **60D** | **80D** (+33%) |

Der Gripper-State in der Action ist notwendig (Greifen unmöglich ohne),
erhöht aber den Suchraum um 33%. CEM benötigt entsprechend mehr Samples
oder Iterationen für gleiche Konvergenzqualität.

### 3. Unterschiedliche Modelle — CEM-Loss nicht direkt vergleichbar

Die beiden Modelle haben unterschiedliche:
- **Latent Spaces** (durch verschiedene Trainings-Datasets, Epochen, Hyperparameter)
- **Action-Normalisierung** (verschiedene mean/std aus unterschiedlichen Datasets)
- **Temporal-Kontext** (num_hist=6 vs 4)

Ein absoluter Loss-Vergleich (1.08 vs 0.34) ist daher nur bedingt aussagekräftig.
Die Loss-Reduktion pro Optimierung (% Konvergenz) ist aussagekräftiger.

## num_hist-Analyse: Kein Mismatch bei Inferenz

**Befürchtung:** Das Modell wurde mit `num_hist=6` trainiert, erhält aber beim Planning
nur 1 Frame als `obs_0`. Führt das zu schlechteren Predictions?

**Ergebnis: Nein.** Die Architektur ist dafür ausgelegt:

### Causal Mask deckt variable Kontextlängen ab

Der ViT-Predictor verwendet eine **kausale Block-Maske** während des Trainings:

```
Position 0 → sieht nur Frame 0         (1 Frame Kontext)
Position 1 → sieht Frames 0–1          (2 Frames Kontext)
Position 2 → sieht Frames 0–2          (3 Frames Kontext)
...
Position 5 → sieht Frames 0–5          (6 Frames Kontext)
```

Der Loss wird **per Position** berechnet. Das Modell lernt also **explizit**,
aus 1, 2, 3, ... bis num_hist Frames zu predicten. Genau das passiert beim
autoregressiven Rollout im CEM:

| Rollout-Step | Frames verfügbar | Predictor sieht | Trainings-Äquivalent |
|---|---|---|---|
| 0 | 1 | 1 Frame | Position 0 ✓ |
| 1 | 2 | 2 Frames | Position 1 ✓ |
| 2 | 3 | 3 Frames | Position 2 ✓ |
| ... | ... | ... | ... |
| 5 | 6 | 6 Frames (voll) | Position 5 ✓ |

### Positional Embeddings sind kompatibel

```python
# ViTPredictor.forward():
x = x + self.pos_embedding[:, :n]  # sliced auf tatsächliche Länge
```

Die Positional Embeddings werden von Index 0 aufwärts gesliced. Position 0
erhält immer dasselbe Embedding — egal ob die Sequenz 1 oder 6 Frames lang ist.
Es gibt keinen Train/Inference-Mismatch.

### Fazit num_hist

- **Mehr num_hist = strikt mehr Information** (sobald aufgewärmt)
- **Kein Degradation** bei weniger als num_hist Frames (Training deckt das ab)
- Bei Horizon=5 und num_hist=6 erreicht das Modell vollen Kontext beim letzten Step
- **num_hist=6 ist eine gute Wahl** — besonders für Tasks mit längeren temporalen Abhängigkeiten

## Zusammenfassung

| Problem | Schwere | Status |
|---------|---------|--------|
| Index-Bug `action[3:6]` → `action[4:7]` | **Kritisch** — Roboter fährt zu falschen Positionen | ✅ Behoben |
| Größerer Suchraum (80D vs 60D) | Moderat — inherent durch Gripper-Action | Akzeptiert |
| num_hist=6 Mismatch bei Inferenz | Keiner — Architektur handhabt das korrekt | Kein Problem |
| CEM-Loss absolut nicht vergleichbar | Info — verschiedene Modelle/Latent Spaces | Bekannt |
