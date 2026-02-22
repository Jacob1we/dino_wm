# Commit Message: CEM Action Bounds Fix

## Kurz-Zusammenfassung (für `git commit -m`)

```
fix: CEM Action Bounds — Clamping, Gripper-Quantisierung, Sigma Floor

CEM sampelte unbegrenzt aus N(mu, sigma²), was zu physikalisch
unmöglichen EEF-Positionen führte (x=1.4m, y=-1.6m statt Workspace
[0.2-0.75, -0.33-0.39]). Drei Fixes:

1. Action Clamping: Alle Samples auf [-3, 3] in normalisiertem Raum
   begrenzt (deckt >99.7% der Trainingsdaten ab)
2. Gripper Quantisierung: Binäre Gripper-Dims (g_s, g_e) werden auf
   {-1, +1} im normalisierten Raum gesnapped (= {0, 1} denormalisiert)
3. Sigma Floor: sigma >= 0.01 verhindert vorzeitige Konvergenz in
   OOD-Regionen

Geänderte Dateien:
- planning/cem.py: Action Bounds, Gripper Quantisierung, Sigma Floor
- conf/planner/cem.yaml: action_lower_bound, action_upper_bound, sigma_min
- planning_server.py: Gripper-Index-Berechnung für 8D Actions
```

---

## Ausführliche Commit Message

### Symptom

Beim Planning mit Modell `260218/11-58` (8D Actions, `action_dim=8`) divergiert der CEM-Planner:

- **CEM-Loss bleibt bei ~0.9-1.0** (statt ~0.3 wie erwartet)
- **EEF-Zielposition driftet** mit jedem MPC-Schritt weiter vom Würfel weg
- **Physikalisch unmögliche Positionen** werden geplant:
  - `target_ee=[0.565, -0.624, -0.004]` (y=-0.624m, z=-0.004m → außerhalb Workspace)
  - `target_ee=[0.836, -1.069, -0.028]` (weit außerhalb)
  - `target_ee=[1.385, -1.596, 0.093]` (komplett unrealistisch)
- **Abstand EEF→Würfel steigt monoton**: 0.27m → 0.43m → 0.60m → ... → >1.0m

### Root Cause Analyse

#### 1. Unbegrenztes CEM-Sampling (Hauptursache)

Der CEM-Planner (`planning/cem.py`) sampelt Actions aus einer unbegrenzten Gaußverteilung:

```python
action = torch.randn(num_samples, horizon, action_dim) * sigma + mu
```

**Keine Begrenzung** — kein `clamp()`, keine Bounds, keine Constraints.

**Warum das zum Problem führt:**

Die **normalisierten Action-Bereiche** aus dem Trainingsdatensatz:

| Dimension | Mean  | Std   | Min (norm.) | Max (norm.) |
|-----------|-------|-------|-------------|-------------|
| x_s       | 0.484 | 0.116 | -2.27       | +2.25       |
| y_s       | -0.026| 0.152 | -2.00       | +2.71       |
| z_s       | 0.169 | 0.071 | -1.44       | +3.67       |
| g_s       | 0.500 | 0.500 | -1.00       | +1.00       |
| x_e       | 0.484 | 0.117 | -2.29       | +2.23       |
| y_e       | -0.027| 0.152 | -2.00       | +2.78       |
| z_e       | 0.168 | 0.070 | -1.51       | +3.75       |
| g_e       | 0.500 | 0.500 | -1.00       | +1.00       |

Der CEM startet mit `mu=0, sigma=1` (var_scale=1). Bei einem **80-dimensionalen Suchraum**
(H=5 × D=16) werden mit 300 Samples regelmäßig Werte von ±5-7σ gesampelt.

**Beispielrechnung** für den divergenten Fall:
```
Normalisierter Wert z = -7.0 für y_e:
  y_denorm = z × std + mean = -7.0 × 0.152 + (-0.027) = -1.09m
  → Das ist der Wert target_ee[1]=-1.069 aus dem Log!
```

Das World Model wurde nur auf Werten in [-2.3, 3.7] trainiert. Für Werte
von ±7 produziert es **beliebige Latent-Embeddings**, die zufällig niedrige
Loss-Werte haben können → CEM konvergiert zu OOD-"Lösungen".

#### 2. Kontinuierliches Gripper-Sampling

Gripper-Dimensionen (g_s, g_e) sind **binär** {0, 1} (open/closed), aber
der CEM sampelt sie als kontinuierliche Werte (z.B. 0.37, 0.82).

- Normalisiert: `g = 0.37` → denormalisiert: `0.37 × 0.5 + 0.5 = 0.685`
- Das WM hat so einen Wert nie gesehen → unzuverlässige Vorhersage

#### 3. Kein Sigma Floor

Nach Elite-Selektion:
```python
sigma[traj] = topk_action.std(dim=0)
```

Sigma kann auf ~0 kollabieren, wenn alle Top-K ähnlich sind.
→ CEM exploriert nicht mehr → bleibt in OOD-Region gefangen.

### Implementierte Fixes

#### Fix 1: Action Bounds (Clamping)

```python
# Nach Sampling, VOR Rollout:
action = action.clamp(min=self.action_lower_bound, max=self.action_upper_bound)
```

- Default: `[-3.0, +3.0]` in normalisiertem Raum
- Deckt >99.7% der Normalverteilung ab
- Verhindert physikalisch unmögliche Werte
- Konfigurierbar in `conf/planner/cem.yaml`

#### Fix 2: Gripper Quantisierung

```python
if self.gripper_indices is not None:
    for gi in self.gripper_indices:
        signs = torch.sign(action[:, :, gi])
        signs[signs == 0] = -1.0  # tie-break: open
        action[:, :, gi] = signs
```

- Gripper-Dims werden auf `{-1, +1}` im normalisierten Raum gesnapped
- Das entspricht `{0, 1}` nach Denormalisierung (mit mean=0.5, std=0.5)
- Gripper-Indices werden dynamisch berechnet:
  - 8D Base → Indices [3, 7]
  - Mit frameskip=2 → Full 16D Indices [3, 7, 11, 15]
  - 6D Base (kein Gripper) → None (keine Quantisierung)

#### Fix 3: Sigma Floor

```python
sigma[traj] = torch.clamp(sigma[traj], min=self.sigma_min)
```

- Default: `sigma_min=0.01`
- Garantiert minimale Exploration in jedem Optimierungsschritt
- Verhindert vorzeitige Konvergenz in OOD-Regionen

### Geänderte Dateien

| Datei | Änderung |
|-------|----------|
| `planning/cem.py` | Action Bounds, Gripper Quantisierung, Sigma Floor — 4 neue Parameter in `__init__`, 3 Codeblöcke in `plan()` |
| `conf/planner/cem.yaml` | Neue Parameter: `action_lower_bound`, `action_upper_bound`, `sigma_min` |
| `planning_server.py` | Gripper-Index-Berechnung aus `base_action_dim` + `frameskip`, Übergabe an CEM, erweiterte Setup-Ausgabe |

### Kompatibilität

- **Rückwärtskompatibel**: Alle neuen Parameter haben Defaults
- **6D Modelle**: Funktionieren wie bisher (gripper_indices=None)
- **Andere Environments**: point_maze, pusht, wall — profitieren von Action Bounds
- **plan.py** (Offline-Planner): Nutzt automatisch die neuen cem.yaml Defaults

### Erwartete Verbesserung

| Metrik | Vorher | Erwartet |
|--------|--------|----------|
| CEM-Loss | ~0.9-1.0 | ~0.3-0.5 |
| Target EEF | Außerhalb Workspace | Innerhalb [0.2-0.75, -0.33-0.39, 0.06-0.43] |
| EEF→Cube Dist | Monoton steigend (0.27→1.0m) | Abnehmend |
