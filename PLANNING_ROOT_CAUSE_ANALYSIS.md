# Root-Cause-Analyse: "Roboter bewegt sich sinnvoll, aber zum falschen Ort"

**Datum:** 2025-02-16  
**Modell:** `outputs/2026-02-14/21-30-33/` (999 Episoden, ActInt2, num_hist=5, frameskip=2)  
**Server-Log:** CEM: 1.903 → 0.978 (48.6% Reduktion), Sub-Action 0: target_ee=[0.614, 0.132, 0.142]

---

## 1. Symptom

> "Der Roboter bewegt sich sinnvoll, aber als würde der zu greifende Würfel wo anders liegen."

- Der Roboter führt physikalisch plausible Armbewegungen aus
- Die Zielposition stimmt räumlich nicht mit der tatsächlichen Würfelposition überein
- CEM-Loss konvergiert, aber auf einen hohen Wert (~0.97 statt ~0.34)

---

## 2. Systematische Untersuchung

| # | Hypothese | Ergebnis | Status |
|---|-----------|----------|--------|
| 1 | Dataset-Statistiken (Full vs. Split) | **IDENTISCH** — TrajSubset delegiert via `__getattr__` | ✅ Eliminiert |
| 2 | Preprocessing-Pipeline (Training vs. Server) | **IDENTISCH** — gleiche Werte für gleichen Frame | ✅ Eliminiert |
| 3 | Kamera/Format-Mismatch | **KONSISTENT** — BGR, cam_0, 224×224, uint8 | ✅ Eliminiert |
| 4 | Robot-Opazität | **KEIN PROBLEM** — RobOpac10 = 100% Opazität | ✅ Eliminiert |
| 5 | World Model Qualität | **GUT** — Recon MSE=0.0016, Pred MSE=0.0048, Ratio=2.94× | ✅ Eliminiert |
| 6 | Action-Format [start,end] | **KORREKT** — [3:6] als Target-EE durchgängig | ✅ Eliminiert |
| 7 | **Goal Proprio (ee_pos)** | **🎯 ROOT CAUSE** — Siehe unten | ❌ **GEFUNDEN** |

---

## 3. Root Cause: Falsches Goal-Proprio (ee_pos)

### 3.1 Das Problem

Der Planning-Server erhielt als Goal-Proprio:

```
Goal ee_pos = [0.429, 0.045, 0.415]   ← z = 0.415 (HOME-Position, Arm oben)
```

Die Dataset-Statistiken zeigen:

```
proprio_mean = [0.480, 0.017, 0.161]   ← z_mean = 0.161 (Manipulations-Höhe)
proprio_std  = [0.123, 0.161, 0.072]   ← z_std  = 0.072
```

**Der Goal z-Wert von 0.415 liegt 3.53 Standardabweichungen über dem Mittelwert!**

```
z_goal_normalized = (0.415 - 0.161) / 0.072 = 3.53σ
```

Typischer z-Bereich im Dataset: [0.09, 0.23] (±1σ)  
Goal z = 0.415 ist weit außerhalb der Trainingsverteilung.

### 3.2 Woher kommt der falsche Wert?

Der Wert `[0.429, 0.045, 0.415]` entspricht exakt `action[0:3]` (= `prev_ee_pos`) 
von Episode 000000, Frame 00 im Dataset:

```
Frame 00.h5: action[0:3] = [0.42919934, 0.04512064, 0.41505504]
```

Das ist die **Start-/Home-Position** des Roboterarms zu Beginn jeder Episode — 
der Arm steht hoch oben, bevor die Manipulation beginnt.

#### Mögliche Ursachen:

**A) `planning_client.py` ohne `--goal_image` (Fallback-Modus):**
```python
# planning_client.py, Zeile ~370:
goal_ee_pos = np.array(env.get_ee_pose()[0], dtype=np.float32)[:3]
# → Erfasst EE-Position DIREKT nach domain_randomization() + settle
# → Das ist die HOME-Position (z ≈ 0.41), NICHT eine Manipulations-Position!
```

**B) `planning_client.py` mit `--goal_image dataset:0:0` (erster Frame):**
```python
# Frame 0 hat eef_states[0:3] = [0.403, 0.023, 0.318]
# ABER die Home-Position vor Frame 0 wäre z ≈ 0.415
```

**C) `planning_eval.py` Stufe 2 mit gespeichertem `final_ee_pos`:**
- Der StackingController hat Phase 9 "Return to start" (Rückkehr zur Home-Position)
- `save_demo_data()` erfasst `final_ee_pos` NACH Controller-Abschluss
- → `final_ee_pos` könnte die retrahierte Home-Position sein (z ≈ 0.41)
- **ABER:** Die überprüften `demo_meta.json` zeigen z ≈ 0.13-0.17 — im Normalbereich

### 3.3 Warum verursacht das "falsches Ziel, richtige Bewegung"?

Die CEM-Objective-Funktion (aus `planning/objectives.py`):

```python
loss = MSE(z_pred["visual"][:,-1:], z_tgt["visual"])      # Visuelles Ziel
     + 0.5 * MSE(z_pred["proprio"][:,-1:], z_tgt["proprio"])  # Proprio-Ziel (α=0.5)
```

**Visuelles Ziel:** Zeigt den Würfel auf dem Tisch → leitet den Arm zur korrekten XY-Position  
**Proprio-Ziel:** Sagt "Arm soll auf z=0.415 sein" → zieht den Arm NACH OBEN

Diese beiden Ziele **konfligieren**:
- Visual: "Geh zum Würfel (z ≈ 0.07)"
- Proprio: "Geh nach oben (z ≈ 0.41)"

Da `α = 0.5` und der Proprio-Fehler quadratisch mit 3.53σ skaliert:
- Proprio-Loss-Beitrag: ~0.5 × 3.53² ≈ **6.2** → dominiert!
- Visual-Loss-Beitrag: ~0.5

**Ergebnis:** Der CEM-Planner findet einen Kompromiss, der weder den Würfel 
korrekt greift NOCH zur Home-Position fährt → "sinnvolle Bewegung, falscher Ort".

---

## 4. Sekundäres Problem: Unvollständiges Training

```
model_latest.pth = model_9.pth (nur 9 von 100 konfigurierten Epochen!)
```

Das Training wurde nach Epoche 9 abgebrochen. Obwohl die WM-Sanity-Check 
akzeptable Vorhersagen zeigt (Ratio 2.94×), könnte ein vollständig trainiertes 
Modell bessere CEM-Konvergenz erreichen.

---

## 5. Lösungsvorschläge

### 5.1 Sofort: Korrektes Goal-Proprio senden (KRITISCH)

**Option A: Goal-ee_pos aus Datensatz korrekt laden**
```python
# Verwende den LETZTEN Frame der Episode (nicht den ersten!)
# planning_client.py --goal_image /pfad/dataset:0:-1
#                                                 ↑ -1 = letzter Frame
```

**Option B: Goal-ee_pos aus `demo_meta.json` laden** (planning_eval.py)
- Stufe 1 speichert `final_ee_pos` → Diese Werte sind korrekt (z ≈ 0.13-0.17)
- Prüfe aber, ob Phase 9 "Return to start" nicht VOR dem Capture abläuft

**Option C: Fallback-Modus reparieren** (planning_client.py ohne --goal_image)
```python
# STATT: goal_ee_pos = env.get_ee_pose()  # HOME-Position!
# RICHTIG: goal_ee_pos von einem relevanten Manipulations-Zustand nehmen
# Zum Beispiel: Würfel-Position + Offset als Ziel-EE-Position
```

### 5.2 Kurzfristig: Robustheit erhöhen

**Option D: Alpha (Proprio-Gewicht) reduzieren**
```python
# In planning_server.py:
objective_fn = hydra.utils.call({
    "_target_": "planning.objectives.create_objective_fn",
    "alpha": 0.1,  # War: 0.5 → Reduziert Proprio-Einfluss
    "base": 2, "mode": "last",
})
```

**Option E: Proprio-Clipping/Validierung im Server**
```python
# In img_to_obs():
# Prüfe ob ee_pos im Trainings-Bereich liegt
ee_z_normalized = (ee_pos[2] - proprio_mean[2]) / proprio_std[2]
if abs(ee_z_normalized) > 2.0:
    print(f"  ⚠ Goal ee_pos z={ee_pos[2]:.3f} ist {ee_z_normalized:.1f}σ "
          f"vom Trainings-Mittelwert entfernt!")
```

### 5.3 Mittelfristig: Training fortsetzen

```bash
# Training von Epoche 9 fortsetzen bis Epoche 100
python train.py --resume outputs/2026-02-14/21-30-33
```

---

## 6. Zusammenfassung

| Faktor | Impact | Fix-Aufwand |
|--------|--------|-------------|
| **Falsches Goal-Proprio** | 🔴 HOCH — dominiert CEM-Loss, falsche Zielrichtung | 🟢 Einfach — korrekten ee_pos senden |
| Unvollständiges Training | 🟡 MITTEL — 9/100 Epochen, WM funktioniert aber akzeptabel | 🟡 Mittel — GPU-Zeit nötig |
| CEM-Hyperparameter | 🟢 GERING — 300 Samples, 30 Steps sind Paper-konform | ⚪ Optional |

**Hauptursache:** Der Planning-Client sendet eine Goal-EE-Position (z=0.415, Home-Position), 
die 3.53σ außerhalb der Trainingsverteilung liegt. Dies verursacht einen dominierenden 
Proprio-Loss im CEM-Objective, der den visuellen Loss überlagert und den Planner 
in die falsche Richtung lenkt.

**Empfohlene sofortige Maßnahme:** Korrektes Goal-Proprio senden (z ≈ 0.07-0.17 für Cube-Manipulation).
