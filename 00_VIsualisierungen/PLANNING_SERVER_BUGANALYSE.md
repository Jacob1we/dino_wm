# Planning Server — Bug-Analyse & Lösungsansätze

**Stand:** 16.02.2026  
**Betrifft:** `planning_server.py` (aktuell: Commit `68aebac`) vs. Referenz `plan.py`  
**Model:** `2026-02-14/21-30-33` (1000 Episoden, ActInt2, 100 Epochen)

---

## Inhaltsverzeichnis

1. [Architektur-Verständnis: Wie CEM wirklich funktioniert](#1-architektur-verständnis)
2. [Bug-Katalog (vollständig, mit Status)](#2-bug-katalog)
3. [Regressions-Analyse: Warum Loss von ~0.3 auf ~0.97?](#3-regressions-analyse)
4. [Lösungsansätze (nach Priorität)](#4-lösungsansätze)

---

## 1. Architektur-Verständnis

### Parallele Environments ≠ Echtzeit-Evaluation

Die `n_evals` parallelen Environments in `plan.py` sind **verschiedene Init/Goal-Paare**, NICHT verschiedene Rollout-Kandidaten für dasselbe Szenario:

```
┌─────────────────────────────────────────────────────────┐
│  CEM-Optimierung (rein im World Model, KEIN Env)        │
│                                                         │
│  300 Samples → WM.rollout() → Latent-Loss → topk       │
│  Ergebnis: mu (bester Plan), sigma (Unsicherheit)       │
│  ❌ Echte Env wird hier NIE berührt                     │
└─────────────────────────┬───────────────────────────────┘
                          │ alle eval_every Steps
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Zwischen-Evaluation (echte Env, NUR Monitoring)        │
│                                                         │
│  mu in echter Env ausführen                             │
│  ❌ Ergebnis fließt NICHT zurück in mu/sigma            │
│  ✅ Nur: Monitoring, Early-Termination, Video-Vergleich │
└─────────────────────────────────────────────────────────┘
```

### Drei Pfade im Original-Code

| Pfad | CEM-Auswahl | Env-Nutzung | Feedback in Planner? |
|------|-------------|-------------|---------------------|
| **300 Samples** | WM-Rollout → Latent-Loss → topk | ❌ Nie in Env | — |
| **eval_every** | — | ✅ mu in Env ausführen | ❌ Nur Monitoring + Early Stop |
| **MPC** (`planning/mpc.py`) | CEM als Sub-Planner | ✅ Env-Rollout nach jedem MPC-Step | ✅ Neuer `obs_0` für nächste CEM-Runde |

### Konsequenz für den Planning Server

Der Server hat **keine Env** → weder `eval_every`-Monitoring noch MPC-Feedback möglich. Das ist architekturbedingt korrekt: Der Isaac-Sim-Client übernimmt die Rolle der Env und sendet nach jeder Aktion ein neues Bild → "Client-seitiges MPC".

---

## 2. Bug-Katalog

### Übersicht

| # | Bug | Schwere | Status | Commit |
|---|-----|---------|--------|--------|
| 1 | `model.eval()` nie aufgerufen | 🔴 Kritisch | ✅ Gefixt | `68aebac` |
| 2 | `evaluator=None` — Kein Early-Stop | 🟡 Performance | ✅ Bewusst akzeptiert | — |
| 3 | Warm-Start füllt mit Nullen → Null-Bias | 🔴 Kritisch | ✅ Gefixt | `68aebac` |
| 4 | `empty_cache()` fragmentiert VRAM | 🟡 Performance | ✅ Gefixt | `68aebac` |
| 5 | `__getattr__`-Fallback maskiert Fehler | 🟡 Wartbarkeit | ✅ Gefixt | `68aebac` |
| 6 | Kein `torch.no_grad()` um CEM-Planner | ✅ Kein Bug | ✅ Korrekt | — |
| 7 | `img_to_obs` — Kein Bildformat-Handling | 🟡 Robustheit | ⚠️ Teilweise | `3b78dcb` |
| P | CEM-Parameter auf Paper-Werte | 🔴 Kritisch | ✅ Gefixt | `68aebac` |

### Bug 1: `model.eval()` wird nie aufgerufen (✅ GEFIXT)

**Datei:** `planning_server.py:130`  
**Problem:** `VWorldModel.train()` (in `models/visual_world_model.py:78-86`) aktiviert Training-Modi für alle Sub-Module (Encoder, Predictor, Proprio/Action-Encoder). Obwohl DINOv2 LayerNorm statt BatchNorm nutzt, können:

- Dropout-Layer im Predictor stochastische Ausgaben erzeugen
- Stochastische Regularisierung in Action/Proprio-Encodern das Ergebnis verfälschen

`plan.py` hat das gleiche Problem — aber dort wird das Modell nur einmal genutzt. Im Server bleibt es persistent, und die Stochastik akkumuliert sich über viele Requests.

**Fix:** 1 Zeile nach Model-Laden:
```python
model.eval()  # WICHTIG: Eval-Modus fuer deterministische Inferenz
```

**Auswirkung auf Loss:** Gering. DINOv2 hat kein BatchNorm, und der Predictor hat (je nach Konfiguration) minimalen Dropout. Erklärt **nicht** den Loss-Anstieg von 0.3 → 0.97.

---

### Bug 2: `evaluator=None` — Keine Early-Termination (✅ Bewusst akzeptiert)

**Datei:** `planning_server.py:285`  
**Im Original:** `planning/cem.py:105-113` prüft `if self.evaluator is not None` und führt die aktuellen besten Actions in der echten Env aus. Bei Erfolg wird CEM vorzeitig beendet.

**Im Server:** `evaluator=None` → CEM läuft **immer** alle `opt_steps` Iterationen durch, auch wenn der Plan bereits nach 3 Schritten konvergiert hat.

**Auswirkung:**
- ❌ Kein Qualitätsproblem (mehr Iterationen können nicht schaden)
- ⚠️ Performance-Problem (unnötige GPU-Zeit bei konvergierten Plänen)
- ❌ Keine Validierung ob der Plan in der echten Physik funktioniert

**Status:** Kein Fix nötig. Early-Termination spart nur Zeit, nicht Qualität. Das Client-seitige MPC übernimmt die Env-Validation.

---

### Bug 3: Warm-Start füllt mit Nullen auf → Null-Bias (✅ GEFIXT)

**Datei:** `planning_server.py:399` (alter Code)  
**Problem:** Beim MPC-Warm-Start wird der vorherige Plan um 1 Step geshiftet. Die fehlende letzte Action wurde mit `torch.zeros()` aufgefüllt:

```python
# ALT (Bug):
zero_tail = torch.zeros(1, 1, warm_start_actions.shape[2])
actions_init = torch.cat([shifted, zero_tail], dim=1)

# NEU (Fix):
last_action = warm_start[:, -1:, :]  # Letzte bekannte Action wiederholen
actions_init = torch.cat([shifted, last_action], dim=1)
```

**Warum ist `[0,0,...,0]` schlecht?** Im z-normalisierten Raum bedeutet Null "bewege dich zum Mittelwert aller Trainingsaktionen". Der CEM startet dann mit einem Plan, dessen letzte Aktion systematisch in Richtung Mittelwert verzerrt ist. Bei `opt_steps=5` (Online-Modus) hat CEM zu wenig Iterationen um diesen Bias zu überwinden.

**Auswirkung auf Loss:** Moderat. Betrifft nur MPC-Sequenzen (ab dem 2. plan()-Aufruf). Erklärt nicht den Loss bei der ersten Planung.

---

### Bug 4: `torch.cuda.empty_cache()` zwischen Chunks fragmentiert VRAM (✅ GEFIXT)

**Datei:** `planning_server.py:187` (alter Code im `ChunkedRolloutWrapper`)  
**Problem:** `empty_cache()` gibt den CUDA-Cache frei, aber die akkumulierten Ergebnis-Tensoren (`all_z_obses`, `all_zs`) bleiben alloziert. Der nächste Chunk muss neuen Speicher anfordern → Fragmentierung. Bei vielen Chunks kann das paradoxerweise zu **mehr** OOM führen statt weniger.

**Fix:** `empty_cache()` nur einmal NACH der gesamten Chunk-Schleife aufrufen:
```python
# Nach der for-Schleife, vor dem return:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Auswirkung auf Loss:** Keine (rein Performance/Stabilität).

---

### Bug 5: `__getattr__`-Fallback im `ChunkedRolloutWrapper` (✅ GEFIXT)

**Datei:** `planning_server.py:162-166`  
**Problem:** Jeder Attributzugriff, der nicht explizit gesetzt ist, wird an `self._model` delegiert. Wenn `self._model` das Attribut auch nicht hat, kommt ein kryptischer Fehler aus dem Model statt aus dem Wrapper. Außerdem: `nn.Module`-Methoden wie `state_dict()`, `parameters()`, `to()` werden stillschweigend durchgereicht, was dazu führen kann, dass z.B. `model.to('cpu')` den Wrapper intakt lässt aber das innere Model verschiebt.

**Fix:** Explizite Forwarding-Methoden für kritische Operationen:
```python
def to(self, *args, **kwargs):
    self._model.to(*args, **kwargs)
    return self

def state_dict(self, *args, **kwargs):
    return self._model.state_dict(*args, **kwargs)
```

**Auswirkung auf Loss:** Keine (Wartbarkeit/Debugging).

---

### Bug 6: Kein `torch.no_grad()` um den CEM-Planner (✅ KEIN BUG)

**Datei:** `planning_server.py:404`  
**Status:** Der äußere `torch.no_grad()` im Server schützt auch den `transform_obs()`- und `encode_obs()`-Aufruf. Innerhalb von `CEMPlanner.plan()` in `planning/cem.py:94` steht ebenfalls `with torch.no_grad()`. Doppelter Kontext ist harmlos.

---

### Bug 7: `img_to_obs` — Kein Handling von Bildgröße/Kamera-Diskrepanzen (⚠️ TEILWEISE)

**Datei:** `planning_server.py:310-326`  
**Problem:** Das Bild wird roh übergeben — keine Überprüfung ob:
- Die Auflösung zur Trainingsauflösung passt (224×224)
- Das Farbformat stimmt (BGR vs RGB)
- Der Wertebereich korrekt ist

Der `Preprocessor.transform_obs()` macht `/255.0` und `transform` (Resize + CenterCrop + Normalize) — aber das nimmt an, dass die Eingabe `uint8 [0-255]` im Format `(B, T, H, W, C)` ist. Wenn das Isaac-Sim-Bild z.B. `float32 [0-1]` oder `(H, W, 4)` (RGBA) liefert, stimmt die Pipeline nicht.

**Aktueller Status:** Teilfix vorhanden — `img_to_obs()` konvertiert jetzt `float32 → uint8`, aber keine RGBA-Erkennung, keine BGR-Prüfung, keine Auflösungs-Validierung.

**Auswirkung auf Loss:** Kann katastrophal sein, wenn das Bildformat nicht stimmt. Aber das war auch im alten Code so → erklärt nicht die Regression.

---

## 3. Regressions-Analyse: Warum Loss von ~0.3 auf ~0.97?

### Fakten

| Metrik | Alter Server (Referenz) | Neuer Server (nach Bugfixes) |
|--------|-------------------------|------------------------------|
| Modell | `2026-02-09/17-59-59` (500ep, ActInt10, 50 Epochen) | `2026-02-14/21-30-33` (1000ep, ActInt2, 100 Epochen) |
| CEM-Parameter | `samples=300, steps=30, topk=30, H=5` | `samples=300, steps=30, topk=30, H=5` |
| Initial Loss | ~0.83 | ~1.90 |
| Final Loss | **~0.34** | **~0.98** |
| Reduktion | 58.9% | 48.6% |

### ⚠️ Die Bugfixes sind NICHT die Ursache der Regression!

Die Bugfixes (`model.eval()`, Warm-Start, Cache) können den Loss nur **senken**, nicht erhöhen. Der Anstieg kommt vom **anderen Modell**:

### Mögliche Ursachen (Modell-bezogen)

**Hypothese A: Anderes Modell, anderer Loss-Raum**
- Das neue Modell (`2026-02-14/21-30-33`) wurde mit 1000 Episoden / ActInt2 / 100 Epochen trainiert
- Das alte Modell (`2026-02-09/17-59-59`) mit 500 Episoden / ActInt10 / 50 Epochen
- Ein anderer Datensatz mit anderer Aktions-Verteilung erzeugt einen **anderen Latent-Raum**
- Die absolute Loss-Höhe ist zwischen Modellen **nicht vergleichbar**
- Ein Loss von 0.97 im neuen Modell kann qualitativ besser sein als 0.34 im alten

**Hypothese B: Zu kurze Trainingszeit / Underfit**
- 100 Epochen bei 1000 Episoden = weniger Passes pro Episode als 50 Epochen bei 500 Episoden
- Der Encoder/Predictor hat den feineren ActInt2-Datensatz möglicherweise nicht genug gelernt
- Prüfbar: Training-Loss-Kurve des neuen Modells ansehen

**Hypothese C: ActInt2 ändert die Aktions-Dynamik**
- ActInt2 (alle 4 Sim-Steps) erzeugt feinere aber kleinere Aktionen als ActInt10 (alle 20 Sim-Steps)
- Die Action-Varianz ist geringer → z-Normalisierung komprimiert den Raum stärker
- CEM muss in einem "dichteren" Raum mit feineren Unterschieden optimieren
- Das erfordert möglicherweise **mehr Samples oder Steps** für gleiche Konvergenz

**Hypothese D: Dataset-Statistik-Diskrepanz**
- **Kritisch zu prüfen:** Die `action_mean`/`action_std` im Output des Servers:
  ```
  Mean: [0.47974253 0.01700846 0.16120689 0.4795267  0.01707228 0.16092643]
  Std:  [0.12250879 0.16102357 0.07198107 0.12270366 0.16116358 0.07203   ]
  ```
- Wenn diese nicht zu den im Training verwendeten Werten passen → falsche Normalisierung → hoher Loss
- **Der Server lädt jetzt alle 999 Episoden** für die Statistik-Berechnung (via `FrankaCubeStackDataset` mit `preload_images=False`), während der alte Server `hydra.utils.call()` nutzte, das ein Train/Val-Split machte
- **→ Unterschiedliche mean/std wenn der Split einen Subset verwendet!**

### 🔴 Wahrscheinlichste Ursache: Hypothese D — Dataset-Split-Diskrepanz

**Alt (`.bak`):**
```python
_datasets, _traj_dset = hydra.utils.call(
    model_cfg.env.dataset,
    num_hist=model_cfg.num_hist,
    num_pred=model_cfg.num_pred,
    frameskip=model_cfg.frameskip,
)
_dset_val = _traj_dset["valid"]
action_mean_base = _dset_val.action_mean.clone()  # Statistik vom VAL-Split
```

**Neu (aktuell):**
```python
_full_dset = FrankaCubeStackDataset(
    n_rollout=_dset_cfg.get("n_rollout", None),
    data_path=_dset_cfg["data_path"],
    preload_images=False,
)
action_mean = _full_dset.action_mean.clone()  # Statistik von ALLEN Episoden
```

**Problem:** Das Modell wurde mit den Statistiken des **Train-Splits** trainiert. Der Server berechnet jetzt die Statistiken über **alle Episoden** (kein Split). Wenn Train- und Full-Statistiken sich unterscheiden:

- `(action - wrong_mean) / wrong_std` ≠ `(action - correct_mean) / correct_std`
- Jede Normalisierung/Denormalisierung ist systematisch verschoben
- Der CEM-Suchraum liegt neben dem tatsächlich gelernten Raum
- → Höherer Loss, weil die Aktionen "daneben" liegen

### Verifikation

Prüfen ob die Statistiken übereinstimmen:
```bash
python -c "
import torch, hydra
from omegaconf import OmegaConf

cfg = OmegaConf.load('outputs/2026-02-14/21-30-33/hydra.yaml')
_, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
dset_val = dset['valid']
print('Val-Split mean:', dset_val.action_mean.numpy())
print('Val-Split std:', dset_val.action_std.numpy())
"
```
Dann vergleichen mit den Server-Werten oben. Wenn sie abweichen → **das ist die Root Cause**.

---

## 4. Lösungsansätze (nach Priorität)

### ✅ Bereits umgesetzt: CEM-Parameter auf Paper-Werte

`samples=300, steps=30, topk=30, horizon=5` — argparse-Defaults korrigiert.

### ✅ Bereits umgesetzt: Ansatz 1 — Minimaler Bugfix

| Fix | Zeilen | Wo | Status |
|-----|--------|----|--------|
| `model.eval()` | 1 | nach `load_model()` | ✅ |
| Warm-Start: `repeat` statt Nullen | 1 | `zero_tail` → `warm_start[:, -1:]` | ✅ |
| `empty_cache()` nur am Ende | 2 | `ChunkedRolloutWrapper.rollout()` | ✅ |
| Explizites Forwarding im Wrapper | 8 | `to()`, `state_dict()`, `eval()` etc. | ✅ |

### Ansatz 2: Loss-basiertes Early Stopping (~20 Zeilen)

Statt `evaluator=None` einen simplen Konvergenz-Check im `LoggingRun` einbauen:

```python
class LoggingRun:
    def __init__(self, patience=5, min_improvement=0.001):
        self._losses = []
        self._patience = patience
        self._min_improvement = min_improvement
    
    def should_stop(self):
        if len(self._losses) < self._patience + 1:
            return False
        recent = self._losses[-self._patience:]
        improvement = (recent[0] - recent[-1]) / recent[0]
        return improvement < self._min_improvement
```

**Aufwand:** ~20 Zeilen. Erfordert Anpassung von `cem.py` oder einen Callback-Hook.  
**Nutzen:** Spart Rechenzeit bei konvergierten Plänen. Kein Qualitätsgewinn.  
**Priorität:** Niedrig.

### 🔴 Ansatz 3: Dataset-Statistiken korrekt laden (DRINGEND)

**Das vermutete Root-Cause-Problem:** Der Server berechnet mean/std über **alle** Episoden, das Training aber nur über den **Train-Split**.

**Fix:** Zurück zum alten Dataset-Loading, aber mit `preload_images=False`:

```python
# Variante A: Hydra-Call wie bisher, aber ohne Bilder
_datasets, _traj_dset = hydra.utils.call(
    model_cfg.env.dataset,
    num_hist=model_cfg.num_hist,
    num_pred=model_cfg.num_pred,
    frameskip=model_cfg.frameskip,
)
_dset_val = _traj_dset["valid"]  # ← GLEICHER Split wie beim Training!
action_mean = _dset_val.action_mean.clone()
action_std = _dset_val.action_std.clone()
# ... etc.
del _dset_val, _traj_dset, _datasets

# Variante B: FrankaCubeStackDataset direkt, ABER mit gleichem Split
# → Erfordert, dass der Split reproduzierbar ist (gleiche n_rollout, gleicher seed)
```

**Aufwand:** ~5 Zeilen ändern.  
**Auswirkung:** Wenn dies die Root Cause ist → Loss sollte wieder auf ~0.3 fallen.  
**Priorität:** 🔴 Höchste Priorität. Sofort testen.

### Ansatz 4: Server-seitige MPC-Logik (~50 Zeilen)

Den bestehenden `plan → execute → plan`-Loop im Client in den Server verlagern:

```
Client                          Server
  │ send(image, goal)             │
  │ ──────────────────────>       │
  │                               │ CEM plan (H=5)
  │                               │ nimm action[0]
  │     <──────────────────       │
  │ receive(action[0])            │
  │ execute in Isaac Sim          │
  │ send(new_image)               │
  │ ──────────────────────>       │
  │                               │ CEM replan ab new_image
  │                               │ mit Warm-Start
  │     <──────────────────       │
  │ receive(action[0])            │
  │ ...                           │
```

Das ist konzeptionell das, was `MPCPlanner` in `planning/mpc.py` tut — aber ohne dass der Server selbst die Env braucht. **Genau dieses Pattern ist bereits im `cmd == "plan"` Handler implementiert** (mit Warm-Start). Der einzige Unterschied: die MPC-Logik liegt im **Client**, nicht im Server.

**Aufwand:** ~50 Zeilen Server-Code.  
**Nutzen:** Beste Qualität, weil der Planner den echten Zustand nach jeder Aktion sieht.  
**Priorität:** Mittel. Funktionell bereits über Client-MPC gelöst.

---

## Zusammenfassung: Nächste Schritte

| Priorität | Aktion | Erwarteter Effekt |
|-----------|--------|-------------------|
| 🔴 **1** | Prüfen ob `action_mean/std` zwischen Full-Dataset und Train-Split abweichen | Root-Cause der Regression identifizieren |
| 🔴 **2** | Falls ja: Dataset-Loading auf `hydra.utils.call()` mit Train/Val-Split zurückstellen | Loss zurück auf ~0.3 Level |
| 🟡 **3** | Training-Loss-Kurve des neuen Modells prüfen (konvergiert?) | Underfit ausschließen |
| 🟢 **4** | Loss-basiertes Early Stopping (optional) | Nur Speedup, keine Qualität |
| 🟢 **5** | Server-seitige MPC-Logik (optional) | Bereits via Client-MPC umgesetzt |
