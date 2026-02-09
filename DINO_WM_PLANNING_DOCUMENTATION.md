# 🎯 DINO World Model - Planning Dokumentation

> Vollständige Dokumentation der Planning-Pipeline für das DINO World Model mit Fokus auf Franka Cube Stacking Integration.

---

## 📑 Inhaltsverzeichnis

1. [Überblick: Planning mit World Models](#1-überblick-planning-mit-world-models)
2. [Architektur-Übersicht](#2-architektur-übersicht)
3. [Schnittstellen und Datenfluss](#3-schnittstellen-und-datenfluss)
4. [Environment Wrapper Interface](#4-environment-wrapper-interface)
5. [CEM Planner im Detail](#5-cem-planner-im-detail)
6. [Online vs. Offline Planning: Computational Bottlenecks](#6-online-vs-offline-planning-computational-bottlenecks)
7. [Integration mit Isaac Sim](#7-integration-mit-isaac-sim)
8. [Konfiguration und Start](#8-konfiguration-und-start)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Überblick: Planning mit World Models

### 1.1 Was ist World Model Planning?

Das DINO World Model wurde trainiert, um **zukünftige visuelle Zustände** vorherzusagen. Beim Planning nutzen wir diese Fähigkeit, um **optimale Aktionssequenzen** zu finden:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WORLD MODEL PLANNING KONZEPT                             │
│                                                                             │
│   ┌─────────┐                                      ┌─────────┐             │
│   │ Aktuell │ ──── Welche Aktionen führen zu? ───► │  Ziel   │             │
│   │  Bild   │                                      │  Bild   │             │
│   └─────────┘                                      └─────────┘             │
│                                                                             │
│   Der Planner:                                                              │
│   1. Generiert viele mögliche Aktionssequenzen                             │
│   2. Simuliert diese im World Model (Latent Space!)                        │
│   3. Vergleicht vorhergesagte Zustände mit Ziel                            │
│   4. Wählt die beste Aktionssequenz aus                                    │
│                                                                             │
│   VORTEIL: Keine echte Simulation nötig - alles im Latent Space!           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Warum kein klassischer Controller?

| Aspekt | Klassischer Controller | World Model Planner |
|--------|----------------------|---------------------|
| **Input** | Explizite Zustandsrepräsentation | Rohe Bilder |
| **Wissen** | Manuell definierte Regeln | Aus Daten gelernt |
| **Flexibilität** | Task-spezifisch | Generalisiert auf neue Situationen |
| **Setup** | Aufwändige Kalibrierung | Nur Training nötig |

### 1.3 Planning-Modi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PLANNING MODI                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODUS 1: Open-Loop Planning                                                │
│  ─────────────────────────────                                              │
│  - Plane einmal am Anfang                                                   │
│  - Führe alle Aktionen blind aus                                            │
│  - Schnell, aber anfällig für Fehlerakkumulation                           │
│                                                                             │
│  [Bild_0] → Planner → [a_0, a_1, a_2, ..., a_T] → Ausführen                │
│                                                                             │
│                                                                             │
│  MODUS 2: MPC (Model Predictive Control) - Receding Horizon                │
│  ──────────────────────────────────────────────────────────                │
│  - Plane bei jedem Schritt neu                                              │
│  - Führe nur erste Aktion(en) aus                                          │
│  - Robuster, aber rechenintensiver                                         │
│                                                                             │
│  [Bild_0] → Planner → [a_0, a_1, ...] → Führe a_0 aus                      │
│  [Bild_1] → Planner → [a_0', a_1', ...] → Führe a_0' aus                   │
│  [Bild_2] → Planner → [a_0'', ...] → ...                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architektur-Übersicht

### 2.1 Komponenten der Planning-Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PLANNING PIPELINE ARCHITEKTUR                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         plan.py (Hauptskript)                         │  │
│  │  - Lädt Konfiguration (Hydra)                                        │  │
│  │  - Initialisiert alle Komponenten                                    │  │
│  │  - Orchestriert den Planning-Prozess                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │ VWorldModel    │  │ CEMPlanner     │  │ Environment    │                │
│  │ (trainiert)    │  │ (planning/     │  │ Wrapper        │                │
│  │                │  │  cem.py)       │  │                │                │
│  │ - Encoder      │  │                │  │ - prepare()    │                │
│  │ - Predictor    │  │ - plan()       │  │ - rollout()    │                │
│  │ - Decoder      │  │ - optimize()   │  │ - eval_state() │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
│         │                    │                   │                          │
│         └────────────────────┼───────────────────┘                          │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      PlanEvaluator                                    │  │
│  │                   (planning/evaluator.py)                             │  │
│  │  - eval_actions(): Führt Aktionen aus und bewertet                   │  │
│  │  - _compute_rollout_metrics(): Berechnet Erfolgsmetriken             │  │
│  │  - _plot_rollout_compare(): Visualisiert Ergebnisse                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Dateien und ihre Rollen

| Datei | Pfad | Beschreibung |
|-------|------|--------------|
| **plan.py** | `dino_wm/plan.py` | Hauptskript, orchestriert alles |
| **cem.py** | `planning/cem.py` | CEM Planner Implementierung |
| **gd.py** | `planning/gd.py` | Gradient Descent Planner (Alternative) |
| **mpc.py** | `planning/mpc.py` | MPC Wrapper für iteratives Planning |
| **evaluator.py** | `planning/evaluator.py` | Evaluiert geplante Aktionen |
| **base_planner.py** | `planning/base_planner.py` | Abstrakte Basis-Klasse |
| **serial_vector_env.py** | `env/serial_vector_env.py` | Wrapper für mehrere Environments |
| **FlexEnvWrapper.py** | `env/deformable_env/` | Referenz-Implementation |

---

## 3. Schnittstellen und Datenfluss

### 3.1 Datenfluss beim Planning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PLANNING DATENFLUSS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCHRITT 1: Ziele vorbereiten                                               │
│  ─────────────────────────────                                              │
│                                                                             │
│  Dataset ──► [obs_0, obs_g, state_0, state_g] ──► PlanWorkspace            │
│              │                                    │                         │
│              │  obs_0: Startbild (B, 1, H, W, C)  │                         │
│              │  obs_g: Zielbild (B, 1, H, W, C)   │                         │
│              │  state_0: Startzustand (B, D)      │                         │
│              │  state_g: Zielzustand (B, D)       │                         │
│              │                                    │                         │
│  Referenz: plan.py Zeile ~200 (prepare_targets)  │                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCHRITT 2: Aktionen planen                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  obs_0, obs_g ──► CEMPlanner.plan() ──► actions (B, T, action_dim)         │
│                   │                                                         │
│                   │  1. Initiale Aktionen samplen                           │
│                   │  2. Im World Model simulieren                           │
│                   │  3. Mit Ziel vergleichen (Objective Function)           │
│                   │  4. Beste Aktionen auswählen (Top-K)                    │
│                   │  5. Wiederholen (CEM Optimierung)                       │
│                   │                                                         │
│  Referenz: planning/cem.py Zeile ~70 (plan)                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCHRITT 3: Aktionen evaluieren                                             │
│  ─────────────────────────────                                              │
│                                                                             │
│  actions ──► PlanEvaluator.eval_actions() ──► logs, successes              │
│              │                                                              │
│              │  1. Rollout im World Model (imaginiert)                      │
│              │  2. Rollout im Environment (real)                            │
│              │  3. Vergleiche final states                                  │
│              │  4. Berechne Metriken                                        │
│              │                                                              │
│  Referenz: planning/evaluator.py Zeile ~85 (eval_actions)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Aktions-Format und Normalisierung

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AKTIONS-TRANSFORMATIONEN                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING (wie Aktionen gespeichert wurden):                                │
│  ───────────────────────────────────────────                                │
│  raw_action: (9,) = [joint_cmd(7), gripper_cmd(2)]                         │
│                                                                             │
│  Mit frameskip=5 während Training:                                          │
│  wm_action: (45,) = [raw_t, raw_t+1, raw_t+2, raw_t+3, raw_t+4]            │
│                                                                             │
│  Normalisiert (Z-Score):                                                    │
│  normalized_action = (wm_action - action_mean) / action_std                 │
│                                                                             │
│                                                                             │
│  PLANNING (wie Aktionen verwendet werden):                                  │
│  ─────────────────────────────────────────                                  │
│                                                                             │
│  Planner Output: normalized_actions (B, T, 45)                              │
│       │                                                                     │
│       │  Referenz: cem.py Zeile ~125 (return mu)                           │
│       ▼                                                                     │
│  Denormalisierung: (Preprocessor)                                           │
│  exec_actions = normalized_actions * action_std + action_mean              │
│       │                                                                     │
│       │  Referenz: evaluator.py Zeile ~112                                 │
│       ▼                                                                     │
│  Reshape für Ausführung:                                                    │
│  exec_actions: (B, T*frameskip, 9) = (B, T*5, 9)                           │
│       │                                                                     │
│       │  Referenz: evaluator.py Zeile ~111                                 │
│       ▼                                                                     │
│  An Environment senden: env.rollout(seed, init_state, exec_actions)        │
│                                                                             │
│       │  Referenz: evaluator.py Zeile ~116                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Environment Wrapper Interface

### 4.1 Erforderliche Methoden

Das Environment muss folgende Schnittstelle implementieren (siehe `FrankaCubeStackWrapper`):

```python
class EnvironmentWrapper:
    """
    Minimale Schnittstelle für DINO WM Planning.
    Referenz: env/deformable_env/FlexEnvWrapper.py
    """
    
    def prepare(self, seed: int, init_state: np.ndarray) -> Tuple[obs, state]:
        """
        Setzt Environment in definierten Anfangszustand.
        
        Aufgerufen von:
        - evaluator.py: eval_actions() Zeile ~110
        - rollout() intern
        
        Returns:
            obs: {"visual": (H,W,3), "proprio": (proprio_dim,)}
            state: (state_dim,)
        """
        pass
    
    def step_multiple(self, actions: np.ndarray) -> Tuple[obses, rewards, dones, infos]:
        """
        Führt Aktionssequenz aus.
        
        Aufgerufen von:
        - rollout() intern
        
        Args:
            actions: (T, action_dim)
            
        Returns:
            obses: {"visual": (T,H,W,3), "proprio": (T,proprio_dim)}
            rewards: float
            dones: bool
            infos: {"state": (T, state_dim)}
        """
        pass
    
    def rollout(self, seed, init_state, actions) -> Tuple[obses, states]:
        """
        Kompletter Rollout = prepare() + step_multiple()
        
        Aufgerufen von:
        - evaluator.py: eval_actions() Zeile ~113-116
        
        WICHTIG: Rückgabe hat T+1 Zeitschritte (inkl. Initial-State)!
        
        Returns:
            obses: {"visual": (T+1,H,W,3), ...}
            states: (T+1, state_dim)
        """
        pass
    
    def eval_state(self, goal_state, cur_state) -> Dict:
        """
        Bewertet ob Ziel erreicht wurde.
        
        Aufgerufen von:
        - evaluator.py: _compute_rollout_metrics() Zeile ~150
        
        Returns:
            {"success": bool, "distance": float, ...}
        """
        pass
    
    def update_env(self, env_info) -> None:
        """
        Aktualisiert Environment-Konfiguration.
        
        Aufgerufen von:
        - plan.py: prepare_targets() Zeile ~230
        """
        pass
```

### 4.2 SerialVectorEnv - Mehrere Environments parallel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERIAL VECTOR ENV                                     │
│                     (env/serial_vector_env.py)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Zweck: Wrapper um mehrere Environment-Instanzen für parallele Evaluation  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SerialVectorEnv                                                     │   │
│  │  ├── env[0]: FrankaCubeStackWrapper                                 │   │
│  │  ├── env[1]: FrankaCubeStackWrapper                                 │   │
│  │  ├── env[2]: FrankaCubeStackWrapper                                 │   │
│  │  └── ...                                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Methoden-Mapping:                                                          │
│                                                                             │
│  vector_env.prepare(seeds, init_states)                                     │
│      → [env[i].prepare(seeds[i], init_states[i]) for i in range(n)]        │
│      → Aggregiert zu (n_envs, ...) Arrays                                  │
│                                                                             │
│  vector_env.rollout(seeds, init_states, actions)                           │
│      → [env[i].rollout(...) for i in range(n)]                             │
│      → obses: {"visual": (n_envs, T+1, H, W, C)}                           │
│      → states: (n_envs, T+1, state_dim)                                    │
│                                                                             │
│  vector_env.eval_state(goal_states, cur_states)                            │
│      → [env[i].eval_state(goal_states[i], cur_states[i]) for i in range(n)]│
│      → {"success": (n_envs,), "distance": (n_envs,)}                       │
│                                                                             │
│  Referenz: env/serial_vector_env.py                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. CEM Planner im Detail

### 5.1 Cross-Entropy Method (CEM)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CEM (Cross-Entropy Method) ALGORITHMUS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CEM ist eine derivative-free Optimierungsmethode:                          │
│  - Keine Gradienten nötig (funktioniert mit Black-Box World Model)         │
│  - Iterative Verbesserung durch Sampling                                   │
│  - Robust gegenüber lokalen Minima                                         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  CEM ALGORITHMUS (planning/cem.py)                                  │    │
│  │                                                                      │    │
│  │  1. INITIALISIERUNG:                                                │    │
│  │     μ = 0 (Mittelwert der Aktionsverteilung)                        │    │
│  │     σ = var_scale (Standardabweichung)                              │    │
│  │                                                                      │    │
│  │  2. FÜR JEDE OPTIMIERUNGS-ITERATION:                               │    │
│  │                                                                      │    │
│  │     a) Sample num_samples Aktionssequenzen:                         │    │
│  │        actions ~ N(μ, σ)                                            │    │
│  │        Shape: (num_samples, horizon, action_dim)                    │    │
│  │                                                                      │    │
│  │     b) Simuliere im World Model:                                    │    │
│  │        z_pred = wm.rollout(obs_0, actions)                          │    │
│  │                                                                      │    │
│  │     c) Berechne Kosten (Distanz zum Ziel):                          │    │
│  │        loss = objective_fn(z_pred, z_goal)                          │    │
│  │                                                                      │    │
│  │     d) Wähle Top-K beste Aktionen:                                  │    │
│  │        topk_actions = actions[argsort(loss)[:topk]]                 │    │
│  │                                                                      │    │
│  │     e) Update Verteilung:                                           │    │
│  │        μ = mean(topk_actions)                                       │    │
│  │        σ = std(topk_actions)                                        │    │
│  │                                                                      │    │
│  │  3. RÜCKGABE: μ (optimierte Aktionssequenz)                         │    │
│  │                                                                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Referenz: planning/cem.py Zeile ~70-125                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 CEM Hyperparameter

```yaml
# Aus conf/planner/cem.yaml
planner:
  name: cem
  
  # Optimierungs-Parameter
  horizon: 5          # Planungshorizont (Anzahl Zeitschritte)
  num_samples: 512    # Anzahl gesampelter Aktionssequenzen pro Iteration
  topk: 64            # Anzahl bester Sequenzen für Update
  var_scale: 1.0      # Initiale Standardabweichung
  opt_steps: 10       # Anzahl Optimierungs-Iterationen
  
  # Evaluation
  eval_every: 5       # Evaluiere alle N Iterationen
```

### 5.3 Objective Function

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBJECTIVE FUNCTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Die Objective Function bewertet, wie nah die vorhergesagten               │
│  Zustände am Ziel sind.                                                    │
│                                                                             │
│  loss = objective_fn(z_pred, z_goal)                                       │
│                                                                             │
│  Standardmäßig: MSE im Latent Space                                        │
│  ─────────────────────────────────────                                      │
│  loss = ||z_pred[:, -1] - z_goal||²                                        │
│                                                                             │
│  Mit alpha-Gewichtung (für proprio):                                       │
│  ─────────────────────────────────────                                      │
│  loss = ||z_visual_pred - z_visual_goal||²                                 │
│       + alpha * ||z_proprio_pred - z_proprio_goal||²                       │
│                                                                             │
│  Referenz: planning/objective.py                                           │
│  Konfiguration: conf/objective/default.yaml                                │
│                                                                             │
│  Parameter:                                                                 │
│  - alpha: Gewichtung von proprio vs. visual (default: 0.1)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 World Model Rollout im Planner

```python
# Pseudocode aus cem.py Zeile ~100-110

def plan(self, obs_0, obs_g, actions=None):
    # 1. Encode Ziel-Observation
    trans_obs_g = self.preprocessor.transform_obs(obs_g)
    z_obs_g = self.wm.encode_obs(trans_obs_g)  # Ziel im Latent Space
    
    # 2. Für jede Optimierungs-Iteration
    for i in range(self.opt_steps):
        # 3. Sample Aktionen aus aktueller Verteilung
        actions = torch.randn(...) * sigma + mu
        
        # 4. Rollout im World Model (KEIN echtes Environment!)
        with torch.no_grad():
            z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,  # Start-Observation
                act=actions,         # Aktionssequenz
            )
        # z_obses: (num_samples, horizon+1, num_patches, emb_dim)
        
        # 5. Berechne Loss zum Ziel
        loss = self.objective_fn(z_obses, z_obs_g)
        
        # 6. Update μ, σ basierend auf Top-K
        ...
    
    return mu  # Optimierte Aktionssequenz
```

---

## 6. Online vs. Offline Planning: Computational Bottlenecks

> **Kernproblem:** Der CEM-Planner ist für Offline-Evaluation konzipiert und nicht direkt für Echtzeit-Robotersteuerung geeignet. Dieses Kapitel dokumentiert die identifizierten Engpässe, deren Ursachen und die notwendigen Anpassungen für Online-Planning.

### 6.1 Problemstellung: Timeout bei Online-Planning

Beim ersten Versuch, den CEM-Planner über die Planning-Server/Client-Architektur (Socket-Kommunikation) mit Isaac Sim zu verbinden, trat folgendes Problem auf:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEOBACHTETES TIMEOUT-PROBLEM                              │
│                                                                             │
│  Isaac Sim Client                           DINO WM Server                  │
│  (planning_client.py)                       (planning_server.py)            │
│                                                                             │
│  1. set_goal(image) ─────────────────────►  Goal encodiert ✓               │
│     ◄──────────────── "ok" ────────────────                                │
│                                                                             │
│  2. plan(image) ─────────────────────────►  CEM läuft...                   │
│     ...                                     ...                             │
│     ... 120s Timeout ...                    ... (noch nicht fertig)         │
│     TimeoutError: timed out ✗              ... (rechnet weiter)            │
│                                                                             │
│  Client gibt auf, Server rechnet noch.                                     │
│  → Keine Aktion zurückgegeben                                               │
│  → Episode abgebrochen                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Ursachenanalyse: Wo geht die Rechenzeit hin?

#### 6.2.1 Der DINO-Encoder als Hauptengpass

Der CEM-Planner führt in jeder Optimierungsiteration einen **World-Model-Rollout** durch. Dieser Rollout beinhaltet drei Schritte:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  WM.ROLLOUT() - KOSTEN PRO AUFRUF                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DINO-Encoder (ViT): obs_0 → z_obs_0                                    │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  TEUER! Kompletter Vision Transformer Forward-Pass             │      │
│     │  - 224×224 Bild → Patch-Embedding → Self-Attention Layers     │      │
│     │  - DINOv2 ViT-Base: 86M Parameter                            │      │
│     │  - Geschätzt: ~5-15ms pro Bild (GPU)                          │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  2. Action-Encoder: action → act_emb                                        │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  GÜNSTIG! Nur 1D-Convolution                                   │      │
│     │  - Conv1d(12, 10, kernel_size=1)                               │      │
│     │  - Geschätzt: <0.1ms                                           │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  3. Predictor: z_concat → z_pred                                            │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  MITTEL: Transformer-basierte Vorhersage im Latent-Space       │      │
│     │  - Arbeitet auf Patch-Embeddings, nicht auf Pixeln             │      │
│     │  - Geschätzt: ~2-5ms                                           │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  PROBLEM: Der DINO-Encoder wird für JEDES Sample JEDE Iteration            │
│  aufgerufen, obwohl obs_0 sich NICHT ändert!                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Quantifizierung: CEM mit Offline-Parametern

Die CEM-Konfiguration in `conf/planner/cem.yaml` ist für **Offline-Evaluation** optimiert:

```yaml
# conf/planner/cem.yaml (Original-Defaults)
num_samples: 300    # Aktionssequenzen pro Iteration
opt_steps: 30       # Optimierungsiterationen  
topk: 30            # Eliten für Verteilungs-Update
```

**Rechenaufwand pro `plan()`-Aufruf (n_evals=1, Online-Fall):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            KOSTENRECHNUNG: CEM MIT OFFLINE-PARAMETERN                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Pro Iteration (opt_step):                                                  │
│    Pro Trajektorie (n_evals=1 für Online):                                  │
│      - 300 Samples werden generiert                                         │
│      - wm.rollout() wird 1× mit Batch=300 aufgerufen                       │
│      - Intern: DINO-Encoder für 300 obs_0-Kopien → 300 ViT-Passes         │
│      - Intern: 300 × horizon Predictor-Passes                              │
│                                                                             │
│  Gesamt-DINO-Encoder-Passes:                                                │
│    num_samples × opt_steps = 300 × 30 = 9.000 ViT-Forward-Passes          │
│                                                                             │
│  Geschätzte Laufzeit (RTX 3090):                                            │
│    9.000 × ~10ms = ~90 Sekunden (nur Encoder!)                             │
│    + Predictor, Objective, Sampling: ~30-60s zusätzlich                     │
│    ≈ 120-150 Sekunden pro plan()-Aufruf                                    │
│                                                                             │
│  → WEIT ÜBER dem Client-Timeout von 120s!                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.3 Redundanz: Gleiche Observation, unterschiedliche Encodings

Der Code in `planning/cem.py` zeigt das Kernproblem:

```python
# planning/cem.py - Zeile ~75-110 (vereinfacht)

def plan(self, obs_0, obs_g, actions=None):
    # obs_0 wird EINMAL transformiert (CPU→GPU, Normalize) ✓
    trans_obs_0 = self.preprocessor.transform_obs(obs_0)  
    
    for i in range(self.opt_steps):        # 30 Iterationen
        for traj in range(n_evals):        # 1 Trajektorie (Online)
            # obs_0 wird auf num_samples KOPIERT
            cur_trans_obs_0 = {
                key: repeat(arr[traj], "... -> n ...", n=self.num_samples)  # 300×
                for key, arr in trans_obs_0.items()
            }
            
            # wm.rollout() ruft intern wm.encode() auf
            # → wm.encode() ruft DINO-Encoder für ALLE 300 Kopien auf!
            i_z_obses, _ = self.wm.rollout(
                obs_0=cur_trans_obs_0,  # 300 identische Bilder werden encodiert
                act=action,
            )
```

**Das identische Bild `obs_0` wird 300 × 30 = 9.000 Mal durch den DINO-Encoder geschickt!**

### 6.3 Offline vs. Online: Zwei unterschiedliche Anforderungsprofile

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              OFFLINE VS. ONLINE PLANNING - VERGLEICH                         │
├─────────────────┬──────────────────────────┬────────────────────────────────┤
│                 │     OFFLINE (plan.py)     │   ONLINE (planning_server.py) │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ Zweck           │ Modell-Evaluation,       │ Echtzeit-Robotersteuerung     │
│                 │ Metriken, Paper          │ in Isaac Sim                  │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ Zeitbudget      │ Unbegrenzt               │ < 30s pro Aktion              │
│ pro plan()      │ (Minuten OK)             │ (idealerweise < 10s)          │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ n_evals         │ 5 (parallel evaluieren)  │ 1 (ein Roboter)               │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ num_samples     │ 300                      │ 32-64                         │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ opt_steps       │ 30                       │ 3-5                           │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ topk            │ 30                       │ 10                            │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ DINO-Passes     │ 300 × 30 = 9.000        │ 64 × 5 = 320                  │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ Geschätzte      │ ~120-150s                │ ~5-15s                        │
│ Laufzeit        │                          │                               │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ Evaluator       │ Ja (eval_actions)        │ Nein (nur plan)               │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ Qualität        │ Bestmöglich              │ Ausreichend für               │
│                 │                          │ geschlossene Regelschleife    │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ Kommentar       │ cem.yaml unverändert     │ CLI-Overrides im Server       │
│                 │                          │ (--num_samples, --opt_steps)  │
└─────────────────┴──────────────────────────┴────────────────────────────────┘
```

**Wichtiger Tradeoff:** Die Online-Parameter liefern suboptimalere Aktionspläne als die Offline-Parameter. Dies wird jedoch durch die **geschlossene Regelschleife** (MPC-Modus) kompensiert: Nach jeder ausgeführten Aktion wird mit frischem Kamerabild neu geplant, sodass Fehler korrigiert werden können.

### 6.4 Implementierte Lösung: Parametrisierter Planning Server

Anstatt den CEM-Planner oder das World Model zu modifizieren, werden die CEM-Parameter im `planning_server.py` über CLI-Argumente überschrieben:

```python
# planning_server.py - CLI-Overrides
parser.add_argument("--num_samples", type=int, default=64)   # statt 300
parser.add_argument("--opt_steps", type=int, default=5)      # statt 30
parser.add_argument("--topk", type=int, default=10)          # statt 30

# Override der cem.yaml-Werte vor Instanziierung
planner_cfg = OmegaConf.load("conf/planner/cem.yaml")
planner_cfg.num_samples = args.num_samples
planner_cfg.opt_steps = args.opt_steps
planner_cfg.topk = args.topk
```

Zudem wurde Timing-Instrumentierung hinzugefügt, um die Planungsdauer pro Aufruf zu messen.

### 6.5 Empfohlene Konfigurationen

```bash
# ─── SCHNELL (< 10s) ─── Für Debugging und schnelle Iterationen
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 32 --opt_steps 3 --topk 5

# ─── STANDARD (10-30s) ─── Empfohlen für Online-Planning
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 64 --opt_steps 5 --topk 10

# ─── QUALITÄT (30-60s) ─── Wenn Zeit weniger kritisch ist
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 128 --opt_steps 10 --topk 20

# ─── OFFLINE (plan.py) ─── Verwendet cem.yaml Defaults direkt
python plan.py --config-name plan_franka model_name=2026-02-09/08-12-44
```

### 6.6 Mögliche zukünftige Optimierungen

Die aktuelle Lösung (Parameter-Reduktion) ist die einfachste, aber nicht die einzige Option. Für weiterführende Arbeiten wären folgende Optimierungen am CEM-Planner oder World Model denkbar:

| Optimierung | Beschreibung | Erwarteter Speedup | Aufwand |
|-------------|-------------|-------------------|---------|
| **Observation Pre-Encoding** | DINO-Encoder 1× aufrufen, Embedding cachen, `rollout_from_z()` nutzen | ~10-30× (eliminiert redundante ViT-Passes) | Mittel (neue Methoden in VWorldModel + CEM) |
| **Warm-Starting** | μ der vorherigen plan()-Runde als Initialisierung für die nächste | ~2× (weniger opt_steps nötig) | Gering |
| **Batched CEM** | Alle n_evals-Trajektorien parallel statt sequentiell | ~n_evals× | Gering (Reshape-Logik) |
| **ONNX/TensorRT Export** | World Model für Inferenz optimieren | ~2-5× | Hoch |
| **Gradient-basiertes Planning** | GDPlanner statt CEM (weniger Forward-Passes nötig) | ~3-10× | Gering (bereits implementiert in planning/gd.py) |

**Observation Pre-Encoding** wäre die wirkungsvollste Einzeloptimierung, da sie das Kernproblem (redundante DINO-Encoder-Aufrufe) direkt adressiert, ohne die Optimierungsqualität zu beeinträchtigen.

---

## 7. Integration mit Isaac Sim

### 7.1 Architektur für Isaac Sim Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ISAAC SIM INTEGRATION ARCHITEKTUR                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DINO World Model                              │   │
│  │                         (Python/PyTorch)                             │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │  │  CEMPlanner  │───►│ VWorldModel  │───►│ FrankaCube-  │          │   │
│  │  │              │    │ (Prediction) │    │ StackWrapper │          │   │
│  │  └──────────────┘    └──────────────┘    └──────┬───────┘          │   │
│  │                                                  │                  │   │
│  └──────────────────────────────────────────────────┼──────────────────┘   │
│                                                     │                      │
│                                                     │ Isaac Sim Interface  │
│                                                     │ (Zu implementieren)  │
│                                                     ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Isaac Sim                                    │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │  │   Franka     │    │    Cubes     │    │   Camera     │          │   │
│  │  │   Robot      │    │              │    │  (256x256)   │          │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 FrankaCubeStackWrapper Implementierung

Der `FrankaCubeStackWrapper` in `env/franka_cube_stack/franka_cube_stack_wrapper.py` implementiert die erforderliche Schnittstelle:

```python
# Verwendung des Wrappers

# 1. Offline-Modus (nur World Model, kein Isaac Sim)
from env.franka_cube_stack import FrankaCubeStackWrapper

wrapper = FrankaCubeStackWrapper(
    offline_mode=True,  # Keine Isaac Sim Verbindung
    img_size=(256, 256)
)

# 2. Online-Modus (mit Isaac Sim)
# Erfordert Implementierung des Isaac Sim Interface
wrapper = FrankaCubeStackWrapper(
    isaac_sim_interface=my_isaac_interface,
    offline_mode=False
)

# 3. Mit SerialVectorEnv für parallele Evaluation
from env.franka_cube_stack.franka_cube_stack_wrapper import create_franka_env_for_planning

env = create_franka_env_for_planning(
    n_envs=5,  # 5 parallele Evaluationen
    offline_mode=True
)
```

### 7.3 Isaac Sim Interface (zu implementieren)

```python
# Beispiel-Struktur für Isaac Sim Interface
# Datei: env/franka_cube_stack/isaac_sim_interface.py

class IsaacSimInterface:
    """
    Interface zwischen FrankaCubeStackWrapper und Isaac Sim.
    
    DIESE KLASSE MUSS AN DEIN ISAAC SIM SETUP ANGEPASST WERDEN!
    """
    
    def __init__(self, config_path: str):
        """Initialisiert Verbindung zu Isaac Sim."""
        # TODO: Verbindung zu laufender Isaac Sim Instanz
        pass
    
    def reset(self) -> None:
        """Setzt Simulation zurück."""
        # TODO: Simulation reset
        pass
    
    def set_robot_state(self, state: np.ndarray) -> None:
        """
        Setzt Roboter in gegebenen Zustand.
        
        Args:
            state: [ee_pos(3), ee_quat(4), gripper(1), joints(7), joint_vel(7)]
        """
        joint_positions = state[8:15]  # joints
        gripper = state[7]
        # TODO: Setze joint positions in Isaac Sim
        pass
    
    def apply_action(self, action: np.ndarray) -> None:
        """
        Wendet Aktion an.
        
        Args:
            action: [joint_cmd(7), gripper_cmd(2)]
        """
        # TODO: Sende Kommandos an Roboter-Controller
        pass
    
    def step(self, dt: float = 1/60) -> None:
        """Führt Simulationsschritt aus."""
        # TODO: world.step()
        pass
    
    def get_camera_image(self) -> np.ndarray:
        """
        Holt aktuelles Kamerabild.
        
        Returns:
            RGB Bild (H, W, 3) uint8
        """
        # TODO: Rendere Kamerabild
        pass
    
    def get_robot_state(self) -> np.ndarray:
        """
        Holt aktuellen Roboterzustand.
        
        Returns:
            state: (22,) - [ee_pos, ee_quat, gripper, joints, joint_vel]
        """
        # TODO: Lese Roboterzustand aus
        pass
```

---

## 8. Konfiguration und Start

### 8.1 Konfigurations-Dateien

```
conf/
├── plan.yaml              # Haupt-Planning-Konfiguration
├── plan_pusht.yaml        # PushT spezifisch
├── plan_wall.yaml         # Wall spezifisch
├── plan_point_maze.yaml   # PointMaze spezifisch
│
├── planner/
│   ├── cem.yaml          # CEM Parameter
│   ├── gd.yaml           # Gradient Descent Parameter
│   └── mpc.yaml          # MPC Parameter
│
├── objective/
│   └── default.yaml      # Objective Function Parameter
│
└── env/
    └── franka_cube_stack.yaml  # Environment-Konfiguration
```

### 8.2 Wichtige Parameter in plan.yaml

```yaml
# conf/plan.yaml - Haupt-Konfiguration

# Checkpoint des trainierten Modells
ckpt_base_path: "."
model_name: "model_50.pth"
model_epoch: "final"

# Planning Parameter
goal_H: 5              # Planungshorizont (wie weit in die Zukunft)
goal_source: "dset"    # Woher kommen Zielbilder?
                       # - "dset": Aus Validation-Dataset
                       # - "random_state": Zufällig generiert
                       # - "file": Aus Datei laden

# Evaluation
n_evals: 5             # Anzahl paralleler Evaluationen
n_plot_samples: 3      # Anzahl zu visualisierender Samples
seed: 42

# Planner (wird aus planner/*.yaml geladen)
planner:
  name: cem
  # ... weitere Parameter aus cem.yaml

# Objective (wird aus objective/*.yaml geladen)  
objective:
  alpha: 0.1           # Gewichtung proprio vs. visual
```

### 8.3 Planning starten

```bash
# Basis-Befehl
python plan.py <checkpoint_ordner> model_name=<modell>.pth goal_H=<horizont>

# Beispiel mit deinem trainierten Modell:
cd ~/Desktop/dino_wm

# Standard-Planning mit CEM
python plan.py outputs/2026-01-31/23-03-37/checkpoints \
    model_name=model_50.pth \
    goal_H=5

# Mit anderen Planern
python plan.py outputs/2026-01-31/23-03-37/checkpoints \
    model_name=model_50.pth \
    goal_H=5 \
    planner=gd  # oder planner=mpc

# Mit verschiedenen goal_sources
python plan.py outputs/2026-01-31/23-03-37/checkpoints \
    model_name=model_50.pth \
    goal_H=5 \
    goal_source=random_state
```

### 8.4 Environment registrieren

Füge zu `env/__init__.py` hinzu:

```python
# Franka Cube Stack Environment registrieren
register(
    id="franka_cube_stack",
    entry_point="env.franka_cube_stack:FrankaCubeStackWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
```

---

## 9. Troubleshooting

### 9.1 MuJoCo Fehler

**Problem:**
```
Exception: You appear to be missing MuJoCo.
```

**Lösung:**
Die `env/__init__.py` wurde bereits angepasst, um MuJoCo-abhängige Imports optional zu machen. Falls der Fehler weiterhin auftritt:

```python
# In env/__init__.py - bereits implementiert
try:
    from .pointmaze import U_MAZE
    _HAS_MUJOCO = True
except Exception:
    _HAS_MUJOCO = False
```

### 9.2 Checkpoint nicht gefunden

**Problem:**
```
FileNotFoundError: model_50.pth not found
```

**Lösung:**
Überprüfe den Pfad:
```bash
ls outputs/2026-01-31/23-03-37/checkpoints/
# Sollte model_X.pth Dateien zeigen
```

### 9.3 CUDA Out of Memory

**Problem:**
```
CUDA out of memory
```

**Lösung:**
Reduziere `num_samples` in der Planner-Konfiguration:
```bash
python plan.py ... planner.num_samples=128
```

### 9.4 Environment nicht gefunden

**Problem:**
```
gym.error.Error: Environment 'franka_cube_stack' doesn't exist
```

**Lösung:**
Registriere das Environment (siehe 7.4) oder verwende direkt den Wrapper:
```python
from env.franka_cube_stack.franka_cube_stack_wrapper import create_franka_env_for_planning
env = create_franka_env_for_planning(n_envs=5)
```

<<<<<<< HEAD
### 8.5 ✅ BEHOBEN: Actions sahen aus wie Pixelkoordinaten (Multi-Robot Grid Offset Problem)
=======
### 9.5 ⚠️ KRITISCH: Actions sehen aus wie Pixelkoordinaten (Multi-Robot Grid Offset Problem)
>>>>>>> 5c13f571789b9ecab3062b8b809dc03c989c63d6

> **Status: BEHOBEN** (Commit `a9af071`, 03.02.2026)  
> **Verifiziert: 09.02.2026** — Beide Logger (`min_data_logger.py`, `primitive_data_logger.py`) subtrahieren `env_offset` korrekt.

**Ursprüngliches Problem:**
Der CEM Planner gab Actions zurück, die unrealistisch große Werte hatten:
```python
# Erwartete Franka Panda Koordinaten (in Metern):
#   X: 0.3 - 0.8 m, Y: -0.5 - 0.5 m, Z: 0.0 - 0.6 m

# Tatsächliche denormalisierte Actions (vor dem Fix):
action = [6.95, 3.98, 0.17, 6.95, 3.98, 0.17]  # ❌ Viel zu groß!
```

**Ursache - Multi-Robot Simulations-Grid:**

Der Franka Cube Stack Datensatz wurde mit **mehreren parallel simulierten Robotern** in Isaac Sim generiert. Jeder Roboter hat einen anderen **Welt-Offset**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ISAAC SIM MULTI-ROBOT GRID LAYOUT                        │
│                                                                             │
│    Y=10 ┤  Robot    Robot    Robot    Robot                                │
│         │  (0,10)   (5,10)   (10,10)  (15,10)                              │
│    Y=5  ┤  Robot    Robot    Robot    Robot                                │
│         │  (0,5)    (5,5)    (10,5)   (15,5)                               │
│    Y=0  ┤  Robot    Robot    Robot    Robot                                │
│         │  (0,0)    (5,0)    (10,0)   (15,0)                               │
│         └──────┴──────┴──────┴──────┴──────►                               │
│              X=0    X=5    X=10   X=15                                     │
│                                                                             │
│    Grid-Spacing: 5 Meter (!) zwischen Robotern                             │
│    Lokaler Arbeitsraum pro Roboter: ca. 0.1-0.8m                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Analyse der Rohdaten (vor dem Fix):**
```
Episode 0:  X = 0.429,  Y = 0.045   → Grid (0, 0)
Episode 1:  X = 5.429,  Y = 0.045   → Grid (5, 0)  
Episode 2:  X = 10.429, Y = 0.045   → Grid (10, 0)
Episode 3:  X = 15.429, Y = 0.045   → Grid (15, 0)
Episode 4:  X = 0.429,  Y = 5.045   → Grid (0, 5)
...
```

**Konsequenz für die Normalisierung (vor dem Fix):**
```python
# Berechnet aus allen Episoden (mit unterschiedlichen Grid-Offsets):
action_mean = [6.96, 3.98, 0.17, 6.96, 3.98, 0.17]  # ← Durchschnitt über Grid!
action_std  = [5.44, 3.83, 0.07, 5.44, 3.83, 0.07]  # ← Hohe Varianz durch Offsets!

# Nach Korrektur der Offsets die korrekten lokalen Statistiken:
local_action_mean = [0.48, 0.01, 0.18, 0.48, 0.01, 0.18]  # ✓ Realistisch!
local_action_std  = [0.12, 0.15, 0.07, 0.12, 0.15, 0.07]  # ✓ Realistisch!
```

**Warum das ein Problem war:**
1. Das World Model wurde mit den **falschen globalen Koordinaten** trainiert
2. Der CEM Planner optimiert im normalisierten Space und gibt z.B. `normalized=0` aus
3. Denormalisierung: `0 * 5.44 + 6.96 = 6.96` → **Keine gültige Roboterposition!**
4. Der Roboter kann diese Position nicht anfahren → **Planning schlägt fehl**

**Implementierter Fix (Commit `a9af071`):**

Beide Data Logger subtrahieren nun den Grid-Offset **vor** dem Speichern aller Koordinaten:

```python
# min_data_logger.py — Offset wird bei start_episode() gespeichert:
if env_offset is not None:
    self.env_offset = np.asarray(env_offset, dtype=np.float64).flatten()[:3]
else:
    self.env_offset = np.zeros(3, dtype=np.float64)

# In log_step() wird der Offset von allen Koordinaten abgezogen:
ee_pos_local = ee_pos.astype(np.float64) - self.env_offset  # EE-Position
corrected = (cp[0] - self.env_offset[0],                     # Cube-Positionen
             cp[1] - self.env_offset[1],
             cp[2] - self.env_offset[2])
action = np.concatenate([prev_ee_pos_local, ee_pos_local])    # Actions
```

```python
# primitive_data_logger.py — Offset in beiden Segmentierungs-Modi:
env_offset = ep.get("env_offset", np.zeros(3))
start_pos_local = start_data["ee_pos"] - env_offset  # Fixed-Mode
end_pos_local = end_data["ee_pos"] - env_offset
action = np.concatenate([start_pos_local, end_pos_local])
```

**Korrigierte Daten (alle 4 Komponenten):**

| Komponente | Vor Fix | Nach Fix |
|-----------|---------|----------|
| EE-Position | Globale Sim-Koordinaten (0–15m) | Lokale Robot-Base-Koordinaten (0.3–0.75m) |
| Cube-Positionen | Globale Sim-Koordinaten | Lokale Koordinaten relativ zum Robot |
| Actions (ee_pos) | `[x_global_start, ..., x_global_end]` | `[x_local_start, ..., x_local_end]` |
| EEF-States | Globale Positionen | Lokale Positionen |

**Diagnose-Kommando (Validierung):**
```bash
cd ~/Desktop/dino_wm
python -c "
import torch, hydra
from omegaconf import OmegaConf

cfg = OmegaConf.load('outputs/2026-02-02/22-50-30/hydra.yaml')
_, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, 
                            num_pred=cfg.num_pred, frameskip=cfg.frameskip)
dset = dset['valid']

print(f'action_mean: {dset.action_mean.numpy()}')
print(f'action_std:  {dset.action_std.numpy()}')
print()
print('✅ Wenn X/Y mean < 1.0 und std < 0.5: Grid-Offset korrekt subtrahiert!')
print('⚠️  Wenn X/Y mean > 1.0 oder std > 1.0: Datensatz muss neu generiert werden!')
"
```

**⚠️ Wichtig:** Datensätze, die **vor** Commit `a9af071` generiert wurden, enthalten noch die falschen globalen Koordinaten und müssen **neu generiert** werden!

---

### 8.6 ✅ KEIN PROBLEM: Pixel-Space (Referenzdatensatz) vs. Meter-Space (Franka)

> **Status: KEIN PROBLEM** — Architektur-Analyse bestätigt am 09.02.2026  
> **Fazit: Die DINO-WM-Architektur ist vollständig einheitsagnostisch.**

**Ursprüngliche Befürchtung:**

Die Referenz-Datensätze (Rope, Push-T, Wall, Point-Maze) verwenden **unterschiedliche Koordinatensysteme** als der Franka Cube Stacking Datensatz. Die Frage war, ob das DINO World Model überhaupt mit Meter-Koordinaten funktionieren kann, wenn es primär mit Pixel-Koordinaten entwickelt und getestet wurde.

**Analyse der Referenz-Datensätze:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KOORDINATENSYSTEME DER DATENSÄTZE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ROPE (Deformable):                                                         │
│  ──────────────────                                                         │
│  Action: [x_start, z_start, x_end, z_end] — 4D                             │
│  Raum:   Physik-Simulator-Einheiten (FleX), Wertebereich ca. ±4            │
│  NICHT Pixel-Space! Sondern Sim-Koordinaten (≈ Meter-Skala)               │
│                                                                             │
│  PUSH-T:                                                                    │
│  ──────────────────                                                         │
│  Action: [dx, dy] — 2D relative Verschiebungen                             │
│  Raum:   Pixel-Space (512×512 pymunk Window), geteilt durch 100            │
│  Effektiver Wertebereich: ca. ±0.2                                         │
│                                                                             │
│  WALL:                                                                      │
│  ──────────────────                                                         │
│  Action: [a1, a2] — 2D                                                      │
│  Raum:   Eigener Sim-Space, mean ≈ 0, std ≈ 0.44–0.47                     │
│                                                                             │
│  FRANKA CUBE STACKING:                                                      │
│  ──────────────────────                                                     │
│  Action: [x_start, y_start, z_start, x_end, y_end, z_end] — 6D            │
│  Raum:   Meter-Space (Isaac Sim), EE-Pos ≈ 0.3–0.75m                      │
│  Effektiver Wertebereich: ca. 0.0–0.8                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Vergleich der Action-Statistiken (alle nach Offset-Korrektur):**

| Datensatz | Action-Dim | Roh-Wertebereich | Nach Z-Score |
|-----------|-----------|-------------------|--------------|
| Rope | 4 | ca. ±4 (Sim-Einheiten) | ~N(0, 1) |
| Push-T | 2 | ca. ±0.2 (Pixel/100) | ~N(0, 1) |
| Wall | 2 | ca. ±0.5 (Sim-Einheiten) | ~N(0, 1) |
| **Franka** | **6** | **ca. 0.0–0.8 (Meter)** | **~N(0, 1)** |

**Warum das KEIN Problem ist — 4 architektonische Gründe:**

**1. Z-Score-Normalisierung als universelle Brücke:**
```python
# Jeder Dataset-Loader normalisiert Actions VOR dem Modell:
normalized_action = (raw_action - action_mean) / action_std

# Egal ob raw_action in Pixeln, Metern, oder Sim-Einheiten:
# → Das Modell sieht IMMER ~N(0, 1)-verteilte Werte
# → Die physikalische Einheit ist nach Normalisierung irrelevant
```

**2. Lernbarer Action Encoder macht Einheiten bedeutungslos:**
```python
# models/proprio.py — ProprioceptiveEmbedding:
self.patch_embed = nn.Conv1d(
    in_chans=action_dim,    # 4 bei Rope, 6 bei Franka
    out_chans=action_emb_dim,  # z.B. 10
    kernel_size=1, stride=1
)
# → Lineare Projektion lernt beliebige Skalierung
# → Keine Annahme über physikalische Einheiten
```

**3. Loss-Funktion ignoriert Actions komplett:**
```python
# Der Embedding-Prediction-Loss berechnet sich NUR über visuelle Patches:
loss = MSE(z_pred[:, :num_visual_patches], z_target[:, :num_visual_patches])
#          └── Action-Embedding-Dims werden NICHT einbezogen ──┘

# Actions dienen ausschließlich als Conditioning-Signal für den Predictor.
# Ihre absolute Skala hat keinen Einfluss auf den Gradienten.
```

**4. Die Referenz-Datensätze sind selbst NICHT einheitlich:**
```
Rope:    ±4.0 Sim-Einheiten  ─┐
Push-T:  ±0.2 Pixel/100       ├── SCHON HETEROGEN!
Wall:    ±0.5 Sim-Einheiten  ─┘
Franka:  0.0–0.8 Meter       ─── Passt problemlos dazu
```
Die Architektur wurde **von Anfang an** so designed, dass sie mit beliebigen Koordinatensystemen funktioniert.

**Zusammenfassung als Diagramm:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 WARUM PIXEL VS. METER KEIN PROBLEM IST                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Rope Actions (±4 Sim)  ──┐                                                │
│  Push-T Actions (±0.2 px) ├──► Z-Score ──► ~N(0,1) ──► nn.Linear ──► Emb  │
│  Wall Actions (±0.5 Sim)  │    Norm.        (alle      (lernbar)     (10D)  │
│  Franka Actions (0-0.8m) ─┘               identisch)                       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════  │
│  Voraussetzungen (beide erfüllt ✅):                                       │
│  1. action_dim ist korrekt konfiguriert (franka: 6)                        │
│  2. action_mean/action_std werden korrekt berechnet (lokale Meter-Werte)   │
│  ═══════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ❌ NICHT erforderlich:                                                     │
│  - Konvertierung Meter→Pixel                                               │
│  - Anpassung der Action-Skala                                              │
│  - Sonderbehandlung im Modell                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Einzige echte Voraussetzung:** Der Grid-Offset muss korrekt subtrahiert sein (→ siehe 8.5). Wenn das der Fall ist, funktioniert die Pipeline mit Meter-Koordinaten genauso wie mit Pixel-Koordinaten.

---

## Anhang: Wichtige Code-Referenzen

| Konzept | Datei | Zeilen |
|---------|-------|--------|
| Planning Hauptloop | plan.py | 430-508 |
| CEM Optimierung | planning/cem.py | 70-125 |
| Evaluator | planning/evaluator.py | 85-150 |
| World Model Rollout | models/visual_world_model.py | rollout() |
| Environment Interface | env/deformable_env/FlexEnvWrapper.py | Alle |
| SerialVectorEnv | env/serial_vector_env.py | Alle |
| Preprocessor | preprocessor.py | Normalisierung |
| FrankaCubeStackWrapper | env/franka_cube_stack/franka_cube_stack_wrapper.py | Alle |

---

*Dokumentation erstellt am 01.02.2026, aktualisiert am 09.02.2026*
