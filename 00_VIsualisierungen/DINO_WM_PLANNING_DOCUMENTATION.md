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
   - 6.1 [Problemstellung: Timeout bei Online-Planning](#61-problemstellung-timeout-bei-online-planning)
   - 6.2 [Ursachenanalyse: Wo geht die Rechenzeit hin?](#62-ursachenanalyse-wo-geht-die-rechenzeit-hin)
   - 6.3 [Offline vs. Online: Zwei Anforderungsprofile](#63-offline-vs-online-zwei-unterschiedliche-anforderungsprofile)
   - 6.4 [Implementierte Lösung: Parametrisierter Planning Server](#64-implementierte-lösung-parametrisierter-planning-server)
   - 6.5 [Empfohlene Konfigurationen](#65-empfohlene-konfigurationen)
   - 6.6 [Mögliche zukünftige Optimierungen](#66-mögliche-zukünftige-optimierungen)
   - **6.7 [Strategische Entscheidung: Warum Online MPC der einzig richtige Ansatz ist](#67-strategische-entscheidung-warum-online-mpc-der-einzig-richtige-ansatz-ist) ← NEU (09.02.2026)**
     - 6.7.1 Das Paper bestätigt: MPC schlägt Open-Loop immer (Table 8)
     - 6.7.2 Warum Offline für Franka Cube Stacking besonders schlecht ist
     - 6.7.3 Warum "Offline planen und zusammensetzen" KEIN guter Kompromiss ist
     - 6.7.4 Die Paper-CEM-Parameter für MPC (Table 10 Inferenzzeit-Analyse)
     - 6.7.5 Die Rolle von Warm-Start im MPC-Kontext
     - 6.7.6 Optimale MPC-Konfiguration: horizon=5, n_taken=1
     - 6.7.7 Konfigurationsübersicht der drei DINO-WM Planner-Configs
     - 6.7.8 Warum wir MPCPlanner nicht direkt verwenden können
     - 6.7.9 Zusammenfassung: Empfohlener Planning-Workflow
7. [Integration mit Isaac Sim](#7-integration-mit-isaac-sim)
8. [Konfiguration und Start](#8-konfiguration-und-start)
   - **8.5 [Planning Server — Vollständige Startbefehl-Übersicht](#85-planning-server--vollständige-startbefehl-übersicht) ← NEU (09.02.2026)**
     - 8.5.1 Alle verfügbaren CLI-Parameter
     - 8.5.2 Parameter-Erklärungen im Detail
     - 8.5.3 Empfohlene Konfigurationen (Configs A–G)
     - 8.5.4 Konfigurations-Vergleichstabelle
     - 8.5.5 CEM-Output lesen und interpretieren
     - 8.5.6 Aktuelle Testergebnisse und Diagnose (09.02.2026)
     - 8.5.7 Zugehöriger Client-Startbefehl (Isaac Sim)
9. [Troubleshooting](#9-troubleshooting)
   - 9.5 [BEHOBEN: Multi-Robot Grid Offset Problem](#95--behoben-actions-sahen-aus-wie-pixelkoordinaten-multi-robot-grid-offset-problem)
   - 9.6 [KEIN PROBLEM: Pixel-Space vs. Meter-Space](#96--kein-problem-pixel-space-referenzdatensatz-vs-meter-space-franka)
10. [WM Sanity-Check: Vorhersagequalität mit GT-Aktionen](#10-wm-sanity-check-vorhersagequalität-mit-gt-aktionen) ← NEU (09.02.2026)
    - 10.1 [Motivation und Problemstellung](#101-motivation-und-problemstellung)
    - 10.2 [Methodik des Sanity-Checks](#102-methodik-des-sanity-checks)
    - 10.3 [Implementierung: wm_sanity_check.py](#103-implementierung-wm_sanity_checkpy)
    - 10.4 [Ergebnisse: Quantitative Analyse](#104-ergebnisse-quantitative-analyse)
    - 10.5 [Diagnose und Interpretation](#105-diagnose-und-interpretation)
    - 10.6 [Kritischer Fund: Visuelle Diskrepanz Training vs. Planning](#106-kritischer-fund-visuelle-diskrepanz-training-vs-planning)
    - 10.7 [Konsequenzen und Handlungsempfehlungen](#107-konsequenzen-und-handlungsempfehlungen)
11. [Finaler Offline-Test: Modell unzureichend für Cube Stacking](#11-finaler-offline-test-modell-unzureichend-für-cube-stacking) ← NEU (14.02.2026)
    - 11.1 [Testaufbau und Parameter](#111-testaufbau-und-parameter)
    - 11.2 [Ergebnisse: CEM-Konvergenz](#112-ergebnisse-cem-konvergenz)
    - 11.3 [Ergebnisse: Ausgeführte Trajektorie](#113-ergebnisse-ausgeführte-trajektorie)
    - 11.4 [Diagnose: Warum das Modell versagt](#114-diagnose-warum-das-modell-versagt)
    - 11.5 [Konsequenz: Neues Training erforderlich](#115-konsequenz-neues-training-erforderlich)
    - 11.6 [Neuer Datensatz und Trainingsplan](#116-neuer-datensatz-und-trainingsplan)
12. [Planning Server Bug-Analyse und Fixes](#12-planning-server-bug-analyse-und-fixes) ← NEU (16.02.2026)
    - 12.1 [Hintergrund: plan.py vs. planning_server.py](#121-hintergrund-planpy-vs-planning_serverpy)
    - 12.2 [Erkenntnisse: Wie plan.py's CEM wirklich funktioniert](#122-erkenntnisse-wie-planpys-cem-wirklich-funktioniert)
    - 12.3 [Bug-Katalog mit Fixes](#123-bug-katalog-mit-fixes)
    - 12.4 [Verbleibende strukturelle Unterschiede](#124-verbleibende-strukturelle-unterschiede)
    - 12.5 [Zusammenfassung der Änderungen](#125-zusammenfassung-der-änderungen)
13. [🚨 KRITISCH: Temporale Alignment-Analyse — Action-Observation Mismatch (20.02.2026)](#13--kritisch-temporale-alignment-analyse--action-observation-mismatch-20022026)
    - 13.1 [Zusammenfassung](#131-zusammenfassung)
    - 13.2 [Auswirkung auf CEM-Planning](#132-auswirkung-auf-cem-planning)
    - 13.3 [Zusammenhang mit CEM-Divergenz](#133-zusammenhang-mit-cem-divergenz)
    - 13.4 [Verifizierungsdaten](#134-verifizierungsdaten)
    - 13.5 [Empfohlener Fix und Workflow](#135-empfohlener-fix-und-workflow)
    - 13.6 [Querverweise](#136-querverweise)
15. [Two-Phase Planning: EEF kommt in Phase 1 nicht tief genug — Root-Cause-Analyse (05.03.2026)](#15-two-phase-planning-eef-kommt-in-phase-1-nicht-tief-genug--root-cause-analyse-05032026)
    - 15.1 [Beobachtung](#151-beobachtung)
    - 15.2 [Ursache 1: OOD-Proprio durch hohe Startposition](#152-ursache-1-ood-proprio-durch-hohe-startposition)
    - 15.3 [Ursache 2: Große z-Distanz überfordert den CEM-Horizont](#153-ursache-2-große-z-distanz-überfordert-den-cem-horizont)
    - 15.4 [Ursache 3: Workspace-Bounds und Sigma-Asymmetrie](#154-ursache-3-workspace-bounds-und-sigma-asymmetrie)
    - 15.5 [Zusammenfassung: Phase 1 vs. Phase 2 im Vergleich](#155-zusammenfassung-phase-1-vs-phase-2-im-vergleich)
    - 15.6 [Lösungsansätze](#156-lösungsansätze)

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

### 6.7 Strategische Entscheidung: Warum Online MPC der einzig richtige Ansatz ist

> **Datum:** 09.02.2026  
> **Kontext:** Nach der BGR-Fix-Iteration (RGB→BGR Konvertierung für korrekte DINO-Features) zeigten die Offline-Testergebnisse eine Verbesserung von 46.3% auf 48.8% Loss-Reduktion — aber die Roboterbewegung blieb suboptimal. Die Frage war: Liegt das Problem in den CEM-Parametern, oder im fundamental falschen Planning-Ansatz?

#### 6.7.1 Das Paper bestätigt: MPC schlägt Open-Loop immer

Die zentrale Evidenz liefert **Table 8 im Appendix A.5.3** des DINO-WM Papers (Zhou et al., 2025):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          PAPER TABLE 8: PLANNING RESULTS OF DINO-WM                         │
│          (Appendix A.5.3, S. 16)                                            │
├─────────────────┬──────────────┬──────────────┬────────────────────────────┤
│                 │  PointMaze   │   Push-T     │    Wall                    │
│                 │  (Sr ↑)      │   (Sr ↑)     │    (Sr ↑)                  │
├─────────────────┼──────────────┼──────────────┼────────────────────────────┤
│  CEM            │    0.80      │    0.86      │    0.74                    │
│  (Open-Loop)    │              │              │                            │
├─────────────────┼──────────────┼──────────────┼────────────────────────────┤
│  GD             │    0.22      │    0.28      │    N/A                     │
│  (Open-Loop)    │              │              │                            │
├─────────────────┼──────────────┼──────────────┼────────────────────────────┤
│  MPC            │  ★ 0.98      │  ★ 0.90      │  ★ 0.96                   │
│  (CEM + Reced.) │              │              │                            │
├─────────────────┼──────────────┼──────────────┼────────────────────────────┤
│  Verbesserung   │   +22.5%     │    +4.7%     │   +29.7%                  │
│  MPC vs. CEM    │              │              │                            │
└─────────────────┴──────────────┴──────────────┴────────────────────────────┘

Quelle: "Table 8. Planning results of DINO-WM" (S. 16, Appendix A.5.3)

Legende:
  CEM   = Plane einmal mit CEM, führe ALLE Actions aus (Open-Loop)
  GD    = Plane einmal mit Gradient Descent, führe ALLE Actions aus
  MPC   = Receding-Horizon mit CEM: Plane, führe k Actions aus, re-plane
  Sr ↑  = Success Rate (höher = besser)
```

**Schlüsselbeobachtungen aus dem Paper:**

1. **MPC verbessert CEM Open-Loop in ALLEN Environments**, besonders bei Wall (+29.7% absolut). Wall ist ein navigationsbasiertes Environment mit Hindernissen — ähnlich wie unser Franka-Setup, wo der Roboterarm um Objekte herum navigieren muss.

2. **Gradient Descent (GD) als Open-Loop ist katastrophal** (0.22 vs. 0.80 bei PointMaze). Das zeigt, dass die Optimierungsqualität eines einzelnen Plans nicht ausreicht — die Feedback-Schleife durch MPC ist entscheidend.

3. **Selbst bei PushT, wo CEM Open-Loop bereits 0.86 erreicht**, verbessert MPC noch auf 0.90. Bei unserem komplexeren 6D Franka-Setup (statt 2D PushT) ist der Unterschied wahrscheinlich noch größer.

**Paper-Zitat (Appendix A.5.1, S. 15):**
> *"After the optimization process is done, the first k actions a₀, ..., aₖ is executed in the environment. The process then repeats at the next time step with the new observation."*

Dies beschreibt exakt den MPC-Ansatz: Plane mit vollem Horizont, führe nur die ersten $k$ Actions aus, beobachte das Ergebnis, plane erneut.

#### 6.7.2 Warum Offline Planning für Franka Cube Stacking besonders schlecht ist

Das Open-Loop-Problem verschärft sich bei unserem Franka-Setup aus mehreren Gründen:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         WARUM OPEN-LOOP BEIM FRANKA BESONDERS PROBLEMATISCH IST             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM 1: Hoher Aktionsraum (6D vs. 2D)                                  │
│  ──────────────────────────────────────────                                 │
│  Push-T Actions:  2D → CEM-Suchraum bei horizon=5: 10 Dimensionen         │
│  Wall Actions:    2D → CEM-Suchraum bei horizon=5: 10 Dimensionen         │
│  Franka Actions:  6D → CEM-Suchraum bei horizon=5: 60 Dimensionen!        │
│                                       (mit frameskip=2)                     │
│                                                                             │
│  Der CEM muss in einem 6× größeren Suchraum optimieren.                   │
│  → Ein einzelner Open-Loop-Plan kann die optimale Lösung in 60D kaum       │
│    finden. MPC erlaubt Korrekturen nach jedem Schritt.                     │
│                                                                             │
│  PROBLEM 2: 3D-Dynamik mit Schwerkraft                                     │
│  ──────────────────────────────────────────                                 │
│  Push-T:  2D-Schiebebewegung auf flacher Oberfläche — Fehler sind          │
│           langsam und korrigierbar.                                         │
│  Franka: 3D-Bewegung mit Schwerkraft — ein falscher Z-Wert kann den        │
│           Greifer in den Tisch rammen oder den Würfel fallen lassen.        │
│           Fehler-Akkumulation ist NICHT reversibel.                          │
│                                                                             │
│  PROBLEM 3: Kontakt-Dynamik                                                │
│  ──────────────────────────────────────────                                 │
│  Das World Model wurde mit nur 200 Episoden trainiert (vgl. Paper           │
│  Push-T: 18.500 Trajektorien, Table 11). Kleine Prädiktionsfehler          │
│  bei Kontakt-Events (Greifen, Ablegen) akkumulieren sich über den           │
│  Horizont. MPC korrigiert nach jedem Kontakt-Event.                        │
│                                                                             │
│  PROBLEM 4: Franka-IK ist nicht perfekt                                     │
│  ──────────────────────────────────────────                                 │
│  Der RMPFlow-IK-Controller erreicht die geplante EE-Position nur           │
│  approximativ (typisch: 3-5mm Fehler). Open-Loop akkumuliert               │
│  diese IK-Fehler über alle Schritte. MPC beobachtet den realen             │
│  Zustand nach IK-Ausführung und korrigiert die nächste Planung.            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Zusammengefasst:** Unser Franka-Setup hat MEHR Gründe für MPC als die Paper-Environments. Wenn MPC dort schon 22-30% besser ist (Table 8), erwarten wir bei Franka einen noch größeren Vorteil.

#### 6.7.3 Warum "Offline planen und zusammensetzen" KEIN guter Kompromiss ist

Eine naheliegende Idee wäre: Offline (mit vollen CEM-Parametern, z.B. 300×30) einen optimalen Plan berechnen, und dann die resultierenden Bilder zu einer flüssigen Bewegung zusammensetzen. **Dies ist aber identisch mit CEM Open-Loop aus Table 8** — also dem schlechteren Ansatz:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│     "OFFLINE PLANEN + ZUSAMMENSETZEN" = CEM OPEN-LOOP                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Ablauf "Offline + Zusammensetzen":                                         │
│                                                                             │
│  1. Startbild erfassen                                                      │
│  2. CEM mit 300×30 laufen lassen (185 Sekunden)                            │
│  3. Alle 10 Actions (5 horizon × 2 frameskip) ausführen                    │
│  4. Video/Bilder speichern                                                  │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════     │
│                                                                             │
│  Das ist EXAKT was das Paper als "CEM" (Open-Loop) in Table 8 misst!       │
│  → Wall: 0.74 Success Rate (vs. 0.96 mit MPC)                             │
│  → PointMaze: 0.80 (vs. 0.98 mit MPC)                                     │
│                                                                             │
│  Das fundamentale Problem bleibt:                                           │
│  Ohne Feedback aus der realen Umgebung akkumulieren sich Prädiktions-      │
│  fehler des World Models über alle Zeitschritte.                            │
│                                                                             │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐          │
│  │  Step 1        │     │  Step 3        │     │  Step 5        │          │
│  │  Fehler: 2mm   │────►│  Fehler: 8mm   │────►│  Fehler: 25mm  │          │
│  │  (OK)          │     │  (spürbar)     │     │  (zu groß!)    │          │
│  └────────────────┘     └────────────────┘     └────────────────┘          │
│                                                                             │
│  vs. MPC:                                                                   │
│                                                                             │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐          │
│  │  Step 1        │     │  Step 3        │     │  Step 5        │          │
│  │  Fehler: 2mm   │────►│  Fehler: 2mm   │────►│  Fehler: 2mm   │          │
│  │  (re-plan) ◄───┘     │  (re-plan) ◄───┘     │  (re-plan) ◄───┘          │
│  └────────────────┘     └────────────────┘     └────────────────┘          │
│                                                                             │
│  MPC hält den Fehler KONSTANT niedrig durch kontinuierliches Re-Planen.    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Mehr CEM-Budget löst das Problem NICHT:**

Selbst mit 300×30 Samples (185 Sekunden Rechenzeit) erreicht CEM Open-Loop nur 0.74 bei Wall (Table 8). Das liegt nicht an zu wenig Optimierung, sondern daran, dass das World Model **systematische Prädiktionsfehler** hat, die sich über den Horizont akkumulieren. Kein noch so gutes CEM-Budget kann Fehler in der Umgebungsdynamik kompensieren — nur echtes Feedback kann das.

#### 6.7.4 Die Paper-CEM-Parameter für MPC (Inferenzzeit-Analyse)

**Table 10 (Appendix A.8, S. 17)** liefert die Referenz-Inferenzzeiten:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          PAPER TABLE 10: INFERENCE TIME AND PLANNING TIME                    │
│          (Appendix A.8, S. 17 — NVIDIA A6000 GPU)                           │
├──────────────────────────────────┬──────────────────────────────────────────┤
│  Metrik                          │  Zeit                                    │
├──────────────────────────────────┼──────────────────────────────────────────┤
│  Inference (Batch=32)            │  0.014s (14ms)                           │
│  Simulation Rollout (Batch=1)    │  3.0s                                    │
│  Planning (CEM, 100×10)          │  53.0s                                   │
├──────────────────────────────────┴──────────────────────────────────────────┤
│                                                                             │
│  Anmerkung: "Planning time is measured with CEM using 100 samples           │
│  per iteration and 10 optimization steps."                                  │
│                                                                             │
│  Das sind die DINO-WM-Autoren selbst, die 100×10 als Standard              │
│  für MPC-Planning nutzen — NICHT die vollen 300×30 aus cem.yaml!           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Wichtige Erkenntnis:** Die Paper-Autoren messen Planning-Zeit mit **100 Samples × 10 Schritte = 1.000 DINO-Passes** und erzielen damit **53 Sekunden auf einer A6000**. Die Default-Config `cem.yaml` (300×30) ist für die **Offline-Evaluation** in `plan.py` gedacht, NICHT für MPC.

**Hochrechnung für unsere Hardware und Setup:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         ZEITBUDGET-RECHNUNG FÜR FRANKA MPC                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Paper-Referenz (A6000):                                                    │
│    100 × 10 = 1.000 Passes → 53 Sekunden                                  │
│    → ~53ms pro DINO-Pass (inkl. Predictor + Overhead)                      │
│                                                                             │
│  Unsere Hardware (vergleichbar, RTX-Klasse):                               │
│    Gemessen: 300×30 = 9.000 Passes → ~185 Sekunden                        │
│    → ~20ms pro DINO-Pass (schneller als Paper, da ViT-S/14 statt          │
│       ViT-Base, und batch-Effekte bei 300 Samples)                         │
│                                                                             │
│  ─── KONFIGURATIONSOPTIONEN FÜR MPC ───                                    │
│                                                                             │
│  Config A: Paper-Standard (100×10)                                          │
│    1.000 Passes × ~20ms = ~20-30s pro MPC-Step                             │
│    ✓ Paper-getestet, nachgewiesene Qualität                                │
│    ✓ Akzeptabel für Masterarbeit (30s Wartezeit pro Schritt)               │
│                                                                             │
│  Config B: Reduziert (64×5)                                                 │
│    320 Passes × ~20ms = ~6-10s pro MPC-Step                                │
│    ✓ Deutlich schneller                                                     │
│    ✓ Warm-Start kompensiert teilweise die geringere Optimierung            │
│    ⚠ Suboptimaler als Config A, aber durch MPC-Feedback ausgeglichen       │
│                                                                             │
│  Config C: Schnell (32×3)                                                   │
│    96 Passes × ~20ms = ~2-3s pro MPC-Step                                  │
│    ✓ Nahe Echtzeit                                                          │
│    ⚠ Niedrige Optimierungsqualität, nur mit starkem Warm-Start sinnvoll   │
│                                                                             │
│  Config D: Qualität (128×10)                                                │
│    1.280 Passes × ~20ms = ~25-35s pro MPC-Step                             │
│    ✓ Hohe Qualität, nahe an Paper-Standard                                 │
│    ⚠ Langsamer, aber für Evaluations-Runs empfohlen                        │
│                                                                             │
│  EMPFEHLUNG: Config A oder B mit Warm-Start                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.7.5 Die Rolle von Warm-Start im MPC-Kontext

**Warm-Start** (bereits implementiert in `planning_server.py`) ist der Schlüssel, der MPC mit reduzierten Parametern ermöglicht:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   WARM-START IM MPC-KONTEXT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OHNE Warm-Start (aktueller Offline-Modus):                                 │
│  ───────────────────────────────────────────                                │
│  plan() Aufruf 1: μ = 0 (Dataset-Durchschnitt)                             │
│                    CEM muss von Null starten → braucht viele Iterationen   │
│  plan() Aufruf 2: μ = 0 (Dataset-Durchschnitt)                             │
│                    IDENTISCH zu Aufruf 1 — kein Wissen vom letzten Plan!   │
│                                                                             │
│  MIT Warm-Start (MPC-Modus):                                               │
│  ────────────────────────────                                               │
│  plan() Aufruf 1: μ = 0 (muss komplett optimieren)                        │
│    Ergebnis: [a₀, a₁, a₂, a₃, a₄] — 5 Horizon-Steps                      │
│    → Führe a₀ aus (1-2 Sub-Actions durch frameskip)                        │
│    → Speichere [a₁, a₂, a₃, a₄, 0] als Warm-Start                         │
│                                                                             │
│  plan() Aufruf 2: μ = [a₁, a₂, a₃, a₄, 0] (geshiftet!)                   │
│    → CEM startet NICHT bei Null, sondern beim vorherigen Plan              │
│    → Die ersten 4 Actions sind bereits gut optimiert                       │
│    → CEM muss nur noch feinjustieren und die letzte Action finden          │
│    → WENIGER Iterationen nötig für gleiches Ergebnis!                      │
│                                                                             │
│  plan() Aufruf 3: μ = [a₂', a₃', a₄', aneu, 0] (erneut geshiftet)        │
│    → Noch weniger Änderung nötig, da sich die Szene nur minimal            │
│      verändert hat (nur 1 Sub-Action wurde ausgeführt)                     │
│    → CEM konvergiert in 3-5 Iterationen statt 30!                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Implementierung in planning_server.py:                                     │
│                                                                             │
│  # Nach plan() Aufruf:                                                      │
│  warm_start_actions = actions.clone()                                       │
│                                                                             │
│  # Vor nächstem plan() Aufruf:                                              │
│  shifted = warm_start_actions[:, 1:, :]       # Ersten Step entfernen      │
│  zero_tail = torch.zeros(1, 1, action_dim)     # Null am Ende anhängen     │
│  actions_init = torch.cat([shifted, zero_tail], dim=1)                     │
│  # → Wird an planner.plan(actions=actions_init) übergeben                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Warum Warm-Start so effektiv ist:**

Das Paper beschreibt in **Appendix A.5.1 (S. 15)** den CEM-Algorithmus:
> *"At each planning iteration, CEM samples a population of N action sequences [...] from a distribution. The initial distribution is set to be Gaussian."*

Ohne Warm-Start ist diese Gaussian-Initialisierung $\mathcal{N}(0, \sigma)$ — also zentriert auf den Dataset-Durchschnitt. Mit Warm-Start ist sie $\mathcal{N}(\mu_{\text{shifted}}, \sigma)$ — bereits nahe an der optimalen Lösung. Das reduziert die benötigten `opt_steps` dramatisch.

#### 6.7.6 Optimale MPC-Konfiguration: horizon=5, n_taken=1

Die Kernparameter des MPC-Ansatzes bestimmen die Balance zwischen Planungsqualität und Reaktionsfähigkeit:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              MPC-PARAMETER UND IHRE WIRKUNG                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HORIZON (planning_horizon / goal_H):                                       │
│  ─────────────────────────────────────                                      │
│  = Wie weit das World Model in die Zukunft schaut                          │
│                                                                             │
│  horizon=1: CEM sieht nur 1 Schritt voraus                                 │
│    → Greedy, kein Vorausdenken                                              │
│    → Kann in Sackgassen laufen (z.B. gegen Hindernisse)                    │
│    → Suchraum: 1 × 12 = 12D (schnell, aber schlecht)                      │
│                                                                             │
│  horizon=5: CEM sieht 5 Schritte voraus (Paper-Standard, Table 11)         │
│    → Plant um Hindernisse herum                                             │
│    → Berücksichtigt Konsequenzen jeder Aktion                              │
│    → Suchraum: 5 × 12 = 60D (langsamer, aber viel besser)                 │
│                                                                             │
│  horizon=10: Zu weit voraus für unser WM                                    │
│    → Prädiktionsfehler dominieren bei Schritt 8-10                         │
│    → Suchraum: 10 × 12 = 120D (zu groß für CEM)                           │
│    → Nicht empfohlen                                                        │
│                                                                             │
│  Paper-Referenz (Table 11, S. 17): Frameskip und History                   │
│    Alle Environments nutzen horizon H=1 oder H=3                            │
│    Franka: H=2 (num_hist), frameskip=2                                     │
│    → Goal-Horizon von 5 ist der Paper-Standard für CEM/MPC-Planning        │
│                                                                             │
│  N_TAKEN (n_taken_actions):                                                 │
│  ─────────────────────────                                                  │
│  = Wie viele der geplanten Horizon-Steps tatsächlich ausgeführt werden     │
│  = Der Rest wird als Warm-Start für den nächsten Plan gespeichert          │
│                                                                             │
│  n_taken=1: Führe nur 1 Horizon-Step aus (= 2 Sub-Actions bei frameskip=2)│
│    → Maximum Feedback (nach jeder Bewegung neu planen)                     │
│    → Best für Franka (IK-Fehler sofort korrigierbar)                       │
│    → EMPFOHLEN: Qualität > Geschwindigkeit                                 │
│                                                                             │
│  n_taken=5 (= horizon): Führe ALLE Steps aus, dann re-plane               │
│    → Equivalent zu Open-Loop mit Warm-Start                                │
│    → Weniger Feedback, mehr Fehlerakkumulation                              │
│    → Das ist was mpc_cem.yaml als Default hat                              │
│    → NICHT empfohlen für Franka (Kontakt-Dynamik erfordert Feedback)       │
│                                                                             │
│  Formel: Gesamtdauer einer Episode                                         │
│    T_episode = (max_steps / n_taken) × T_plan                              │
│    Bei horizon=5, n_taken=1, Config A (100×10, ~30s):                      │
│      50 MPC-Steps × 30s = 25 Minuten pro Episode                          │
│    Bei horizon=5, n_taken=1, Config B (64×5, ~10s):                        │
│      50 MPC-Steps × 10s ≈ 8 Minuten pro Episode                           │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  EMPFOHLENE KONFIGURATION:                                                  │
│  horizon=5, n_taken=1, num_samples=100, opt_steps=10, topk=20             │
│  → Paper-nah, Warm-Start-kompatibel, akzeptable Dauer (~30s/Step)          │
│                                                                             │
│  ALTERNATIVE FÜR SCHNELLERES ITERIEREN:                                    │
│  horizon=5, n_taken=1, num_samples=64, opt_steps=5, topk=10               │
│  → Halbierte Rechenzeit (~10s/Step), Warm-Start kompensiert                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.7.7 Konfigurationsübersicht der drei DINO-WM Planner-Configs

Die existierenden Config-Dateien im Repository bestätigen die Strategie:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         BESTEHENDE PLANNER-KONFIGURATIONEN IM DINO-WM REPO                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  conf/planner/cem.yaml (Standalone CEM — Open-Loop):                       │
│  ──────────────────────────────────────────────────                         │
│  _target_: planning.cem.CEMPlanner                                         │
│  horizon: 5, num_samples: 300, opt_steps: 30, topk: 30                    │
│  var_scale: 1                                                               │
│  → Für Offline-Evaluation in plan.py                                       │
│  → NICHT für MPC geeignet (zu langsam, kein Warm-Start-Support)            │
│                                                                             │
│  conf/planner/mpc_cem.yaml (MPC mit CEM Sub-Planner):                     │
│  ──────────────────────────────────────────────────                         │
│  _target_: planning.mpc.MPCPlanner                                         │
│  n_taken_actions: 5  ← Alle Horizon-Steps ausführen (= Open-Loop-ähnlich) │
│  sub_planner:                                                               │
│    _target_: planning.cem.CEMPlanner                                       │
│    horizon: 5, num_samples: 300, opt_steps: 30, topk: 30                  │
│  → MPC-Wrapper, aber mit n_taken=5 de facto Open-Loop                      │
│  → Benötigt env + evaluator (für lokalen Sim-Rollout)                      │
│                                                                             │
│  conf/planner/mpc_gd.yaml (MPC mit Gradient Descent):                     │
│  ──────────────────────────────────────────────────                         │
│  _target_: planning.mpc.MPCPlanner                                         │
│  n_taken_actions: 1  ← NUR 1 Step ausführen, dann re-planen               │
│  sub_planner:                                                               │
│    _target_: planning.gd.GDPlanner                                         │
│  → Zeigt: Die Autoren nutzen n_taken=1 für GD-basiertes MPC               │
│  → Bestätigt: n_taken=1 ist der richtige Ansatz für maximales Feedback     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEOBACHTUNG: Die mpc_gd.yaml nutzt n_taken_actions=1 — das bestätigt,    │
│  dass die Paper-Autoren bei MPC möglichst häufig re-planen wollen.         │
│  Für CEM-MPC ist n_taken=5 in mpc_cem.yaml gesetzt, was aber mehr          │
│  ein "MPC-Warm-Start" als echtes MPC ist.                                  │
│                                                                             │
│  UNSERE STRATEGIE: CEM mit n_taken=1 (wie GD-MPC) — kombiniert die        │
│  Robustheit von CEM mit dem maximalen Feedback von n_taken=1.              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.7.8 Warum wir MPCPlanner nicht direkt verwenden können

Der existierende `MPCPlanner` (in `planning/mpc.py`) kann in unserer Socket-Architektur **nicht direkt** instanziiert werden:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│       WARUM MPCPlanner NICHT DIREKT FUNKTIONIERT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MPCPlanner.__init__() erwartet:                                            │
│    - env: SerialVectorEnv (lokale Simulation für Rollouts)                 │
│    - evaluator: PlanEvaluator (bewertet Actions im lokalen Env)            │
│                                                                             │
│  MPCPlanner.plan() macht intern:                                            │
│    1. sub_planner.plan(obs_0, obs_g)     → Plan im World Model             │
│    2. evaluator.eval_actions(actions)     → Rollout in LOKALEM Env         │
│    3. Neues obs_0 aus env.rollout()       → Neues Bild aus LOKALEM Env     │
│    4. Wiederhole mit neuem obs_0                                            │
│                                                                             │
│  PROBLEM FÜR UNSERE ARCHITEKTUR:                                           │
│  ───────────────────────────────                                            │
│  Unser "Environment" ist Isaac Sim — in einem ANDEREN PROZESS auf einem    │
│  ANDEREN Python-Environment (python.sh). Es gibt kein lokales env-Objekt   │
│  das MPCPlanner aufrufen könnte.                                            │
│                                                                             │
│  ┌─────────────────┐          ┌─────────────────┐                          │
│  │ planning_server  │ ◄─TCP─► │ planning_client  │                          │
│  │ (conda dino_wm)  │         │ (Isaac Sim)      │                          │
│  │                   │         │                   │                          │
│  │ MPCPlanner        │         │ MinimalFrankaEnv  │                          │
│  │ benötigt env ──── ╳ ──────►│ (ist HIER, nicht  │                          │
│  │                   │         │  im Server!)      │                          │
│  └─────────────────┘          └─────────────────┘                          │
│                                                                             │
│  LÖSUNG: MPC-Logik ist im Client/Server-Protokoll implementiert.           │
│  ─────────────────────────────────────────────────────────────              │
│  Der Client übernimmt die MPC-Schleife:                                    │
│    1. Client holt Bild von Isaac Sim Kamera                                │
│    2. Client sendet Bild an Server → Server plant mit CEM                  │
│    3. Server gibt n_taken Sub-Actions zurück (+ Warm-Start intern)         │
│    4. Client führt Sub-Actions in Isaac Sim aus (RMPFlow IK)               │
│    5. Client holt neues Bild → zurück zu Schritt 2                         │
│                                                                             │
│  Dies ist funktional IDENTISCH mit MPCPlanner, nur verteilt über TCP.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.7.9 Zusammenfassung: Empfohlener Planning-Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                EMPFOHLENER PLANNING-WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. IMMER Online MPC verwenden (--mode online)                             │
│     Begründung: Paper Table 8 — MPC > Open-Loop in ALLEN Environments      │
│                                                                             │
│  2. Horizon=5 beibehalten                                                   │
│     Begründung: Paper Table 11 — Standard für alle Environments            │
│     Vorteil: Langfristiges Vorausdenken (5 Steps = 10 Sub-Actions)         │
│                                                                             │
│  3. n_taken=1 (nur 1 Horizon-Step ausführen, dann re-planen)               │
│     Begründung: mpc_gd.yaml nutzt n_taken=1; maximales Feedback            │
│     Praxis: 2 Sub-Actions pro MPC-Step (frameskip=2)                       │
│                                                                             │
│  4. CEM-Parameter: 100×10 (Paper-Standard) oder 64×5 (schneller)          │
│     Begründung: Table 10 — 100×10 → 53s auf A6000                         │
│     Unsere HW: 100×10 → ~25-35s, 64×5 → ~8-12s                           │
│                                                                             │
│  5. Warm-Start IMMER aktiviert (bereits implementiert)                     │
│     Begründung: Shifted μ konvergiert in weniger Iterationen               │
│     Praxis: Reduziert effektive opt_steps um ~50%                          │
│                                                                             │
│  ─── STARTBEFEHLE ───                                                       │
│                                                                             │
│  # Server (empfohlene Paper-nahe Konfiguration):                            │
│  python planning_server.py --model_name 2026-02-09/08-12-44 \              │
│      --num_samples 100 --opt_steps 10 --topk 20 --goal_H 5                │
│                                                                             │
│  # Client (Online MPC):                                                     │
│  ../../python.sh planning_client.py \                                       │
│      --goal_image /pfad/dataset:0:-1 \                                      │
│      --mode online --max_steps 50                                           │
│                                                                             │
│  → Erwartete Dauer: 50 Steps × ~30s = ~25 Minuten pro Episode             │
│  → Alternative: --num_samples 64 --opt_steps 5 → ~8 Min/Episode           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Fazit für die Masterarbeit:**

Die Entscheidung für Online MPC statt Offline Open-Loop ist keine Kompromisslösung, sondern **der im Paper als optimal identifizierte Ansatz**. Die vermeintlich höhere Rechenzeit pro Episode (~25 min statt ~3 min für Offline) wird dadurch kompensiert, dass:

1. **Jede Episode deutlich höhere Erfolgsraten hat** (Table 8: bis zu +30% bei Wall)
2. **Weniger Episoden für aussagekräftige Evaluation nötig sind** (höhere Konsistenz)
3. **Die Ergebnisse für die Masterarbeit wissenschaftlich besser vergleichbar sind** mit den Paper-Resultaten, da wir denselben MPC-Ansatz verwenden

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

### 8.5 Planning Server — Vollständige Startbefehl-Übersicht

> **Datum:** 09.02.2026
> **Aktueller Modell-Checkpoint:** `2026-02-09/08-12-44` (frameskip=2, num_hist=2, img_size=224, normalize_action=true)

Der `planning_server.py` ist der zentrale Entry-Point für Online-MPC-Planning mit dem Franka-Roboter in Isaac Sim. Er läuft in der `dino_wm` Conda-Umgebung und kommuniziert via TCP-Socket (Port 5555) mit dem Isaac Sim Client (`planning_client.py`).

#### 8.5.1 Alle verfügbaren CLI-Parameter

```bash
python planning_server.py \
    --model_name <PFAD>           # PFLICHT: Modell relativ zu outputs/
    --mode online|offline          # Planning-Modus (default: online)
    --port <INT>                   # TCP-Port (default: 5555)
    --goal_H <INT>                 # Planning-Horizon (default: online=2, offline=5)
    --num_samples <INT>            # CEM Samples pro Iteration (default: online=64)
    --opt_steps <INT>              # CEM Optimierungsschritte (default: online=5)
    --topk <INT>                   # CEM Elite-Samples (default: online=10)
    --wandb                        # W&B Dashboard-Logging aktivieren
    --wandb_project <STR>          # W&B Projektname (default: dino_wm_planning)
```

#### 8.5.2 Parameter-Erklärungen im Detail

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PARAMETER-REFERENZ                                                          │
├──────────────┬───────────────────────────────────────────────────────────────┤
│              │                                                               │
│  --model_name│  Pflichtparameter. Pfad zum Modell-Checkpoint relativ         │
│              │  zu outputs/. Enthält hydra.yaml + checkpoints/.              │
│              │  Beispiel: 2026-02-09/08-12-44                                │
│              │                                                               │
│  --mode      │  online (default): MPC-Loop. Client sendet nach jeder         │
│              │    ausgeführten Aktion ein neues Bild → re-plane.             │
│              │    CEM-Parameter werden reduziert für schnellere Planung.     │
│              │  offline: Open-Loop. Einmaliger Plan, alle Aktionen werden    │
│              │    auf einmal zurückgegeben (via plan_all Befehl).            │
│              │    Nutzt volle cem.yaml Parameter (300/30/30).               │
│              │                                                               │
│  --goal_H    │  Planning-Horizon: Wie viele Zeitschritte das World Model     │
│              │  in die Zukunft simuliert.                                     │
│              │  Online-Default: 2 (24D Suchraum — schnell konvergierend)    │
│              │  Offline-Default: 5 (60D Suchraum — mehr Vorausdenken)       │
│              │  Paper-Standard: 5 (Table 11, Appendix A.8)                   │
│              │                                                               │
│  --num_samples│  Anzahl zufällig gesampelter Aktionssequenzen pro CEM-       │
│              │  Iteration. Mehr Samples = bessere Abdeckung des Suchraums,  │
│              │  aber linear mehr Rechenzeit.                                  │
│              │  Online-Default: 64 | Offline/cem.yaml: 300                  │
│              │  Paper MPC (Table 10): 100                                    │
│              │                                                               │
│  --opt_steps │  Anzahl CEM-Optimierungsiterationen. In jeder Iteration:     │
│              │  Sample → Evaluate → Top-K → Update μ/σ.                     │
│              │  Mehr Steps = bessere Konvergenz, aber linear mehr Zeit.      │
│              │  Online-Default: 5 | Offline/cem.yaml: 30                    │
│              │  Paper MPC (Table 10): 10                                     │
│              │                                                               │
│  --topk      │  Anzahl der Elite-Samples für μ/σ-Update. Muss < num_samples │
│              │  sein. Kleinere Werte = aggressivere Fokussierung,           │
│              │  aber Risiko auf lokale Minima.                               │
│              │  Online-Default: 10 | Offline/cem.yaml: 30                   │
│              │  Faustregel: topk ≈ num_samples / 5–10                       │
│              │                                                               │
│  --wandb     │  Aktiviert Weights & Biases Logging. Loggt:                   │
│              │  - cem/loss pro CEM-Iteration (für Konvergenz-Plots)         │
│              │  - plan_summary/initial, final, reduction pro plan()-Aufruf   │
│              │  - plan_summary/time_s Planungsdauer                          │
│              │  Ohne --wandb: Nur stdout-Ausgabe (weiterhin aktiv).         │
│              │                                                               │
└──────────────┴───────────────────────────────────────────────────────────────┘
```

**Suchraum-Dimensionalität** — bestimmt die CEM-Schwierigkeit:

$$\text{SearchDim} = \text{goal\_H} \times \text{action\_dim} \times \text{frameskip}$$

| goal_H | Franka (6D, frameskip=2) | Push-T (2D) | Wall (2D) |
|--------|--------------------------|-------------|-----------|
| 1 | **12D** | 2D | 2D |
| 2 | **24D** | 4D | 4D |
| 5 | **60D** | 10D | 10D |
| 10 | **120D** (nicht empfohlen) | 20D | 20D |

→ Bei Franka ist der Suchraum **6× größer** als bei den Paper-Environments. Das erklärt, warum man mehr Samples und Iterationen braucht.

#### 8.5.3 Empfohlene Konfigurationen (Copy-Paste-fertig)

Alle Befehle gehen davon aus, dass man sich im `dino_wm`-Verzeichnis befindet mit aktivierter Conda-Umgebung:

```bash
cd ~/Desktop/dino_wm
conda activate dino_wm
```

**Config A — Debug (Minimal, ~3-5s/plan)**

```bash
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 32 --opt_steps 3 --topk 5 --goal_H 2
```

| Eigenschaft | Wert |
|-------------|------|
| Suchraum | 24D |
| DINO-Passes | 32 × 3 = 96 |
| Geschätzte Zeit/plan | ~3-5s |
| Verwendung | Socket-Debugging, Verbindungstests, schnelle Iteration |
| Qualität | Niedrig — CEM findet nur grobe Richtung |

**Config B — Standard Online MPC (~8-12s/plan)**

```bash
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 64 --opt_steps 5 --topk 10
```

| Eigenschaft | Wert |
|-------------|------|
| Suchraum | 24D (default goal_H=2) |
| DINO-Passes | 64 × 5 = 320 |
| Geschätzte Zeit/plan | ~8-12s |
| Verwendung | Standard-MPC mit kurzen Horizont |
| Qualität | Mittel — Warm-Start kompensiert kurzen Horizont |

**Config C — Erweitert mit langem Horizont (~25-30s/plan) ← AKTUELL IM EINSATZ**

```bash
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 128 --opt_steps 10 --goal_H 5
```

| Eigenschaft | Wert |
|-------------|------|
| Suchraum | **60D** |
| DINO-Passes | 128 × 10 = 1.280 |
| Geschätzte Zeit/plan | ~25-30s |
| topk | 10 (default, da kein --topk angegeben) |
| Verwendung | Aktuelle Testlauf-Konfiguration |
| Beobachtete Ergebnisse (09.02.2026) | Siehe 8.5.5 |

> **⚠️ Beobachtung:** `topk=10` bei `num_samples=128` bedeutet, dass nur die besten 7.8% der Samples das μ/σ-Update bestimmen. Das ist recht selektiv. `topk=20` wäre weniger aggressiv.

**Config D — Paper-nah (~30-40s/plan) ← EMPFOHLEN**

```bash
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 100 --opt_steps 10 --topk 20 --goal_H 5
```

| Eigenschaft | Wert |
|-------------|------|
| Suchraum | 60D |
| DINO-Passes | 100 × 10 = 1.000 |
| Geschätzte Zeit/plan | ~30-40s |
| Verwendung | Am nächsten an Paper Table 10 (53s auf A6000) |
| Qualität | Hoch — Paper-validierte Parameter |

**Config E — Qualität (~50-70s/plan)**

```bash
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 200 --opt_steps 15 --topk 30 --goal_H 5
```

| Eigenschaft | Wert |
|-------------|------|
| Suchraum | 60D |
| DINO-Passes | 200 × 15 = 3.000 |
| Geschätzte Zeit/plan | ~50-70s |
| Verwendung | Bestmögliche Online-Qualität, wenn Zeit unkritisch |
| Qualität | Sehr hoch — 3× mehr Budget als Paper-Standard |

**Config F — Offline Evaluation (~180s/plan)**

```bash
python planning_server.py --model_name 2026-02-09/08-12-44 --mode offline
```

| Eigenschaft | Wert |
|-------------|------|
| Suchraum | 60D (default goal_H=5) |
| DINO-Passes | 300 × 30 = 9.000 |
| Geschätzte Zeit/plan | ~180s (3 Minuten) |
| Verwendung | Open-Loop Baseline, plan_all Befehl |
| Qualität | Maximale CEM-Qualität, aber kein Feedback (Open-Loop) |

**Config G — Jede Config mit W&B Dashboard**

```bash
# Einfach --wandb an jede Config anhängen:
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 128 --opt_steps 10 --goal_H 5 \
    --wandb --wandb_project dino_wm_planning

# W&B Dashboard öffnet sich automatisch im Browser.
# Metriken: cem/loss, plan_summary/initial, plan_summary/final, 
#           plan_summary/reduction, plan_summary/time_s
```

#### 8.5.4 Konfigurations-Vergleichstabelle

```
┌──────────┬──────────┬───────────┬───────┬────────┬────────────┬────────────┐
│ Config   │ Samples  │ OptSteps  │ TopK  │ goalH  │ Passes     │ ~Zeit/plan │
├──────────┼──────────┼───────────┼───────┼────────┼────────────┼────────────┤
│ A Debug  │    32    │     3     │   5   │   2    │      96    │   3-5s     │
│ B Std    │    64    │     5     │  10   │   2    │     320    │   8-12s    │
│ C Erw.   │   128    │    10     │  10   │   5    │   1.280    │  25-30s    │
│ D Paper  │   100    │    10     │  20   │   5    │   1.000    │  30-40s    │
│ E Qual.  │   200    │    15     │  30   │   5    │   3.000    │  50-70s    │
│ F Offl.  │   300    │    30     │  30   │   5    │   9.000    │  ~180s     │
├──────────┼──────────┼───────────┼───────┼────────┼────────────┼────────────┤
│ Paper    │   100    │    10     │   ?   │   5    │   1.000    │   53s      │
│ (Table10)│          │           │       │        │            │  (A6000)   │
└──────────┴──────────┴───────────┴───────┴────────┴────────────┴────────────┘

Alle Zeiten geschätzt für unsere Hardware (RTX-Klasse GPU).
Paper-Referenz: Table 10, Appendix A.8, S. 17.
```

#### 8.5.5 CEM-Output lesen und interpretieren

Die Server-Ausgabe bei jedem `plan`-Befehl folgt diesem Schema:

```
  [Plan] Running CEM (samples=128, steps=10, horizon=5)...
    [CEM] Step 1: loss=3.970347       ← Anfangsloss (je niedriger, desto besser)
    [CEM] Step 2: loss=3.039177       ← Sollte sinken
    ...
    [CEM] Step 10: loss=2.161562      ← Endloss
  [Plan] loss: 3.970347 -> 2.161562 (45.6% Reduktion) (26.4s)
  [Plan] Actions shape: torch.Size([1, 5, 12])
  [Plan] mu L2-Norm (normalized): 9.8762 (0=Mittelwert, >1=signifikant)
  [Plan] 2 Sub-Actions (frameskip=2):
    sub 0: [0.4520, -0.0878, 0.1408, 0.4053, 0.3890, 0.1465]
    sub 1: [0.5423, -0.1669, 0.2493, 0.3591, -0.0387, 0.1373]
```

**Was die Metriken bedeuten:**

| Metrik | Gut | Schlecht | Interpretation |
|--------|-----|----------|----------------|
| Loss-Reduktion | > 30% | < 10% | CEM konvergiert gut vs. stagniert |
| Anfangsloss (kalt) | < 3.0 | > 5.0 | Wie schwer das Planungsproblem ist |
| Anfangsloss (warm) | < vorheriger Endloss + 0.5 | >> vorheriger Endloss | Warm-Start hilft vs. neue Szene zu anders |
| mu L2-Norm | 3-10 | > 15 | Plan weicht moderat vs. extrem vom Mittelwert ab |
| Sub-Action Werte | 0.1 - 0.8 (typischer Franka-Arbeitsraum) | > 1.0 oder < 0.0 | Plan im vs. außerhalb des Arbeitsraums |

**Typische Muster und ihre Bedeutung:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MUSTER 1: Gute Konvergenz (erwartet bei korrektem Setup)                   │
│  Plan 1 (kalt):  4.0 → 2.0 (50% Reduktion)                                │
│  Plan 2 (warm):  2.3 → 1.8 (22% Reduktion)  ← Startet nahe Plan 1 Ende   │
│  Plan 3 (warm):  2.0 → 1.6 (20% Reduktion)  ← Kontinuierliche Verbesser. │
│  → Roboter nähert sich dem Ziel.                                            │
│                                                                             │
│  MUSTER 2: Divergierende Starts (aktuell beobachtet!)                       │
│  Plan 1 (kalt):  3.97 → 2.16 (46%)                                         │
│  Plan 2 (warm):  2.85 → 2.39 (16%)  ← Start HÖHER als Plan 1 Ende!       │
│  Plan 3 (warm):  3.07 → 2.84 (7%)   ← Start noch HÖHER, kaum Reduktion!  │
│  Plan 4 (warm):  3.31 → 2.33 (30%)  ← Start weiter steigend              │
│  → Roboter bewegt sich NICHT zum Ziel. Jeder Schritt verschlechtert die    │
│    Ausgangslage. Warm-Start wird ungültig weil reale Szene nach Action-    │
│    Ausführung zu stark abweicht von WM-Prediktion.                         │
│                                                                             │
│  MUSTER 3: Loss stagniert                                                   │
│  Plan N: 4.5 → 4.3 (4% Reduktion)                                          │
│  → CEM findet keine bessere Lösung im 60D-Suchraum.                       │
│    Mögliche Ursachen: zu wenig Samples, goal zu weit entfernt,             │
│    oder WM-Qualität unzureichend.                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.6 Aktuelle Testergebnisse und Diagnose (09.02.2026)

**Getestete Konfiguration:** Config C (128/10/10, goal_H=5)

**Beobachtete Server-Ausgabe (4 MPC-Schritte):**

| Plan # | Warm-Start | Start-Loss | End-Loss | Reduktion | Zeit |
|--------|-----------|------------|----------|-----------|------|
| 1 | Nein (kalt) | 3.970 | 2.162 | 45.6% | 26.4s |
| 2 | Ja | 2.849 | 2.389 | 16.1% | 26.6s |
| 3 | Ja | 3.069 | 2.842 | 7.4% | 26.8s |
| 4 | Ja | 3.314 | 2.327 | 29.8% | 26.7s |

**Diagnose — Muster 2 (Divergierende Starts):**

```
Start-Loss-Entwicklung:  3.97 → 2.85 → 3.07 → 3.31
                         ─────────────────────────────► steigend!
                         
Das bedeutet: Nach Ausführung jeder Aktion ist die Szene WEITER
vom Ziel entfernt als vorher. Der Roboter bewegt sich nicht
zielgerichtet.
```

**Mögliche Ursachen (Reihenfolge nach Wahrscheinlichkeit):**

1. **Modellqualität (200 Episoden vs. Paper 1.000-18.500)**
   Das WM wurde mit nur 200 Episoden trainiert. Die Paper-Environments nutzen deutlich mehr Daten (Table 11: PushT 18.500, Wall 100 aber einfacheres 2D-Environment). Bei 200 Episoden mit 6D-Aktionsraum hat das WM möglicherweise keine genaue Dynamik gelernt → Prädiktionsfehler → CEM optimiert auf falsche Vorhersagen.

2. **topk zu aggressiv für 60D-Suchraum**
   `topk=10` bei `num_samples=128` = 7.8% Eliten. Im 60D-Suchraum kann dies zu schneller Konvergenz auf lokale Minima führen. **Empfehlung: `--topk 20` oder `--topk 25` testen.**

3. **Goal-Bild zu weit entfernt**
   Wenn das Goal-Bild einen Zustand zeigt, der viele Schritte entfernt ist, kann der CEM bei horizon=5 den Weg nicht finden. **Empfehlung: Einfacheres Goal testen (z.B. nur leichte Positionsänderung).**

4. **BGR-Konvertierung im Client korrekt?**
   Das Modell wurde mit BGR-Bildern trainiert. Der Client muss RGB→BGR konvertieren bevor er das Bild an den Server sendet. **Prüfen: `get_obs_for_planner()` in planning_client.py.**

**Nächste empfohlene Schritte:**

```bash
# 1. Gleiche Config aber mit mehr topk (weniger aggressiv):
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 128 --opt_steps 10 --topk 25 --goal_H 5 --wandb

# 2. Paper-nahe Config:
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 100 --opt_steps 10 --topk 20 --goal_H 5 --wandb

# 3. Kürzerer Horizont (weniger Dimensionen, leichter für CEM):
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 128 --opt_steps 10 --topk 20 --goal_H 3 --wandb

# 4. Maximale Qualität (Referenz-Baseline):
python planning_server.py --model_name 2026-02-09/08-12-44 \
    --num_samples 200 --opt_steps 20 --topk 30 --goal_H 5 --wandb
```

#### 8.5.7 Zugehöriger Client-Startbefehl (Isaac Sim)

```bash
# Terminal 2: Isaac Sim Client (in separater Shell)
cd ~/Desktop/isaacsim
./python.sh 00_Franka_Cube_Stack/Franka_Cube_Stacking/planning_client.py \
    --goal_image /pfad/zum/dataset:0:-1 \
    --mode online \
    --max_steps 50

# Erwartete Episodendauer bei Config C (128/10, ~27s/plan):
#   50 MPC-Steps × 27s = ~22 Minuten pro Episode
#
# Erwartete Episodendauer bei Config D (100/10, ~35s/plan):
#   50 MPC-Steps × 35s = ~29 Minuten pro Episode
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

**Lösung 1 — CEM-Parameter reduzieren:**
```bash
python plan.py ... planner.num_samples=128
```

**Lösung 2 — Dataset-Bilder im Server freigeben (HAUPTURSACHE bei großen Datasets):**

**Root Cause (gefunden 2026-02-10):** `FrankaCubeStackDataset` lädt mit `preload_images=True` **ALLE** Episoden-Bilder in den RAM. Ein `TrajSubset` (dset_val) hält eine Referenz zum **vollen** Dataset — d.h. auch `dset["valid"]` hält alle 500 Episoden im Speicher!

| Dataset-Größe | Geschätzter RAM | OOM bei 500 samples? |
|---|---|---|
| 200 Episoden | ~2-3 GB | ❌ kein Problem |
| 500 Episoden | ~6-8 GB | ✅ OOM! |

**Der Server braucht aber nur:**
- `action_mean/std` (6 Floats), `state_mean/std` (14 Floats), `proprio_mean/std` (3 Floats)
- `transform` (torchvision Transform)
- `action_dim` (1 Integer)

**Fix in `planning_server.py`:** Stats extrahieren + `.clone()`, dann sofort `del` + `gc.collect()`:
```python
# Statistiken extrahieren und KLONEN
base_action_dim = _dset_val.action_dim
action_mean_base = _dset_val.action_mean.clone()
# ... etc.

# SOFORT freigeben
del _dset_val, _traj_dset, _datasets
gc.collect()
torch.cuda.empty_cache()
```

**Warum werden die Episoden überhaupt geladen?**
Weil die Normalisierungs-Statistiken (`action_mean/std` etc.) im Dataset-Objekt berechnet werden und NICHT im Checkpoint gespeichert sind. Die Bilder selbst werden im Server NIE verwendet (Goals kommen vom Isaac Sim Client), aber `preload_images=True` lädt sie trotzdem mit.

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

### 9.5 ✅ BEHOBEN: Actions sahen aus wie Pixelkoordinaten (Multi-Robot Grid Offset Problem)

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

### 9.6 ✅ KEIN PROBLEM: Pixel-Space (Referenzdatensatz) vs. Meter-Space (Franka)

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

## 10. WM Sanity-Check: Vorhersagequalität mit GT-Aktionen

> **Datum:** 09.02.2026 | **Skript:** `wm_sanity_check.py` | **Model:** `2026-02-09/08-12-44`

### 10.1 Motivation und Problemstellung

Nach mehreren MPC-Planning-Tests mit dem Franka-Roboter in Isaac Sim zeigte sich ein wiederkehrendes Muster (**"Muster 2: Divergierende Starts"**):

```
Planning-Ergebnisse (128 Samples, 10 Opt-Steps, topk=25, H=5):
  Plan 1 (cold): Start-Loss 3.980 → Final-Loss 2.289 (−42.5%)
  Plan 2 (warm): Start-Loss 2.945 → Final-Loss 2.510 (−14.8%)
  Plan 3 (warm): Start-Loss 3.097 → Final-Loss 2.878 (−7.1%)
  Plan 4 (warm): Start-Loss 3.398 → Final-Loss 2.778 (−18.3%)
  Plan 5 (warm): Start-Loss 3.116 → ...

  ⚠️ Start-Losses STEIGEN trotz Warm-Start (2.945 → 3.097 → 3.398)
  ⚠️ CEM-Reduktion wird immer kleiner (42% → 14% → 7%)
```

**Kernfrage:** Liegt das Problem an der CEM-Parametrisierung oder am World Model selbst?

Die Hypothese: Wenn die WM-Vorhersagen mit Ground-Truth-Aktionen bereits schlecht sind, dann sind CEM-optimierte Aktionen zwangsläufig kontraproduktiv — der Planner optimiert gegen ein fehlerhaftes Modell.

**Zusätzlicher Kontext:** Das Franka-Modell wurde mit **200 Episoden** trainiert. Die Referenz-Datensätze im DINO-WM-Paper verwenden **1000 Episoden** (Rope, Wall) bzw. **18.500 Trajektorien** (Push-T).

### 10.2 Methodik des Sanity-Checks

Der WM Sanity-Check prüft die Vorhersagequalität des World Models, indem er **Ground-Truth-Aktionen aus dem Trainingsdatensatz** durch das Modell rollt und die vorhergesagten Bilder mit den tatsächlichen vergleicht:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WM SANITY-CHECK METHODIK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Für jede Test-Episode aus dem Validierungs-Set:                           │
│                                                                             │
│  1. RECONSTRUCTION-TEST (Basislinie):                                       │
│     ┌───────┐    DINO      ┌───────┐    VQVAE     ┌───────┐               │
│     │ Bild  │ ──Encoder──► │Latent │ ──Decoder──► │ Bild' │               │
│     │  (GT) │              │ Space │              │(Recon)│               │
│     └───────┘              └───────┘              └───────┘               │
│     → Misst: Wie gut rekonstruiert der Decoder?                            │
│                                                                             │
│  2. PREDICTION-TEST (Kerntest):                                             │
│     ┌───────┐    Encode     ┌───────┐   Predict    ┌───────┐  Decode      │
│     │obs_0  │ ──────────► │ z_0   │ ──────────► │z_pred │ ──────►Bild   │
│     │(2 Hist│    + GT     │       │  ViT-Trans- │       │  VQVAE        │
│     │Frames)│  Actions    └───────┘  former      └───────┘               │
│     └───────┘                                                              │
│     → Misst: Wie gut sagt das WM den nächsten Zustand vorher?              │
│                                                                             │
│  3. HORIZONT-ANALYSE:                                                       │
│     Wiederhole Prediction über mehrere Schritte (autoregressive Rollout)   │
│     → Misst: Wie schnell akkumulieren Vorhersagefehler?                    │
│                                                                             │
│  Metriken:                                                                  │
│  - MSE im normalisierten Bildraum ([-1, 1])                               │
│  - PSNR (Peak Signal-to-Noise Ratio) in dB                                │
│  - Prediction/Reconstruction Ratio (Schlüsselmetrik)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Wichtig:** Der Test verwendet **ausschließlich Validierungs-Episoden** (20 Episoden, 10% Split) — das Modell hat diese Daten nie gesehen.

### 10.3 Implementierung: wm_sanity_check.py

**Datei:** `dino_wm/wm_sanity_check.py`

**Aufruf:**
```bash
conda activate dino_wm
python wm_sanity_check.py --model_name 2026-02-09/08-12-44 --n_episodes 5 --rollout_len 5
```

**CLI-Parameter:**

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--model_name` | (required) | Checkpoint-Name (z.B. `2026-02-09/08-12-44`) |
| `--n_episodes` | 5 | Anzahl zu testender Validierungs-Episoden |
| `--rollout_len` | 5 | Anzahl Vorhersage-Schritte nach `num_hist` |

**Technische Details der Implementierung:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATENFLUSS IM SANITY-CHECK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dataset (Validierung, 20 Episoden)                                         │
│    │                                                                        │
│    ├── obses.pth: (T, H, W, C) float32 BGR 0-255                          │
│    ├── H5-Files:  actions (6D, z-normalisiert), eef_states (14D)           │
│    │                                                                        │
│    ▼                                                                        │
│  get_frames(idx, frame_indices)                                             │
│    │  Frame-Indizes: [0, fs, 2*fs, ...] mit fs=frameskip=2                 │
│    │  → obs['visual']: (T, C, H, W) normalisiert + transformiert           │
│    │  → act: (T, 6) z-normalisiert                                         │
│    │                                                                        │
│    ▼                                                                        │
│  Action-Concatenation für Frameskip:                                        │
│    Einzelaktionen (6D) → je frameskip=2 concateniert → (12D)               │
│    WM-Step i: act[i*2] ∥ act[i*2+1] → (12,)                               │
│    │                                                                        │
│    ▼  (Action-Encoder erwartet Conv1d(12, 10, kernel_size=1))              │
│                                                                             │
│  WM.rollout(obs_0, all_acts)                                               │
│    │  obs_0: (1, num_hist=2, 3, 224, 224) — erste 2 Frames                │
│    │  all_acts: (1, total_steps, 12) — concatenierte GT-Aktionen           │
│    │                                                                        │
│    ├── z = encode(obs_0, act_0)        [DINO + Action Encoder]             │
│    ├── for each step:                                                       │
│    │   ├── z_pred = predict(z[-2:])    [ViT Transformer]                   │
│    │   ├── z_new = replace_actions(z_pred, action_t)                       │
│    │   └── z = cat(z, z_new)                                               │
│    └── final predict → z_final                                              │
│    │                                                                        │
│    ▼                                                                        │
│  decode_obs(z) → predicted images                                           │
│    │  VQVAE Decoder: latent → (B, T, 3, 224, 224) normalisiert             │
│    │                                                                        │
│    ▼                                                                        │
│  Vergleich: predicted vs. GT (MSE, PSNR)                                   │
│  Visualisierung: 3-Zeilen Side-by-Side (GT | Prediction | Differenz)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Output-Verzeichnis:** `wm_sanity_outputs/<model_name>/`
- `episode_XXXX.png` — Side-by-Side-Vergleich pro Episode (3 Zeilen × T Spalten)
- `mse_over_horizon.png` — MSE-Verlauf über Vorhersage-Horizont (aggregiert)
- `metrics.json` — Alle Metriken als strukturiertes JSON

### 10.4 Ergebnisse: Quantitative Analyse

**Testbedingungen:**
- Model: `2026-02-09/08-12-44` (Epoch 50, 200 Episoden, frameskip=2, num_hist=2)
- 5 Validierungs-Episoden, je 5 Vorhersage-Schritte + 1 Extra
- Device: NVIDIA RTX A5000 (24 GB)

#### Reconstruction-Qualität (Basislinie)

| Metrik | Wert | Bewertung |
|--------|------|-----------|
| **Ø MSE** | 0.0035 | Gut |
| **Ø PSNR** | 30.5 dB | Akzeptabel |

→ Der VQVAE-Decoder rekonstruiert Bilder aus DINO-Embeddings recht gut. Die Encoder→Decoder-Pipeline (ohne Prediction) funktioniert.

#### Prediction-Qualität mit GT-Aktionen

| Horizont-Schritt | Ø MSE | Ø PSNR (dB) | Degradation vs. Recon |
|------------------|-------|-------------|----------------------|
| Reconstruction | 0.0035 | 30.5 | (Basislinie) |
| **pred_1** | 0.0088 | 26.6 | 2.5× |
| **pred_2** | 0.0116 | 25.4 | 3.3× |
| **pred_3** | 0.0133 | 24.8 | 3.8× |
| **pred_4** | 0.0107 | 25.7 | 3.1× |
| **pred_5** | 0.0145 | 24.4 | 4.1× |
| **pred_6** | 0.0159 | 24.0 | 4.5× |

**Gesamt:**
| Metrik | Wert |
|--------|------|
| Ø Prediction MSE | 0.0125 |
| Ø Prediction PSNR | 25.1 dB |
| **Prediction/Reconstruction Ratio** | **3.52×** |

#### MSE-Verlauf über Horizont

```
MSE
  │
  0.016 ─┤                                          ╱ pred_6
  0.014 ─┤                              ╱──────────╱  pred_5
  0.012 ─┤                   ╱─────────╱
  0.010 ─┤           ╱──────╱              pred_3, pred_4
  0.008 ─┤     ╱────╱   pred_1, pred_2
  0.006 ─┤    ╱
  0.004 ─┤───╱  recon (Basislinie: 0.0035)
  0.002 ─┤
  0.000 ─┼──────┬──────┬──────┬──────┬──────┬──────┬──────
          0      1      2      3      4      5      6
              Vorhersage-Schritt (0 = Reconstruction)
```

#### Ergebnisse pro Episode (Detail)

| Episode | Recon MSE | Pred_1 MSE | Pred_5 MSE | Trend |
|---------|-----------|------------|------------|-------|
| Ep 0 | 0.0100 | 0.0184 | 0.0078 | Schwankend |
| Ep 4 | 0.0021 | 0.0046 | 0.0126 | Steigend |
| Ep 8 | 0.0026 | 0.0080 | 0.0279 | Stark steigend |
| Ep 12 | 0.0015 | 0.0067 | 0.0181 | Steigend |
| Ep 16 | 0.0015 | 0.0061 | 0.0061 | Stabil |

→ **Hohe Varianz** zwischen Episoden: Manche (Ep 8) degradieren stark, andere (Ep 16) bleiben stabil.

### 10.5 Diagnose und Interpretation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIAGNOSE-ZUSAMMENFASSUNG                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Prediction/Reconstruction MSE Ratio: 3.52×                                 │
│                                                                             │
│  ⚠️  BEWERTUNG: MODERATE DEGRADATION                                       │
│                                                                             │
│  Interpretation:                                                            │
│  ──────────────                                                             │
│  • Ratio < 2.0×: ✅ WM gut trainiert — Problem liegt woanders              │
│  • Ratio 2.0–5.0×: ⚠️ Moderate Degradation — WM hat teilweise gelernt     │
│  • Ratio > 5.0×: ❌ WM schlecht — kann Dynamik nicht vorhersagen           │
│                                                                             │
│  Mit 3.52× liegt das Franka-Modell im mittleren Bereich.                   │
│                                                                             │
│  Was bedeutet das für CEM-Planning?                                         │
│  ─────────────────────────────────                                          │
│  1. Der ViT-Predictor (Transformer) ist die Schwachstelle,                 │
│     NICHT der VQVAE-Decoder (Reconstruction ist gut)                       │
│                                                                             │
│  2. Selbst mit PERFEKTEN GT-Aktionen ist die Vorhersage schon              │
│     2.5× schlechter als Reconstruction — bei Schritt 1!                    │
│                                                                             │
│  3. Für CEM-Planning ist das fatal:                                         │
│     - CEM vergleicht vorhergesagte Bilder mit Zielbildern                  │
│     - Bei MSE ~0.01-0.02 ist die Vorhersage zu unscharf                    │
│     - CEM kann feine Aktionsunterschiede nicht diskriminieren              │
│     - Optimierte Aktionen sind daher quasi zufällig                        │
│                                                                             │
│  4. Fehlerakkumulation über den Horizont:                                   │
│     MSE verdoppelt sich von Schritt 1 (0.009) zu Schritt 6 (0.016)        │
│     → Längere Horizonte (H>3) sind unzuverlässig                           │
│                                                                             │
│  Hauptursache: Unzureichende Trainingsdaten                                │
│  ─────────────────────────────────────────                                  │
│  • Franka: 200 Episoden (aktuell)                                          │
│  • DINO-WM Paper Referenz: 1000 Episoden (Rope, Wall)                     │
│  • DINO-WM Paper Referenz: 18.500 Trajektorien (Push-T)                   │
│  → Das Modell hat die Franka-Dynamik noch nicht ausreichend gelernt        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.6 Kritischer Fund: Visuelle Diskrepanz Training vs. Planning

Bei der Analyse des Sanity-Checks wurde eine **zweite, unabhängige Problemquelle** identifiziert: Die visuelle Umgebung im Planning Client (Isaac Sim) unterscheidet sich **erheblich** von der Trainingsumgebung.

#### Vergleich: Trainings-Szene vs. Planning-Szene

| Szenenkomponente | Training (fcs_main_parallel.py) | Planning (planning_client.py) | Status |
|------------------|-------------------------------|-------------------------------|--------|
| **Default Ground Plane** | ✅ Vorhanden | ✅ Vorhanden | ✅ Match |
| **Custom Bodenplatte** | ✅ 0.60m × 0.75m Quad vor Robot | ❌ **FEHLT** | 🔴 **MISMATCH** |
| **Material-Randomisierung** | ✅ 7 Materialien (Stahl, Holz, Gummi, ...) | ❌ Keine | 🔴 **MISMATCH** |
| **Beleuchtung** | ✅ Randomisierte SphereLight (5500–7000) | ❌ Nur Default-Licht | 🟡 MISMATCH |
| **Roboter-Sichtbarkeit** | ✅ Opacity 1.0 | ✅ Opacity 1.0 | ✅ Match |
| **Würfel-Farben** | ✅ Randomisierte Farben | ⚠️ Festes Rot | 🟡 Minor |
| **Kamera-Position** | ✅ Aus camera_configs.py | ✅ Aus camera_configs.py | ✅ Match |
| **Bildformat** | ✅ 224×224 BGR | ✅ 224×224 BGR | ✅ Match |

#### Das Bodenplatten-Problem im Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              VISUELLE DISKREPANZ: BODENPLATTE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING (fcs_main_parallel.py → min_data_logger.py):                     │
│  ┌─────────────────────────────────────────┐                               │
│  │          Kamera-Blick                    │                               │
│  │  ┌─────────────────────────────────┐    │                               │
│  │  │  ████████████████████████████   │    │  ← Farbige Bodenplatte        │
│  │  │  ███ Würfel auf Platte ██████   │    │    (Stahl/Holz/Gummi/...)     │
│  │  │  ████████████████████████████   │    │    0.60m × 0.75m              │
│  │  └─────────────────────────────────┘    │                               │
│  │        Grauer Isaac Sim Ground           │                               │
│  └─────────────────────────────────────────┘                               │
│                                                                             │
│  PLANNING (planning_client.py → MinimalFrankaEnv):                         │
│  ┌─────────────────────────────────────────┐                               │
│  │          Kamera-Blick                    │                               │
│  │                                          │                               │
│  │       Würfel auf grauem Ground           │  ← KEINE Bodenplatte!        │
│  │                                          │    Nur Default Ground Plane   │
│  │                                          │                               │
│  └─────────────────────────────────────────┘                               │
│                                                                             │
│  KONSEQUENZ FÜR DINO ENCODER:                                              │
│  Der DINOv2-Encoder erzeugt für diese visuell verschiedenen Szenen         │
│  UNTERSCHIEDLICHE Embeddings — auch bei identischer Roboter/Würfel-Pose.   │
│  → Das trainierte World Model bekommt Input aus einer anderen Verteilung   │
│  → "Distribution Shift" führt zu schlechten Vorhersagen beim Planning      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Relevanter Code

**Training** (`min_data_logger.py` → `add_or_update_plane()`):
```python
def add_or_update_plane(self, seed):
    # Erstellt 0.60m × 0.75m Quad vor Robot-Base
    mesh = UsdGeom.Mesh.Define(self.stage, f"{self.task_root}/Plane")
    # Wählt zufälliges Material (7 Optionen):
    ALLOWED_AREA_MATS = ["Steel", "Aluminum", "Oak_Wood", "Birch_Plywood",
                          "Black_HDPE", "Rubber_Mat", "Frosted_Acrylic"]
    material = self.materials[random_material_index]
    UsdShade.MaterialBindingAPI(plane_prim).Bind(material)
```

**Planning** (`planning_client.py` → `MinimalFrankaEnv.setup()`):
```python
def setup(self):
    self.world = World(stage_units_in_meters=1.0)
    self.world.scene.add_default_ground_plane()  # ← NUR default ground!
    self._add_franka()
    self._add_cubes()
    self._add_camera()
    # KEIN add_or_update_plane() — keine Bodenplatte!
```

### 10.7 Konsequenzen und Handlungsempfehlungen

Es gibt **zwei unabhängige Probleme**, die beide zur schlechten Planning-Performance beitragen:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ZWEI UNABHÄNGIGE PROBLEME                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM 1: UNZUREICHENDE TRAININGSDATEN (WM-Qualität)                     │
│  ══════════════════════════════════════════════════════                     │
│  Symptom:   Prediction/Reconstruction Ratio = 3.52×                        │
│  Ursache:   200 Episoden vs. Paper-Referenz 1000                           │
│  Wirkung:   WM kann Franka-Dynamik nicht genau vorhersagen                 │
│  Lösung:    Mehr Trainingsdaten sammeln (Ziel: 1000 Episoden)              │
│  Priorität: HOCH — aber zeitaufwändig                                      │
│                                                                             │
│  PROBLEM 2: VISUELLE DISKREPANZ (Distribution Shift)                       │
│  ══════════════════════════════════════════════════════                     │
│  Symptom:   Planning-Bilder sehen anders aus als Trainingsbilder           │
│  Ursache:   MinimalFrankaEnv hat keine Bodenplatte + Default-Licht         │
│  Wirkung:   DINO-Encoder erzeugt andere Embeddings → WM-Vorhersagen       │
│             basieren auf falscher Eingabeverteilung                         │
│  Lösung:    Bodenplatte + Beleuchtung im Planning Client nachbauen         │
│  Priorität: KRITISCH — schnell umsetzbar, großer Impact                   │
│                                                                             │
│  EMPFOHLENE REIHENFOLGE:                                                    │
│  ─────────────────────                                                      │
│  1. ✅ Bodenplatte im Planning Client hinzufügen (sofort umsetzbar)        │
│  2. ✅ Beleuchtung im Planning Client anpassen                             │
│  3. 🔄 Erneut testen mit korrekter visueller Umgebung                      │
│  4. 📊 Sanity-Check wiederholen zur Basislinie                             │
│  5. 📈 Falls immer noch schlecht: Mehr Trainingsdaten sammeln              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Zusammenfassung:** Das WM-Training mit 200 Episoden zeigt moderate Vorhersagequalität (3.52× Degradation vs. Reconstruction). Das ist ein bekanntes Symptom für untertrainierte World Models. **Zusätzlich** wurde die fehlende Bodenplatte im Planning Client als kritische visuelle Diskrepanz identifiziert, die zu einem Distribution Shift im DINO-Encoder führt. Beide Probleme müssen adressiert werden, aber die Bodenplatte ist der schnellere Fix.

---

## 11. Finaler Offline-Test: Modell unzureichend für Cube Stacking (14.02.2026)

### 11.1 Testaufbau und Parameter

Nach allen Bugfixes (Proprio-Fix, Goal-Image-Fix, OOM-Fix) wurde ein finaler Offline-Test mit dem 500-Episoden-Modell durchgeführt:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINALER OFFLINE-TEST (14.02.2026)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Modell:     2026-02-09/17-59-59 (500 Episoden, ActInt10, 50 Epochen)      │
│  GPU:        NVIDIA RTX A5000 (24 GB VRAM)                                 │
│                                                                             │
│  CEM-Parameter (maximale Qualität):                                        │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  Horizon (H):      5     (5 Schritte vorausplanen)      │               │
│  │  Samples:          300   (300 Aktionssequenzen/Iteration)│               │
│  │  Opt Steps:        30    (30 CEM-Iterationen)           │               │
│  │  Top-K:            30    (30 beste Samples für Refit)   │               │
│  │  MPC:              Aus   (Open-Loop, 10 Aktionen)       │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Cube-Position: (0.396, -0.215, 0.025)                                     │
│  Start-EEF:     (0.468, 0.079, 0.368)                                      │
│  Goal-EEF:      (0.586, -0.196, 0.069)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Ergebnisse: CEM-Konvergenz

```
CEM Optimierung:
  Start-Loss:    0.828
  End-Loss:      0.340
  Reduktion:     58.9%
  Dauer:         530 Sekunden (30 Steps)
  
  Loss-Verlauf:
  Step  1: 0.828 ──► Step 10: 0.405 ──► Step 20: 0.351 ──► Step 30: 0.340
           │                                                          │
           └─── Schnelle Konvergenz ──► Plateau ab Step ~15 ──────────┘
```

**Interpretation:** Der CEM konvergiert zwar (58.9% Reduktion), aber der finale Loss von 0.340 ist zu hoch. Ein Loss-Plateau bei 0.34 bedeutet: **Keine der 300 gesampelten Aktionssequenzen kann das Modell von einem „guten" Zielzustand überzeugen.** Das Modell hat keine interne Repräsentation für eine sinnvolle Trajektorie zum Cube.

### 11.3 Ergebnisse: Ausgeführte Trajektorie

Der Client führte die 10 geplanten Aktionen aus. **Keine einzige Aktion nähert sich dem Cube:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AUSGEFÜHRTE TRAJEKTORIE                                  │
├──────┬──────────────────────────────┬───────────────────────────────────────┤
│ Step │ EEF-Position (x, y, z)       │ Analyse                              │
├──────┼──────────────────────────────┼───────────────────────────────────────┤
│  0   │ (0.468,  0.079, 0.368)       │ Startposition                        │
│  1   │ (0.447, -0.001, 0.350)       │ Leichte Y-Korrektur, zu hoch         │
│  2   │ (0.441, -0.074, 0.327)       │ Weiter in -Y, immer noch hoch       │
│  3   │ (0.427, -0.059, 0.317)       │ Y springt zurück, kein Ziel         │
│  4   │ (0.378,  0.031, 0.265)       │ ← SPRUNG: Y wechselt auf +0.03!    │
│  5   │ (0.391,  0.117, 0.300)       │ ← CHAOTISCH: Y = +0.117            │
│  6   │ (0.426,  0.019, 0.311)       │ Y springt wieder                     │
│  7   │ (0.480, -0.043, 0.340)       │ Zurück Richtung Start!              │
│  8   │ (0.526, -0.111, 0.212)       │ Endlich tiefer, aber zu weit rechts │
│  9   │ (0.556, -0.090, 0.251)       │ Z wieder hoch                       │
│ 10   │ (0.576, -0.120, 0.193)       │ Nächster Punkt an Goal, aber miss   │
├──────┼──────────────────────────────┼───────────────────────────────────────┤
│ Cube │ (0.396, -0.215, 0.025)       │ NIEMALS ERREICHT                    │
│ Goal │ (0.586, -0.196, 0.069)       │ Step 10 nähert sich, aber zu hoch   │
└──────┴──────────────────────────────┴───────────────────────────────────────┘
```

**Kritische Beobachtungen:**
1. **Y-Koordinate chaotisch:** Springt zwischen -0.12 und +0.12 (±24 cm!)
2. **Z immer zu hoch:** Minimum 0.193m, Cube steht bei 0.025m — nie tief genug
3. **Kein Annäherungsverhalten:** Die Trajektorie zeigt kein zielgerichtetes Verhalten
4. **Rückwärtsbewegung:** Step 7 bewegt sich Richtung Startposition zurück

### 11.4 Diagnose: Warum das Modell versagt

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        URSACHENANALYSE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. LOSS-PLATEAU BEI 0.34                                                  │
│  ═══════════════════════                                                   │
│  Nach 30 CEM-Iterationen mit je 300 Samples (= 9000 Bewertungen)          │
│  konvergiert der Loss auf 0.34 und bewegt sich nicht weiter.               │
│  → Im gesamten Aktionsraum existiert KEINE Sequenz, die das Modell        │
│    von einer sinnvollen Cube-Manipulation überzeugt.                       │
│                                                                             │
│  2. MODELL-KAPAZITÄT UNZUREICHEND                                          │
│  ═════════════════════════════════                                         │
│  Training: 500 Episoden × 25 Frames, frameskip=2                           │
│  Action-Interval: 10 (alle 10 Sim-Steps ein Frame)                        │
│  → Zwischen 2 Frames passiert VIEL Bewegung                               │
│  → Modell kann die feingranulare Dynamik nicht lernen                      │
│  → Vorhersagen sind "verschwommen" — kein klarer Zielzustand              │
│                                                                             │
│  3. ZEITLICHE AUFLÖSUNG ZU GROB                                           │
│  ════════════════════════════════                                          │
│  ActInt10 + frameskip=2 = effektiv alle 20 Sim-Steps ein Datenpunkt        │
│  Bei ~150 Physics-Steps pro Episode:                                       │
│  → Nur ~7-8 effektive Zeitschritte pro Trajektorie                        │
│  → Zu wenig Information für das Modell, um feine Bewegungen zu lernen     │
│                                                                             │
│  4. ZU WENIG EPOCHEN                                                       │
│  ════════════════════                                                      │
│  50 Epochen bei 500 Episoden = begrenzte Konvergenz                        │
│  Paper-Referenz: Rope-Dataset trainiert mit mehr Iterationen               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.5 Konsequenz: Neues Training erforderlich

Der finale Test bestätigt definitiv: **Das aktuelle Modell (500 Episoden, ActInt10, 50 Epochen) kann keine sinnvollen Aktionssequenzen für Cube Manipulation produzieren.** Alle Code-Bugfixes (Proprio, Goal Image, OOM) waren korrekt und notwendig, aber das Modell selbst hat zu wenig gelernt.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENTSCHEIDUNG: RETRAINING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Status aller Bugfixes:                                                     │
│  ✅ OOM-Fix (ChunkedRolloutWrapper)                                        │
│  ✅ Proprio-Fix (ee_pos statt Nullen)                                      │
│  ✅ Goal-Image-Fix (Dataset-Bild statt Start-Bild)                         │
│  ✅ Goal-EEF-Fix (H5 eef_states statt aktuelle Position)                   │
│                                                                             │
│  Alle Fixes sind implementiert und verifiziert.                            │
│  Das Problem ist NICHT der Code, sondern das MODELL.                       │
│                                                                             │
│  → RETRAINING MIT BESSEREN DATEN UND PARAMETERN ERFORDERLICH              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.6 Neuer Datensatz und Trainingsplan

Basierend auf der Diagnose werden folgende Änderungen für das Retraining vorgenommen:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VERGLEICH: ALT vs. NEU                                   │
├─────────────────────────┬──────────────────┬────────────────────────────────┤
│ Parameter               │ Alt (500ep)      │ Neu (1000ep)                   │
├─────────────────────────┼──────────────────┼────────────────────────────────┤
│ Episoden                │ 500              │ 1000  (2× mehr)               │
│ Epochen                 │ 50               │ 100   (2× mehr)               │
│ ActionInterval          │ 10               │ 2     (5× feiner)            │
│ Frames/Episode          │ 25               │ 25    (gleich)                │
│ frameskip               │ 2                │ 2     (gleich)                │
│ action_dim              │ 6                │ 6     (gleich)                │
│ proprio_dim             │ 3                │ 3     (gleich)                │
│ Robot Opacity           │ 1.0              │ 1.0   (gleich)                │
│ img_size                │ 224              │ 224   (gleich)                │
├─────────────────────────┼──────────────────┼────────────────────────────────┤
│ Effektive Sim-Steps/    │ 10×2 = 20        │ 2×2 = 4                       │
│ Datenpunkt              │                  │                                │
│ Trainingssamples        │ ~9.900           │ ~19.800 (2× mehr Episoden)    │
│ Geschätzte Dauer        │ ~2h (RTX A5000)  │ ~8h (RTX A5000)              │
├─────────────────────────┼──────────────────┼────────────────────────────────┤
│ Datensatzpfad           │ primLogger_      │ primLogger_                    │
│                         │ NEps500_ActInt10 │ NEps1000_ActInt2               │
│                         │ _RobOpac10_      │ _RobOpac10_                    │
│                         │ NCams4_NCube1    │ NCams4_NCube1                  │
└─────────────────────────┴──────────────────┴────────────────────────────────┘
```

**Erwartete Verbesserungen:**

| Änderung | Wirkung |
|----------|---------|
| **1000 statt 500 Episoden** | Doppelt so viel Varianz → bessere Generalisierung |
| **100 statt 50 Epochen** | Mehr Trainingsiterationen → bessere Konvergenz |
| **ActInt2 statt ActInt10** | 5× feinere zeitliche Auflösung → Modell lernt feinere Dynamik |
| **ActInt2 + frameskip=2** | Effektiv alle 4 Sim-Steps ein Datenpunkt (statt 20) → 5× feinere Bewegungsinformation zwischen Frames |

**Trainingsbefehl:**
```bash
cd /home/tsp_jw/Desktop/dino_wm
conda activate dino_wm
FRANKA_DATA_PATH=/home/tsp_jw/Desktop/fcs_datasets/primLogger_NEps1000_ActInt2_RobOpac10_NCams4_NCube1 \
python train.py env=franka_cube_stack frameskip=2 training.epochs=100
```

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
| WM Sanity-Check | wm_sanity_check.py | Alle |
| Preprocessor | preprocessor.py | Normalisierung |
| FrankaCubeStackWrapper | env/franka_cube_stack/franka_cube_stack_wrapper.py | Alle |

---

*Dokumentation erstellt am 01.02.2026, aktualisiert am 09.02.2026 (Sektion 6.7: Strategische MPC-Entscheidung, Sektion 8.5: Startbefehl-Übersicht, Sektion 10: WM Sanity-Check mit Bodenplatten-Analyse), aktualisiert am 16.02.2026 (Sektion 12: Planning Server Bug-Analyse und Fixes)*

---

## 12. Planning Server Bug-Analyse und Fixes

> Datum: 16.02.2026. Systematische Analyse der Qualitätsunterschiede zwischen
> `plan.py` (Offline-Evaluation) und `planning_server.py` (Online Isaac Sim
> Steuerung). Identifizierte Bugs wurden direkt im Code behoben.

### 12.1 Hintergrund: plan.py vs. planning_server.py

| Eigenschaft | `plan.py` | `planning_server.py` |
|-------------|-----------|---------------------|
| **Zweck** | Offline Batch-Evaluation, Jobs, WandB | Persistenter TCP-Server für Isaac Sim |
| **Start** | `hydra.main()` / submitit | `argparse` + Socket-Loop |
| **Lebenszyklus** | Kurzlebig pro Run | Permanent, viele Requests |
| **Dataset** | Komplett geladen (für Targets + Eval) | Nur Statistiken (mean/std), dann freigegeben |
| **Env** | `SerialVectorEnv` mit `n_evals` gym-Envs | Keine Env (Client steuert Isaac Sim) |
| **Evaluator** | `PlanEvaluator` (Env-Rollout + Metriken) | `None` (Client evaluiert) |
| **Logging** | WandB + `logs.json` | `LoggingRun` auf stdout |
| **OOM-Schutz** | Keiner (GPU wird nicht geteilt) | `ChunkedRolloutWrapper` |

### 12.2 Erkenntnisse: Wie plan.py's CEM wirklich funktioniert

**Häufiges Missverständnis:** "Die `n_evals` parallelen Envs werden als Echtzeit-
Evaluation verwendet, um die beste Aktion unter den Rollouts zu wählen."

**Tatsächlicher Ablauf:**

Die `n_evals` Environments repräsentieren **verschiedene Init/Goal-Paare** (verschiedene
Szenarien), NICHT verschiedene Rollout-Kandidaten für dasselbe Szenario:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  WIE CEM IN plan.py WIRKLICH FUNKTIONIERT                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: CEM-Optimierung (rein im World Model, KEIN Env-Kontakt)          │
│  ─────────────────────────────────────────────────────────────────           │
│                                                                             │
│  for traj in range(n_evals):          # z.B. 5 versch. Szenarien           │
│      for i in range(opt_steps):        # z.B. 30 CEM-Iterationen           │
│          sample 300 Action-Kandidaten                                       │
│          → wm.rollout() im LATENT SPACE (nicht in der Env!)                │
│          → objective_fn() bewertet Distanz zum Ziel-Embedding              │
│          → topk=30 beste Kandidaten → neues mu/sigma                       │
│                                                                             │
│  Referenz: planning/cem.py Zeile 80-99                                     │
│                                                                             │
│  Die 300 Samples werden AUSSCHLIESSLICH durch das World Model              │
│  gerollt — die echte Env wird hier NIE berührt.                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 2: Zwischen-Evaluation (echte Env, nur Monitoring)                  │
│  ──────────────────────────────────────────────────────────                  │
│                                                                             │
│  if evaluator is not None and i % eval_every == 0:                         │
│      logs, successes = evaluator.eval_actions(mu, ...)                     │
│      if np.all(successes): break   # Early Termination                     │
│                                                                             │
│  Referenz: planning/cem.py Zeile 105-113                                   │
│                                                                             │
│  Hier werden die aktuellen besten Actions (mu) IN DER ECHTEN ENV           │
│  ausgeführt — aber das Ergebnis fliesst NICHT zurück in mu/sigma.          │
│  Es dient nur:                                                              │
│    1. Monitoring: Wie gut ist der aktuelle Plan in der echten Welt?         │
│    2. Early Termination: Wenn alle Szenarien erfolgreich → stoppen         │
│    3. Video-Erzeugung: Side-by-side WM-Prediction vs. Env-Realität        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 3: MPC (nur wenn MPCPlanner konfiguriert, nicht CEM direkt)         │
│  ─────────────────────────────────────────────────────────────────           │
│                                                                             │
│  Der MPCPlanner (planning/mpc.py) ist der EINZIGE Pfad, der echtes         │
│  Env-Feedback für Replanning nutzt. Er führt CEM aus, nimmt die            │
│  ersten n_taken_actions, führt sie in der Env aus und plant dann            │
│  vom neuen Zustand weiter.                                                  │
│                                                                             │
│  Der planning_server.py implementiert MPC manuell über den Socket-Loop:    │
│  Client sendet neues Bild → Server plant → Client führt aus → repeat       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Zusammenfassung der Rollen:**

| | CEM Auswahl | Env-Evaluation | Feedback in Planner? |
|---|---|---|---|
| **300 Samples** | WM-Rollout → Latent-Loss → topk | ❌ Nie in Env | — |
| **eval_every** | — | ✅ mu in Env ausführen | ❌ Nur Monitoring + Early Stop |
| **MPC** | CEM als Sub-Planner | ✅ Env-Rollout nach jedem MPC-Step | ✅ Neuer obs_0 für nächste CEM-Runde |

### 12.3 Bug-Katalog mit Fixes

#### 12.3.1 🔴 Bug 1: `model.eval()` wird nie aufgerufen

**Datei:** `planning_server.py` Zeile 107 (nach Fix)

**Problem:** Nach `load_model()` bleibt das Model im `train()`-Modus.
`VWorldModel.train()` (models/visual_world_model.py Zeile 78-86) aktiviert
Training-Modi für alle Sub-Module (Encoder, Predictor, Proprio/Action-Encoder).
Folgen:
- **Dropout-Layer** im Predictor erzeugen stochastische Ausgaben
- **Stochastische Regularisierung** verfälscht Ergebnisse
- Im persistenten Server akkumuliert sich die Stochastik über viele Requests

**Mögliche Lösungen:**
1. ✅ `model.eval()` einmal nach dem Laden aufrufen (1 Zeile)
2. ○ `torch.no_grad()` um jeden Planner-Call (existiert bereits, schützt aber nicht gegen Dropout)

**Gewählte Lösung:** Ansatz 1 — simpelste und nachhaltigste Lösung.

```python
model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
model.eval()  # WICHTIG: Eval-Modus fuer deterministische Inferenz
```

#### 12.3.2 🔴 Bug 2: Warm-Start füllt mit Nullen auf → Null-Bias

**Datei:** `planning_server.py` Zeile 374-380 (nach Fix)

**Problem (alt):**
```python
zero_tail = torch.zeros(1, 1, warm_start.shape[2], device=warm_start.device)
actions_init = torch.cat([shifted, zero_tail], dim=1)
```

Die letzte Action im Warm-Start war immer `[0, 0, ..., 0]`. Im normalisierten
Raum bedeutet Null: "bewege dich zum Mittelwert aller Trainingsaktionen". Der CEM
startet mit einem Plan, dessen letzte Aktion systematisch in Richtung Dataset-
Mittelwert verzerrt ist.

Bei wenigen opt_steps (z.B. 5 im Online-Modus) hat CEM zu wenig Iterationen,
um diesen Bias zu überwinden. Effekt: Roboter-Arm "driftet" in Richtung
Mittelposition des Workspace nach mehreren MPC-Schritten.

**Mögliche Lösungen:**
1. ✅ Letzte bekannte Action wiederholen statt Null (1 Zeile)
2. ○ Lineare Extrapolation der letzten 2 Actions
3. ○ Kein Warm-Start (jedes Mal von Null starten → schlechtere Konvergenz)

**Gewählte Lösung:** Ansatz 1 — physikalisch am sinnvollsten (Trägheitsannahme).

```python
last_action = warm_start[:, -1:, :]  # Letzte bekannte Action
actions_init = torch.cat([shifted, last_action], dim=1)
```

#### 12.3.3 🟡 Bug 3: `torch.cuda.empty_cache()` zwischen Chunks fragmentiert VRAM

**Datei:** `planning_server.py`, `ChunkedRolloutWrapper.rollout()` Zeile 168 (nach Fix)

**Problem (alt):**
```python
for start in range(0, B, self.chunk_size):
    ...
    z_obses, zs = self._model.rollout(chunk_obs, chunk_act)
    all_z_obses.append(z_obses)
    all_zs.append(zs)
    torch.cuda.empty_cache()  # ← ZWISCHEN Chunks!
```

`empty_cache()` gibt den CUDA-Cache frei, aber die akkumulierten Ergebnis-Tensoren
bleiben alloziert. Der nächste Chunk muss neuen Speicher anfordern → Fragmentierung.
Bei vielen Chunks kann das paradoxerweise zu MEHR OOM führen statt weniger.

**Mögliche Lösungen:**
1. ✅ `empty_cache()` nur einmal NACH der Schleife (2 Zeilen)
2. ○ `.detach().cpu()` für Zwischen-Ergebnisse (komplexer, mehr Code)
3. ○ Chunking komplett entfernen (riskant bei großen Batches)

**Gewählte Lösung:** Ansatz 1 — keine Fragmentierung, volle Kontrolle.

```python
for start in range(0, B, self.chunk_size):
    ...
    all_z_obses.append(z_obses)
    all_zs.append(zs)
# GPU-Cache erst NACH allen Chunks freigeben
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### 12.3.4 🔴 Bug 4: `evaluator=None` → Kein Early-Stopping im CEM

**Datei:** `planning_server.py` Zeile 278

**Problem:** Im Server ist `evaluator=None`. In `cem.py` Zeile 105-113 wird
der gesamte Evaluierungs- und Early-Termination-Block übersprungen:

```python
if self.evaluator is not None and i % self.eval_every == 0:
    logs, successes, _, _ = self.evaluator.eval_actions(...)
    if np.all(successes):
        break  # ← Existiert nicht im Server
```

**Auswirkung:** Kein Qualitätsproblem (Early Stop spart nur Zeit), aber der Server
optimiert immer für alle opt_steps, auch wenn der Loss bereits konvergiert ist.

**Mögliche Lösungen:**
1. ✅ Akzeptieren — ist im Server kein kritisches Problem, da Online-Modus
   ohnehin wenige Schritte nutzt. Bei Bedarf: Loss-Konvergenz-Check im
   `LoggingRun` hinzufügen (Eigenentwicklung, nicht aus plan.py übertragbar,
   da kein Env im Server vorhanden ist).
2. ○ Socket-basierter Evaluator (Server fragt Client nach Env-Feedback) —
   würde bidirektionale Kommunikation erfordern, hohe Komplexität
3. ○ Dummy-Evaluator der nur Loss-Konvergenz prüft

**Gewählte Lösung:** Ansatz 1 — bewusste Entscheidung: der CEM-Loss auf stdout
zeigt dem Nutzer bereits die Konvergenz. Ein automatischer Early-Stop wäre
Premature Optimization bei den aktuellen Online-Parametern (5-15 Steps).

#### 12.3.5 🟡 Bug 5: `__getattr__`-Fallback maskiert Fehler im ChunkedRolloutWrapper

**Datei:** `planning_server.py`, `ChunkedRolloutWrapper` Zeile 141-152 (nach Fix)

**Problem:** Jeder Attributzugriff, der nicht explizit gesetzt ist, wird stumm
an `self._model` delegiert. `nn.Module`-Methoden wie `to()`, `state_dict()`
werden durchgereicht, was dazu führen kann, dass z.B. `model.to('cpu')` den
Wrapper intakt lässt aber das innere Model verschiebt.

**Gewählte Lösung:** Explizite Forwarding-Methoden für kritische Operationen.

```python
def to(self, *args, **kwargs):
    self._model.to(*args, **kwargs)
    return self

def state_dict(self, *args, **kwargs):
    return self._model.state_dict(*args, **kwargs)
```

### 12.4 Verbleibende strukturelle Unterschiede

Diese Unterschiede sind **architekturbedingt** und können nicht durch einfache
Bugfixes behoben werden:

| Aspekt | `plan.py` | `planning_server.py` | Anmerkung |
|--------|-----------|---------------------|-----------|
| **Evaluator** | `PlanEvaluator` mit Env-Rollout | `None` | Server hat keine Env — Client-seitige MPC-Loop ersetzt dies |
| **MPC** | `MPCPlanner` mit Env-Feedback | Socket-Loop im Client | Funktional äquivalent, aber ohne server-seitige Action-Maskierung |
| **Multi-Eval** | `n_evals=5` parallele Szenarien | Immer `n_evals=1` | Ok für Online-Betrieb (1 Szenario = 1 Roboter) |
| **Dataset-Targets** | `prepare_targets()` aus Dset | Socket-basierte Goals | Architektureller Unterschied, kein Bug |
| **Reproduzierbarkeit** | `dump_targets()` + `logs.json` | Nur stdout | Ggf. JSON-Log hinzufügen |

### 12.5 Zusammenfassung der Änderungen

**Geänderte Datei:** `planning_server.py`

| Bug | Schwere | Fix | Zeilen geändert | Erwartete Auswirkung |
|-----|---------|-----|-----------------|---------------------|
| `model.eval()` fehlt | 🔴 Hoch | 1 Zeile hinzugefügt | +1 | Deterministische Inferenz |
| Warm-Start Null-Bias | 🔴 Hoch | `zero_tail` → `last_action` | ~3 | Kein Drift zum Dataset-Mittelwert |
| `empty_cache()` Fragmentierung | 🟡 Mittel | Nach statt in der Schleife | ~3 | Stabilere GPU-Nutzung |
| Evaluator fehlt | 🔴 Info | Bewusst akzeptiert | 0 | — (kein Fix nötig) |
| Wrapper-Forwarding | 🟡 Mittel | `to()` + `state_dict()` | +6 | Robustere Wrapper-Nutzung |

**Gesamt: ~13 Zeilen geändert, 0 neue Abhängigkeiten, 0 API-Änderungen.**

Die CEM-Parameter-Korrektur (num_samples/opt_steps/topk auf cem.yaml-Defaults)
wurde bereits separat durchgeführt und ist hier nicht erneut dokumentiert.
---

## 13. 🚨 KRITISCH: Temporale Alignment-Analyse — Action-Observation Mismatch (20.02.2026)

### 13.1 Zusammenfassung

Bei der Analyse der CEM-Divergenz wurde ein **fundamentaler Off-by-One-Fehler** in der zeitlichen Zuordnung von Actions und Observations im FCS-Datensatz identifiziert. Die Konvention im FCS-Datensatz (`primitive_data_logger.py`) weicht von der Referenz-Konvention (Rope/Deformable Environment) des DINO-WM Papers ab.

| Eigenschaft | Rope (Referenz) | FCS (aktuell) |
|-------------|-----------------|---------------|
| `obs[t]` zeigt | Zustand **VOR** `act[t]` | Zustand **NACH** `act[t]` |
| `act[t]` bedeutet | "Auszuführen VON `obs[t]`" | "Hat `obs[t]` PRODUZIERT" |
| Initiales Bild | ✓ (als `obs[0]`) | ❌ (fehlt) |
| Semantik | Vorwärtsblickend | Rückwärtsblickend |

### 13.2 Auswirkung auf CEM-Planning

#### Das Problem im Modell-Rollout

In `VWorldModel.rollout()` ([models/visual_world_model.py](models/visual_world_model.py#L261)):
```python
z = self.encode(obs_0, act_0)     # Historische obs + gepaarte Actions kodieren
while t < action.shape[1]:
    z_pred = self.predict(z[:, -num_hist:])
    z_new = z_pred[:, -1:, ...]
    z_new = self.replace_actions_from_z(z_new, action[:, t:t+1, :])  # CEM-Action einsetzen
    z = torch.cat([z, z_new], dim=1)
```

1. **CEM schlägt Action vor** als "was soll der Roboter **als nächstes tun**" → vorwärtsblickend
2. **Das Modell wurde trainiert** mit Actions die bedeuten "was hat diesen Zustand **produziert**" → rückwärtsblickend
3. **Semantischer Mismatch**: Das Modell interpretiert die CEM-Action anders als beabsichtigt

#### Konkretes Beispiel mit frameskip=2

**Rope (korrekt):**
```
Training-Window: obs[0] obs[2] obs[4] obs[6] obs[8] obs[10] obs[12]
Action-Groups:   (a0,a1) (a2,a3) (a4,a5) (a6,a7) (a8,a9) (a10,a11) (a12,a13)

act_group[0] = (a0, a1):
  a0: obs[0] → obs[1]     ← vorwärts VON obs[0]
  a1: obs[1] → obs[2]     ← vorwärts
  Kombiniert: obs[0] → obs[2] = obs_window[0] → obs_window[1] ✓
```

**FCS (fehlerhaft):**
```
Training-Window: obs[0] obs[2] obs[4] obs[6] obs[8] obs[10] obs[12]
Action-Groups:   (a0,a1) (a2,a3) (a4,a5) (a6,a7) (a8,a9) (a10,a11) (a12,a13)

act_group[0] = (a0, a1):
  a0: obs[-1] → obs[0]    ← RÜCKWÄRTS! (obs[-1] nicht im Window)
  a1: obs[0] → obs[1]     ← nur EIN Schritt vorwärts
  Kombiniert: obs[-1] → obs[1], NICHT obs[0] → obs[2] ❌
```

**Das Modell erhält Action-Groups die NICHT zur beobachteten Obs-Transition passen.** Die erste Action jeder Gruppe ist rückwärtsblickend (beschreibt die Vergangenheit), die zweite reicht nur einen Schritt statt zwei.

### 13.3 Zusammenhang mit CEM-Divergenz

Die CEM-Divergenz (EEF driftet zu unmöglichen Positionen wie x=1.385, y=-1.596) hat vermutlich **zwei Ursachen**:

1. **Fehlende Action Bounds** (bereits gefixt: Clamping auf [-3,3])
2. **Temporaler Mismatch** (dieser Bug): Das Modell kann die CEM-Actions nicht korrekt als Zustandstransitionen interpretieren, da es mit einer anderen semantischen Konvention trainiert wurde

### 13.4 Verifizierungsdaten

Aus dem Verifikationsskript:
```
Episode 0: 20 Bilder (obses.pth), 20 H5-Dateien — GLEICHE Anzahl (kein initiales Bild)

Timing-Check: action[t].start_pos ≈ eef_states[t-1][:3]
  t=1: start=[0.489,0.090,0.417] vs eef[0]=[0.485,0.085,0.418] → d=0.007 OK
  t=2: start=[0.524,0.182,0.366] vs eef[1]=[0.521,0.171,0.374] → d=0.014 OK
  t=5: start=[0.589,0.110,0.187] vs eef[4]=[0.585,0.114,0.188] → d=0.005 OK
```

**Bestätigt:** `action[t].start_pos ≈ eef[t-1]` = Action t startet wo Action t-1 endete → `act[t]` transitiert `obs[t-1] → obs[t]`.

### 13.5 Implementierter Fix und Workflow (21.02.2026)

#### Schritt 1: Data Logger gefixt

**`primitive_data_logger.py`** → `_save_primitive_h5()`: Bild/EEF/Würfelpositionen werden jetzt vom **START** des Primitivs gespeichert (statt vom Ende). Damit ist `obs[t]` = Zustand **VOR** `action[t]` — identisch mit der Rope-Referenz.

**`min_data_logger.py`** → Buffer-Ansatz: Observation wird gepuffert und erst beim nächsten Step zusammen mit der Forward-Action `[buffered_pos → curr_pos]` als H5 gespeichert. Der letzte Buffer wird in `end_episode()` mit Dummy-Action gespeichert. Neue Hilfsmethoden: `_save_step_h5()`, `_flush_buffer_final()`.

**Kein Datenverlust** — weiterhin T Bilder + T Actions pro Episode bei beiden Loggern.

#### Schritt 1b: Anwendungscode angepasst

**`fcs_main_parallel.py`**: Keine Code-Änderungen nötig (log_step()-API unverändert). Dokumentation aktualisiert in `collect_timestep_data()`, `save_successful_episode()` und Hauptschleife.

**`planning_client.py`**: Keine Code-Änderungen nötig (PlanningLogger = separater, simpler Logger). Dokumentation aktualisiert in `PlanningLogger`-Docstring und `log_step_if_active()`.

#### Schritt 2: Datensatz NEU generieren

Der bestehende Datensatz wurde mit der alten Konvention (END-Bild) generiert und ist inkompatibel.

#### Schritt 3: Modell NEU trainieren

Das aktuell trainierte Modell (260218/11-58) hat die falsche Konvention gelernt. **Neutraining erforderlich.**

#### Schritt 4: CEM-Planning mit neuem Modell

Nach dem Neutraining interpretiert das Modell CEM-Actions korrekt als vorwärtsblickende Transitionen.

### 13.6 Querverweise

- **Training-Dokumentation**: Detaillierte Code-Analyse und Fix-Implementierung → siehe Abschnitt "KRITISCH: Action-Observation Temporale Alignment-Analyse"
- **CEM Fixes**: Action Bounds, Gripper-Quantisierung, Sigma Floor → siehe Abschnitt 10

---

## 14. Action-Dim Mismatch: Model vs. Dataset (04.03.2026)

### 14.1 Problem

Beim Start des Planning Servers mit einem älteren Model (z.B. `260216/23-42`, trainiert mit `action_dim=6`)
crashte der CEM-Rollout mit:

```
RuntimeError: Given groups=1, weight of size [10, 12, 1], expected input[39, 16, 1]
to have 12 channels, but got 16 channels instead
```

### 14.2 Root Cause

1. Das Model wurde mit `action_dim=6` trainiert (6D Actions ohne Gripper-State, aus `hydra.yaml`)
2. Das zugehörige Dataset im Archiv (`00_Archiv/NEps1000_RobOpac0_NPrim20_NCams4_NCube1`) wurde
   **nach dem Training** mit 8D Actions (Gripper-Tracking) regeneriert
3. Der Planning Server las `base_action_dim = _full_dset.action_dim` direkt vom Dataset → erhielt 8
4. `full_action_dim = 8 × frameskip(2) = 16` → CEM erzeugte 16D Actions
5. Das Model's Action-Encoder (`Conv1d(12, 10)`) erwartet aber `6 × 2 = 12` Kanäle → Crash

**Kernproblem:** Der Server vertraute der Dataset-Action-Dim statt der Model-Config. Wenn sich
das Dataset nachträglich ändert (z.B. durch Regenerierung), divergieren die Dimensionen.

### 14.3 Fix (planning_server.py)

1. **Autoritative Dimension:** `model_cfg.env.action_dim` aus `hydra.yaml` bestimmt `base_action_dim`,
   nicht die Dataset-Klasse
2. **Stats-Adaption:** Wenn Dataset 8D hat aber Model 6D erwartet, werden die Normalisierungs-Stats
   angepasst (Gripper-Dims an Index 3,7 entfernt)
3. **Workspace Bounds:** 8D-Bounds aus `plan_franka.yaml` werden automatisch auf 6D reduziert
   (Gripper-Dims entfernt)
4. **Gripper-Config:** Bei 6D-Modellen wird die Gripper-Konfiguration übersprungen (kein Gripper
   in der Action)
5. **Diagnostik-Print:** `target_ee` Index passt sich an (8D: `[4:7]`, 6D: `[3:6]`)

### 14.4 Empfehlung

- **Normalisierungs-Stats im Checkpoint speichern** (langfristiger Fix): Dann ist der Server
  unabhängig vom aktuellen Dataset-Zustand
- **Datasets nicht nachträglich regenerieren**, wenn trainierte Models darauf basieren
- **Immer `model_cfg.env.action_dim`** als Ground Truth für die Dimension verwenden

---

## 15. Two-Phase Planning: EEF kommt in Phase 1 nicht tief genug — Root-Cause-Analyse (05.03.2026)

> **Datum:** 05.03.2026  
> **Kontext:** Beim Two-Phase-Planning (Phase 1 = Approach/Grasp, Phase 2 = Transport/Place) wurde
> wiederholt beobachtet, dass der End-Effektor in Phase 1 nicht tief genug kommt, um den Würfel zu
> greifen. In Phase 2 erreicht der EEF dagegen zuverlässig die korrekte z-Tiefe. Diese Analyse
> identifiziert drei zusammenwirkende Ursachen.

### 15.1 Beobachtung

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              BEOBACHTETES VERHALTEN (REPRODUZIERBAR)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1 (P1-GRASP): CEM plant Anfahrt zum Würfel                         │
│  ─────────────────────────────────────────────────                          │
│  - EEF startet bei z ≈ 0.42 (Franka Ruhepose)                             │
│  - Goal-Proprio: z ≈ 0.05–0.07 (Tischniveau, aus Dataset)                 │
│  - ERGEBNIS: EEF bleibt bei z ≈ 0.10–0.15 hängen                          │
│    → Gripper schließt sich UM den Würfel herum = VERFEHLT                  │
│    → Zu hoch, Fingerspitzen greifen nicht um den Würfel                    │
│                                                                             │
│  GRIPPER-CLOSE (deterministisch, 15 settle steps)                          │
│                                                                             │
│  PHASE 2 (P2-PLACE): CEM plant Transport zum Ziel                         │
│  ─────────────────────────────────────────────────                          │
│  - EEF startet bei z ≈ 0.10–0.15 (wo Phase 1 endete)                      │
│  - Goal-Proprio: z ≈ 0.05–0.07 (Ablageposition, aus Dataset)              │
│  - ERGEBNIS: EEF erreicht z ≈ 0.05–0.07 zuverlässig ✓                     │
│    → Korrekte Tiefe, aber Würfel wurde in Phase 1 nicht gegriffen          │
│                                                                             │
│  PARADOX: Phase 2 kann die Tiefe, Phase 1 nicht — obwohl beide            │
│  den gleichen CEM, das gleiche World Model und ähnliche Goals nutzen.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Schlüsselfrage:** Warum funktioniert die z-Konvergenz in Phase 2, aber nicht in Phase 1?

### 15.2 Ursache 1: OOD-Proprio durch hohe Startposition (Hauptursache)

Die **Startposition des EEF** ist der entscheidende Unterschied zwischen beiden Phasen.

**Phase 1** startet nach `env.reset()` von der **Franka-Ruhepose** mit z ≈ 0.42m, also ca. 40cm
über dem Tisch. Das World Model wurde aber auf **Manipulationsdaten** trainiert, in denen der
EEF sich ausschließlich nahe der Tischoberfläche bewegt (z ∈ [0.03, 0.15]).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PROPRIO-NORMALISIERUNG: IN-DISTRIBUTION VS. OOD                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Proprio-Statistiken aus dem Trainings-Dataset:                             │
│    proprio_mean_z ≈ 0.050                                                   │
│    proprio_std_z  ≈ 0.070                                                   │
│                                                                             │
│  PHASE 1 START: z = 0.42 (Ruhepose)                                        │
│    proprio_z_norm = (0.42 - 0.050) / 0.070 = +5.3σ                         │
│    → EXTREMER AUSREISSER! 5.3 Standardabweichungen über dem Mittelwert     │
│    → Das World Model hat NIEMALS einen Proprio-Wert in dieser Größen-      │
│      ordnung während des Trainings gesehen                                  │
│    → Predictions für z-Dynamik sind unzuverlässig (Extrapolation)          │
│                                                                             │
│  PHASE 2 START: z ≈ 0.10–0.15 (wo Phase 1 endete)                         │
│    proprio_z_norm = (0.12 - 0.050) / 0.070 = +1.0σ                         │
│    → NORMAL! Innerhalb der Trainingsdaten-Verteilung                       │
│    → World Model kann z-Dynamik gut prädizieren                            │
│                                                                             │
│                                                                             │
│  Verteilung der Trainings-Proprio (z-Komponente):                          │
│                                                                             │
│     ╎                     ████                                              │
│     ╎                   ████████                                            │
│     ╎                 ████████████                                          │
│     ╎               ████████████████                                        │
│     ╎             ████████████████████                                      │
│     ╎           ████████████████████████                                    │
│     ╎         ███████████████████████████                                   │
│     ╎───────████████████████████████████───────────────────────────         │
│     0.00   0.05   0.10   0.15   0.20          ...          0.42            │
│     ▲       ▲                                                ▲             │
│     │       │                                                │             │
│     │    Trainings-                                     Phase-1-Start      │
│     │    Bereich                                        (5.3σ OOD!)        │
│     │                                                                      │
│  Phase-2-Start (1.0σ, in-distribution)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Auswirkung auf die Objective-Funktion:**

Die Objective-Funktion ist `loss = MSE_visual + 0.5 × MSE_proprio` (siehe `planning/objectives.py`,
`alpha=0.5`, `mode="last"`). Der `MSE_proprio`-Term treibt den CEM dazu, Actions zu finden, deren
**prädizierte Proprio** (EEF-Position am Horizont-Ende) nahe am Goal-Proprio liegt. Wenn aber die
WM-Predictions bei OOD-Proprio unzuverlässig sind, liefert der Proprio-Term **irreführende
Gradienten** — der CEM optimiert auf Basis falscher Zustandsvorhersagen.

### 15.3 Ursache 2: Große z-Distanz überfordert den CEM-Horizont

Der CEM plant mit **Horizon H=5** Actions. Jede Action spezifiziert eine **absolute
EEF-Zielposition** (nicht eine Verschiebung relativ zur aktuellen Position). Das heißt, die
z-Komponente jeder Action liegt im Bereich `[0.00, 0.12]` (Workspace Bounds).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           CEM-AKTIONEN: ABSOLUTE POSITIONEN VS. PHYSISCHE DISTANZ           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Die CEM-Actions sind absolute EEF-Zielpositionen:                         │
│    action[4:7] = [target_x, target_y, target_z]                            │
│                                                                             │
│  Workspace Bounds: z ∈ [0.00, 0.12]                                        │
│  → Jede geplante Action zielt auf z ∈ [0.00, 0.12]                         │
│  → Das ist KORREKT — der Tisch ist bei z ≈ 0.04, Würfel-Höhe bei z ≈ 0.07│
│                                                                             │
│  ABER: Das RMPFlow-IK braucht PHYSISCHEN WEG, um dort hinzukommen!         │
│                                                                             │
│  Phase 1: EEF bei z=0.42 → Target z=0.05                                  │
│    → IK muss 0.37m Weg zurücklegen                                         │
│    → Bei settle_steps=20 und threshold=0.005 konvergiert IK                │
│      möglicherweise NICHT in 20 Steps auf so große Distanzen               │
│    → move_ee_to() gibt converged=False zurück → dist bleibt > 0            │
│    → Der nächste MPC-Step plant mit dem REALEN z (immer noch zu hoch)      │
│    → Erst nach mehreren MPC-Steps ist der EEF nahe genug am Target         │
│                                                                             │
│  Phase 2: EEF bei z=0.12 → Target z=0.05                                  │
│    → IK muss nur 0.07m Weg zurücklegen                                     │
│    → Konvergiert in 5-10 Steps → EEF ist sofort am Target                  │
│    → Nächster MPC-Step plant mit korrektem z                               │
│                                                                             │
│  Konsequenz: In Phase 1 hinkt die REALE EEF-Position mehrere MPC-Steps    │
│  hinter der GEPLANTEN Position her. Das World Model sieht ein Bild,        │
│  in dem der EEF noch hoch ist, und muss erneut "nach unten" planen.        │
│  Die 30 MPC-Steps reichen aus, um den EEF schrittweise herunter-          │
│  zubringen — aber nicht immer bis ganz auf Greif-Tiefe (z ≈ 0.05).        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Zusammenhang mit Ursache 1:** Die IK-Konvergenz-Problematik verstärkt die OOD-Problematik.
Selbst wenn der CEM in Iteration 1 korrekt `z=0.05` plant, braucht der physische Roboter
mehrere MPC-Steps, um dort anzukommen. In der Zwischenzeit beobachtet das World Model
eine immer noch hohe z-Position → OOD-Input → schlechte Predictions → suboptimale nächste
Planung → Teufelskreis.

### 15.4 Ursache 3: Workspace-Bounds und Sigma-Asymmetrie

Die CEM-Exploration in der z-Dimension wird durch **Workspace-Bounds** begrenzt und ist
**asymmetrisch um den Mittelwert** verteilt:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           CEM Z-EXPLORATION: BOUNDS VS. SIGMA                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Action-Statistiken (z-Dimension):                                          │
│    action_mean_z = 0.050                                                    │
│    action_std_z  = 0.070                                                    │
│                                                                             │
│  CEM-Initialisierung (normalisierter Raum):                                 │
│    mu_z = 0.0       (= action_mean = 0.050 in Weltkoord.)                  │
│    sigma_z = 2.0    (sigma_scale_z = 2.0 aus plan_franka.yaml)             │
│                                                                             │
│  Workspace Bounds (z-Dimension, Weltkoord.):                               │
│    z_lower = 0.00   → norm: (0.00 - 0.050) / 0.070 = -0.714              │
│    z_upper = 0.12   → norm: (0.12 - 0.050) / 0.070 = +1.000              │
│                                                                             │
│  Gültige Bandbreite im normalisierten Raum: [-0.714, +1.000] = 1.714σ     │
│  Initiale Exploration (sigma_z = 2.0): ±2.0σ                              │
│                                                                             │
│  → ~60% der initialen CEM-Samples werden durch Bounds geclampt!            │
│    Samples < -0.714 → geclampt auf -0.714 (z=0.00)                         │
│    Samples > +1.000 → geclampt auf +1.000 (z=0.12)                         │
│                                                                             │
│  Das ist an sich kein Problem (die Bounds SOLLEN den Suchraum              │
│  einschränken), aber es bedeutet:                                          │
│                                                                             │
│  Die CEM-Verteilung nach Clamping ist FLACHER als eine Gauss-Verteilung    │
│  → mehr Samples bei z=0.00 und z=0.12 (an den Rändern)                    │
│  → weniger Samples bei z=0.05 (am Optimum, Würfel-Höhe)                   │
│  → TopK-Selektion muss aus dieser verzerrten Verteilung wählen            │
│                                                                             │
│  Bei Phase 2 (Start z=0.12, Goal z=0.05): Die Verzerrung ist gering,      │
│  weil der aktuelle Zustand nah am Goal ist. Das WM kann gut prädizieren,   │
│  welche z-Werte zur Goal-Proprio passen → der TopK-Filter findet schnell   │
│  die richtigen Samples.                                                     │
│                                                                             │
│  Bei Phase 1 (Start z=0.42, Goal z=0.05): Das WM prädiziert schlecht von   │
│  OOD-Proprio aus → ALLE z-Werte bekommen ähnliche (hohe) Losses            │
│  → TopK wählt quasi zufällig → mu konvergiert langsam/gar nicht richtig    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15.5 Zusammenfassung: Phase 1 vs. Phase 2 im Vergleich

| Faktor | Phase 1 (P1-GRASP) | Phase 2 (P2-PLACE) |
|--------|-------------------|-------------------|
| **Start-z** | ~0.42m (Franka-Ruhepose) | ~0.05–0.12m (Ende von Phase 1) |
| **Goal-z** | ~0.05–0.07m (Greif-Position) | ~0.05–0.07m (Ablage-Position) |
| **z-Distanz zum Goal** | ~0.35m (groß) | ~0.00–0.07m (klein) |
| **Proprio normalisiert** | +5.3σ (extremer OOD-Ausreißer) | ~0–1σ (in-distribution) |
| **WM-Prediction-Qualität** | Schlecht (Extrapolation von OOD) | Gut (Interpolation in ID) |
| **IK-Konvergenz** | Langsam (0.37m Weg pro Step) | Schnell (0.07m Weg pro Step) |
| **CEM-Konvergenz in z** | Langsam, oft unvollständig | Schnell, zuverlässig |
| **Ergebnis** | EEF bleibt bei z ≈ 0.10–0.15 hängen | EEF erreicht z ≈ 0.05–0.07 ✓ |

**Ursachenkette Phase 1:**

```
Ruhepose (z=0.42)
  → Proprio 5.3σ OOD
    → WM-Predictions unzuverlässig
      → CEM-Objective irreführend (visual + proprio MSE basiert auf falschen Predictions)
        → CEM konvergiert langsam in z
          → Geplantes z-Target wird vom IK nur langsam angefahren (große Distanz)
            → Nächster MPC-Step: z immer noch zu hoch → Teufelskreis
              → Nach 30 MPC-Steps: z ≈ 0.10–0.15, nicht tief genug zum Greifen
```

### 15.6 Lösungsansätze

#### Ansatz A: Pre-Phase — EEF vor Phase 1 auf niedrige z-Position fahren

Statt Phase 1 von der Ruhepose (z=0.42) starten zu lassen, könnte eine **deterministische
Pre-Phase** den EEF auf eine Startposition nahe des Tisches fahren (z.B. z ≈ 0.15):

```python
# planning_client.py — vor Phase 1:
# EEF deterministisch auf eine Position oberhalb des Arbeitsbereichs fahren
pre_phase_target = [current_x, current_y, 0.15]  # nur z ändern
env.move_ee_to(pre_phase_target, target_orientation=EE_DEFAULT_ORIENT,
               max_steps=50, threshold=0.005)
```

- **Vorteil:** Kein WM-/CEM-Änderung nötig, rein client-seitig, sofort umsetzbar
- **Nachteil:** Pre-Phase-Position muss gewählt werden; der Würfel ist aber noch nicht lokalisiert
  → x/y-Position des EEF passt evtl. nicht zum Würfel

#### Ansatz B: Datensammlung mit hoher Startposition (z=0.42 inkludieren)

Den Datensammlungs-Controller so anpassen, dass er **von der Ruhepose aus** startet (nicht nur
nahe Tischoberfläche). Dann wäre z=0.42 in-distribution für das World Model:

- **Vorteil:** WM lernt die vollständige Bewegungsdynamik von oben nach unten
- **Nachteil:** Deutlich aufwändigerer Fix (Datensammlung + Retraining nötig),
  und die meisten Trainingsdaten wären "Abstieg"-Bewegungen, die nicht task-relevant sind

#### Empfehlung

**Ansatz A (Pre-Phase)** ist kurzfristig am vielversprechendsten: Schnell implementierbar,
keine Änderung am World Model oder Training nötig, und bringt den EEF in den ID-Bereich
des World Models, bevor die CEM-Planung beginnt.

**Querverweise:**
- Abschnitt 5: CEM Planner im Detail (Sigma-Initialisierung, Bounds)
- Abschnitt 6.7.5: Warm-Start im MPC-Kontext
- Abschnitt 11: Finaler Offline-Test (WM-Qualitätsprobleme)
- `plan_franka.yaml`: Workspace Bounds, Sigma-Scale
- `planning_client.py`: `_run_mpc_phase()`, Two-Phase-Logik
- **Datensatz-Verifikation**: Actions, Proprio, RGB-Check → siehe Abschnitt 11