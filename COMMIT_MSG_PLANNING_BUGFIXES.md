# Commit Message — Planning Server Bugfixes

```
fix(planning_server): 5 Bugs behoben + Dataset-Loading optimiert + Dokumentation

KONTEXT:
  planning_server.py produzierte qualitativ schlechtere Roboter-Bewegungen
  als plan.py (Offline-Evaluation). Systematische Code-Analyse beider Pfade
  identifizierte 5 Bugs und eine Ineffizienz im Dataset-Loading.

BUGFIXES IN planning_server.py:
  1. model.eval() nach load_model() hinzugefügt (Zeile 130)
     - Model blieb im train()-Modus → Dropout aktiv bei Inferenz
     - Folge: Stochastische, nicht-reproduzierbare Predictions
     - Fix: 1 Zeile → deterministische Inferenz

  2. Warm-Start Null-Bias behoben (Zeile 393-399)
     - Alt: Letzte Action im shifted Plan = torch.zeros() 
       → Im z-normalisierten Raum = "bewege zum Dataset-Mittelwert"
       → Systematischer Drift zur Workspace-Mitte nach mehreren MPC-Steps
     - Neu: warm_start[:, -1:, :] (letzte bekannte Action wiederholen)
       → Physikalisch sinnvolle Trägheitsannahme

  3. CUDA Cache-Fragmentierung behoben (Zeile 187-190)
     - Alt: torch.cuda.empty_cache() INNERHALB der Chunk-Schleife
       → Fragmentiert VRAM, paradoxerweise mehr OOM-Risiko
     - Neu: empty_cache() nur einmal NACH der Schleife

  4. evaluator=None bewusst akzeptiert (kein Code-Change)
     - Server hat keine Env → kein Early-Stop via Evaluator
     - Client-seitige MPC-Loop übernimmt diese Rolle
     - Dokumentiert als architekturelle Entscheidung

  5. ChunkedRolloutWrapper: explizite to()/state_dict() (Zeile 169-175)
     - Alt: Nur __getattr__-Fallback → maskierte Fehler bei to('cpu') etc.
     - Neu: Explizite Forwarding-Methoden für kritische nn.Module-Operationen

DATASET-LOADING OPTIMIERUNG in planning_server.py:
  - Alt: hydra.utils.call(model_cfg.env.dataset) lud ALLE 1000 Episoden
    MIT Bildern in RAM (mehrere GB, ~60s Startzeit) — nur um 6 Statistik-
    Tensoren (action/state/proprio mean+std) zu extrahieren
  - Neu: FrankaCubeStackDataset(preload_images=False) direkt instanziiert
    → Identische Statistiken, ~10s Startzeit, minimaler RAM-Verbrauch
  - H5-Dateien (Actions, EEF-States, ~KB/Episode) werden weiterhin geladen,
    damit mean/std über alle Episoden korrekt berechnet werden

CEM-PARAMETER DEFAULTS:
  - argparse-Defaults auf Paper-Werte gesetzt statt None:
    num_samples=300, opt_steps=30, topk=30, goal_H=5, chunk_size=0
  - Vorher: None-Defaults erforderten immer CLI-Argumente oder
    fielen auf reduzierte Online-Werte (64/5/10) zurück

DOKUMENTATION:
  DINO_WM_PLANNING_DOCUMENTATION.md:
    - Neue Sektion 12: "Planning Server Bug-Analyse und Fixes"
    - 12.1: Architektur-Vergleich plan.py vs. planning_server.py (Tabelle)
    - 12.2: Wie CEM wirklich funktioniert (n_evals ≠ Eval-Kandidaten,
            300 Samples NUR im Latent Space, Env nur für Monitoring)
    - 12.3: Bug-Katalog mit je: Problem, Lösungsalternativen, gewählter Fix
    - 12.4: Verbleibende strukturelle Unterschiede (architekturbedingt)
    - 12.5: Zusammenfassungstabelle aller Änderungen
    - Inhaltsverzeichnis aktualisiert

  DINO_WM_PLANNING_PIPELINE.md:
    - Neue Sektion: "BUGFIXES IM PLANNING SERVER (16.02.2026)"
    - ASCII-Diagramme: Fix-Übersicht, Warm-Start Alt vs. Neu,
      CEM-Missverständnis-Korrektur, Architektur-Unterschiede

WEITERE GEÄNDERTE DATEIEN (aus vorherigem Commit-Scope):
  conf/env/franka_cube_stack.yaml → conf/env/fcs.yaml:
    - Umbenannt (kürzerer Name für Hydra-Config)
    - data_path auf NEps500_RobOpac10_NPrim10_NCams4_NCube1 aktualisiert

  conf/train.yaml:
    - Hydra run/sweep dir: %Y-%m-%d/%H-%M-%S → %y%m%d/%H-%M (kürzere Pfade)
    - SLURM gres: gpu:h100:1 → gpu:A5000:1 (lokale Hardware)
    - epochs: bereits auf 100 (aus Retraining-Commit)
    - frameskip, num_hist: Defaults in YAML statt CLI-Overrides

  conf/plan_franka.yaml:
    - Kommentar mit Modell-Referenz hinzugefügt (whitespace-only sonst)

  datasets/franka_cube_stack_dset.py:
    - Action-Format Dokumentation vereinfacht: nur noch ee_pos (6D)
    - Entfernt: delta_pose/velocity Modi (nie verwendet in unserer Pipeline)
    - Entfernt: H5 info/action_mode Auto-Detection (unnötige Komplexität)
    - action_mode hardcoded auf "ee_pos"

  COMMIT_MSG_RETRAINING.md:
    - Whitespace-only Change (Leerzeile)

Gesamt: ~13 Zeilen Code geändert in planning_server.py,
        0 neue Abhängigkeiten, 0 API-Änderungen.
```
