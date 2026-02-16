# Commit Message — DINO WM Repository

```
docs: Finaler Offline-Test dokumentiert — Retraining mit 1000ep/100epochs/ActInt2

ERGEBNIS DES FINALEN OFFLINE-TESTS (14.02.2026):
  Modell 2026-02-09/17-59-59 (500 Episoden, ActInt10, 50 Epochen)
  CEM: H=5, samples=300, steps=30, topk=30 (maximale Qualität)

  CEM-Loss:  0.828 → 0.340 (58.9% Reduktion, Plateau ab Step 15)
  
  Trajektorie: 10 Aktionen ausgeführt — KEINE nähert sich dem Cube
    - Cube bei (0.396, -0.215, 0.025) — niemals erreicht
    - Y-Koordinate chaotisch (±24 cm Sprünge)
    - Z immer zu hoch (min 0.193m, Cube bei 0.025m)
    - Kein zielgerichtetes Verhalten

DIAGNOSE:
  Alle Code-Bugfixes (Proprio, Goal Image, OOM) korrekt implementiert.
  Problem ist das MODELL selbst:
  - ActInt10 + frameskip=2 = alle 20 Sim-Steps ein Datenpunkt (zu grob)
  - 50 Epochen = unzureichende Konvergenz
  - 500 Episoden = zu wenig Varianz
  - Loss-Plateau bei 0.34 = keine sinnvolle Trajektorie im Suchraum

RETRAINING-PLAN:
  Alt:  500 Episoden, ActInt10, 50 Epochen → alle 20 Sim-Steps
  Neu: 1000 Episoden, ActInt2, 100 Epochen → alle  4 Sim-Steps

  Neuer Datensatz: primLogger_NEps1000_ActInt2_RobOpac10_NCams4_NCube1
  Datensammlung läuft (gestartet 14.02.2026)

Dokumentation:
  DINO_WM_PLANNING_DOCUMENTATION.md:
    - Neue Sektion 11: Finaler Offline-Test (6 Unterabschnitte)
    - Testaufbau, CEM-Konvergenz, Trajektorie, Diagnose, Trainingsplan

  Konfiguration:
    - conf/train.yaml: epochs 50→100 (für neues Training)
    - conf/env/franka_cube_stack.yaml: data_path auf neuen Datensatz
```
