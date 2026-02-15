# Commit Message

```
feat: Hyperparameter-Analyse mit VRAM-Modell, Validierungs-Lastspitze
      und Episodenlängen-Sweep (T=10–50) für Masterarbeit

────────────────────────────────────────────────────────────────────────
KONTEXT
────────────────────────────────────────────────────────────────────────

Für die Masterarbeit wird eine wissenschaftlich fundierte Begründung
der Hyperparameter-Wahl benötigt. Die zentrale Frage:

  "Warum batch_size=4, num_hist=6, frameskip=2?"

Die Antwort erfordert ein kalibriertes VRAM-Modell, das zeigt, dass
diese Konfiguration das Maximum an temporalem Kontext (num_hist)
ausschöpft, das die GPU (A5000, 24564 MiB) tragen kann — inklusive
der Validierungs-Lastspitze, die mehr VRAM verbraucht als das Training.

────────────────────────────────────────────────────────────────────────
NEUE DATEIEN
────────────────────────────────────────────────────────────────────────

hyperparameter_analysis.py  (1881 Zeilen)
══════════════════════════════════════════

  Vollständiges Analyse-Script mit empirisch kalibriertem VRAM-Modell,
  Optimierungssolver und 10 Plot-Generatoren für die Masterarbeit.

  Kernkomponenten:

  1. VRAM-Schätzung (estimate_vram_mib)
     - Analytisches Modell: Feste Kosten (559 MiB) + Aktivierungen
       + CUDA Overhead (2000 MiB)
     - Aktivierungen zerlegt in: DINOv2 Encoder (linear in B×num_frames),
       ViT Predictor Attention (QUADRATISCH in num_hist×196),
       VQVAE Decoder (linear), Misc (Loss, Tiling)
     - Empirische Kalibrierung: Faktor 5.51× auf theoretische Werte
       Referenz: bs=4, nh=6, fs=2 → 16467 MiB (gemessen)
       Kreuzvalidierung: bs=8, nh=3 → 59.4% (train.yaml: "~60%") ✅

  2. Validierungs-Lastspitze (estimate_vram_validation_peak_mib)
     - Entdeckung: val() in train.py verbraucht ~17% MEHR VRAM
       als Training, weil:
       (a) model(obs, act) OHNE torch.no_grad() → voller Graph
       (b) openloop_rollout() fragmentiert CUDA-Allocator
       (c) Kein torch.cuda.empty_cache() dazwischen
     - Formel: (F + C + A_val + R_rollout + P_plot) × 1.12
     - Konsequenz: num_hist=7 passt ins Training (88.7%), aber
       VAL OOM bei 103.7%!

  3. Optimierungssolver (find_optimal_config)
     - Greedy: num_hist absteigend, dann frameskip aufsteigend,
       dann batch_size absteigend
     - Harte Grenze: Val Peak ≤ VRAM_TOTAL (nicht Training VRAM!)
     - Ergebnis: bs=4, nh=6, fs=2 = optimale Konfiguration

  4. Parameter-Sweep (compute_sweep)
     - Vollständiger Grid-Sweep über alle Parameter-Kombinationen
     - Score-Funktion: num_hist×1000 + Effizienz×100
     - Export als CSV

  5. Plot-Generatoren (10 Typen, 17 Dateien)
     - Plot 01: Feasibility Heatmap (T=22, T=25)
     - Plot 02: VRAM vs batch_size × num_hist mit Val Peak
     - Plot 03: Samples-Effizienz (Slices + Steps)
     - Plot 04: Optimal Frontier (Pareto-Front)
     - Plot 05: VRAM Breakdown (gestapelt mit Val-Overhead)
     - Plot 06: Attention Scaling (quadratisch, O(n²))
     - Plot 07: Paper-Vergleich (Zhou et al. vs. unsere Konfig)
     - Plot 08: Sweep-Tabelle mit Status-Codes
     - Plot 09: Validierungs-Peak (3 Subplots vertikal)
     - Plot 10: Episodenlängen-Sweep T=10–50 (2×2 Grid)   ← NEU

hyperparameter_analysis/  (33 Dateien)
═══════════════════════════════════════

  Generierte Outputs:
  - 15 Plots × (PDF + PNG) = 30 Bilddateien
  - 1 CSV: parameter_sweep.csv (vollständiger Grid)
  - 2 ältere CSV-Dateien

────────────────────────────────────────────────────────────────────────
PLOT 10: EPISODENLÄNGEN-SWEEP (T=10–50)     ← HAUPTÄNDERUNG
────────────────────────────────────────────────────────────────────────

  Neuer Plot: Wie beeinflusst die Episodenlänge T die Parameterwahl?

  Layout: 2×2 Grid (14×11 Zoll)

  ┌─────────────────────────┬──────────────────────────────┐
  │ Oben-links:             │ Oben-rechts:                 │
  │ Max num_hist pro T      │ Slices pro Episode über T    │
  │ für fs=1..5             │ für 5 Konfigurationen        │
  │ (Rollout + VRAM Limit)  │ (H=6/fs=2 hervorgehoben)    │
  ├─────────────────────────┼──────────────────────────────┤
  │ Unten-links:            │ Unten-rechts:                │
  │ Gesamte Train-Samples   │ Heatmap: T × frameskip       │
  │ über T (500 Episoden)   │ → max H (Val-sicher)         │
  │ 5 Konfigurationen       │ mit Zellen-Beschriftung      │
  └─────────────────────────┴──────────────────────────────┘

  Erkenntnisse:
  - T=22 mit fs=2: max H=7 (Rollout), davon H=6 Val-sicher
  - T<14 bei fs=2: gar kein Training möglich (0 Slices)
  - T>30 bei fs=2: H=8+ möglich, aber VRAM wird zum Engpass
  - fs=1 erlaubt das höchste H, verbraucht aber am meisten VRAM
    (weil num_frames = num_hist + 1 Bilder durch DINOv2 müssen)
  - Unsere Datensätze (T=22, T=25) sind mit ★ markiert

────────────────────────────────────────────────────────────────────────
PLOT 09: ÜBERARBEITUNG (visuell kaputt → repariert)
────────────────────────────────────────────────────────────────────────

  Vorher (defekt):
  - 2 Subplots nebeneinander (16×7), gequetscht
  - \\n Literal-Strings statt echte Zeilenumbrüche
  - x-Achsen-Labels überlagert
  - Ursachen-Textbox abgeschnitten
  - Balken-Labels überlagern Limit-Linien

  Nachher (repariert):
  - 3 Subplots vertikal (10×13), übersichtlich
  - Oben: Linienplot Training vs Val Peak über H
  - Mitte: Horizontale Balken mit MiB/%-Annotation
  - Unten: Gestapelter Balken (6 Komponenten) mit Legende
  - Saubere Zeilenumbrüche, k-Formatter auf y-Achse
  - Ursachen-Box in eigenem Subplot-Bereich

────────────────────────────────────────────────────────────────────────
DOKUMENTATION
────────────────────────────────────────────────────────────────────────

DINO_WM_TRAINING_DOCUMENTATION.md
  + Neuer Abschnitt 3.7 "VRAM-Analyse und Validierungs-Lastspitze"
    (8 Unterabschnitte, ~200 Zeilen):
    3.7.1  VRAM-Modell: Drei Kostenklassen
    3.7.2  Empirische Kalibrierung (Faktor 5.51×)
    3.7.3  Validierungs-Lastspitze (val() Bug-Analyse)
    3.7.4  Maximales num_hist nach VRAM (Tabelle)
    3.7.5  Optimale Konfigurationen (Solver)
    3.7.6  Vergleich mit Paper (Zhou et al. 2025)
    3.7.7  Generierte Analyse-Plots (Übersicht)
    3.7.8  Potentielle Code-Fixes (nicht angewendet)

DEVELOPMENT_PROTOCOL.md (isaacsim, kein Git)
  + Neuer Meilenstein in Übersichtstabelle
  + Neue Sektion: "DINO WM: VRAM-ANALYSE & HYPERPARAMETER-
    OPTIMIERUNG (15.02.2026)" mit Methodik, Ergebnissen, Outputs

────────────────────────────────────────────────────────────────────────
GEÄNDERTE DATEIEN (Übersicht)
────────────────────────────────────────────────────────────────────────

  Geändert:
    M  DINO_WM_TRAINING_DOCUMENTATION.md     (+382 Zeilen, Abschnitt 3.7)
    M  conf/train.yaml                       (minor)
    M  conf/env/franka_cube_stack.yaml       (minor)

  Neu (untracked):
    ?? hyperparameter_analysis.py            (1881 Zeilen)
    ?? hyperparameter_analysis/              (33 Dateien, 15 Plots)

  Extern (isaacsim, kein Git):
    ~  DEVELOPMENT_PROTOCOL.md               (+Sektion 15.02.2026)

────────────────────────────────────────────────────────────────────────
ZUSAMMENFASSUNG DER ERGEBNISSE
────────────────────────────────────────────────────────────────────────

  Gewählte Konfiguration: batch_size=4, num_hist=6, frameskip=2, T=22

  ┌────────────────────┬───────────┬──────────────────────────────────┐
  │ Metrik             │ Wert      │ Bewertung                        │
  ├────────────────────┼───────────┼──────────────────────────────────┤
  │ Training VRAM      │ 16467 MiB │ 67.0% — komfortabel              │
  │ Val Peak VRAM      │ 19239 MiB │ 78.3% — sicher unter 90%-Limit   │
  │ Val Overhead       │ +17%      │ durch fehlende torch.no_grad()   │
  │ Max H (Val-sicher) │ 6         │ H=7 → Val OOM (103.7%)          │
  │ Slices/Episode     │ 9         │ gute Dateneffizienz              │
  │ Train-Samples      │ 3771      │ bei 499 Episoden, 419 nutzbar    │
  │ Steps/Epoch        │ 943       │ ~30 min/Epoch (geschätzt)        │
  └────────────────────┴───────────┴──────────────────────────────────┘

  Vergleich mit Paper (Zhou et al. 2025):
  - Paper: bs=32, H=1–3 auf A6000 (48 GB)
  - Wir:   bs=4, H=6 auf A5000 (24.5 GB)
  → 2× mehr temporaler Kontext bei ~50% GPU-Budget
```
