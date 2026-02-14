# Commit Message — DINO WM Repository

```
fix(planning_server): Proprio-Bug + OOM-Schutz (Phase 13)

ROOT CAUSE — Roboter fährt zu falscher Position:
  Der Planning Server hat die Propriozeption (EEF-Position) des Roboters
  NICHT an das World Model weitergeleitet. Stattdessen wurde ein
  Null-Vektor [0, 0, 0] als Proprio gesetzt (img_to_obs, Zeile 298).

  Das DINO World Model verwendet Propriozeption als Eingabe:
  Die 3D EEF-Position [x, y, z] wird durch den ProprioceptiveEmbedding-
  Encoder zu einem 10D-Embedding kodiert (proprio_emb_dim=10) und mit
  den visuellen DINO-Features per Dimension-Concatenation verbunden
  (concat_dim=1). Dieses kombinierte Embedding geht in den ViT-Predictor.

  Mit proprio=[0,0,0] berechnet der Preprocessor nach z-Normalisierung:
    normalized = (0 - proprio_mean) / proprio_std
              ≈ (-[0.47, 0.02, 0.16]) / [0.12, 0.16, 0.07]
              ≈ [-3.8, -0.1, -2.2]
  Das ist ein physisch unmöglicher Zustand — kein Trainingsbeispiel hatte
  je solche Werte. Der ViT-Predictor generiert dadurch unsinnige Latent-
  Trajektorien, und CEM konvergiert zu Aktionen die den Roboter auf den
  Boden drücken statt zum Ziel zu fahren.

FIX — Proprio-Pipeline:
  1. img_to_obs() akzeptiert jetzt ee_pos als Parameter
  2. Alle 3 Socket-Handler (set_goal, plan, plan_all) extrahieren
     ee_pos aus der Client-Nachricht: msg["ee_pos"]
  3. Abwärtskompatibel: Wenn kein ee_pos gesendet wird (alter Client),
     fällt der Server auf Nullen zurück mit Warnung auf stdout
  4. Diagnostik-Logging: Denormalisierte Ziel-Positionen werden pro
     Sub-Action auf stdout ausgegeben

OOM-FIX — ChunkedRolloutWrapper (ebenfalls in diesem Commit):
  Problem: CEM sampelt z.B. 300 Trajektorien und schickt ALLE auf
  einmal durch wm.rollout(). Die Self-Attention im ViT-Predictor
  skaliert O(N²) mit N = num_hist × patches_per_frame:
    - num_hist=2: N = 2 × 256 = 512 Tokens → ~5 GB VRAM → OK
    - num_hist=4: N = 4 × 256 = 1024 Tokens → ~20 GB VRAM → OOM

  Die Attention-Matrix hat Größe (B × heads × N × N):
    300 × 16 × 1024 × 1024 × 4 Bytes ≈ 20 GB
  Das übersteigt die 24 GB der RTX A5000.

  Lösung: ChunkedRolloutWrapper teilt den CEM-Batch (300 Samples)
  in GPU-sichere Sub-Batches (z.B. 25 Stück). Auto-Detection der
  chunk_size basierend auf freiem VRAM und num_hist:
    gb_per_sample ≈ 0.02 × num_hist²
    chunk_size = clamp(free_GB / gb_per_sample, 8, 64)

  Der CEM-Planner sieht keinen Unterschied (gleiches Interface).
  Steuerbar über --chunk_size CLI-Argument (0 = deaktiviert).

Geänderte Datei:
  planning_server.py  337 → 447 Zeilen (+110)

Neue Klasse:
  ChunkedRolloutWrapper — Transparenter Batch-Splitting-Wrapper

Neue CLI-Argumente:
  --chunk_size INT    Max Batch-Size pro Rollout (None=auto, 0=aus)

Konfiguration (500-Episoden-Modell, 2026-02-09/17-59-59):
  num_hist=4, frameskip=2, action_dim=6, proprio_dim=3
  proprio_emb_dim=10, concat_dim=1, num_proprio_repeat=1
```
