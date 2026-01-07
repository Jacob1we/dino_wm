# DINO World Model - Training Log

## Experiment Übersicht

| Feld | Wert |
|------|------|
| **Experiment ID** | `franka_cube_stack_f5_h3_p1` |
| **Start Datum** | 2026-01-07 09:33:48 |
| **Status** | 🟡 Running |
| **WandB Run** | [lively-violet-5](https://wandb.ai/jacob-weyer-rwth-aachen-university/dino_wm/runs/mhlk8hxd) |
| **Checkpoint Pfad** | `outputs/2026-01-07/09-33-48/` |

---

## 1. Datensatz Parameter

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `env` | `franka_cube_stack` | Environment/Dataset Name |
| `data_path` | `/home/tsp_jw/Desktop/isaacsim/.../franka_cube_stack_ds` | Datensatz-Pfad |
| `n_rollouts` | 20 | Anzahl Trajektorien |
| `max_timesteps` | 935 | Max. Schritte pro Episode |
| `state_dim` | 22 | Zustandsdimension |
| `action_dim` | 9 | Aktionsdimension |
| `image_size` | 256×256 → 224×224 | Bildgröße (Original → Resize) |
| `split_ratio` | 0.9 | Train/Val Split (90/10) |
| `normalize_action` | True | Aktionen z-normalisiert |

---

## 2. Training Hyperparameter

### 2.1 Basis-Konfiguration

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `epochs` | 100 | Anzahl Epochen |
| `batch_size` | 32 | Batch-Größe |
| `seed` | 0 | Random Seed |
| `save_every_x_epoch` | 1 | Checkpoint-Frequenz |

### 2.2 Temporal Parameter

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `frameskip` | 5 | Frames überspringen |
| `num_hist` | 3 | Historie-Frames (Kontext) |
| `num_pred` | 1 | Prädiktions-Frames |

### 2.3 Learning Rates

| Komponente | Learning Rate | Trainierbar |
|------------|---------------|-------------|
| Encoder (DINO) | 1e-6 | ❌ Frozen |
| Predictor (ViT) | 5e-4 | ✅ Ja |
| Decoder (VQ-VAE) | 3e-4 | ✅ Ja |
| Action Encoder | 5e-4 | ✅ Ja |

### 2.4 Embedding Dimensionen

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `action_emb_dim` | 10 | Action Embedding |
| `proprio_emb_dim` | 10 | Proprioception Embedding |
| `num_action_repeat` | 1 | Action Repeat |
| `num_proprio_repeat` | 1 | Proprio Repeat |
| **Total emb_dim** | 404 | Gesamt-Embedding (384 + 10 + 10) |

---

## 3. Modell-Architektur

| Komponente | Typ | Details |
|------------|-----|---------|
| **Encoder** | DINOv2-ViT-S/14 | Pretrained, Frozen |
| **Predictor** | Vision Transformer | Trainierbar |
| **Decoder** | VQ-VAE | Trainierbar |
| **Proprio Encoder** | Conv1d(3→10) | EE-Position |
| **Action Encoder** | Conv1d(45→10) | 9×5 (frameskip) |

---

## 4. Training Fortschritt

### Aktueller Status

| Metrik | Wert |
|--------|------|
| Aktuelle Epoch | 1 / 100 |
| Batches pro Epoch | 515 |
| Train Loss | - |
| Val Loss | - |

### Epoch Log

| Epoch | Train Loss | Val Loss | z_err_pred | Datum | Notizen |
|-------|------------|----------|------------|-------|---------|
| 1 | - | - | - | 2026-01-07 | Training gestartet |
| 2 | | | | | |
| 3 | | | | | |
| ... | | | | | |

---

## 5. Befehl zum Starten

```bash
cd /home/tsp_jw/Desktop/dino_wm
conda activate dino_wm
python train.py --config-name train.yaml env=franka_cube_stack frameskip=5 num_hist=3
```

### Resume Training

```bash
# Training fortsetzen vom letzten Checkpoint
python train.py --config-name train.yaml env=franka_cube_stack frameskip=5 num_hist=3
# (Lädt automatisch model_latest.pth wenn vorhanden)
```

---

## 6. Troubleshooting

### 6.1 CUDA Out of Memory (OOM)

**Problem:** Training mit `batch_size=32` führt zu OOM auf RTX A4000 (16GB VRAM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 442.00 MiB.
```

**Ursache:** VQ-VAE Decoder ist speicherintensiv bei hoher Batch-Size

**Lösung:** Batch-Size reduzieren

| Batch Size | VRAM (geschätzt) | Status |
|------------|------------------|--------|
| 32 | ~18GB | ❌ OOM |
| 16 | ~10GB | ✅ Empfohlen |
| 8 | ~6GB | ✅ Sicher |

**Befehl mit reduzierter Batch-Size:**
```bash
python train.py --config-name train.yaml env=franka_cube_stack frameskip=5 num_hist=3 training.batch_size=16
```

### 6.2 Segfault / DataLoader Deadlock ✅ GELÖST

**Problem:** Training crasht mit Segfault oder friert ein bei ~30-40%
```
segfault in libtorch_cpu.so
resource_tracker: leaked semaphore objects
```

**Ursache:** `torch.load()` in `get_frames()` wird bei jedem Zugriff aufgerufen → Deadlock mit multiprocessing Workers

**Lösung: Bilder vorab in RAM laden**

Dataset-Loader wurde modifiziert (`preload_images=True` default):
- Alle Bilder werden beim Start in RAM geladen (~3.5 GB für 20 Episoden)
- Kein `torch.load()` mehr während des Trainings
- Multiprocessing-Worker können sicher auf gecachte Daten zugreifen

| num_workers | preload_images | Zeit/Epoch | Status |
|-------------|----------------|------------|--------|
| 8 | False | ~25 min | ❌ Deadlock |
| 4 | False | ~35 min | ❌ Deadlock |
| 0 | False | ~90 min | ✅ Stabil |
| 4-8 | **True** | ~25-30 min | ✅ **Empfohlen** |

---

### 6.3 Torch Versionskonflikt ✅ GELÖST

**Problem:** `RuntimeError: operator torchvision::nms does not exist`

**Ursache:** Konflikt zwischen lokaler torch-Installation (`~/.local/lib/python3.10`) und conda-Umgebung

**Lösung (angewendet):**
```bash
# Lokale Installation entfernt
pip uninstall torch torchvision --user
```

**Alternative Workarounds (nicht mehr nötig):**
```bash
# Option 1: Für einzelnen Befehl
PYTHONNOUSERSITE=1 python train.py ...

# Option 2: Permanent in ~/.bashrc
export PYTHONNOUSERSITE=1
```

---

## 7. Notizen

- **2026-01-07 09:33:** Erstes Training gestartet → OOM bei batch_size=32
- **2026-01-07:** Torch-Konflikt behoben durch Deinstallation der User-Installation (`pip uninstall torch torchvision --user`)
- **2026-01-07:** ✅ Training läuft mit batch_size=8
- 20 Rollouts verwendet (von 36 generierten)
- GPU: NVIDIA RTX A4000 (16GB VRAM)

---

*Log erstellt: 2026-01-07*
*Letzte Aktualisierung: 2026-01-07*

