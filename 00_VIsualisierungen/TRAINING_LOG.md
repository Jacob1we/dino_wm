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

## 0. System Hardware

### GPU
| Feld | Wert |
|------|------|
| **Modell** | NVIDIA RTX A4000 |
| **VRAM Total** | 16,376 MiB (~16 GB) |
| **Driver Version** | 570.195.03 |
| **CUDA Version** | 12.8 |

### CPU
| Feld | Wert |
|------|------|
| **Modell** | Intel Core i9-14900K |
| **Kerne** | 24 Cores (32 Threads) |
| **Sockets** | 1 |
| **Max Takt** | 6.0 GHz |
| **Min Takt** | 0.8 GHz |

### RAM
| Feld | Wert |
|------|------|
| **Total** | 188 GB |
| **Verfügbar** | ~169 GB |
| **Swap** | 2 GB |

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

### 6.2 Training Freeze ⚠️ UNGELÖST

**Problem:** Training friert nach zufälliger Zeit ein (15min - 60min+)
- Keine Fehlermeldung, kein Crash
- GPU-Temperatur normal (~48°C)
- Zombie-Prozess `[python] <defunct>` entsteht

**Bereits getestete Lösungen (OHNE Erfolg):**

| Versuch | Ergebnis |
|---------|----------|
| `num_workers=0` | ❌ Freeze bleibt |
| `preload_images=True` | ❌ Freeze bleibt |
| Kleinerer Datensatz (10 statt 100 Episoden) | ❌ Freeze bleibt |
| GPU Thermal Management (Power Limit, Lüfter) | ❌ Freeze bleibt (Temp war nie das Problem) |

**Ausgeschlossene Ursachen:**
- ~~DataLoader Multiprocessing~~ → Freeze auch mit `num_workers=0`
- ~~GPU Überhitzung~~ → Temp bei Freeze: 48°C
- ~~Datensatz-Größe~~ → Freeze mit 10 und 100 Episoden
- ~~OOM~~ → Keine Fehlermeldung, genug RAM/VRAM

**WandB Analyse (25 Runs):**
- RAM ist KONSTANT (~4-5GB) - kein Speicherleck
- Crashes zeitlich verstreut (15-60+ min) - kein Muster
- GPU Power fällt plötzlich ab (80W → 20W)
- Ursache weiterhin unbekannt

**Noch zu testen:**
- [ ] WandB deaktivieren (`WANDB_MODE=disabled`) - **PRIORITÄT 1**
- [ ] Visualisierungen deaktivieren (`training.num_reconstruct_samples=0`)
- [ ] RAM-Monitoring parallel zum Training
- [ ] gc.collect() nach jeder Epoche

**Debug-Befehl:**
```bash
# RAM-Leak Test ohne WandB und Visualisierungen
WANDB_MODE=disabled python train.py env=franka_cube_stack training.num_reconstruct_samples=0
```

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

### 6.4 GPU Thermal Throttling ✅ GELÖST

**Problem:** Training friert nach 15-25 Minuten ein, GPU-Auslastung fällt auf 0%

**Symptome (WandB Monitoring):**
- GPU-Temperatur steigt kontinuierlich auf 82-83°C
- Bei ~83°C: Plötzlicher Abfall aller Metriken (Power, Memory Access, Utilization)
- Training-Prozess hängt, reagiert nicht mehr

**Ursache:** RTX A4000 überhitzt → Thermal Throttling / System-Schutzabschaltung
- Blower-Style Kühlung (Einzellüfter)
- VBIOS-Lüfterkurve zu konservativ: Nur 62% Lüfter bei 83°C!
- Maximale Betriebstemperatur: 83-86°C (zu nah am Limit)

**Diagnose:**
```bash
nvidia-smi
# Ausgabe zeigte: 83°C, Fan 64%, Power 72W/140W
```

**Lösung (angewendet):**

1. **Power Limit reduziert:**
   ```bash
   sudo nvidia-smi -pl 100  # Von 140W auf 100W
   ```

2. **GreenWithEnvy (GWE) installiert** für custom Lüfterkurve:
   ```bash
   sudo apt install flatpak
   flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
   flatpak install flathub com.leinardi.gwe
   flatpak run com.leinardi.gwe
   ```

3. **Custom Fan Profile "JW" erstellt:**
   | Temperatur | Lüfter |
   |------------|--------|
   | 30°C | 0% |
   | 40°C | 40% |
   | 60°C | 60% |
   | 70°C | 75% |
   | 75°C | 100% |

4. **GWE Autostart aktiviert:** Settings → Launch on login

**Ergebnis:**
| Vorher | Nachher |
|--------|---------|
| 83°C @ 62% Fan | <80°C @ 100% Fan |
| Training friert ein | Training stabil |

**Empfohlene Konfiguration für RTX A4000:**
- Power Limit: 100-110W (statt 140W)
- Lüfter: 100% ab 70-75°C
- Gehäuse offen oder Zusatzlüfter bei langen Trainings

---

## 7. Notizen

- **2026-01-07 09:33:** Erstes Training gestartet → OOM bei batch_size=32
- **2026-01-07:** Torch-Konflikt behoben durch Deinstallation der User-Installation (`pip uninstall torch torchvision --user`)
- **2026-01-07:** ✅ Training läuft mit batch_size=8
- **2026-01-13:** ❌ Training friert regelmäßig nach 15-25 Min ein (auch mit kleinem 10-Episode Dataset)
- **2026-01-13:** 🔍 Verdacht: GPU Thermal Throttling (83°C, Lüfter nur 62%)
- **2026-01-13:** ✅ GreenWithEnvy installiert, Custom Fan Profile "JW", Power Limit 100W
- **2026-01-13:** ❌ Freeze bleibt trotz GPU-Kühlung (Temp bei Freeze: 48°C)
- **2026-01-13:** ❌ `num_workers=0` getestet → Freeze bleibt
- **2026-01-13:** ❌ Kleiner Datensatz (10 Episoden) → Freeze bleibt
- **2026-01-13:** ⚠️ **Ursache weiterhin unbekannt** - weder Temperatur noch DataLoader noch Datensatz-Größe
- GPU: NVIDIA RTX A4000 (16GB VRAM)

### Nächste Schritte
1. WandB deaktivieren testen (`WANDB_MODE=offline`)
2. CUDA synchron testen (`CUDA_LAUNCH_BLOCKING=1`)
3. Accelerator/DDP deaktivieren
4. Auf anderem System testen

---

*Log erstellt: 2026-01-07*
*Letzte Aktualisierung: 2026-01-13*

