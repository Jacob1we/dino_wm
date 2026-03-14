# DINO-WM Training Metriken: Vollständige Referenz

Diese Dokumentation beschreibt alle Metriken, die während des DINO World Model Trainings auf Weights & Biases geloggt werden. Die Definitionen basieren auf dem Paper [Zhou et al., 2025] und der Codebase.

---

## 1. Loss-Metriken (Hauptverluste)

| Metrik | Definition | Quelle |
|--------|------------|--------|
| **`train_loss` / `val_loss`** | Gesamtverlust = `z_loss` + `decoder_loss_reconstructed`. Die Hauptoptimierungsmetrik. | `visual_world_model.py#L241-L271` |
| **`train_z_loss` / `val_z_loss`** | Latent-Space-Vorhersageverlust: MSE zwischen vorhergesagten und tatsächlichen DINOv2-Embeddings. Entspricht Gleichung (1) im Paper. | Paper Eq. (1), `visual_world_model.py#L236-L242` |
| **`train_z_visual_loss`** | Verlust nur für die visuellen Patch-Embeddings (ohne Propriozeption und Aktion). | `visual_world_model.py#L224-L232` |
| **`train_z_proprio_loss`** | Verlust nur für die Propriozeptionskomponente der Embeddings. | `visual_world_model.py#L225-L235` |

### Paper-Referenz (Gleichung 1):

$$\mathcal{L}_{pred} = \|p_\theta(\text{enc}_\theta(o_{t-H:t}), \phi(a_{t-H:t})) - \text{enc}_\theta(o_{t+1})\|^2$$

wobei:
- $p_\theta$ = Transition Model (ViT Predictor)
- $\text{enc}_\theta$ = DINOv2 Encoder (frozen)
- $\phi$ = Action Encoder (MLP)
- $H$ = Context Length (Anzahl historischer Frames)

---

## 2. Decoder-Metriken

Der Decoder ist **optional** und dient primär der Visualisierung. Sein Training ist vom Predictor entkoppelt.

| Metrik | Definition | Quelle |
|--------|------------|--------|
| **`decoder_loss_pred`** | Decoder-Verlust auf **vorhergesagten** Latents: `recon_loss_pred` + λ · `vq_loss_pred`. Der Decoder rekonstruiert Bilder aus den vom Predictor geschätzten zukünftigen Zuständen. | `visual_world_model.py#L213-L218` |
| **`decoder_loss_reconstructed`** | Decoder-Verlust auf **echten** Latents: Rekonstruktion der Eingabebilder aus ihren eigenen DINOv2-Embeddings. Testet die Decoder-Qualität isoliert. | `visual_world_model.py#L252-L266` |
| **`decoder_recon_loss_pred`** | Rekonstruktionsverlust (MSE) zwischen dekodierten vorhergesagten Bildern und Ground Truth. | `visual_world_model.py#L210-L217` |
| **`decoder_recon_loss_reconstructed`** | Rekonstruktionsverlust auf echten Latents. | `visual_world_model.py#L255` |
| **`decoder_vq_loss_pred`** | VQ-VAE Commitment Loss auf vorhergesagten Latents. Bei Wert 0.0 wird kein VQ-VAE verwendet (reiner Transposed-Conv-Decoder). | `visual_world_model.py#L214` |
| **`decoder_vq_loss_reconstructed`** | VQ-VAE Commitment Loss auf echten Latents. | `visual_world_model.py#L261` |

### Paper-Referenz (Gleichung 2):

$$\mathcal{L}_{rec} = \|q_\theta(z_t) - o_t\|^2, \quad \text{wobei } z_t = \text{enc}_\theta(o_t)$$

**Wichtig:** Das Decoder-Training beeinflusst NICHT die Planungsfähigkeit des Weltmodells!

---

## 3. Bildqualitäts-Metriken

Diese Metriken werden in `metrics/image_metrics.py` berechnet und vergleichen rekonstruierte/vorhergesagte Bilder mit Ground Truth.

| Metrik | Definition | Formel | Interpretation |
|--------|------------|--------|----------------|
| **`img_l1`** | Mean Absolute Error | $\frac{1}{N}\sum\|I_{pred} - I_{gt}\|$ | Niedriger = besser |
| **`img_l2`** | Mean Squared Error | $\frac{1}{N}\sum(I_{pred} - I_{gt})^2$ | Niedriger = besser |
| **`img_mse`** | Identisch mit L2 (Alias) | wie L2 | Niedriger = besser |
| **`img_psnr`** | Peak Signal-to-Noise Ratio | $20 \cdot \log_{10}\left(\frac{1}{\sqrt{MSE}}\right)$ | **Höher = besser** (in dB) |
| **`img_ssim`** | Structural Similarity Index | Luminanz × Kontrast × Struktur | **Höher = besser** (0-1) |
| **`img_lpips`** | Learned Perceptual Image Patch Similarity | VGG-basierte perzeptuelle Distanz | **Niedriger = besser** |

### Suffixe:

- **`_pred`**: Berechnet auf **vorhergesagten** Bildern (Predictor → Decoder → Bild)
- **`_reconstructed`**: Berechnet auf **rekonstruierten** Bildern (echte Latents → Decoder → Bild)

### Paper-Hauptmetrik:

**LPIPS** ist die primäre Bildmetrik im Paper. Die Autoren berichten eine **56% Verbesserung** gegenüber Baselines auf den schwierigsten Tasks.

### Code-Referenz (`metrics/image_metrics.py`):

```python
def eval_images(img1, img2):
    metrics = {}
    metrics['l1'] = torch.abs((img1 - img2)).mean()
    metrics['l2'] = ((img1 - img2) ** 2).mean()
    metrics['ssim'] = ssim(img1, img2)
    metrics['mse'] = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean()
    metrics['psnr'] = 20 * torch.log10(1.0 / torch.sqrt(mse))
    metrics['lpips'] = lpips(img1, img2, net_type='vgg')
    return metrics
```

---

## 4. Embedding-Fehler-Metriken (z_err)

Diese messen den Vorhersagefehler **direkt im Latent-Space** ohne Dekodierung. Sie sind aussagekräftiger für die Planungsqualität als Bildmetriken.

| Metrik | Definition | Zeitfenster |
|--------|------------|-------------|
| **`z_visual_err_full`** | MSE über alle vorhergesagten visuellen Patch-Embeddings | Gesamte Sequenz |
| **`z_visual_err_pred`** | MSE nur über die Vorhersage-Frames (die letzten `num_pred` Frames) | Nur vorhergesagte Frames |
| **`z_visual_err_next1`** | MSE nur für den nächsten Frame (1-Schritt-Vorhersage) | Nur Frame t+1 |
| **`z_proprio_err_full`** | MSE über alle Propriozeptionswerte | Gesamte Sequenz |
| **`z_proprio_err_pred`** | MSE nur über Vorhersage-Frames | Nur vorhergesagte Frames |
| **`z_proprio_err_next1`** | MSE für nächsten Frame | Nur Frame t+1 |

### Slices (aus `train.py#L449-L459`):

```python
slices = {
    "full": (None, None),                    # Alle Frames
    "pred": (-num_pred, None),               # Letzte num_pred Frames
    "next1": (-num_pred, -num_pred + 1),     # Nur erster Vorhersage-Frame
}
```

### Interpretation:

- **`err_full`**: Gesamtperformance inklusive Teacher-Forcing-Anteil
- **`err_pred`**: Reine Vorhersagequalität (relevanter für Planung)
- **`err_next1`**: 1-Schritt-Vorhersage (zeigt `nan` wenn `num_pred=1`)

---

## 5. Rollout-Metriken

Diese evaluieren die **autoregressive Vorhersage** über mehrere Zeitschritte hinweg. Sie sind der beste Indikator für die Planungsqualität.

| Metrik | Definition | Kontext |
|--------|------------|---------|
| **`z_visual_err_rollout`** | MSE zwischen final vorhergesagtem und echtem Ziel-Zustand nach vollständigem Rollout | Start mit `num_hist` Frames als Kontext |
| **`z_visual_err_rollout_1framestart`** | Wie oben, aber Start mit nur **1 Frame** als Kontext | Härterer Test: weniger Kontext |
| **`z_proprio_err_rollout`** | Propriozeptionsfehler nach Rollout | Start mit `num_hist` Frames |
| **`z_proprio_err_rollout_1framestart`** | Propriozeptionsfehler mit 1 Frame Start | Härterer Test |

### Berechnung (aus `train.py#L660-L740`):

1. Wähle zufällige Trajektorie aus Datensatz
2. Extrahiere Startframes und Aktionssequenz
3. Führe autoregressiven Rollout durch:
   $$\hat{z}_{t+1} = p_\theta(\hat{z}_t, a_t)$$
4. Vergleiche finalen Zustand $\hat{z}_T$ mit Ground Truth $z_T$

### Warum zwei Varianten?

- **`_rollout`**: Testet mit vollem Kontext (realistisches Szenario)
- **`_1framestart`**: Testet Robustheit bei minimalem Kontext (härter)

---

## 6. Metriken-Hierarchie für Evaluation

### Priorität für Modellbewertung:

| Priorität | Metrik | Warum wichtig |
|-----------|--------|---------------|
| 🔴 **Kritisch** | `val_z_loss` | Haupttrainings-Metrik, Generalisierung |
| 🔴 **Kritisch** | `val_z_visual_err_rollout` | Autoregressive Vorhersagequalität → Planungsfähigkeit |
| 🔴 **Kritisch** | `val_img_lpips_pred` | Paper-Hauptmetrik, perzeptuelle Qualität |
| 🟡 **Wichtig** | `val_img_ssim_pred` | Strukturelle Bildähnlichkeit |
| 🟡 **Wichtig** | `val_img_psnr_pred` | Signal-Rausch-Verhältnis |
| 🟢 **Diagnostik** | `decoder_loss_reconstructed` | Decoder-Qualität isoliert |
| 🟢 **Diagnostik** | `z_visual_err_rollout_1framestart` | Robustheit bei wenig Kontext |

### Train vs. Val Vergleich:

- **Gesund**: Train ≈ Val (leicht besser Train ist OK)
- **Overfitting**: Train << Val (3× oder mehr Unterschied)
- **Underfitting**: Train ≈ Val, beide hoch

---

## 7. Beispiel-Interpretation

Gegeben folgende Werte nach Epoch 50:

| Metrik | Train | Val | Bewertung |
|--------|-------|-----|-----------|
| `z_loss` | 0.111 | 0.346 | ⚠️ Overfitting (3× Unterschied) |
| `img_ssim_pred` | 0.957 | 0.819 | ✅ Train sehr gut, Val akzeptabel |
| `img_lpips_pred` | 0.030 | 0.069 | ✅ Train exzellent, Val gut |
| `z_visual_err_rollout` | 0.369 | 1.443 | ⚠️ Rollout akkumuliert Fehler (4×) |
| `img_psnr_pred` | 28.8 dB | 19.0 dB | ⚠️ ~10 dB Unterschied signifikant |

### Diagnose:

Das Modell zeigt **klassisches Overfitting**:
- Validierungs-Rollout-Fehler sind 4× höher als Training
- PSNR-Differenz von 10 dB ist erheblich

### Empfehlungen:

1. **Mehr Trainingsdaten** (Environment Sweep erhöhen)
2. **Stärkere Regularisierung** (Dropout, Weight Decay)
3. **Data Augmentation** (Domain Randomization erweitern)
4. **Early Stopping** basierend auf `val_z_loss`

---

## 8. Referenzen

- **Paper**: Zhou et al. (2025). "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"
- **Code**: `models/visual_world_model.py`, `train.py`, `metrics/image_metrics.py`
- **LPIPS**: Zhang et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
- **SSIM**: Wang et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity"
