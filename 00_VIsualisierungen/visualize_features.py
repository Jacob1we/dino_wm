#!/usr/bin/env python3
"""
Feature-Visualisierung für DINOv2 Encoder und ViT Predictor.

Erzeugt für die Masterarbeit anschauliche Bilder, die zeigen, wie das
DINO World Model visuelle Features in Franka-Cube-Stacking-Szenen erkennt.

Visualisierungen:
  1. DINOv2 Attention Maps       – Self-Attention der letzten Encoder-Schicht
  2. DINOv2 PCA Feature Maps     – K-Means Segmentierung + PCA pro Cluster
  3. DINOv2 Feature Similarity   – Cosine-Similarity eines Referenz-Patches zu allen anderen
  4. ViT Predictor Attention     – Cross-Attention-Maps des trainierten Predictors
  5. VQ-VAE Rekonstruktion       – Decoder-Ausgabe vs. Original (wenn Decoder vorhanden)

Nutzung:
  conda activate dino_wm
  python visualize_features.py --model_name 260305/07-56 [--episode_idx 0] [--frame_idx 5]
  python visualize_features.py --image_path /pfad/zu/bild.png   # Einzelbild (nur DINOv2)

Ausgabe:
  feature_visualizations/<model_name>/
    ├── original_<ep>_<fr>.png
    ├── dino_attention_<ep>_<fr>.png
    ├── dino_pca_<ep>_<fr>.png
    ├── dino_similarity_<ep>_<fr>.png
    ├── vit_attention_<ep>_<fr>.png
    ├── reconstruction_<ep>_<fr>.png
    └── summary_<ep>_<fr>.png
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNorm
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── Projekt-Root auf sys.path ──
DINO_WM_DIR = os.path.dirname(os.path.abspath(__file__))
if DINO_WM_DIR not in sys.path:
    sys.path.insert(0, DINO_WM_DIR)

from masterarbeit_style import apply_style, TEXTWIDTH_IN, FONT_SIZE, save_ma_figure
apply_style()

from plan import load_model
from preprocessor import Preprocessor
from utils import seed


# =====================================================================
# Hilfsfunktionen
# =====================================================================

def denorm_image(img_tensor):
    """(C, H, W) normalisiertes Bild → (H, W, C) uint8 numpy."""
    img = (img_tensor + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def load_single_image(image_path, img_size=224):
    """Lädt ein einzelnes Bild und bereitet es für DINOv2 vor."""
    from torchvision import transforms
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor, np.array(img.resize((img_size, img_size)))


def apply_colormap(heatmap, cmap="inferno"):
    """Numpy-Heatmap (H, W) float [0,1] → (H, W, 3) uint8 RGB."""
    cm = plt.get_cmap(cmap)
    colored = cm(heatmap)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(image, heatmap, alpha=0.5, cmap="inferno"):
    """Überlagert eine Heatmap auf ein RGB-Bild."""
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = apply_colormap(heatmap_norm, cmap)

    if image.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = np.array(Image.fromarray(heatmap_color).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR))

    blended = (alpha * heatmap_color.astype(float) +
               (1 - alpha) * image.astype(float)).clip(0, 255).astype(np.uint8)
    return blended


def build_aligned_context_window(obs_visual, obs_proprio, actions, num_hist, frameskip,
                                 target_frame_idx):
    """
    Baut einen WM-konformen Kontext aus derselben Episode wie das DINO-Bild.

    Der Ziel-Frame wird nach Möglichkeit exakt in den Kontext eingebettet. Die
    Auswahl folgt derselben Zeitkonvention wie TrajSlicerDataset:
    obs[start : start + num_hist*frameskip : frameskip]
    act[start : start + num_hist*frameskip] -> pro Kontext-Frame konkateniert.
    """
    if obs_proprio is None:
        raise ValueError("Proprio fehlt im Dataset; ViT-Visualisierung nicht möglich.")

    total_obs = obs_visual.shape[0]
    total_actions = actions.shape[0]

    for candidate_frame in range(target_frame_idx, -1, -1):
        for frame_idx_in_seq in range(num_hist - 1, -1, -1):
            start_idx = candidate_frame - frame_idx_in_seq * frameskip
            end_obs = start_idx + num_hist * frameskip

            if start_idx < 0:
                continue
            if end_obs > total_obs or end_obs > total_actions:
                continue

            sampled_indices = list(range(start_idx, end_obs, frameskip))
            context_visual = obs_visual[sampled_indices]
            context_proprio = obs_proprio[sampled_indices]
            context_actions = []

            for step_idx in range(num_hist):
                action_start = start_idx + step_idx * frameskip
                action_end = action_start + frameskip
                context_actions.append(actions[action_start:action_end].reshape(-1))

            return {
                "context_visual": context_visual.unsqueeze(0),
                "context_proprio": context_proprio.unsqueeze(0),
                "context_actions": torch.stack(context_actions).unsqueeze(0),
                "frame_idx_in_seq": frame_idx_in_seq,
                "actual_frame_idx": candidate_frame,
                "sampled_indices": sampled_indices,
                "start_idx": start_idx,
            }

    raise ValueError(
        "Kein gültiges WM-Kontextfenster für diesen Frame gefunden. "
        "Der Frame liegt vermutlich zu nah am Episodenende."
    )


# =====================================================================
# 1. DINOv2  Attention Maps
# =====================================================================

def extract_dino_attention(encoder, img_tensor, device, encoder_transform=None):
    """
    Extrahiert die Self-Attention-Maps aus der letzten Schicht des DINOv2.

    Returns:
        attn_maps: (num_heads, H_patches, W_patches)  – Attention vom CLS-Token
        patch_tokens: (num_patches, emb_dim)
    """
    encoder.eval()
    model = encoder.base_model
    img = img_tensor.to(device)
    # Wende dieselbe Transformation an wie das World Model
    if encoder_transform is not None:
        img = encoder_transform(img)

    # DINOv2 ViT: ersetze forward durch Hook auf den letzten Block
    attn_weights = []

    def hook_fn(module, input, output):
        # Bei DINOv2 ist die Attention innerhalb von Block.attn
        # Wir holen die QKV direkt
        pass

    # Nutze die get_intermediate_layers API von DINOv2
    # Alternativ: manuell durch forward_features mit Hook
    with torch.no_grad():
        # Prepare input like DINOv2 expects
        x = model.prepare_tokens_with_masks(img)

        # Forward through all blocks except last, then extract attention from last
        for i, blk in enumerate(model.blocks[:-1]):
            x = blk(x)

        # Last block: extract attention
        last_block = model.blocks[-1]
        # Get attention from last block
        x_norm = last_block.norm1(x)
        B, N, C = x_norm.shape
        qkv = last_block.attn.qkv(x_norm).reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (C // last_block.attn.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)  # (B, heads, N, N)

        # CLS-Token Attention auf alle Patches (ohne CLS selbst)
        # Beim DINOv2 ViT-S/14: Token 0 = CLS, Token 1..N-1 = Patches
        # Optional: auch register tokens (bei DINOv2 v2)
        n_register = getattr(model, 'num_register_tokens', 0)
        cls_attn = attn[0, :, 0, 1 + n_register:]  # (heads, num_patches)

        # Patch-Tokens aus dem Output
        x_out = last_block(x)  # finish the block
        patch_tokens_out = x_out[0, 1 + n_register:]  # (num_patches, emb_dim)

    # DINOv2-Features über forward_features für die PCA-Analyse
    with torch.no_grad():
        features = model.forward_features(img)
        patch_tokens = features["x_norm_patchtokens"][0]  # (num_patches, emb_dim)

    patch_size = model.patch_size
    h_patches = img.shape[2] // patch_size
    w_patches = img.shape[3] // patch_size

    cls_attn = cls_attn.reshape(-1, h_patches, w_patches)  # (heads, h, w)

    return cls_attn.cpu(), patch_tokens.cpu(), h_patches, w_patches


def visualize_dino_attention(img_np, cls_attn, h_patches, w_patches, save_path):
    """
    Erzeugt eine Multi-Head Attention-Visualisierung.

    Args:
        img_np: (H, W, 3) uint8
        cls_attn: (num_heads, h_patches, w_patches)
    """
    num_heads = cls_attn.shape[0]

    # Mittlere Attention über alle Heads
    mean_attn = cls_attn.mean(0).numpy()  # (h, w)
    mean_attn_resized = np.array(Image.fromarray(
        (mean_attn * 255 / mean_attn.max()).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

    # Subplot: Original + Mean-Attention + einzelne Heads
    n_show = min(num_heads, 6)
    fig, axes = plt.subplots(2, n_show + 1, figsize=(3 * (n_show + 1), 6))

    # Obere Reihe: Heatmaps
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original",  fontweight='bold')
    axes[0, 0].axis("off")

    axes[1, 0].imshow(overlay_heatmap(img_np, mean_attn_resized, alpha=0.55))
    axes[1, 0].set_title("Mean Attention\n(Overlay)", )
    axes[1, 0].axis("off")

    for i in range(n_show):
        head_attn = cls_attn[i].numpy()
        head_resized = np.array(Image.fromarray(
            (head_attn * 255 / (head_attn.max() + 1e-8)).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

        axes[0, i + 1].imshow(head_attn, cmap="inferno", interpolation="nearest")
        axes[0, i + 1].set_title(f"Head {i}", )
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(overlay_heatmap(img_np, head_resized, alpha=0.55))
        axes[1, i + 1].set_title(f"Head {i}\n(Overlay)", )
        axes[1, i + 1].axis("off")

    fig.suptitle("DINOv2 Self-Attention (CLS → Patches, letzte Schicht)",  fontweight='bold')
    plt.tight_layout()
    save_ma_figure(fig, os.path.splitext(save_path)[0])
    plt.close(fig)
    print(f"  ✓ DINOv2 Attention Map → {save_path}")



# =====================================================================
# Hilfsfunktion: FG/BG-Trennung + K-Means Segmentierung + Per-Cluster PCA
# =====================================================================

# Distinkte Cluster-Farben (ColorBrewer Set1)
CLUSTER_COLORS = np.array([
    [228,  26,  28],   # rot
    [ 55, 126, 184],   # blau
    [ 77, 175,  74],   # grün
    [152,  78, 163],   # lila
    [255, 127,   0],   # orange
    [255, 255,  51],   # gelb
    [166,  86,  40],   # braun
    [247, 129, 191],   # pink
], dtype=np.uint8)


def _otsu_threshold(values_uint8):
    """Otsu-Threshold auf uint8-Array. Gibt Schwellwert zurück."""
    hist, _ = np.histogram(values_uint8, bins=256, range=(0, 256))
    total = values_uint8.size
    sum_all = np.sum(np.arange(256) * hist)
    sum_bg, w_bg, max_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def compute_segmented_pca(patch_tokens, h_patches, w_patches,
                          n_pca_pre=30, n_fg_clusters=3, n_components=3):
    """
    Segmentierungs-basierte PCA nach Amir et al. 2022:

      1. PCA(1) → Otsu-Threshold → FG/BG-Trennung
      2. PCA(n_pca_pre) Dimensionsreduktion der FG-Tokens
      3. K-Means auf PCA-reduzierten FG-Features
      4. PCA(3) pro Cluster → Intra-Cluster RGB

    Returns:
        fg_mask:         (num_patches,) bool
        labels:          (num_patches,) int (-1 = BG)
        seg_map:         (h, w, 3) uint8 – Cluster-Farben (BG = grau)
        per_cluster_pca: (h, w, 3) float [0,1] – PCA pro Cluster (BG = dunkelgrau)
        cluster_info:    dict {cluster_id: {n_patches, var_ratios}}
        n_clusters_used: int – tatsächliche Anzahl Cluster
    """
    tokens = patch_tokens.numpy() if isinstance(patch_tokens, torch.Tensor) else patch_tokens
    n_patches = tokens.shape[0]

    # ── Schritt 1: FG/BG-Trennung via PC1 + Otsu ──
    pca1 = PCA(n_components=1)
    pc1 = pca1.fit_transform(tokens)[:, 0]
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    pc1_uint8 = (pc1_norm * 255).astype(np.uint8)
    threshold = _otsu_threshold(pc1_uint8)

    fg_mask = pc1_uint8 > threshold
    if fg_mask.sum() > len(fg_mask) // 2:
        fg_mask = ~fg_mask

    fg_indices = np.where(fg_mask)[0]
    if len(fg_indices) < n_components + 1:
        fg_indices = np.arange(n_patches)
        fg_mask[:] = True

    # ── Schritt 2: PCA-Vorreduktion der FG-Tokens ──
    fg_tokens = tokens[fg_indices]
    n_pca = min(n_pca_pre, fg_tokens.shape[0] - 1, fg_tokens.shape[1])
    pca_pre = PCA(n_components=n_pca)
    fg_reduced = pca_pre.fit_transform(fg_tokens)

    # ── Schritt 3: K-Means auf PCA-reduzierten FG-Features ──
    from sklearn.metrics import silhouette_score
    n_fg = len(fg_indices)
    best_k, best_sil = 2, -1
    max_k = min(n_fg_clusters + 2, n_fg // 3, 8)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(fg_reduced)
        sil = silhouette_score(fg_reduced, labs)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    fg_labels = kmeans.fit_predict(fg_reduced)

    # Labels für alle Patches (-1 = BG)
    labels = np.full(n_patches, -1, dtype=int)
    labels[fg_indices] = fg_labels

    # ── Segmentierungsmaske ──
    bg_color = np.array([180, 180, 180], dtype=np.uint8)
    seg_flat = np.tile(bg_color, (n_patches, 1))
    seg_flat[fg_indices] = CLUSTER_COLORS[fg_labels % len(CLUSTER_COLORS)]
    seg_map = seg_flat.reshape(h_patches, w_patches, 3)

    # ── Schritt 4: PCA(3) pro Cluster ──
    per_cluster_pca = np.full((n_patches, n_components), 0.3)
    cluster_info = {}

    for ci in range(best_k):
        mask = labels == ci
        cluster_tokens = tokens[mask]
        info = {"n_patches": int(mask.sum()), "var_ratios": None}

        if cluster_tokens.shape[0] < n_components + 1:
            cluster_info[ci] = info
            continue

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(cluster_tokens)
        info["var_ratios"] = pca.explained_variance_ratio_

        for c in range(n_components):
            col = pca_result[:, c]
            pca_result[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)

        per_cluster_pca[mask] = pca_result
        cluster_info[ci] = info

    per_cluster_pca = per_cluster_pca.reshape(h_patches, w_patches, n_components)
    return fg_mask, labels, seg_map, per_cluster_pca, cluster_info, best_k


# =====================================================================
# 2. DINOv2  PCA Feature Map (Segmentierungs-basiert)
# =====================================================================

def visualize_dino_pca(img_np, patch_tokens, h_patches, w_patches, save_path,
                       n_components=3):
    """
    Segmentierungs-basierte PCA der DINOv2 Patch-Tokens.

    Verfahren (nach Amir et al. 2022):
      1. PC1 → Otsu-Threshold → Foreground/Background-Trennung
      2. PCA(30) Dimensionsreduktion auf FG-Tokens
      3. K-Means in PCA-Raum (auto-k via Silhouette)
      4. Per-Cluster PCA(3) → RGB-Farbkodierung

    Zeigt:
      Oben:   Original | Naive PCA | FG-Maske | FG-PCA (alle FG)
      Unten:  Segmentierung Overlay | Einzelne Cluster-Masken
    """
    tokens = patch_tokens.numpy() if isinstance(patch_tokens, torch.Tensor) else patch_tokens

    # ── Naive PCA (alle Patches, als Referenz) ──
    pca_all = PCA(n_components=n_components)
    pca_all_result = pca_all.fit_transform(tokens)
    for c in range(n_components):
        col = pca_all_result[:, c]
        pca_all_result[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    pca_all_img = pca_all_result.reshape(h_patches, w_patches, n_components)

    # ── FG/BG + K-Means + Per-Cluster PCA ──
    fg_mask, labels, seg_map, per_cluster_pca, cluster_info, n_k = compute_segmented_pca(
        tokens, h_patches, w_patches, n_components=n_components
    )

    # ── FG-only PCA (alle FG-Patches zusammen) ──
    fg_indices = np.where(fg_mask)[0]
    pca_fg = PCA(n_components=n_components)
    fg_pca_result = pca_fg.fit_transform(tokens[fg_indices])
    for c in range(n_components):
        col = fg_pca_result[:, c]
        fg_pca_result[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    fg_pca_img = np.full((len(tokens), n_components), 0.3)
    fg_pca_img[fg_indices] = fg_pca_result
    fg_pca_img = fg_pca_img.reshape(h_patches, w_patches, n_components)

    # ── Resize ──
    fg_mask_2d = fg_mask.reshape(h_patches, w_patches).astype(np.uint8) * 255
    mask_resized = np.array(Image.fromarray(fg_mask_2d).resize(
        (img_np.shape[1], img_np.shape[0]), Image.NEAREST))
    seg_resized = np.array(Image.fromarray(seg_map).resize(
        (img_np.shape[1], img_np.shape[0]), Image.NEAREST))
    fg_pca_resized = np.array(Image.fromarray(
        (fg_pca_img * 255).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR))

    # ── Plot: 2 Reihen ──
    n_cols = max(4, n_k + 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))

    # Obere Reihe: Original | Naive PCA | FG-Maske | FG-PCA
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original", fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pca_all_img, interpolation="nearest")
    var_all = pca_all.explained_variance_ratio_
    axes[0, 1].set_title(
        f"Naive PCA (alle Patches)\nPC1={var_all[0]:.0%} PC2={var_all[1]:.0%} PC3={var_all[2]:.0%}")
    axes[0, 1].axis("off")

    n_fg = fg_mask.sum()
    axes[0, 2].imshow(mask_resized, cmap="gray")
    axes[0, 2].set_title(
        f"FG-Maske (PC1+Otsu)\n{n_fg}/{len(fg_mask)} = {n_fg/len(fg_mask):.0%} Foreground")
    axes[0, 2].axis("off")

    fg_overlay = (0.45 * fg_pca_resized.astype(float) +
                  0.55 * img_np.astype(float)).clip(0, 255).astype(np.uint8)
    axes[0, 3].imshow(fg_overlay)
    var_fg = pca_fg.explained_variance_ratio_
    axes[0, 3].set_title(
        f"FG-PCA (Overlay)\nPC1={var_fg[0]:.0%} PC2={var_fg[1]:.0%} PC3={var_fg[2]:.0%}")
    axes[0, 3].axis("off")

    for ci in range(4, n_cols):
        axes[0, ci].axis("off")

    # Untere Reihe: Segmentierung Overlay + einzelne Cluster
    seg_overlay = (0.45 * seg_resized.astype(float) +
                   0.55 * img_np.astype(float)).clip(0, 255).astype(np.uint8)
    axes[1, 0].imshow(seg_overlay)
    axes[1, 0].set_title(f"K-Means Segmentierung\n(k={n_k}, auto via Silhouette)")
    axes[1, 0].axis("off")

    for ci in range(min(n_k, n_cols - 1)):
        mask_ci = (labels == ci).reshape(h_patches, w_patches).astype(np.uint8) * 255
        mask_ci_resized = np.array(Image.fromarray(mask_ci).resize(
            (img_np.shape[1], img_np.shape[0]), Image.NEAREST))
        colored = img_np.copy()
        hi_mask = mask_ci_resized > 128
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        colored[hi_mask] = (
            0.5 * img_np[hi_mask].astype(float) +
            0.5 * color.astype(float)
        ).clip(0, 255).astype(np.uint8)

        info = cluster_info.get(ci, {})
        n_p = info.get("n_patches", 0)
        var_str = ""
        vr = info.get("var_ratios")
        if vr is not None:
            var_str = f"\nVar: {vr[0]:.0%}/{vr[1]:.0%}/{vr[2]:.0%}"
        axes[1, ci + 1].imshow(colored)
        axes[1, ci + 1].set_title(f"Cluster {ci} ({n_p} Patches){var_str}")
        axes[1, ci + 1].axis("off")

    for ci in range(n_k, n_cols - 1):
        axes[1, ci + 1].axis("off")

    fig.suptitle(
        f"DINOv2 PCA Feature Map ({h_patches}×{w_patches} Patches) "
        f"— FG/BG-Trennung + K-Means (k={n_k}) + Per-Cluster PCA",
        fontweight='bold'
    )
    plt.tight_layout()
    save_ma_figure(fig, os.path.splitext(save_path)[0])
    plt.close(fig)
    print(f"  ✓ DINOv2 PCA Feature Map → {save_path}")


# =====================================================================
# 3. DINOv2  Feature Similarity Map
# =====================================================================

def visualize_dino_similarity(img_np, patch_tokens, h_patches, w_patches, save_path,
                               ref_positions=None):
    """
    Cosine-Similarity von ausgewählten Referenz-Patches zu allen anderen.
    Zeigt, welche Bildregionen semantisch ähnlich sind.

    Args:
        ref_positions: Liste von (row, col) Patch-Positionen. Default: Würfel-Mitte + EEF + Hintergrund
    """
    tokens = patch_tokens.numpy()  # (num_patches, emb_dim)

    if ref_positions is None:
        # Automatische Referenz-Patches: Mitte, oben-links, unten-Mitte
        ref_positions = [
            (h_patches // 2, w_patches // 2),     # Mitte (Würfel/Roboter)
            (h_patches // 4, w_patches // 4),      # Oben-links (Hintergrund)
            (3 * h_patches // 4, w_patches // 2),   # Unten-Mitte (Tisch)
            (h_patches // 4, 3 * w_patches // 4),   # Oben-rechts
        ]

    num_refs = len(ref_positions)
    fig, axes = plt.subplots(2, num_refs + 1, figsize=(4 * (num_refs + 1), 8))

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original",  fontweight='bold')
    # Zeichne Referenz-Positionen als Punkte
    patch_h = img_np.shape[0] / h_patches
    patch_w = img_np.shape[1] / w_patches
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for idx, (r, c) in enumerate(ref_positions):
        y = (r + 0.5) * patch_h
        x = (c + 0.5) * patch_w
        axes[0, 0].plot(x, y, 'o', color=colors[idx % len(colors)],
                        markersize=10, markeredgecolor='white', markeredgewidth=2)
        axes[0, 0].text(x + 5, y - 5, f"R{idx}", color=colors[idx % len(colors)],
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Tokens normalisieren für Cosine-Similarity
    tokens_norm = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)

    for idx, (r, c) in enumerate(ref_positions):
        patch_idx = r * w_patches + c
        ref_token = tokens_norm[patch_idx:patch_idx + 1]  # (1, emb_dim)

        # Cosine-Similarity
        similarity = (tokens_norm @ ref_token.T).squeeze()  # (num_patches,)
        sim_map = similarity.reshape(h_patches, w_patches)

        # Patch-Auflösung
        axes[0, idx + 1].imshow(sim_map, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0,
                                interpolation="nearest")
        axes[0, idx + 1].set_title(f"Ref {idx} ({r},{c})\nPatch-Auflösung",
color=colors[idx % len(colors)])
        axes[0, idx + 1].axis("off")

        # Overlay auf Originalbild
        sim_resized = np.array(Image.fromarray(
            ((sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

        axes[1, idx + 1].imshow(overlay_heatmap(img_np, sim_resized, alpha=0.55, cmap="RdYlBu_r"))
        axes[1, idx + 1].set_title(f"Ref {idx} (Overlay)", )
        axes[1, idx + 1].axis("off")

    fig.suptitle("DINOv2 Feature-Ähnlichkeit (Cosine Similarity pro Referenz-Patch)",
                 fontweight='bold')
    plt.tight_layout()
    save_ma_figure(fig, os.path.splitext(save_path)[0])
    plt.close(fig)
    print(f"  ✓ DINOv2 Similarity Map → {save_path}")


# =====================================================================
# 4. ViT Predictor  Attention Maps
# =====================================================================

def extract_vit_attention(model, obs_dict, act, device):
    """
    Extrahiert Attention-Maps aus dem ViT Predictor.
    Verwendet Hooks auf die Attention-Module.

    Returns:
        attn_maps: Liste von (heads, seq_len, seq_len) pro Layer
    """
    model.eval()
    attn_maps = []

    def make_hook(storage):
        def hook_fn(module, input, output):
            x = module.norm(input[0])
            B, T, C = x.size()
            qkv = module.to_qkv(x).chunk(3, dim=-1)
            q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=module.heads) for t in qkv]
            dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
            # Causal Mask anwenden — identisch zu Attention.forward() in models/vit.py
            dots = dots.masked_fill(module.bias[:, :, :T, :T] == 0, float("-inf"))
            attn = module.attend(dots)
            storage.append(attn[0].detach().cpu())  # (heads, seq, seq)
        return hook_fn

    # Register hooks on all Attention modules in the predictor
    hooks = []
    for layer_idx, (attn_module, _) in enumerate(model.predictor.transformer.layers):
        h = attn_module.register_forward_hook(make_hook(attn_maps))
        hooks.append(h)

    with torch.no_grad():
        obs_device = {
            'visual': obs_dict['visual'].to(device),
            'proprio': obs_dict['proprio'].to(device),
        }
        act_device = act.to(device)
        z = model.encode(obs_device, act_device)
        z_src = z[:, :model.num_hist, :, :]
        _ = model.predict(z_src)

    for h in hooks:
        h.remove()

    return attn_maps


def visualize_vit_attention(img_np, attn_maps, num_hist, h_patches, w_patches,
                            save_path, frame_idx_in_seq=0, sampled_indices=None):
    """
    Visualisiert die Attention-Maps des ViT Predictors.

    Args:
        attn_maps: Liste von (heads, seq_len, seq_len) pro Layer
        frame_idx_in_seq: Welcher Frame im Kontext visualisiert wird
    """
    n_layers = len(attn_maps)
    if n_layers == 0:
        print("  ⚠ Keine ViT Attention Maps extrahiert.")
        return

    # Letzten Layer verwenden
    last_attn = attn_maps[-1]  # (heads, seq, seq)
    num_heads = last_attn.shape[0]
    total_tokens = last_attn.shape[1]
    patches_per_frame = h_patches * w_patches
    # Bei concat_dim=0: +2 Tokens (proprio + action) pro Frame
    # Bei concat_dim=1: gleiche Anzahl Patches
    tokens_per_frame = total_tokens // num_hist

    # Mittlere Attention über alle Heads
    mean_attn = last_attn.mean(0).numpy()  # (seq, seq)

    # Zeige Attention Matrix + pro-Frame Self-Attention
    n_show_heads = min(num_heads, 4)
    fig, axes = plt.subplots(2, n_show_heads + 2, figsize=(4 * (n_show_heads + 2), 8))

    # Original-Bild
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Frame",  fontweight='bold')
    axes[0, 0].axis("off")

    # Volle Attention Matrix (Mean)
    axes[1, 0].imshow(mean_attn, cmap="viridis", aspect="auto")
    axes[1, 0].set_title("Mean Attn Matrix\n(alle Tokens)", )
    axes[1, 0].set_xlabel("Key")
    axes[1, 0].set_ylabel("Query")

    # Attention des ausgewählten Kontext-Frames auf alle Frames
    start_frame = frame_idx_in_seq * tokens_per_frame
    end_frame = (frame_idx_in_seq + 1) * tokens_per_frame

    # Self-Attention innerhalb des ausgewählten Frames
    last_frame_self_attn = mean_attn[start_frame:end_frame, start_frame:end_frame]

    # Nur visulle Patches (ohne proprio/action Token falls concat_dim=0)
    visual_patches = min(patches_per_frame, last_frame_self_attn.shape[0])

    if visual_patches == patches_per_frame:
        self_attn_visual = last_frame_self_attn[:patches_per_frame, :patches_per_frame]
        # Summiere Attention pro Query-Patch
        attn_sum = self_attn_visual.sum(axis=0).reshape(h_patches, w_patches)
    else:
        attn_sum = last_frame_self_attn.sum(axis=0)[:patches_per_frame].reshape(h_patches, w_patches)

    axes[0, 1].imshow(attn_sum, cmap="inferno", interpolation="nearest")
    frame_label = f"Kontext-Frame {frame_idx_in_seq + 1}/{num_hist}"
    if sampled_indices is not None and frame_idx_in_seq < len(sampled_indices):
        frame_label = f"Episode-Frame {sampled_indices[frame_idx_in_seq]}\nKontext-Pos {frame_idx_in_seq + 1}/{num_hist}"
    axes[0, 1].set_title(f"{frame_label}\nSelf-Attn (Summe)", )
    axes[0, 1].axis("off")

    attn_sum_resized = np.array(Image.fromarray(
        ((attn_sum - attn_sum.min()) / (attn_sum.max() - attn_sum.min() + 1e-8) * 255).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0
    axes[1, 1].imshow(overlay_heatmap(img_np, attn_sum_resized, alpha=0.55))
    axes[1, 1].set_title("Self-Attn\n(Overlay)", )
    axes[1, 1].axis("off")

    # Pro-Head Self-Attention für letzten Frame
    for hi in range(n_show_heads):
        head_attn = attn_maps[-1][hi].numpy()
        head_last = head_attn[start_frame:end_frame, start_frame:end_frame]
        vp = min(patches_per_frame, head_last.shape[0])
        head_sum = head_last.sum(axis=0)[:vp].reshape(h_patches, w_patches)

        axes[0, hi + 2].imshow(head_sum, cmap="inferno", interpolation="nearest")
        axes[0, hi + 2].set_title(f"Head {hi}\n(Patch-Auflösung)", )
        axes[0, hi + 2].axis("off")

        head_resized = np.array(Image.fromarray(
            ((head_sum - head_sum.min()) / (head_sum.max() - head_sum.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0
        axes[1, hi + 2].imshow(overlay_heatmap(img_np, head_resized, alpha=0.55))
        axes[1, hi + 2].set_title(f"Head {hi}\n(Overlay)", )
        axes[1, hi + 2].axis("off")

    fig.suptitle("ViT Predictor Attention (letzte Schicht, ausgewählter Kontext-Frame)",
                 fontweight='bold')
    plt.tight_layout()
    save_ma_figure(fig, os.path.splitext(save_path)[0])
    plt.close(fig)
    print(f"  ✓ ViT Predictor Attention → {save_path}")


# =====================================================================
# 5. VQ-VAE Rekonstruktion
# =====================================================================

def visualize_reconstruction(model, obs_dict, act, device, img_np, save_path):
    """VQ-VAE Decoder Rekonstruktion vs. Original."""
    if model.decoder is None:
        print("  ⚠ Kein Decoder vorhanden — Rekonstruktion übersprungen.")
        return

    model.eval()
    with torch.no_grad():
        obs_device = {
            'visual': obs_dict['visual'].to(device),
            'proprio': obs_dict['proprio'].to(device),
        }
        act_device = act.to(device)

        z = model.encode(obs_device, act_device)
        obs_recon, _ = model.decode(z)
        visual_recon = obs_recon['visual']  # (1, T, 3, H, W)

    n_frames = visual_recon.shape[1]
    n_show = min(n_frames, 5)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        orig = denorm_image(obs_dict['visual'][0, i])
        recon = denorm_image(visual_recon[0, i].cpu())

        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"Original t={i}", )
        axes[0, i].axis("off")

        axes[1, i].imshow(recon)
        mse = ((obs_dict['visual'][0, i] - visual_recon[0, i].cpu()) ** 2).mean().item()
        axes[1, i].set_title(f"Rekonstruktion t={i}\nMSE={mse:.4f}", )
        axes[1, i].axis("off")

    fig.suptitle("VQ-VAE Decoder: Original vs. Rekonstruktion",  fontweight='bold')
    plt.tight_layout()
    save_ma_figure(fig, os.path.splitext(save_path)[0])
    plt.close(fig)
    print(f"  ✓ VQ-VAE Rekonstruktion → {save_path}")


# =====================================================================
# 6. Zusammenfassung (Summary Grid)
# =====================================================================

def create_summary(img_np, cls_attn, patch_tokens, h_patches, w_patches, save_path):
    """Kompakte Zusammenfassung: Original + Attention + Segmentierung + Similarity."""
    tokens = patch_tokens.numpy() if isinstance(patch_tokens, torch.Tensor) else patch_tokens

    # Mean Attention
    mean_attn = cls_attn.mean(0).numpy()
    mean_attn_resized = np.array(Image.fromarray(
        (mean_attn * 255 / (mean_attn.max() + 1e-8)).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

    # FG/BG + K-Means Segmentierung
    _, _, seg_map, _, _, n_k = compute_segmented_pca(
        tokens, h_patches, w_patches
    )
    seg_resized = np.array(Image.fromarray(seg_map).resize(
        (img_np.shape[1], img_np.shape[0]), Image.NEAREST))
    seg_overlay = (0.45 * seg_resized.astype(float) +
                   0.55 * img_np.astype(float)).clip(0, 255).astype(np.uint8)

    # Similarity: Mitte
    tokens_norm = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)
    center_idx = (h_patches // 2) * w_patches + w_patches // 2
    ref_token = tokens_norm[center_idx:center_idx + 1]
    sim = (tokens_norm @ ref_token.T).squeeze().reshape(h_patches, w_patches)
    sim_resized = np.array(Image.fromarray(
        ((sim - sim.min()) / (sim.max() - sim.min() + 1e-8) * 255).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(overlay_heatmap(img_np, mean_attn_resized, alpha=0.55))
    axes[1].set_title("DINOv2 Attention\n(CLS → Patches)")
    axes[1].axis("off")

    axes[2].imshow(seg_overlay)
    axes[2].set_title(f"K-Means Segmentierung\n(FG k={n_k}, Overlay)")
    axes[2].axis("off")

    axes[3].imshow(overlay_heatmap(img_np, sim_resized, alpha=0.55, cmap="RdYlBu_r"))
    axes[3].set_title("Feature-Ähnlichkeit\n(Referenz: Bildmitte)")
    axes[3].axis("off")

    fig.suptitle("DINOv2 Feature-Visualisierung — Franka Cube Stacking",
                 fontweight='bold')
    plt.tight_layout()
    save_ma_figure(fig, os.path.splitext(save_path)[0])
    plt.close(fig)
    print(f"  ✓ Summary → {save_path}")


# =====================================================================
# Hauptprogramm
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 & ViT Feature-Visualisierung für die Masterarbeit"
    )
    parser.add_argument("--model_name", type=str, default=None,
                        help="Modell-Ordner unter outputs/, z.B. '260305/07-56'")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Pfad zu einem einzelnen Bild (nur DINOv2-Visualisierungen)")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Episode-Index im Dataset")
    parser.add_argument("--frame_idx", type=int, default=5,
                        help="Frame-Index innerhalb der Episode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Ausgabe-Ordner (default: feature_visualizations/<model_name>/)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda wenn verfügbar)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    seed(42)
    print(f"Device: {device}")

    # ── Einzelbild-Modus (nur DINOv2) ──
    if args.image_path is not None:
        print(f"\n[Einzelbild-Modus] Lade: {args.image_path}")
        img_tensor, img_np = load_single_image(args.image_path)

        # DINOv2 laden (pretrained, ohne trainiertes Modell)
        from models.dino import DinoV2Encoder
        encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
        encoder.to(device)
        encoder.eval()

        out_dir = args.output_dir or os.path.join(DINO_WM_DIR, "feature_visualizations", "single_image")
        os.makedirs(out_dir, exist_ok=True)
        basename = Path(args.image_path).stem

        cls_attn, patch_tokens, hp, wp = extract_dino_attention(encoder, img_tensor, device)
        visualize_dino_attention(img_np, cls_attn, hp, wp,
                                os.path.join(out_dir, f"dino_attention_{basename}.png"))
        visualize_dino_pca(img_np, patch_tokens, hp, wp,
                           os.path.join(out_dir, f"dino_pca_{basename}.png"))
        visualize_dino_similarity(img_np, patch_tokens, hp, wp,
                                  os.path.join(out_dir, f"dino_similarity_{basename}.png"))
        create_summary(img_np, cls_attn, patch_tokens, hp, wp,
                       os.path.join(out_dir, f"summary_{basename}.png"))
        print(f"\n✅ Fertig! Ausgabe in: {out_dir}")
        return

    # ── Modell-Modus (DINOv2 + ViT + VQ-VAE) ──
    if args.model_name is None:
        parser.error("Entweder --model_name oder --image_path muss angegeben werden.")

    model_path = os.path.join(DINO_WM_DIR, "outputs", args.model_name)
    print(f"\n[Modell-Modus] Lade Config aus: {model_path}")

    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    # Dataset laden (roh: volle Episoden)
    print("[1/4] Lade Dataset...")
    datasets, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dset["valid"]          # raw: TrajSubset, returns full episodes
    print(f"  Validierungs-Episoden: {len(dset)}")

    # Modell laden
    print("[2/4] Lade Modell...")
    torch.cuda.empty_cache()
    model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
    model.eval()

    num_hist = model_cfg.num_hist
    frameskip = model_cfg.frameskip

    # Ausgabe-Ordner
    out_dir = args.output_dir or os.path.join(
        DINO_WM_DIR, "feature_visualizations", args.model_name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # ── Daten aus Dataset laden ──
    ep_idx = min(args.episode_idx, len(dset) - 1)
    print(f"[3/4] Episode {ep_idx}, Frame {args.frame_idx}")

    # TrajSubset returns (obs_dict, act, state, info)
    obs_raw, actions, _, _ = dset[ep_idx]
    obs_visual = obs_raw["visual"]           # (T, 3, H, W)
    obs_proprio = obs_raw.get("proprio", None)  # (T, 3)

    # Frame auswählen
    fr_idx = min(args.frame_idx, obs_visual.shape[0] - 1)
    tag = f"ep{ep_idx:03d}_fr{fr_idx:02d}"
    created_files = [f"original_{tag}.png"]  # Original bleibt PNG (PIL-Bild)

    # Einzelbild für DINOv2-Visualisierungen
    single_frame = obs_visual[fr_idx:fr_idx + 1]  # (1, 3, H, W)
    img_np = denorm_image(single_frame[0])

    # Original speichern
    Image.fromarray(img_np).save(os.path.join(out_dir, f"original_{tag}.png"))
    print(f"  Original gespeichert: original_{tag}.png")

    # ── DINOv2 Visualisierungen ──
    print("[4/4] Erzeuge Visualisierungen...")

    encoder = model.encoder
    cls_attn, patch_tokens, hp, wp = extract_dino_attention(
        encoder, single_frame, device, encoder_transform=model.encoder_transform)

    visualize_dino_attention(img_np, cls_attn, hp, wp,
                            os.path.join(out_dir, f"dino_attention_{tag}.png"))
    created_files.append(f"dino_attention_{tag}.svg")
    visualize_dino_pca(img_np, patch_tokens, hp, wp,
                       os.path.join(out_dir, f"dino_pca_{tag}.png"))
    created_files.append(f"dino_pca_{tag}.svg")
    visualize_dino_similarity(img_np, patch_tokens, hp, wp,
                              os.path.join(out_dir, f"dino_similarity_{tag}.png"))
    created_files.append(f"dino_similarity_{tag}.svg")
    create_summary(img_np, cls_attn, patch_tokens, hp, wp,
                   os.path.join(out_dir, f"summary_{tag}.png"))
    created_files.append(f"summary_{tag}.svg")

    # ── ViT Predictor Attention + VQ-VAE Rekonstruktion ──
    if model.predictor is not None and num_hist > 0:
        context = build_aligned_context_window(
            obs_visual=obs_visual,
            obs_proprio=obs_proprio,
            actions=actions,
            num_hist=num_hist,
            frameskip=frameskip,
            target_frame_idx=fr_idx,
        )

        context_visual = context["context_visual"].float()
        context_proprio = context["context_proprio"].float()
        context_actions = context["context_actions"].float()
        frame_idx_in_seq = context["frame_idx_in_seq"]
        actual_frame_idx = context["actual_frame_idx"]
        sampled_indices = context["sampled_indices"]

        if actual_frame_idx != fr_idx:
            print(
                f"  ⚠ ViT-Kontext auf Frame {actual_frame_idx} verschoben, "
                f"weil für Frame {fr_idx} kein vollständiges WM-Fenster existiert."
            )

        ref_img_np = denorm_image(context_visual[0, frame_idx_in_seq])
        vit_tag = f"ep{ep_idx:03d}_fr{actual_frame_idx:02d}"

        obs_dict = {
            "visual": context_visual,
            "proprio": context_proprio,
        }

        vit_attn = extract_vit_attention(model, obs_dict, context_actions, device)
        visualize_vit_attention(
            ref_img_np,
            vit_attn,
            num_hist,
            hp,
            wp,
            os.path.join(out_dir, f"vit_attention_{vit_tag}.png"),
            frame_idx_in_seq=frame_idx_in_seq,
            sampled_indices=sampled_indices,
        )
        created_files.append(f"vit_attention_{vit_tag}.svg")

        # ── VQ-VAE Rekonstruktion ──
        visualize_reconstruction(model, obs_dict, context_actions,
                                 device, ref_img_np,
                                 os.path.join(out_dir, f"reconstruction_{vit_tag}.png"))
        created_files.append(f"reconstruction_{vit_tag}.svg")

    print(f"\n✅ Fertig! Alle Visualisierungen in: {out_dir}")
    print(f"   Dateien:")
    for f in created_files:
        print(f"     • {f}")


if __name__ == "__main__":
    main()
