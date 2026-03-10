#!/usr/bin/env python3
"""
Feature-Visualisierung für DINOv2 Encoder und ViT Predictor.

Erzeugt für die Masterarbeit anschauliche Bilder, die zeigen, wie das
DINO World Model visuelle Features in Franka-Cube-Stacking-Szenen erkennt.

Visualisierungen:
  1. DINOv2 Attention Maps       – Self-Attention der letzten Encoder-Schicht
  2. DINOv2 PCA Feature Maps     – PCA auf Patch-Token → Falschfarben-Überlagerung
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

# ── Projekt-Root auf sys.path ──
DINO_WM_DIR = os.path.dirname(os.path.abspath(__file__))
if DINO_WM_DIR not in sys.path:
    sys.path.insert(0, DINO_WM_DIR)

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
    axes[0, 0].set_title("Original", fontsize=10, fontweight='bold')
    axes[0, 0].axis("off")

    axes[1, 0].imshow(overlay_heatmap(img_np, mean_attn_resized, alpha=0.55))
    axes[1, 0].set_title("Mean Attention\n(Overlay)", fontsize=9)
    axes[1, 0].axis("off")

    for i in range(n_show):
        head_attn = cls_attn[i].numpy()
        head_resized = np.array(Image.fromarray(
            (head_attn * 255 / (head_attn.max() + 1e-8)).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

        axes[0, i + 1].imshow(head_attn, cmap="inferno", interpolation="nearest")
        axes[0, i + 1].set_title(f"Head {i}", fontsize=9)
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(overlay_heatmap(img_np, head_resized, alpha=0.55))
        axes[1, i + 1].set_title(f"Head {i}\n(Overlay)", fontsize=9)
        axes[1, i + 1].axis("off")

    fig.suptitle("DINOv2 Self-Attention (CLS → Patches, letzte Schicht)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ DINOv2 Attention Map → {save_path}")


# =====================================================================
# 2. DINOv2  PCA Feature Map
# =====================================================================

def visualize_dino_pca(img_np, patch_tokens, h_patches, w_patches, save_path, n_components=3):
    """
    PCA-Reduktion der Patch-Tokens auf 3 Komponenten → RGB-Falschfarbenbild.
    Zeigt semantische Regionen: gleichfarbige Patches = ähnliche Features.
    """
    tokens = patch_tokens.numpy()  # (num_patches, emb_dim)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(tokens)  # (num_patches, 3)

    # Normalisiere jede Komponente auf [0, 1]
    for c in range(n_components):
        col = pca_result[:, c]
        pca_result[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    pca_img = pca_result.reshape(h_patches, w_patches, n_components)
    pca_img_resized = np.array(Image.fromarray(
        (pca_img * 255).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR))

    # Varianz-Erklärung
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(pca_img, interpolation="nearest")
    axes[1].set_title(f"PCA Feature Map\n(Patch-Auflösung {h_patches}×{w_patches})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(pca_img_resized)
    axes[2].set_title("PCA Feature Map\n(bilinear interpoliert)", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(
        f"DINOv2 Patch-Token PCA (Varianz: PC1={var_explained[0]:.1%}, "
        f"PC2={var_explained[1]:.1%}, PC3={var_explained[2]:.1%})",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    axes[0, 0].set_title("Original", fontsize=11, fontweight='bold')
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
                        fontsize=8, fontweight='bold',
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
                                   fontsize=9, color=colors[idx % len(colors)])
        axes[0, idx + 1].axis("off")

        # Overlay auf Originalbild
        sim_resized = np.array(Image.fromarray(
            ((sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

        axes[1, idx + 1].imshow(overlay_heatmap(img_np, sim_resized, alpha=0.55, cmap="RdYlBu_r"))
        axes[1, idx + 1].set_title(f"Ref {idx} (Overlay)", fontsize=9)
        axes[1, idx + 1].axis("off")

    fig.suptitle("DINOv2 Feature-Ähnlichkeit (Cosine Similarity pro Referenz-Patch)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
            attn = dots.softmax(dim=-1)
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
                            save_path, frame_idx_in_seq=0):
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
    axes[0, 0].set_title("Original Frame", fontsize=10, fontweight='bold')
    axes[0, 0].axis("off")

    # Volle Attention Matrix (Mean)
    axes[1, 0].imshow(mean_attn, cmap="viridis", aspect="auto")
    axes[1, 0].set_title("Mean Attn Matrix\n(alle Tokens)", fontsize=9)
    axes[1, 0].set_xlabel("Key")
    axes[1, 0].set_ylabel("Query")

    # Attention des letzten Frames auf alle Frames (Prediction-Attention)
    # Letzter Frame: Query-Tokens am Ende
    start_last_frame = (num_hist - 1) * tokens_per_frame
    end_last_frame = num_hist * tokens_per_frame

    # Self-Attention innerhalb des letzten Frames
    last_frame_self_attn = mean_attn[start_last_frame:end_last_frame,
                                      start_last_frame:end_last_frame]

    # Nur visulle Patches (ohne proprio/action Token falls concat_dim=0)
    visual_patches = min(patches_per_frame, last_frame_self_attn.shape[0])

    if visual_patches == patches_per_frame:
        self_attn_visual = last_frame_self_attn[:patches_per_frame, :patches_per_frame]
        # Summiere Attention pro Query-Patch
        attn_sum = self_attn_visual.sum(axis=0).reshape(h_patches, w_patches)
    else:
        attn_sum = last_frame_self_attn.sum(axis=0)[:patches_per_frame].reshape(h_patches, w_patches)

    axes[0, 1].imshow(attn_sum, cmap="inferno", interpolation="nearest")
    axes[0, 1].set_title("Letzer Frame\nSelf-Attn (Summe)", fontsize=9)
    axes[0, 1].axis("off")

    attn_sum_resized = np.array(Image.fromarray(
        ((attn_sum - attn_sum.min()) / (attn_sum.max() - attn_sum.min() + 1e-8) * 255).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0
    axes[1, 1].imshow(overlay_heatmap(img_np, attn_sum_resized, alpha=0.55))
    axes[1, 1].set_title("Self-Attn\n(Overlay)", fontsize=9)
    axes[1, 1].axis("off")

    # Pro-Head Self-Attention für letzten Frame
    for hi in range(n_show_heads):
        head_attn = attn_maps[-1][hi].numpy()
        head_last = head_attn[start_last_frame:end_last_frame,
                              start_last_frame:end_last_frame]
        vp = min(patches_per_frame, head_last.shape[0])
        head_sum = head_last.sum(axis=0)[:vp].reshape(h_patches, w_patches)

        axes[0, hi + 2].imshow(head_sum, cmap="inferno", interpolation="nearest")
        axes[0, hi + 2].set_title(f"Head {hi}\n(Patch-Auflösung)", fontsize=9)
        axes[0, hi + 2].axis("off")

        head_resized = np.array(Image.fromarray(
            ((head_sum - head_sum.min()) / (head_sum.max() - head_sum.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0
        axes[1, hi + 2].imshow(overlay_heatmap(img_np, head_resized, alpha=0.55))
        axes[1, hi + 2].set_title(f"Head {hi}\n(Overlay)", fontsize=9)
        axes[1, hi + 2].axis("off")

    fig.suptitle("ViT Predictor Attention (letzte Schicht, letzter Kontext-Frame)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
        axes[0, i].set_title(f"Original t={i}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(recon)
        mse = ((obs_dict['visual'][0, i] - visual_recon[0, i].cpu()) ** 2).mean().item()
        axes[1, i].set_title(f"Rekonstruktion t={i}\nMSE={mse:.4f}", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("VQ-VAE Decoder: Original vs. Rekonstruktion", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ VQ-VAE Rekonstruktion → {save_path}")


# =====================================================================
# 6. Zusammenfassung (Summary Grid)
# =====================================================================

def create_summary(img_np, cls_attn, patch_tokens, h_patches, w_patches, save_path):
    """Kompakte Zusammenfassung: Original + Mean-Attention + PCA auf einem Bild."""
    # Mean Attention
    mean_attn = cls_attn.mean(0).numpy()
    mean_attn_resized = np.array(Image.fromarray(
        (mean_attn * 255 / (mean_attn.max() + 1e-8)).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)).astype(float) / 255.0

    # PCA
    tokens = patch_tokens.numpy()
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(tokens)
    for c in range(3):
        col = pca_result[:, c]
        pca_result[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    pca_img = np.array(Image.fromarray(
        (pca_result.reshape(h_patches, w_patches, 3) * 255).astype(np.uint8)
    ).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR))

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
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(overlay_heatmap(img_np, mean_attn_resized, alpha=0.55))
    axes[1].set_title("DINOv2 Attention\n(CLS → Patches)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(pca_img)
    axes[2].set_title("DINOv2 PCA\n(3 Hauptkomponenten)", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(overlay_heatmap(img_np, sim_resized, alpha=0.55, cmap="RdYlBu_r"))
    axes[3].set_title("Feature-Ähnlichkeit\n(Referenz: Bildmitte)", fontsize=11)
    axes[3].axis("off")

    fig.suptitle("DINOv2 Feature-Visualisierung — Franka Cube Stacking",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
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

    # Dataset laden (sliced + raw)
    print("[1/4] Lade Dataset...")
    datasets, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dset["valid"]          # raw: TrajSubset, returns full episodes
    dset_sliced = datasets["valid"]    # sliced: TrajSlicerDataset, ready for model
    print(f"  Validierungs-Episoden: {len(dset)}, Sliced samples: {len(dset_sliced)}")

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
    visualize_dino_pca(img_np, patch_tokens, hp, wp,
                       os.path.join(out_dir, f"dino_pca_{tag}.png"))
    visualize_dino_similarity(img_np, patch_tokens, hp, wp,
                              os.path.join(out_dir, f"dino_similarity_{tag}.png"))
    create_summary(img_np, cls_attn, patch_tokens, hp, wp,
                   os.path.join(out_dir, f"summary_{tag}.png"))

    # ── ViT Predictor Attention + VQ-VAE Rekonstruktion ──
    if model.predictor is not None and num_hist > 0 and len(dset_sliced) > 0:
        # Verwende sliced dataset (Actions sind bereits korrekt konkatiert für frameskip)
        sliced_idx = min(args.episode_idx, len(dset_sliced) - 1)
        sliced_obs, sliced_act, sliced_state = dset_sliced[sliced_idx]
        # sliced_obs: {"visual": (num_hist+num_pred, 3, H, W), "proprio": (num_hist+num_pred, 3)}
        # sliced_act: (num_hist+num_pred, action_dim*frameskip)

        n_frames = sliced_obs["visual"].shape[0]
        context_visual = sliced_obs["visual"][:num_hist].unsqueeze(0)      # (1, num_hist, 3, H, W)
        context_proprio = sliced_obs["proprio"][:num_hist].unsqueeze(0)    # (1, num_hist, 3)
        context_actions = sliced_act[:num_hist].unsqueeze(0)               # (1, num_hist, act_dim*fs)

        # Verwende letztes Frame des Kontexts als Referenzbild
        ref_img_np = denorm_image(sliced_obs["visual"][num_hist - 1])

        obs_dict = {
            "visual": context_visual.float(),
            "proprio": context_proprio.float(),
        }

        vit_attn = extract_vit_attention(model, obs_dict, context_actions.float(), device)
        visualize_vit_attention(ref_img_np, vit_attn, num_hist, hp, wp,
                                os.path.join(out_dir, f"vit_attention_{tag}.png"))

        # ── VQ-VAE Rekonstruktion ──
        visualize_reconstruction(model, obs_dict, context_actions.float(),
                                 device, ref_img_np,
                                 os.path.join(out_dir, f"reconstruction_{tag}.png"))

    print(f"\n✅ Fertig! Alle Visualisierungen in: {out_dir}")
    print(f"   Dateien:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png") and tag in f:
            print(f"     • {f}")


if __name__ == "__main__":
    main()
