#!/usr/bin/env python3
"""
Feature-Visualisierung für DINO WM Planning Inferenz.

Leichtgewichtiges Modul, das die Kernvisualisierungen aus visualize_features.py
für den Einsatz während der Planning-Inferenz extrahiert.

Visualisierungen (alle PNG):
  1. Naive PCA Feature Map     – 3-Komponenten PCA über alle Patches
  2. K-Means Clustering + PCA  – Segmentierung + Per-Cluster PCA (3 Cluster)
  3. DINO Self-Attention       – CLS-Token Attention auf alle Patches

Verwendung in planning_server.py:
    from planning_feature_visualizer import PlanningFeatureVisualizer
    
    visualizer = PlanningFeatureVisualizer(model.encoder, device)
    visualizer.visualize(img_np, out_dir, step_idx, prefix="goal")

Ausgabe-Struktur:
    <out_dir>/
        <prefix>_step<N>_naive_pca.png
        <prefix>_step<N>_kmeans_pca.png
        <prefix>_step<N>_attention.png
"""

import os
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Matplotlib Backend setzen (für headless Server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────
# Farben und Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────

CLUSTER_COLORS = np.array([
    [228,  26,  28],   # rot
    [ 55, 126, 184],   # blau
    [ 77, 175,  74],   # grün
    [152,  78, 163],   # lila
    [255, 127,   0],   # orange
], dtype=np.uint8)


def apply_colormap(heatmap: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Numpy-Heatmap (H, W) float [0,1] → (H, W, 3) uint8 RGB."""
    cm = plt.get_cmap(cmap)
    colored = cm(heatmap)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, cmap: str = "inferno") -> np.ndarray:
    """Überlagert eine Heatmap auf ein RGB-Bild."""
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = apply_colormap(heatmap_norm, cmap)

    if image.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = np.array(Image.fromarray(heatmap_color).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR))

    blended = (alpha * heatmap_color.astype(float) +
               (1 - alpha) * image.astype(float)).clip(0, 255).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────────
# DINO Feature Extraction
# ─────────────────────────────────────────────────────────────────

def extract_dino_features(encoder, img_tensor: torch.Tensor, device: torch.device,
                          encoder_transform=None):
    """
    Extrahiert DINO Patch-Tokens und Self-Attention Maps.
    
    Args:
        encoder: DINOv2 Encoder (model.encoder)
        img_tensor: (1, 3, H, W) normalisiertes Bild [-1, 1]
        device: torch.device
        encoder_transform: Optional encoder transform (model.encoder_transform)
    
    Returns:
        patch_tokens: (num_patches, emb_dim) Feature-Vektoren
        cls_attn: (num_heads, h_patches, w_patches) Attention Maps
        h_patches, w_patches: Patch-Grid Dimensionen
    """
    encoder.eval()
    model = encoder.base_model
    img = img_tensor.to(device)
    
    if encoder_transform is not None:
        img = encoder_transform(img)

    with torch.no_grad():
        # Prepare input
        x = model.prepare_tokens_with_masks(img)

        # Forward through all blocks except last
        for blk in model.blocks[:-1]:
            x = blk(x)

        # Extract attention from last block
        last_block = model.blocks[-1]
        x_norm = last_block.norm1(x)
        B, N, C = x_norm.shape
        qkv = last_block.attn.qkv(x_norm).reshape(
            B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (C // last_block.attn.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)

        # CLS-Token Attention (ohne register tokens)
        n_register = getattr(model, 'num_register_tokens', 0)
        cls_attn = attn[0, :, 0, 1 + n_register:]

        # Features via forward_features
        features = model.forward_features(img)
        patch_tokens = features["x_norm_patchtokens"][0]

    patch_size = model.patch_size
    h_patches = img.shape[2] // patch_size
    w_patches = img.shape[3] // patch_size
    cls_attn = cls_attn.reshape(-1, h_patches, w_patches)

    return patch_tokens.cpu(), cls_attn.cpu(), h_patches, w_patches


# ─────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────

def create_naive_pca_image(patch_tokens: torch.Tensor, h_patches: int, w_patches: int,
                           img_size: tuple = (224, 224), n_components: int = 3,
                           pca_data: dict = None) -> tuple:
    """
    Erzeugt ein Naive PCA Feature-Map Bild mit optionaler PCA-Konsistenz.
    
    Args:
        pca_data: Optional - Dictionary mit gespeichertem PCA-Objekt und Min/Max-Werten.
                  Wenn None, wird PCA neu berechnet und pca_data zurückgegeben.
    
    Returns:
        (image, pca_data): 
            - image: (H, W, 3) uint8 RGB Bild
            - pca_data: Dictionary mit 'pca', 'minmax' für Wiederverwendung
    """
    tokens = patch_tokens.numpy() if isinstance(patch_tokens, torch.Tensor) else patch_tokens
    
    if pca_data is None:
        # Goal-Bild: PCA neu fitten und Normalisierungswerte speichern
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(tokens)
        
        # Min/Max pro Komponente speichern
        minmax = []
        for c in range(n_components):
            col = pca_result[:, c]
            min_val, max_val = col.min(), col.max()
            minmax.append((min_val, max_val))
            pca_result[:, c] = (col - min_val) / (max_val - min_val + 1e-8)
        
        pca_data = {'pca': pca, 'minmax': minmax}
    else:
        # Spätere Bilder: Gespeicherte PCA-Komponenten und Normalisierung verwenden
        pca = pca_data['pca']
        minmax = pca_data['minmax']
        pca_result = pca.transform(tokens)
        
        # Dieselbe Normalisierung wie beim Goal-Bild
        for c in range(n_components):
            min_val, max_val = minmax[c]
            pca_result[:, c] = (pca_result[:, c] - min_val) / (max_val - min_val + 1e-8)
            # Clipping für Werte außerhalb des Goal-Bereichs
            pca_result[:, c] = np.clip(pca_result[:, c], 0, 1)
    
    # Reshape to patch grid
    pca_img = pca_result.reshape(h_patches, w_patches, n_components)
    
    # Resize to original image size
    pca_resized = np.array(Image.fromarray(
        (pca_img * 255).astype(np.uint8)
    ).resize(img_size, Image.NEAREST))
    
    return pca_resized, pca_data


def create_kmeans_pca_image(patch_tokens: torch.Tensor, h_patches: int, w_patches: int,
                            img_np: np.ndarray, n_clusters: int = 3, 
                            n_components: int = 3,
                            goal_data: dict = None) -> tuple:
    """
    Erzeugt K-Means Clustering + Per-Cluster PCA Visualisierung.
    
    Erzeugt PRO CLUSTER ein separates Bild. In jedem Bild ist nur der jeweilige
    Cluster mit PCA-Farben sichtbar, alle anderen Patches sind schwarz.
    
    Args:
        goal_data: Optional - Dictionary mit gespeicherten Goal-Referenzen:
                   'centroids': K-Means Zentroide
                   'pca_all': PCA-Objekt und minmax für Naive PCA
                   'cluster_pcas': {cluster_id: {'pca': ..., 'minmax': ...}}
                   Wenn None, werden diese vom Goal-Bild berechnet.
    
    Returns:
        (grids, goal_data): 
            - grids: Liste von n_clusters Grids, jedes (H*2, W*2, 3) uint8 RGB
            - goal_data: Dictionary mit allen Referenz-Daten für Wiederverwendung
    """
    tokens = patch_tokens.numpy() if isinstance(patch_tokens, torch.Tensor) else patch_tokens
    img_h, img_w = img_np.shape[:2]
    is_goal = goal_data is None
    
    if is_goal:
        goal_data = {'centroids': None, 'pca_all': None, 'cluster_pcas': {}}
    
    # Naive PCA (gleich für alle Grids) - mit Goal-Referenz für Konsistenz
    if is_goal:
        pca_all = PCA(n_components=n_components)
        pca_all_result = pca_all.fit_transform(tokens)
        pca_all_minmax = []
        for c in range(n_components):
            col = pca_all_result[:, c]
            min_val, max_val = col.min(), col.max()
            pca_all_minmax.append((min_val, max_val))
            pca_all_result[:, c] = (col - min_val) / (max_val - min_val + 1e-8)
        goal_data['pca_all'] = {'pca': pca_all, 'minmax': pca_all_minmax}
    else:
        pca_all = goal_data['pca_all']['pca']
        pca_all_minmax = goal_data['pca_all']['minmax']
        pca_all_result = pca_all.transform(tokens)
        for c in range(n_components):
            min_val, max_val = pca_all_minmax[c]
            pca_all_result[:, c] = (pca_all_result[:, c] - min_val) / (max_val - min_val + 1e-8)
            pca_all_result[:, c] = np.clip(pca_all_result[:, c], 0, 1)
    
    pca_all_img = pca_all_result.reshape(h_patches, w_patches, n_components)
    pca_all_resized = np.array(Image.fromarray(
        (pca_all_img * 255).astype(np.uint8)
    ).resize((img_w, img_h), Image.NEAREST))
    
    # K-Means Clustering
    tokens_norm = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)
    
    if is_goal:
        # Neues K-Means: Zentroide berechnen (bei Goal-Bild)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tokens_norm)
        goal_data['centroids'] = kmeans.cluster_centers_
    else:
        # Gespeicherte Zentroide verwenden: Patches dem nächsten Zentroid zuweisen
        centroids = goal_data['centroids']
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        similarities = tokens_norm @ centroids_norm.T  # (n_patches, n_clusters)
        labels = np.argmax(similarities, axis=1)
    
    # Segmentation map (gleich für alle Grids)
    seg_map = CLUSTER_COLORS[labels % len(CLUSTER_COLORS)].reshape(h_patches, w_patches, 3)
    seg_resized = np.array(Image.fromarray(seg_map).resize((img_w, img_h), Image.NEAREST))
    
    # Per-Cluster PCA berechnen (für alle Cluster) - mit Goal-Referenz
    cluster_pca_results = {}
    for ci in range(n_clusters):
        mask = labels == ci
        cluster_tokens = tokens[mask]
        
        if cluster_tokens.shape[0] >= n_components + 1:
            if is_goal:
                # Goal: Neue PCA fitten und speichern
                pca_cl = PCA(n_components=n_components)
                pca_result = pca_cl.fit_transform(cluster_tokens)
                cluster_minmax = []
                for c in range(n_components):
                    col = pca_result[:, c]
                    min_val, max_val = col.min(), col.max()
                    cluster_minmax.append((min_val, max_val))
                    pca_result[:, c] = (col - min_val) / (max_val - min_val + 1e-8)
                goal_data['cluster_pcas'][ci] = {'pca': pca_cl, 'minmax': cluster_minmax}
                cluster_pca_results[ci] = (mask, pca_result)
            else:
                # Später: Gespeicherte PCA-Projektion verwenden
                if ci in goal_data['cluster_pcas']:
                    pca_cl = goal_data['cluster_pcas'][ci]['pca']
                    cluster_minmax = goal_data['cluster_pcas'][ci]['minmax']
                    pca_result = pca_cl.transform(cluster_tokens)
                    for c in range(n_components):
                        min_val, max_val = cluster_minmax[c]
                        pca_result[:, c] = (pca_result[:, c] - min_val) / (max_val - min_val + 1e-8)
                        pca_result[:, c] = np.clip(pca_result[:, c], 0, 1)
                    cluster_pca_results[ci] = (mask, pca_result)
                else:
                    # Fallback: Cluster existierte nicht im Goal
                    cluster_pca_results[ci] = (mask, np.full((mask.sum(), n_components), 0.5))
        else:
            # Fallback: graue Patches wenn zu wenig Tokens
            cluster_pca_results[ci] = (mask, np.full((mask.sum(), n_components), 0.5))
    
    # Erzeuge ein Grid pro Cluster
    grids = []
    for ci in range(n_clusters):
        # Single-Cluster PCA: nur dieser Cluster sichtbar, Rest schwarz
        single_cluster_pca = np.zeros((tokens.shape[0], n_components))
        mask, pca_result = cluster_pca_results[ci]
        single_cluster_pca[mask] = pca_result
        
        single_cluster_img = single_cluster_pca.reshape(h_patches, w_patches, n_components)
        single_cluster_resized = np.array(Image.fromarray(
            (single_cluster_img * 255).astype(np.uint8)
        ).resize((img_w, img_h), Image.NEAREST))
        
        # Create 2x2 grid für diesen Cluster
        grid = np.zeros((img_h * 2, img_w * 2, 3), dtype=np.uint8)
        grid[:img_h, :img_w] = img_np
        grid[:img_h, img_w:] = pca_all_resized
        grid[img_h:, :img_w] = seg_resized
        grid[img_h:, img_w:] = single_cluster_resized
        
        grids.append(grid)
    
    return grids, goal_data


def create_attention_image(cls_attn: torch.Tensor, h_patches: int, w_patches: int,
                           img_np: np.ndarray, n_heads_show: int = 4) -> np.ndarray:
    """
    Erzeugt DINO Self-Attention Visualisierung.
    
    Returns:
        (H, W*cols, 3) uint8 RGB Bild (Original | Mean Attn | Einzelne Heads)
    """
    num_heads = cls_attn.shape[0]
    img_h, img_w = img_np.shape[:2]
    
    # Mean attention
    mean_attn = cls_attn.mean(0).numpy()
    mean_attn_resized = np.array(Image.fromarray(
        (mean_attn * 255 / (mean_attn.max() + 1e-8)).astype(np.uint8)
    ).resize((img_w, img_h), Image.BILINEAR)).astype(float) / 255.0
    
    n_show = min(num_heads, n_heads_show)
    n_cols = n_show + 2  # Original + Mean Overlay + Heads
    
    # Create horizontal grid
    grid = np.zeros((img_h, img_w * n_cols, 3), dtype=np.uint8)
    grid[:, :img_w] = img_np
    grid[:, img_w:2*img_w] = overlay_heatmap(img_np, mean_attn_resized, alpha=0.55)
    
    for i in range(n_show):
        head_attn = cls_attn[i].numpy()
        head_resized = np.array(Image.fromarray(
            (head_attn * 255 / (head_attn.max() + 1e-8)).astype(np.uint8)
        ).resize((img_w, img_h), Image.BILINEAR)).astype(float) / 255.0
        
        col_start = (i + 2) * img_w
        col_end = (i + 3) * img_w
        grid[:, col_start:col_end] = overlay_heatmap(img_np, head_resized, alpha=0.55)
    
    return grid


# ─────────────────────────────────────────────────────────────────
# Main Visualizer Class
# ─────────────────────────────────────────────────────────────────

class PlanningFeatureVisualizer:
    """
    Feature-Visualisierer für Planning Inferenz.
    
    Verwendung:
        visualizer = PlanningFeatureVisualizer(model.encoder, device)
        visualizer.set_encoder_transform(model.encoder_transform)
        visualizer.visualize(img_np, out_dir, step_idx=0, prefix="current")
    """
    
    def __init__(self, encoder, device: torch.device, enabled: bool = True):
        """
        Args:
            encoder: DINO Encoder (model.encoder)
            device: torch.device
            enabled: Ob Visualisierungen erzeugt werden sollen
        """
        self.encoder = encoder
        self.device = device
        self.enabled = enabled
        self.encoder_transform = None
        self.n_clusters = 3
        self.n_heads_show = 4
        # Gespeicherte Goal-Referenzen für konsistente Farben
        self.goal_pca_data = None      # Für Naive PCA
        self.goal_kmeans_data = None   # Für K-Means + Per-Cluster PCA
        
    def set_encoder_transform(self, transform):
        """Setzt den Encoder-Transform (aus model.encoder_transform)."""
        self.encoder_transform = transform
        
    def disable(self):
        """Deaktiviert Visualisierungen."""
        self.enabled = False
        
    def enable(self):
        """Aktiviert Visualisierungen."""
        self.enabled = True
    
    def reset_goal_data(self):
        """Setzt alle gespeicherten Goal-Referenzen zurück (bei neuem Goal)."""
        self.goal_pca_data = None
        self.goal_kmeans_data = None
    
    def reset_centroids(self):
        """Alias für reset_goal_data() - Rückwärtskompatibilität."""
        self.reset_goal_data()
    
    def _img_to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """Konvertiert uint8 RGB Bild zu normalisierten Tensor."""
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        
        # (H, W, 3) -> (1, 3, H, W), normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor * 2.0 - 1.0
        return img_tensor.unsqueeze(0)
    
    def visualize(self, img_np: np.ndarray, out_dir: str, step_idx: int = 0,
                  prefix: str = "frame", save_attention: bool = True,
                  save_naive_pca: bool = True, save_kmeans_pca: bool = True,
                  is_goal: bool = False) -> dict:
        """
        Erzeugt und speichert Feature-Visualisierungen.
        
        Args:
            img_np: (H, W, 3) uint8 RGB Bild
            out_dir: Ausgabe-Verzeichnis
            step_idx: Schritt-Index für Dateinamen
            prefix: Präfix für Dateinamen (z.B. "pick", "place", "pick_current")
            save_attention: Ob DINO Self-Attention gespeichert werden soll
            save_naive_pca: Ob Naive PCA gespeichert werden soll
            save_kmeans_pca: Ob K-Means + Per-Cluster PCA gespeichert werden soll
            is_goal: Ob dies ein Goal-Bild ist (PCA/K-Means neu fitten und speichern)
            
        Returns:
            dict mit Pfaden zu erzeugten Dateien:
            {
                "naive_pca": "path/to/file.png" oder None,
                "kmeans_pca": "path/to/file.png" oder None,
                "attention": "path/to/file.png" oder None,
            }
        """
        result = {"naive_pca": None, "kmeans_pca": None, "attention": None}
        
        if not self.enabled:
            return result
            
        os.makedirs(out_dir, exist_ok=True)
        
        # Convert image and extract features
        img_tensor = self._img_to_tensor(img_np)
        patch_tokens, cls_attn, h_patches, w_patches = extract_dino_features(
            self.encoder, img_tensor, self.device, self.encoder_transform
        )
        
        img_size = (img_np.shape[1], img_np.shape[0])
        
        # Bei Goal-Bild: PCA/K-Means neu fitten und speichern (für konsistente Farben)
        is_goal_image = is_goal
        
        if save_naive_pca:
            # Bei Goal-Bild: PCA neu fitten, sonst gespeicherte verwenden
            use_pca_data = None if is_goal_image else self.goal_pca_data
            pca_img, pca_data = create_naive_pca_image(patch_tokens, h_patches, w_patches, img_size, pca_data=use_pca_data)
            
            if is_goal_image:
                self.goal_pca_data = pca_data
                print(f"  Naive PCA vom Goal-Bild gespeichert")
            
            path = os.path.join(out_dir, f"{prefix}_naive_pca_step{step_idx:03d}.png")
            Image.fromarray(pca_img).save(path)
            result["naive_pca"] = path
            
        if save_kmeans_pca:
            # Bei Goal-Bild: Alles neu berechnen und speichern
            # Bei anderen Bildern: gespeicherte Goal-Daten verwenden (konsistente Farben)
            use_goal_data = None if is_goal_image else self.goal_kmeans_data
            
            kmeans_grids, goal_data = create_kmeans_pca_image(
                patch_tokens, h_patches, w_patches, img_np, self.n_clusters,
                goal_data=use_goal_data
            )
            
            # Bei Goal-Bild: Alle Daten speichern für spätere Verwendung
            if is_goal_image:
                self.goal_kmeans_data = goal_data
                print(f"  K-Means + PCA vom Goal-Bild gespeichert ({self.n_clusters} Cluster)")
            
            # Speichere ein Bild pro Cluster
            kmeans_paths = []
            for ci, grid in enumerate(kmeans_grids):
                path = os.path.join(out_dir, f"{prefix}_kmeans_pca_cluster{ci}_step{step_idx:03d}.png")
                Image.fromarray(grid).save(path)
                kmeans_paths.append(path)
            result["kmeans_pca"] = kmeans_paths
            
        if save_attention:
            attn_img = create_attention_image(
                cls_attn, h_patches, w_patches, img_np, self.n_heads_show
            )
            path = os.path.join(out_dir, f"{prefix}_attention_step{step_idx:03d}.png")
            Image.fromarray(attn_img).save(path)
            result["attention"] = path
            
        return result
    
    def visualize_comparison(self, current_img: np.ndarray, goal_img: np.ndarray,
                             out_dir: str, step_idx: int = 0) -> dict:
        """
        Erzeugt vergleichende Visualisierung: Current vs Goal.
        
        Args:
            current_img: (H, W, 3) uint8 RGB Bild (aktueller Zustand)
            goal_img: (H, W, 3) uint8 RGB Bild (Zielzustand)
            out_dir: Ausgabe-Verzeichnis
            step_idx: Schritt-Index
            
        Returns:
            dict mit Pfaden zu erzeugten Dateien
        """
        result = {}
        
        if not self.enabled:
            return result
            
        os.makedirs(out_dir, exist_ok=True)
        
        # Extract features for both
        current_tensor = self._img_to_tensor(current_img)
        goal_tensor = self._img_to_tensor(goal_img)
        
        current_tokens, current_attn, hp, wp = extract_dino_features(
            self.encoder, current_tensor, self.device, self.encoder_transform
        )
        goal_tokens, goal_attn, _, _ = extract_dino_features(
            self.encoder, goal_tensor, self.device, self.encoder_transform
        )
        
        img_h, img_w = current_img.shape[:2]
        
        # Side-by-side comparison: Current | Goal
        # Each with: Image | Naive PCA | K-Means
        
        # Naive PCA comparison (jeweils unabhängig gefittet für Vergleich)
        current_pca, _ = create_naive_pca_image(current_tokens, hp, wp, (img_w, img_h))
        goal_pca, _ = create_naive_pca_image(goal_tokens, hp, wp, (img_w, img_h))
        
        comparison = np.zeros((img_h * 2, img_w * 2, 3), dtype=np.uint8)
        comparison[:img_h, :img_w] = current_img
        comparison[:img_h, img_w:] = current_pca
        comparison[img_h:, :img_w] = goal_img
        comparison[img_h:, img_w:] = goal_pca
        
        path = os.path.join(out_dir, f"comparison_pca_step{step_idx:03d}.png")
        Image.fromarray(comparison).save(path)
        result["comparison_pca"] = path
        
        return result

    def visualize_wm_prediction(self, predicted_img: np.ndarray, goal_img: np.ndarray,
                                 current_img: np.ndarray,
                                 out_dir: str, step_idx: int = 0,
                                 prefix: str = "wm_pred") -> dict:
        """
        Erzeugt Vergleich: Aktueller Zustand, WM-Vorhersage und Zielbild.
        
        Layout (1 Zeile, 4 Bilder):
          | Goal | Current | Predicted | Difference |
        
        Args:
            predicted_img: (H, W, 3) uint8 RGB - Vom World Model vorhergesagter nächster Zustand
            goal_img: (H, W, 3) uint8 RGB - Zielbild
            current_img: (H, W, 3) uint8 RGB - Aktueller realer Zustand
            out_dir: Ausgabe-Verzeichnis
            step_idx: Schritt-Index für Dateinamen
            prefix: Präfix für Dateinamen
            
        Returns:
            dict mit:
                - "wm_prediction": Pfad zum gespeicherten Bild
                - "metrics": dict mit Differenz-Metriken:
                    - "mae_goal_current": MAE zwischen Goal und Current
                    - "mae_goal_predicted": MAE zwischen Goal und Predicted
                    - "mse_goal_current": MSE zwischen Goal und Current
                    - "mse_goal_predicted": MSE zwischen Goal und Predicted
        """
        result = {"wm_prediction": None, "metrics": None}
        
        if not self.enabled:
            return result
            
        os.makedirs(out_dir, exist_ok=True)
        
        # Sicherstellen, dass alle Bilder uint8 sind
        if predicted_img.dtype != np.uint8:
            predicted_img = (predicted_img * 255).clip(0, 255).astype(np.uint8)
        if goal_img.dtype != np.uint8:
            goal_img = (goal_img * 255).clip(0, 255).astype(np.uint8)
        if current_img.dtype != np.uint8:
            current_img = (current_img * 255).clip(0, 255).astype(np.uint8)
            
        img_h, img_w = predicted_img.shape[:2]
        
        # ─── Metriken berechnen ───
        goal_f = goal_img.astype(float)
        current_f = current_img.astype(float)
        predicted_f = predicted_img.astype(float)
        
        # MAE (Mean Absolute Error) - normalisiert auf [0, 1] (teilen durch 255)
        mae_goal_current = np.mean(np.abs(goal_f - current_f)) / 255.0
        mae_goal_predicted = np.mean(np.abs(goal_f - predicted_f)) / 255.0
        
        # MSE (Mean Squared Error) - normalisiert
        mse_goal_current = np.mean((goal_f - current_f) ** 2) / (255.0 ** 2)
        mse_goal_predicted = np.mean((goal_f - predicted_f) ** 2) / (255.0 ** 2)
        
        metrics = {
            "mae_goal_current": mae_goal_current,
            "mae_goal_predicted": mae_goal_predicted,
            "mse_goal_current": mse_goal_current,
            "mse_goal_predicted": mse_goal_predicted,
        }
        result["metrics"] = metrics
        
        # ─── Visualisierung ───
        # Difference: Absoluter Unterschied zwischen Prediction und Goal
        diff_raw = np.abs(predicted_f - goal_f)
        diff_gray = np.mean(diff_raw, axis=2)  # Mittlere Differenz über Kanäle
        
        # Normalisiere auf [0, 1] für Colormap
        diff_norm = diff_gray / (diff_gray.max() + 1e-8)
        
        # Colormap anwenden (rot = große Differenz, blau = kleine)
        cm = plt.get_cmap("jet")
        diff_colored = (cm(diff_norm)[:, :, :3] * 255).astype(np.uint8)
        
        # 1x4 Grid erstellen: Goal | Current | Predicted | Difference
        grid = np.zeros((img_h, img_w * 4, 3), dtype=np.uint8)
        grid[:, 0*img_w:1*img_w] = goal_img
        grid[:, 1*img_w:2*img_w] = current_img
        grid[:, 2*img_w:3*img_w] = predicted_img
        grid[:, 3*img_w:4*img_w] = diff_colored
        
        path = os.path.join(out_dir, f"{prefix}_step{step_idx:03d}.png")
        Image.fromarray(grid).save(path)
        result["wm_prediction"] = path
        
        print(f"  WM-Pred: MAE(G↔C)={mae_goal_current:.4f}, MAE(G↔P)={mae_goal_predicted:.4f}")
        
        return result

    def visualize_rollout_strip(self, current_img: np.ndarray, rollout_images: list,
                                  goal_img: np.ndarray, out_dir: str, step_idx: int = 0,
                                  prefix: str = "rollout") -> dict:
        """
        Erzeugt horizontalen Streifen mit dem kompletten WM-Rollout.
        
        Layout (1 Zeile):
          | Current | t=1 | t=2 | ... | t=H | Goal |
        
        Args:
            current_img: (H, W, 3) uint8 RGB - Aktueller realer Zustand
            rollout_images: Liste von (H, W, 3) uint8 RGB - Vom WM vorhergesagte Zustände
            goal_img: (H, W, 3) uint8 RGB - Zielbild
            out_dir: Ausgabe-Verzeichnis
            step_idx: Schritt-Index für Dateinamen
            prefix: Präfix für Dateinamen
            
        Returns:
            dict mit "rollout_strip": Pfad zum gespeicherten Bild oder None
        """
        result = {"rollout_strip": None}
        
        if not self.enabled or len(rollout_images) == 0:
            return result
            
        os.makedirs(out_dir, exist_ok=True)
        
        # Sicherstellen, dass alle Bilder uint8 sind
        def to_uint8(img):
            if img.dtype != np.uint8:
                return (img * 255).clip(0, 255).astype(np.uint8)
            return img
        
        current_img = to_uint8(current_img)
        goal_img = to_uint8(goal_img)
        rollout_images = [to_uint8(img) for img in rollout_images]
        
        img_h, img_w = current_img.shape[:2]
        horizon = len(rollout_images)
        
        # Gesamtbreite: Current + H×Rollout + Goal = (H+2) Bilder
        n_images = horizon + 2
        total_w = img_w * n_images
        
        # Grid erstellen
        grid = np.zeros((img_h, total_w, 3), dtype=np.uint8)
        
        # Bilder einfügen
        grid[:, 0:img_w] = current_img  # Current (Index 0)
        for t, rollout_img in enumerate(rollout_images):
            x_start = (t + 1) * img_w
            grid[:, x_start:x_start + img_w] = rollout_img
        grid[:, (horizon + 1) * img_w:] = goal_img  # Goal (letztes Bild)
        
        path = os.path.join(out_dir, f"{prefix}_step{step_idx:03d}.png")
        Image.fromarray(grid).save(path)
        result["rollout_strip"] = path
        
        print(f"  Rollout-Strip: {horizon} Schritte → {path}")
        
        return result


# ─────────────────────────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PlanningFeatureVisualizer - Standalone Test")
    print("Dieses Modul ist für den Import in planning_server.py gedacht.")
    print("")
    print("Verwendung:")
    print("  from planning_feature_visualizer import PlanningFeatureVisualizer")
    print("  visualizer = PlanningFeatureVisualizer(model.encoder, device)")
    print("  visualizer.set_encoder_transform(model.encoder_transform)")
    print("  visualizer.visualize(img_np, './out', step_idx=0, prefix='current')")
