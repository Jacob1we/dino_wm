#!/usr/bin/env python3
"""
Batch-Script: Erzeugt Feature-Visualisierungen und WM-Sanity-Checks
für ALLE trainierten Modelle in outputs/.

Für jedes Modell werden die Ergebnisse abgelegt:
  feature_visualizations/<model_name>/
  outputs/<model_name>/wm_sanity_check/

Nutzung:
  conda activate dino_wm
  python run_all_visualizations.py [--skip_existing] [--only MODEL_NAME]
"""

import os
import sys
import argparse
import traceback
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import OmegaConf

# ── Projekt-Root ──
DINO_WM_DIR = os.path.dirname(os.path.abspath(__file__))
if DINO_WM_DIR not in sys.path:
    sys.path.insert(0, DINO_WM_DIR)

from masterarbeit_style import apply_style, TEXTWIDTH_IN, FONT_SIZE, save_ma_figure
apply_style()


# =====================================================================
# Datensatz-Pfad-Auflösung
# =====================================================================

# Bekannte alternative Pfade für verschobene/archivierte Datensätze
DATASET_SEARCH_PATHS = [
    # Externe Disk: Hauptverzeichnis
    "/media/tsp_jw/data/DINO_WM/fcs_datasets",
    # Externe Disk: Archiv
    "/media/tsp_jw/data/DINO_WM/fcs_datasets/00_Archiv",
    # Desktop: Hauptverzeichnis
    "/home/tsp_jw/Desktop/fcs_datasets",
    # Desktop: Archiv
    "/home/tsp_jw/Desktop/fcs_datasets/00_Archiv",
]

# Manuelle Zuordnung für Datensätze die mit _BGR Suffix archiviert wurden
DATASET_NAME_ALIASES = {
    "primLogger_NEps200_ActInt10_RobOpac10_NCams4_NCube1":
        "primLogger_NEps200_ActInt10_RobOpac10_NCams4_NCube1_BGR",
    "primLogger_NEps500_ActInt10_RobOpac10_NCams4_NCube1":
        "primLogger_NEps500_ActInt10_RobOpac10_NCams4_NCube1_BGR",
    "primLogger_NEps1000_ActInt2_RobOpac10_NCams4_NCube1":
        "primLogger_NEps1000_ActInt2_RobOpac10_NCams4_NCube1_BGR",
    "NEps200_ActInt10_RobOpac10_NCams4_NCube1":
        "NEps200_ActInt10_RobOpac10_NCams4_NCube1_BGR",
    "NEps500_RobOpac10_NPrim10_NCams4_NCube1":
        "NEps500_RobOpac10_NPrim10_NCams4_NCube1_BGR",
}


def resolve_dataset_path(original_path: str) -> str:
    """
    Versucht den Datensatz-Pfad aufzulösen, auch wenn der Datensatz
    verschoben oder archiviert wurde.

    Returns:
        Aufgelöster Pfad oder None wenn nicht gefunden.
    """
    # 1. Originalpfad direkt prüfen
    if os.path.isdir(original_path):
        return original_path

    dataset_name = os.path.basename(original_path.rstrip("/"))

    # 2. In allen Suchpfaden nach dem exakten Namen suchen
    for search_dir in DATASET_SEARCH_PATHS:
        candidate = os.path.join(search_dir, dataset_name)
        if os.path.isdir(candidate):
            return candidate

    # 3. Alias-Namen versuchen (z.B. ohne/mit _BGR)
    alias_name = DATASET_NAME_ALIASES.get(dataset_name)
    if alias_name:
        for search_dir in DATASET_SEARCH_PATHS:
            candidate = os.path.join(search_dir, alias_name)
            if os.path.isdir(candidate):
                return candidate

    return None


# =====================================================================
# Modell-Discovery
# =====================================================================

def find_all_models():
    """
    Findet alle trainierten Modelle in outputs/.
    Returns: Liste von (model_name, model_path) Tupeln.
    """
    outputs_dir = os.path.join(DINO_WM_DIR, "outputs")
    models = []

    for root, dirs, files in os.walk(outputs_dir):
        if "hydra.yaml" in files and "checkpoints" in dirs:
            # Prüfe ob model_latest.pth existiert
            ckpt = os.path.join(root, "checkpoints", "model_latest.pth")
            if os.path.exists(ckpt):
                # model_name = relativer Pfad ab outputs/
                rel = os.path.relpath(root, outputs_dir)
                models.append((rel, root))

    # Sortiere nach Name
    models.sort(key=lambda x: x[0])
    return models


# =====================================================================
# Feature-Visualisierung für ein Modell
# =====================================================================

def run_feature_visualization(model_name, model_path, output_dir, device,
                              episode_indices=None, frame_indices=None):
    """Erzeugt Feature-Visualisierungen für ein Modell."""
    from plan import load_model
    from utils import seed
    from visualize_features import (
        build_aligned_context_window,
        denorm_image, extract_dino_attention, visualize_dino_attention,
        visualize_dino_pca, visualize_dino_similarity, create_summary,
        extract_vit_attention, visualize_vit_attention,
        visualize_reconstruction,
    )
    from PIL import Image as PILImage

    seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # Config laden
    cfg_path = os.path.join(model_path, "hydra.yaml")
    with open(cfg_path, "r") as f:
        model_cfg = OmegaConf.load(f)

    # Datensatz-Pfad auflösen
    orig_data_path = model_cfg.env.dataset.data_path
    resolved_path = resolve_dataset_path(orig_data_path)
    if resolved_path is None:
        print(f"  ⚠ SKIP: Datensatz nicht gefunden: {orig_data_path}")
        return False

    if resolved_path != orig_data_path:
        print(f"  ℹ Datensatz umgeleitet: {os.path.basename(resolved_path)}")
        model_cfg.env.dataset.data_path = resolved_path

    # Dataset laden
    print("  [1/4] Lade Dataset...")
    datasets, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dset["valid"]
    print(f"    Validierungs-Episoden: {len(dset)}")

    # Modell laden
    print("  [2/4] Lade Modell...")
    torch.cuda.empty_cache()
    model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
    model.eval()

    num_hist = model_cfg.num_hist
    frameskip = model_cfg.frameskip

    # Default: 3 Episoden, je 2 Frames
    if episode_indices is None:
        n_eps = min(3, len(dset))
        step = max(1, len(dset) // n_eps)
        episode_indices = list(range(0, len(dset), step))[:n_eps]
    if frame_indices is None:
        frame_indices = [3, 8]

    print(f"  [3/4] Episoden: {episode_indices}, Frames: {frame_indices}")

    for ep_idx in episode_indices:
        if ep_idx >= len(dset):
            continue
        obs_raw, actions, _, _ = dset[ep_idx]
        obs_visual = obs_raw["visual"]  # (T, 3, H, W)

        for fr_idx in frame_indices:
            fr_idx = min(fr_idx, obs_visual.shape[0] - 1)
            tag = f"ep{ep_idx:03d}_fr{fr_idx:02d}"

            single_frame = obs_visual[fr_idx:fr_idx + 1]
            img_np = denorm_image(single_frame[0])

            # Original speichern
            PILImage.fromarray(img_np).save(os.path.join(output_dir, f"original_{tag}.png"))

            # DINOv2 Visualisierungen
            encoder = model.encoder
            cls_attn, patch_tokens, hp, wp = extract_dino_attention(
                encoder, single_frame, device, encoder_transform=model.encoder_transform)

            visualize_dino_attention(img_np, cls_attn, hp, wp,
                                    os.path.join(output_dir, f"dino_attention_{tag}.png"))
            visualize_dino_pca(img_np, patch_tokens, hp, wp,
                               os.path.join(output_dir, f"dino_pca_{tag}.png"))
            visualize_dino_similarity(img_np, patch_tokens, hp, wp,
                                      os.path.join(output_dir, f"dino_similarity_{tag}.png"))
            create_summary(img_np, cls_attn, patch_tokens, hp, wp,
                           os.path.join(output_dir, f"summary_{tag}.png"))

    # ViT Predictor + VQ-VAE für den ersten Episode/Frame
    print("  [4/4] ViT Predictor + VQ-VAE...")
    if model.predictor is not None and num_hist > 0 and episode_indices:
        ep_idx = min(episode_indices[0], len(dset) - 1)
        obs_raw, actions, _, _ = dset[ep_idx]
        obs_visual = obs_raw["visual"]
        obs_proprio = obs_raw.get("proprio", None)

        available_frames = [fr for fr in frame_indices if fr < obs_visual.shape[0]] or [obs_visual.shape[0] - 1]
        target_frame_idx = available_frames[0]
        context = build_aligned_context_window(
            obs_visual=obs_visual,
            obs_proprio=obs_proprio,
            actions=actions,
            num_hist=num_hist,
            frameskip=frameskip,
            target_frame_idx=target_frame_idx,
        )

        context_visual = context["context_visual"].float()
        context_proprio = context["context_proprio"].float()
        context_actions = context["context_actions"].float()
        frame_idx_in_seq = context["frame_idx_in_seq"]
        actual_frame_idx = context["actual_frame_idx"]
        sampled_indices = context["sampled_indices"]

        ref_img_np = denorm_image(context_visual[0, frame_idx_in_seq])
        tag = f"ep{ep_idx:03d}_fr{actual_frame_idx:02d}"

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
            os.path.join(output_dir, f"vit_attention_{tag}.png"),
            frame_idx_in_seq=frame_idx_in_seq,
            sampled_indices=sampled_indices,
        )

        visualize_reconstruction(model, obs_dict, context_actions,
                                 device, ref_img_np,
                                 os.path.join(output_dir, f"reconstruction_{tag}.png"))

    # GPU-Speicher freigeben
    del model
    torch.cuda.empty_cache()

    return True


# =====================================================================
# WM Sanity-Check für ein Modell
# =====================================================================

def run_sanity_check_for_model(model_name, model_path, output_dir, device,
                                n_episodes=5, rollout_len=5):
    """Führt den WM Sanity-Check für ein Modell aus."""
    from plan import load_model
    from utils import seed
    from wm_sanity_check import denorm_image, compute_mse, compute_psnr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # Config laden
    cfg_path = os.path.join(model_path, "hydra.yaml")
    with open(cfg_path, "r") as f:
        model_cfg = OmegaConf.load(f)

    # Datensatz-Pfad auflösen
    orig_data_path = model_cfg.env.dataset.data_path
    resolved_path = resolve_dataset_path(orig_data_path)
    if resolved_path is None:
        print(f"  ⚠ SKIP: Datensatz nicht gefunden: {orig_data_path}")
        return False

    if resolved_path != orig_data_path:
        print(f"  ℹ Datensatz umgeleitet: {os.path.basename(resolved_path)}")
        model_cfg.env.dataset.data_path = resolved_path

    # Dataset laden
    print("  [1/5] Lade Dataset...")
    _, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dset["valid"]
    print(f"    Validierungs-Episoden: {len(dset)}")

    # Modell laden
    print("  [2/5] Lade Modell...")
    torch.cuda.empty_cache()
    model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
    model.eval()

    num_hist = model_cfg.num_hist
    num_pred = model_cfg.num_pred
    frameskip = model_cfg.frameskip
    action_dim = dset.action_dim

    # Rollout-Länge anpassen
    all_seq_lengths = [dset.get_seq_length(i) for i in range(len(dset))]
    max_seq_len = max(all_seq_lengths) if all_seq_lengths else 0
    # Need max_frame = (num_hist + rollout_len) * frameskip, plus frames 0..max_frame → max_frame+1 frames
    min_frames_needed = (num_hist + rollout_len) * frameskip + 1

    if min_frames_needed > max_seq_len:
        max_possible = (max_seq_len - 1) // frameskip - num_hist
        if max_possible <= 0:
            print(f"  ⚠ SKIP: Episoden zu kurz für Sanity-Check")
            del model
            torch.cuda.empty_cache()
            return False
        rollout_len = max_possible
        min_frames_needed = (num_hist + rollout_len) * frameskip + 1
        print(f"    Rollout-Länge reduziert auf {rollout_len}")

    # Geeignete Episoden finden
    valid_eps = [i for i in range(len(dset)) if dset.get_seq_length(i) >= min_frames_needed]
    if not valid_eps:
        print(f"  ⚠ SKIP: Keine geeigneten Episoden")
        del model
        torch.cuda.empty_cache()
        return False

    n_episodes = min(n_episodes, len(valid_eps))
    step = max(1, len(valid_eps) // n_episodes)
    selected = valid_eps[:n_episodes * step:step][:n_episodes]
    print(f"  [3/5] {n_episodes} Episoden, {rollout_len} Vorhersage-Schritte")

    # Rollout pro Episode
    all_metrics = []

    for ep_num, dset_idx in enumerate(selected):
        seq_len = dset.get_seq_length(dset_idx)
        total_wm_steps = num_hist + rollout_len
        max_frame = total_wm_steps * frameskip

        if max_frame >= seq_len:
            # Reduziere rollout_len für diese Episode
            local_rollout = (seq_len - 1) // frameskip - num_hist
            if local_rollout <= 0:
                continue
            total_wm_steps = num_hist + local_rollout
            max_frame = total_wm_steps * frameskip
            if max_frame >= seq_len:
                continue

        all_frame_indices = list(range(max_frame + 1))
        obs_all, act_all, state_all, _ = dset.get_frames(dset_idx, all_frame_indices)

        img_indices = list(range(0, max_frame + 1, frameskip))
        obs_visual = obs_all['visual'][img_indices]
        obs_proprio = obs_all['proprio'][img_indices]

        wm_actions = []
        for s in range(total_wm_steps):
            start = s * frameskip
            step_acts = act_all[start:start + frameskip]
            wm_actions.append(step_acts.reshape(-1))
        wm_actions = torch.stack(wm_actions)

        obs_0_visual = obs_visual[:num_hist].unsqueeze(0).to(device)
        obs_0_proprio = obs_proprio[:num_hist].unsqueeze(0).to(device)
        obs_0 = {'visual': obs_0_visual, 'proprio': obs_0_proprio}
        all_acts = wm_actions[:total_wm_steps].unsqueeze(0).to(device)

        with torch.no_grad():
            z_obses, z_full = model.rollout(obs_0, all_acts)
            z_visual = z_obses['visual']
            pred_visuals = []
            for t_idx in range(z_visual.shape[1]):
                z_t = z_visual[:, t_idx:t_idx + 1, :, :]
                obs_decoded, _ = model.decode_obs(
                    {"visual": z_t, "proprio": z_obses['proprio'][:, t_idx:t_idx + 1]}
                )
                pred_visuals.append(obs_decoded['visual'][:, 0])
            pred_visuals = torch.cat(pred_visuals, dim=0)

        gt_visuals = obs_visual.to(device)

        ep_metrics = {"episode_idx": dset_idx, "steps": []}
        for t_idx in range(pred_visuals.shape[0]):
            mse = compute_mse(pred_visuals[t_idx], gt_visuals[t_idx])
            psnr = compute_psnr(mse)
            step_type = "recon" if t_idx < num_hist else f"pred_{t_idx - num_hist + 1}"
            ep_metrics["steps"].append({"t": t_idx, "type": step_type, "mse": mse, "psnr": psnr})
        all_metrics.append(ep_metrics)

        # Side-by-Side Visualisierung
        T = pred_visuals.shape[0]
        n_show = min(T, num_hist + rollout_len + 1)
        fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
        if n_show == 1:
            axes = axes.reshape(3, 1)

        for t_idx in range(n_show):
            gt_img = denorm_image(gt_visuals[t_idx])
            axes[0, t_idx].imshow(gt_img)
            axes[0, t_idx].set_title(f"GT t={t_idx}")
            axes[0, t_idx].axis("off")

            pred_img = denorm_image(pred_visuals[t_idx])
            axes[1, t_idx].imshow(pred_img)
            mse_val = ep_metrics["steps"][t_idx]["mse"]
            label = "Recon" if t_idx < num_hist else f"Pred {t_idx - num_hist + 1}"
            axes[1, t_idx].set_title(f"{label}\nMSE={mse_val:.4f}")
            axes[1, t_idx].axis("off")

            diff_img = np.abs(gt_img.astype(float) - pred_img.astype(float))
            diff_max = max(diff_img.max(), 1)
            diff_img = (diff_img / diff_max * 255).astype(np.uint8)
            axes[2, t_idx].imshow(diff_img)
            axes[2, t_idx].set_title(f"Diff")
            axes[2, t_idx].axis("off")

        fig.suptitle(
            f"WM Sanity-Check — Episode {dset_idx}\n"
            f"Model: {model_name} | num_hist={num_hist}, frameskip={frameskip}"
        )
        plt.tight_layout()
        save_ma_figure(fig, os.path.join(output_dir, f"episode_{dset_idx:04d}"))
        plt.close(fig)

    # Aggregierte Metriken
    recon_mses = []
    pred_mses_by_step = {}
    for ep in all_metrics:
        for step in ep["steps"]:
            if step["type"] == "recon":
                recon_mses.append(step["mse"])
            else:
                pred_mses_by_step.setdefault(step["type"], []).append(step["mse"])

    all_pred_mses = [m for ms in pred_mses_by_step.values() for m in ms]

    # MSE-über-Horizont Plot
    if pred_mses_by_step:
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.5))
        steps = sorted(pred_mses_by_step.keys())
        step_nums = [int(s.split("_")[1]) for s in steps]
        mean_mses = [np.mean(pred_mses_by_step[s]) for s in steps]
        std_mses = [np.std(pred_mses_by_step[s]) for s in steps]

        if recon_mses:
            step_nums = [0] + step_nums
            mean_mses = [np.mean(recon_mses)] + mean_mses
            std_mses = [np.std(recon_mses)] + std_mses

        ax.errorbar(step_nums, mean_mses, yerr=std_mses, marker='o', capsize=4, linewidth=2)
        ax.set_xlabel("Vorhersage-Schritt (0 = Reconstruction)")
        ax.set_ylabel("MSE")
        ax.set_title(f"WM Prediction Quality\nModel: {model_name}")
        ax.grid(True, alpha=0.3)
        save_ma_figure(fig, os.path.join(output_dir, "mse_over_horizon"))
        plt.close(fig)

    # Metriken als JSON
    metrics_data = {
        "model_name": model_name,
        "n_episodes": n_episodes,
        "rollout_len": rollout_len,
        "num_hist": num_hist,
        "frameskip": frameskip,
        "action_dim": action_dim,
        "episodes": all_metrics,
        "summary": {
            "recon_mse_mean": float(np.mean(recon_mses)) if recon_mses else None,
            "pred_mse_mean": float(np.mean(all_pred_mses)) if all_pred_mses else None,
            "pred_mse_by_step": {k: float(np.mean(v)) for k, v in pred_mses_by_step.items()},
        }
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Zusammenfassung ausgeben
    if recon_mses:
        print(f"    Recon MSE: {np.mean(recon_mses):.6f}")
    if all_pred_mses:
        print(f"    Pred MSE:  {np.mean(all_pred_mses):.6f}")

    # GPU-Speicher freigeben
    del model
    torch.cuda.empty_cache()

    return True


# =====================================================================
# Hauptprogramm
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch-Visualisierung für alle trainierten Modelle"
    )
    parser.add_argument("--skip_existing", action="store_true",
                        help="Überspringe Modelle die bereits Visualisierungen haben")
    parser.add_argument("--only", type=str, default=None,
                        help="Nur dieses eine Modell verarbeiten (z.B. '260305/07-56')")
    parser.add_argument("--skip_features", action="store_true",
                        help="Feature-Visualisierungen überspringen")
    parser.add_argument("--skip_sanity", action="store_true",
                        help="WM Sanity-Check überspringen")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Anzahl Episoden für Sanity-Check (default: 5)")
    parser.add_argument("--rollout_len", type=int, default=5,
                        help="Rollout-Länge für Sanity-Check (default: 5)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda wenn verfügbar)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"DINO WM Dir: {DINO_WM_DIR}")

    # Modelle finden
    all_models = find_all_models()
    print(f"\n{len(all_models)} trainierte Modelle gefunden:")
    for name, path in all_models:
        print(f"  • {name}")

    # Filter
    if args.only:
        all_models = [(n, p) for n, p in all_models if n == args.only]
        if not all_models:
            print(f"\nModell '{args.only}' nicht gefunden!")
            return

    results = {"success": [], "skipped": [], "failed": []}

    for idx, (model_name, model_path) in enumerate(all_models):
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(all_models)}] Modell: {model_name}")
        print(f"{'='*70}")

        # Dataset-Name aus Config lesen für Ordnerbenennung
        cfg_path = os.path.join(model_path, "hydra.yaml")
        try:
            with open(cfg_path, "r") as f:
                _cfg = OmegaConf.load(f)
            dataset_name = os.path.basename(_cfg.env.dataset.data_path.rstrip("/"))
        except Exception:
            dataset_name = "unknown"

        dir_name = model_name.replace("/", "_") + "_" + dataset_name
        feat_dir = os.path.join(DINO_WM_DIR, "feature_visualizations", dir_name)
        sanity_dir = os.path.join(model_path, "wm_sanity_check")

        # Skip-Prüfung
        if args.skip_existing:
            feat_exists = os.path.isdir(feat_dir) and len(os.listdir(feat_dir)) > 0
            sanity_exists = os.path.isdir(sanity_dir) and len(os.listdir(sanity_dir)) > 0
            if feat_exists and sanity_exists:
                print(f"  ⏭ Übersprungen (bereits vorhanden)")
                results["skipped"].append(model_name)
                continue

        # Feature-Visualisierung
        if not args.skip_features:
            print(f"\n  ── Feature-Visualisierung ──")
            try:
                if args.skip_existing and os.path.isdir(feat_dir) and len(os.listdir(feat_dir)) > 0:
                    print(f"  ⏭ Features bereits vorhanden")
                else:
                    ok = run_feature_visualization(
                        model_name, model_path, feat_dir, device)
                    if ok:
                        print(f"  ✓ Feature-Visualisierung abgeschlossen")
                    else:
                        print(f"  ⚠ Feature-Visualisierung übersprungen")
            except Exception as e:
                print(f"  ✗ FEHLER bei Feature-Visualisierung: {e}")
                traceback.print_exc()

        # WM Sanity-Check
        if not args.skip_sanity:
            print(f"\n  ── WM Sanity-Check ──")
            try:
                if args.skip_existing and os.path.isdir(sanity_dir) and len(os.listdir(sanity_dir)) > 0:
                    print(f"  ⏭ Sanity-Check bereits vorhanden")
                else:
                    ok = run_sanity_check_for_model(
                        model_name, model_path, sanity_dir, device,
                        n_episodes=args.n_episodes, rollout_len=args.rollout_len)
                    if ok:
                        print(f"  ✓ Sanity-Check abgeschlossen")
                    else:
                        print(f"  ⚠ Sanity-Check übersprungen")
            except Exception as e:
                print(f"  ✗ FEHLER bei Sanity-Check: {e}")
                traceback.print_exc()

        results["success"].append(model_name)

    # Zusammenfassung
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"  Erfolgreich: {len(results['success'])}")
    for m in results["success"]:
        print(f"    ✓ {m}")
    if results["skipped"]:
        print(f"  Übersprungen: {len(results['skipped'])}")
        for m in results["skipped"]:
            print(f"    ⏭ {m}")
    if results["failed"]:
        print(f"  Fehlgeschlagen: {len(results['failed'])}")
        for m in results["failed"]:
            print(f"    ✗ {m}")


if __name__ == "__main__":
    main()
