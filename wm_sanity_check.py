#!/usr/bin/env python3
"""
WM Sanity-Check: Prüft die Vorhersagequalität des World Models mit
Ground-Truth Aktionen aus dem Trainingsdatensatz.

Ablauf:
  1. Lade trainiertes World Model + Dataset (gleicher Checkpoint wie Planning)
  2. Wähle N Episoden aus dem Validierungs-Set
  3. Für jede Episode: nimm obs_0 (num_hist Frames) + GT-Aktionen
  4. Rolle GT-Aktionen durch das WM → predicted next frames
  5. Vergleiche predicted vs. actual frames (MSE, SSIM, visuell)
  6. Speichere Side-by-Side PNGs + aggregierte Metriken

Nutzung:
  conda activate dino_wm
  python wm_sanity_check.py --model_name 2026-02-09/08-12-44 --n_episodes 5 --rollout_len 5

Ausgabe:  wm_sanity_outputs/<model_name>/
"""

import os
import sys
import json
import argparse
import torch
import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf

# ── Projekt-Root auf sys.path ──
dino_wm_dir = os.path.dirname(os.path.abspath(__file__))
if dino_wm_dir not in sys.path:
    sys.path.insert(0, dino_wm_dir)

from plan import load_model
from preprocessor import Preprocessor
from utils import seed


# =========================================================================
# Hilfsfunktionen
# =========================================================================

def denorm_image(img_tensor):
    """
    Konvertiert ein normalisiertes Bild-Tensor (C, H, W) zurück zu
    uint8 numpy (H, W, C) für Visualisierung.
    
    Die default_transform normalisiert mit mean=0.5, std=0.5:
        img_norm = (img - 0.5) / 0.5 = 2*img - 1   →   img ∈ [-1, 1]
    Rücktransformation:
        img = (img_norm + 1) / 2   →   img ∈ [0, 1]
    """
    img = (img_tensor + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
    return (img * 255).astype(np.uint8)


def compute_mse(img_a, img_b):
    """MSE zwischen zwei float-Tensoren (gleiche Shape)."""
    return ((img_a - img_b) ** 2).mean().item()


def compute_psnr(mse, max_val=2.0):
    """PSNR aus MSE. max_val=2.0 weil Bilder in [-1, 1]."""
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(max_val ** 2 / mse)


# =========================================================================
# Hauptlogik
# =========================================================================

def run_sanity_check(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(42)
    print(f"Device: {device}")

    # ── 1. Config + Model laden ──
    model_path = os.path.join(dino_wm_dir, "outputs", args.model_name)
    print(f"\n[1/5] Lade Model-Config aus: {model_path}")
    
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    # Dataset laden
    print("[2/5] Lade Dataset...")
    _, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    # Verwende Validierungs-Set (wie beim Planning)
    dset = traj_dset["valid"]
    print(f"  Validierungs-Episoden: {len(dset)}")
    print(f"  Action dim: {dset.action_dim}, Frameskip: {model_cfg.frameskip}")
    print(f"  num_hist: {model_cfg.num_hist}, num_pred: {model_cfg.num_pred}")

    # Model laden — erst auf CPU, dann auf GPU
    print("[3/5] Lade Model-Checkpoint...")
    torch.cuda.empty_cache()
    model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
    model.eval()

    num_hist = model_cfg.num_hist      # 2
    num_pred = model_cfg.num_pred      # 1
    frameskip = model_cfg.frameskip    # 2
    action_dim = dset.action_dim       # 6

    # ── 2. Episoden auswählen ──
    n_episodes = min(args.n_episodes, len(dset))
    rollout_len = args.rollout_len  # Anzahl Vorhersage-Schritte (nach num_hist)
    
    # Minimale Episodenlänge: (num_hist + rollout_len) * frameskip + 1 Original-Frames
    min_frames_needed = (num_hist + rollout_len) * frameskip + 1
    
    # Finde geeignete Episoden
    valid_episode_idxs = []
    for i in range(len(dset)):
        if dset.get_seq_length(i) >= min_frames_needed:
            valid_episode_idxs.append(i)
    
    if len(valid_episode_idxs) < n_episodes:
        print(f"  WARNUNG: Nur {len(valid_episode_idxs)} Episoden lang genug "
              f"(brauche {min_frames_needed} Frames)")
        n_episodes = len(valid_episode_idxs)
    
    # Wähle gleichmäßig verteilte Episoden
    step = max(1, len(valid_episode_idxs) // n_episodes)
    selected_idxs = valid_episode_idxs[:n_episodes * step:step][:n_episodes]
    print(f"  Ausgewählte Episoden: {selected_idxs}")

    # ── 3. Output-Verzeichnis ──
    output_dir = os.path.join(dino_wm_dir, "wm_sanity_outputs", args.model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[4/5] Output-Verzeichnis: {output_dir}")

    # ── 4. Pro Episode: Rollout + Vergleich ──
    all_metrics = []
    
    print(f"\n[5/5] Starte Sanity-Check für {n_episodes} Episoden, "
          f"je {rollout_len} Vorhersage-Schritte...\n")
    
    for ep_idx, dset_idx in enumerate(selected_idxs):
        print(f"{'='*60}")
        print(f"Episode {ep_idx + 1}/{n_episodes}  (Dataset-Index: {dset_idx})")
        print(f"{'='*60}")
        
        seq_len = dset.get_seq_length(dset_idx)
        
        # Wir brauchen (num_hist + rollout_len) WM-Schritte.
        # Jeder WM-Schritt entspricht frameskip Originalframes.
        # Bilder: an Positionen [0, fs, 2*fs, ...] → (num_hist + rollout_len + 1) Bilder
        # Aktionen: ALLE Originalframes laden, dann je frameskip konsekutive concatenieren
        total_wm_steps = num_hist + rollout_len
        max_frame = total_wm_steps * frameskip  # letzter benötigter Frame-Index
        
        if max_frame >= seq_len:
            print(f"  SKIP: Zu wenig Frames (brauche {max_frame+1}, haben {seq_len})")
            continue
        
        # Lade ALLE Frames von 0 bis max_frame (inkl.)
        all_frame_indices = list(range(max_frame + 1))
        obs_all, act_all, state_all, _ = dset.get_frames(dset_idx, all_frame_indices)
        # obs_all['visual']: (max_frame+1, C, H, W) — normalisiert
        # act_all: (max_frame+1, action_dim=6) — z-normalisiert
        
        # Bilder: nur an frameskip-Positionen [0, fs, 2*fs, ...]
        img_indices = list(range(0, max_frame + 1, frameskip))  # total_wm_steps + 1 Bilder
        obs_visual = obs_all['visual'][img_indices]   # (total_wm_steps+1, C, H, W)
        obs_proprio = obs_all['proprio'][img_indices]  # (total_wm_steps+1, proprio_dim)
        
        # Aktionen: je frameskip konsekutive concatenieren → (total_wm_steps, frameskip*action_dim)
        # WM-Step i benutzt Aktionen [i*fs, i*fs+1, ..., i*fs+fs-1]
        wm_actions = []
        for step in range(total_wm_steps):
            start = step * frameskip
            step_acts = act_all[start : start + frameskip]  # (frameskip, action_dim)
            concat_act = step_acts.reshape(-1)  # (frameskip * action_dim,) = (12,)
            wm_actions.append(concat_act)
        wm_actions = torch.stack(wm_actions)  # (total_wm_steps, 12)
        
        T = obs_visual.shape[0]  # total_wm_steps + 1
        actual_rollout_steps = T - num_hist - 1
        
        print(f"  Bilder: {T} (an Frameskip-Positionen {img_indices[:5]}...)")
        print(f"  obs visual shape: {obs_visual.shape}")
        print(f"  wm_actions shape: {wm_actions.shape} (frameskip={frameskip} × action_dim={action_dim})")
        print(f"  Effektive Rollout-Schritte: {actual_rollout_steps}")

        # ── Vorbereitung für WM ──
        # obs_0: die ersten num_hist Frames → (1, num_hist, C, H, W)
        obs_0_visual = obs_visual[:num_hist].unsqueeze(0).to(device)
        obs_0_proprio = obs_proprio[:num_hist].unsqueeze(0).to(device)
        obs_0 = {'visual': obs_0_visual, 'proprio': obs_0_proprio}
        
        # Aktionen für den Rollout:
        # WM.rollout erwartet act shape: (b, t+n, full_action_dim)
        # wobei n = num_hist (Aktionen für obs_0), t = rollout_len
        # Alle WM-Aktionen (ohne letzten Step, da der kein Action hat)
        all_acts = wm_actions[:total_wm_steps]  # (total_wm_steps, 12)
        all_acts = all_acts.unsqueeze(0).to(device)  # (1, total_wm_steps, 12)
        
        print(f"  obs_0 visual shape: {obs_0['visual'].shape}")
        print(f"  all_acts shape: {all_acts.shape}")

        # ── WM Rollout ──
        with torch.no_grad():
            z_obses, z_full = model.rollout(obs_0, all_acts)
            # z_obses: dict mit 'visual' (b, T, num_patches, emb_dim) und 'proprio'
            # z_full: (b, T, num_patches, emb_dim) — full embeddings inkl. action/proprio dims
            
            # Dekodiere alle Frames (predicted)
            z_visual = z_obses['visual']  # (1, T, num_patches, visual_emb_dim)
            pred_visuals = []
            for t_idx in range(z_visual.shape[1]):
                z_t = z_visual[:, t_idx:t_idx+1, :, :]  # (1, 1, p, d)
                obs_decoded, _ = model.decode_obs(
                    {"visual": z_t, "proprio": z_obses['proprio'][:, t_idx:t_idx+1]}
                )
                pred_visuals.append(obs_decoded['visual'][:, 0])  # (1, C, H, W)
            pred_visuals = torch.cat(pred_visuals, dim=0)  # (T, C, H, W) auf GPU

        # GT Bilder (alle Bilder an frameskip-Positionen)
        gt_visuals = obs_visual.to(device)  # (T, C, H, W)

        # ── Metriken berechnen ──
        ep_metrics = {"episode_idx": dset_idx, "steps": []}
        
        print(f"\n  {'Step':>5} | {'MSE':>10} | {'PSNR (dB)':>10} | {'Type':>10}")
        print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        for t_idx in range(pred_visuals.shape[0]):
            mse = compute_mse(pred_visuals[t_idx], gt_visuals[t_idx])
            psnr = compute_psnr(mse)
            
            # Bestimme ob Reconstruction (t < num_hist) oder Prediction
            if t_idx < num_hist:
                step_type = "recon"
            else:
                step_type = f"pred_{t_idx - num_hist + 1}"
            
            ep_metrics["steps"].append({
                "t": t_idx,
                "type": step_type,
                "mse": mse,
                "psnr": psnr,
            })
            print(f"  {t_idx:>5} | {mse:>10.6f} | {psnr:>10.2f} | {step_type:>10}")
        
        all_metrics.append(ep_metrics)
        
        # ── Visualisierung: Side-by-Side ──
        n_show = min(pred_visuals.shape[0], num_hist + actual_rollout_steps + 1)
        fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
        if n_show == 1:
            axes = axes.reshape(3, 1)
        
        for t_idx in range(n_show):
            # GT
            gt_img = denorm_image(gt_visuals[t_idx])
            axes[0, t_idx].imshow(gt_img)
            axes[0, t_idx].set_title(f"GT t={t_idx}", fontsize=8)
            axes[0, t_idx].axis("off")
            
            # Predicted
            pred_img = denorm_image(pred_visuals[t_idx])
            axes[1, t_idx].imshow(pred_img)
            mse_val = ep_metrics["steps"][t_idx]["mse"]
            psnr_val = ep_metrics["steps"][t_idx]["psnr"]
            label = "Recon" if t_idx < num_hist else f"Pred {t_idx - num_hist + 1}"
            axes[1, t_idx].set_title(f"{label}\nMSE={mse_val:.4f}", fontsize=7)
            axes[1, t_idx].axis("off")
            
            # Differenz (verstärkt)
            diff_img = np.abs(gt_img.astype(float) - pred_img.astype(float))
            diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8) if diff_img.max() > 0 else diff_img.astype(np.uint8)
            axes[2, t_idx].imshow(diff_img)
            axes[2, t_idx].set_title(f"Diff (×{255/max(diff_img.max(),1):.0f})", fontsize=7)
            axes[2, t_idx].axis("off")

        # Markiere Grenze zwischen Reconstruction und Prediction
        for row in range(3):
            if num_hist < n_show:
                axes[row, num_hist].spines['left'].set_visible(True)
                axes[row, num_hist].spines['left'].set_color('red')
                axes[row, num_hist].spines['left'].set_linewidth(3)

        fig.suptitle(
            f"WM Sanity-Check — Episode {dset_idx}\n"
            f"Model: {args.model_name} | num_hist={num_hist}, frameskip={frameskip}\n"
            f"Zeile 1: Ground-Truth | Zeile 2: WM-Vorhersage | Zeile 3: Differenz",
            fontsize=10
        )
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f"episode_{dset_idx:04d}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Gespeichert: {fig_path}")

    # ── 5. Aggregierte Metriken ──
    print(f"\n{'='*60}")
    print("AGGREGIERTE ERGEBNISSE")
    print(f"{'='*60}")
    
    # Sammle alle prediction MSEs (nicht reconstruction)
    recon_mses = []
    pred_mses_by_step = {}
    
    for ep in all_metrics:
        for step in ep["steps"]:
            if step["type"] == "recon":
                recon_mses.append(step["mse"])
            else:
                pred_step = step["type"]
                if pred_step not in pred_mses_by_step:
                    pred_mses_by_step[pred_step] = []
                pred_mses_by_step[pred_step].append(step["mse"])
    
    print(f"\nReconstruction (encode → decode, ohne Prediction):")
    if recon_mses:
        print(f"  Ø MSE: {np.mean(recon_mses):.6f}  "
              f"(PSNR: {compute_psnr(np.mean(recon_mses)):.2f} dB)")
    
    print(f"\nPrediction mit GT-Aktionen (pro Horizont-Schritt):")
    for step_name in sorted(pred_mses_by_step.keys()):
        mses = pred_mses_by_step[step_name]
        mean_mse = np.mean(mses)
        print(f"  {step_name}: Ø MSE = {mean_mse:.6f}  "
              f"(PSNR: {compute_psnr(mean_mse):.2f} dB)  "
              f"[n={len(mses)}]")
    
    # Gesamtbewertung
    all_pred_mses = [m for ms in pred_mses_by_step.values() for m in ms]
    if all_pred_mses:
        overall_pred_mse = np.mean(all_pred_mses)
        print(f"\n  Gesamt Prediction: Ø MSE = {overall_pred_mse:.6f}  "
              f"(PSNR: {compute_psnr(overall_pred_mse):.2f} dB)")
        
        # Diagnose
        print(f"\n{'─'*60}")
        print("DIAGNOSE:")
        if recon_mses:
            recon_mean = np.mean(recon_mses)
            pred_mean = overall_pred_mse
            ratio = pred_mean / recon_mean if recon_mean > 0 else float('inf')
            print(f"  Prediction/Reconstruction MSE Ratio: {ratio:.2f}x")
            
            if ratio < 2.0:
                print("  ✅ WM-Vorhersagen sind gut — Prediction ≈ Reconstruction.")
                print("     → Problem liegt wahrscheinlich NICHT am World Model.")
                print("     → Mögliche Ursachen: CEM-Parametrisierung, Action-Space Mismatch")
            elif ratio < 5.0:
                print("  ⚠️  WM-Vorhersagen degradieren moderat.")
                print("     → World Model hat teilweise gelernt, aber Fehler akkumulieren.")
                print("     → Empfehlung: Mehr Trainingsdaten (aktuell 200, Paper 1000)")
            else:
                print("  ❌ WM-Vorhersagen sind schlecht — Prediction >> Reconstruction.")
                print("     → World Model kann Dynamik nicht vorhersagen.")
                print("     → Dringend: Mehr Trainingsdaten sammeln (Ziel: 1000 Episoden)")
                print("     → Alternativ: Trainings-Hyperparameter prüfen")
    
    # Speichere Metriken als JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "n_episodes": n_episodes,
            "rollout_len": rollout_len,
            "num_hist": num_hist,
            "frameskip": frameskip,
            "action_dim": action_dim,
            "episodes": all_metrics,
            "summary": {
                "recon_mse_mean": float(np.mean(recon_mses)) if recon_mses else None,
                "pred_mse_mean": float(np.mean(all_pred_mses)) if all_pred_mses else None,
                "pred_mse_by_step": {
                    k: float(np.mean(v)) for k, v in pred_mses_by_step.items()
                },
            }
        }, f, indent=2)
    print(f"\n  Metriken gespeichert: {metrics_path}")
    
    # Zusammenfassungs-Plot: MSE über Horizont
    if pred_mses_by_step:
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = sorted(pred_mses_by_step.keys())
        step_nums = [int(s.split("_")[1]) for s in steps]
        mean_mses = [np.mean(pred_mses_by_step[s]) for s in steps]
        std_mses = [np.std(pred_mses_by_step[s]) for s in steps]
        
        # Füge Reconstruction als Step 0 hinzu
        if recon_mses:
            step_nums = [0] + step_nums
            mean_mses = [np.mean(recon_mses)] + mean_mses
            std_mses = [np.std(recon_mses)] + std_mses
        
        ax.errorbar(step_nums, mean_mses, yerr=std_mses, 
                    marker='o', capsize=4, linewidth=2)
        ax.set_xlabel("Vorhersage-Schritt (0 = Reconstruction)")
        ax.set_ylabel("MSE (normalisierter Bildraum)")
        ax.set_title(f"WM Prediction Quality über Horizont\n"
                     f"Model: {args.model_name} | {n_episodes} Episoden")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Recon ↔ Pred')
        ax.legend()
        
        summary_path = os.path.join(output_dir, "mse_over_horizon.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Zusammenfassung: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"Fertig! Alle Outputs in: {output_dir}")
    print(f"{'='*60}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WM Sanity-Check: Vorhersagequalität mit GT-Aktionen prüfen"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model-Name (z.B. '2026-02-09/08-12-44')"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=5,
        help="Anzahl zu testender Episoden (default: 5)"
    )
    parser.add_argument(
        "--rollout_len", type=int, default=5,
        help="Anzahl Vorhersage-Schritte nach num_hist (default: 5)"
    )
    
    args = parser.parse_args()
    run_sanity_check(args)
