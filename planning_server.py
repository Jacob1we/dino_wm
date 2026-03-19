"""
DINO WM Planning Server — Minimaler Socket-Wrapper um plan.py's CEM-Planner.

Nutzt DIREKT plan.py's Infrastruktur (load_model, Preprocessor, CEMPlanner).
Der Server ist nur eine duenne Socket-Schicht: Bild rein → Actions raus.

Das World Model und der Planner kommen 1:1 aus dem DINO WM Codebase —
identisch zu plan.py fuer deformable_env, wall, etc.

Architektur:
  Isaac Sim (planning_eval.py) ←TCP Socket→ Dieser Server (planning_server.py)
  
  Alternative: FrankaCubeStackWrapper als gym.Env direkt in plan.py nutzen
  (siehe env/franka_cube_stack/), dann ist kein Server noetig.

Verwendung:
    cd ~/Desktop/dino_wm && conda activate dino_wm
    python planning_server.py --model_name 2026-02-09/17-59-59
    python planning_server.py --model_name 2026-02-09/17-59-59 --mode offline --goal_H 5
"""

import os
import sys
import gc
import socket
import pickle
import time
import numpy as np
import argparse
from einops import rearrange

# DINO WM Pfad
dino_wm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dino_wm_dir)
os.chdir(dino_wm_dir)

import torch
import hydra
import wandb
from pathlib import Path
from omegaconf import OmegaConf

from plan import load_model, DummyWandbRun
from preprocessor import Preprocessor
from utils import seed
from planning_feature_visualizer import PlanningFeatureVisualizer

# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="DINO WM Planning Server")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--mode", type=str, default="online", choices=["online", "offline"])
parser.add_argument("--goal_H", type=int, default=5)
parser.add_argument("--num_samples", type=int, default=300)
parser.add_argument("--opt_steps", type=int, default=30)
parser.add_argument("--topk", type=int, default=30)
parser.add_argument("--chunk_size", type=int, default=None,
                    help="Max Batch-Size pro WM Rollout (OOM-Schutz). "
                         "None=auto (GPU-abhaengig), 0=kein Chunking.")
parser.add_argument("--n_sub_actions", type=int, default=1,
                    help="Anzahl Sub-Actions pro Plan-Step (default: 1). "
                         "Max = frameskip. 0 = alle (=frameskip).")
parser.add_argument("--planning_config", type=str, default="plan_franka",
                    help="Name der Planning-Config (ohne .yaml). Default: plan_franka")
parser.add_argument("--checkpoint", type=str, default="latest",
                    help="Checkpoint-Name: 'latest' fuer model_latest.pth, "
                         "oder Epoch-Nummer (z.B. '50', '100') fuer model_50.pth etc.")
args = parser.parse_args()

# # Default Parameter aus dem Paper:
# horizon: 5
# topk: 30
# num_samples: 300
# var_scale: 1
# opt_steps: 30

# =============================================================================
# SETUP — identisch zu plan.py's planning_main()
# =============================================================================

print("=" * 60)
print("DINO WM Planning Server")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed(42)

# 1. Config laden
model_path = f"{dino_wm_dir}/outputs/{args.model_name}/"
with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
    model_cfg = OmegaConf.load(f)

# 2. Dataset-Statistiken extrahieren (dann sofort freigeben → OOM-Fix)
#
# WICHTIG: Wir brauchen NUR die Normalisierungs-Statistiken (mean/std) vom
# Dataset, nicht die Bilder selbst. Daher laden wir das Dataset MIT ALLEN
# Episoden (fuer identische Statistiken wie beim Training), aber OHNE
# Bilder im RAM (preload_images=False). Das spart mehrere GB RAM und
# reduziert die Startzeit von ~60s auf ~10s.
#
# Die H5-Dateien (Actions, EEF-States) sind klein (~KB pro Episode)
# und muessen geladen werden, damit mean/std korrekt berechnet werden.
print("Lade Dataset-Statistiken (ohne Bilder)...")

from datasets.franka_cube_stack_dset import FrankaCubeStackDataset

_dset_cfg = OmegaConf.to_container(model_cfg.env.dataset, resolve=True)
_transform = hydra.utils.call(model_cfg.env.dataset.transform)

_full_dset = FrankaCubeStackDataset(
    n_rollout=_dset_cfg.get("n_rollout", None),
    data_path=_dset_cfg["data_path"],
    normalize_action=_dset_cfg.get("normalize_action", True),
    transform=_transform,
    preload_images=False,  # ← KEIN Bild-RAM! Nur Actions/EEF fuer Statistiken
)

dset_action_dim = _full_dset.action_dim
dset_transform = _full_dset.transform
action_mean = _full_dset.action_mean.clone()
action_std = _full_dset.action_std.clone()
state_mean = _full_dset.state_mean.clone()
state_std = _full_dset.state_std.clone()
proprio_mean = _full_dset.proprio_mean.clone()
proprio_std = _full_dset.proprio_std.clone()
dataset_path = _dset_cfg["data_path"]

del _full_dset, _transform, _dset_cfg
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- Action-Dim-Konsistenzcheck: model_cfg vs. Dataset ---
# Das Model wurde mit model_cfg.env.action_dim trainiert. Wenn das Dataset
# inzwischen eine andere action_dim hat (z.B. 8D statt 6D weil es nach dem
# Training mit Gripper-Tracking regeneriert wurde), muessen wir die Stats
# auf die Trainings-Dimension anpassen. Sonst bekommt der CEM die falsche
# Dimension und der Action-Encoder crasht (Conv1d Kanal-Mismatch).
model_action_dim = model_cfg.env.action_dim
if dset_action_dim != model_action_dim:
    print(f"\n  ⚠ ACTION-DIM MISMATCH: Dataset hat {dset_action_dim}D, "
          f"Model wurde mit {model_action_dim}D trainiert!")
    if dset_action_dim == 8 and model_action_dim == 6:
        # 8D → 6D: Gripper-Dims (Index 3, 7) entfernen
        # 8D: [x_s, y_s, z_s, g_s, x_e, y_e, z_e, g_e]
        # 6D: [x_s, y_s, z_s, x_e, y_e, z_e]
        keep_idx = [0, 1, 2, 4, 5, 6]
        action_mean = action_mean[keep_idx]
        action_std = action_std[keep_idx]
        print(f"  → Stats auf 6D reduziert (Gripper-Dims 3,7 entfernt)")
        print(f"  ⚠ WARNUNG: Stats stammen aus regeneriertem 8D-Dataset und "
              f"koennten leicht von den Original-Trainings-Stats abweichen!")
    elif dset_action_dim == 6 and model_action_dim == 8:
        # 6D → 8D: Gripper-Dims (g_s=0.5, g_e=0.5) einfuegen
        _new_mean = torch.zeros(8)
        _new_std = torch.ones(8)
        _new_mean[[0, 1, 2, 4, 5, 6]] = action_mean
        _new_std[[0, 1, 2, 4, 5, 6]] = action_std
        _new_mean[[3, 7]] = 0.5   # Gripper Midpoint
        _new_std[[3, 7]] = 0.5    # Gripper Std (binaer 0/1)
        action_mean = _new_mean
        action_std = _new_std
        print(f"  → Stats auf 8D erweitert (Gripper-Dims 3,7 mit default mean=0.5, std=0.5)")
    else:
        raise ValueError(
            f"Kann action_dim Mismatch ({dset_action_dim}D→{model_action_dim}D) "
            f"nicht automatisch aufloesen. Bitte Dataset neu generieren oder "
            f"passendes Model verwenden."
        )
    base_action_dim = model_action_dim
else:
    base_action_dim = dset_action_dim

print(f"  action_dim={base_action_dim}, Stats extrahiert, Dataset freigegeben ✓")

# 3. Model laden (wie plan.py)
if args.checkpoint == "latest":
    ckpt_filename = "model_latest.pth"
else:
    ckpt_filename = f"model_{args.checkpoint}.pth"
model_ckpt = Path(model_path) / "checkpoints" / ckpt_filename
if not model_ckpt.exists():
    available = sorted(model_ckpt.parent.glob("model_*.pth"))
    avail_str = ", ".join(p.stem.replace("model_", "") for p in available)
    raise FileNotFoundError(
        f"Checkpoint nicht gefunden: {model_ckpt}\n"
        f"Verfuegbare Checkpoints: {avail_str}"
    )
print(f"Lade Model (checkpoint={ckpt_filename})...")
model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
model.eval()  # WICHTIG: Eval-Modus fuer deterministische Inferenz (kein Dropout etc.)

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"  GPU: {alloc:.0f}/{total:.0f} MB")

# 3b. Chunked Rollout Wrapper (OOM-Schutz fuer grosse CEM Batches)
#
# Problem: CEM sampelt z.B. 300 Trajektorien und schickt ALLE auf einmal
# durch wm.rollout(). Die Self-Attention im ViT-Predictor skaliert O(N²)
# mit N = num_hist × patches_per_frame. Bei num_hist=4 und 300 Samples
# braucht allein die Attention-Matrix ~20 GB → OOM auf RTX A5000 (24 GB).
#
# Loesung: Wir wrappen das Model und splitten den Batch in Sub-Batches.
# Der CEM-Planner sieht keinen Unterschied (gleiches Interface).

class ChunkedRolloutWrapper:
    """Transparenter Wrapper: teilt grosse Batches in GPU-sichere Chunks."""

    def __init__(self, model, chunk_size):
        self._model = model
        self.chunk_size = chunk_size
        # Alle Attribute des Original-Models durchreichen
        for attr in ['num_hist', 'num_pred', 'encoder', 'predictor',
                     'decoder', 'action_encoder', 'proprio_encoder',
                     'concat_dim', 'num_action_repeat', 'num_proprio_repeat',
                     'emb_dim', 'action_dim', 'proprio_dim',
                     'encoder_transform', 'emb_criterion',
                     'decoder_criterion', 'decoder_latent_loss_weight']:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

    def __getattr__(self, name):
        # Fallback: alles was nicht explizit gesetzt ist, vom Model holen
        if name.startswith('_') or name == 'chunk_size':
            raise AttributeError(name)
        return getattr(self._model, name)

    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def rollout(self, obs_0, act):
        B = act.shape[0]
        if B <= self.chunk_size:
            return self._model.rollout(obs_0, act)

        all_z_obses = []
        all_zs = []
        for start in range(0, B, self.chunk_size):
            end = min(start + self.chunk_size, B)
            chunk_obs = {k: v[start:end] for k, v in obs_0.items()}
            chunk_act = act[start:end]
            z_obses, zs = self._model.rollout(chunk_obs, chunk_act)
            all_z_obses.append(z_obses)
            all_zs.append(zs)

        # GPU-Cache erst NACH allen Chunks freigeben (vermeidet Fragmentierung)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            {k: torch.cat([z[k] for z in all_z_obses], dim=0) for k in all_z_obses[0]},
            torch.cat(all_zs, dim=0),
        )

    def encode_obs(self, obs):
        return self._model.encode_obs(obs)

    def eval(self):
        return self._model.eval()

    def train(self, mode=True):
        return self._model.train(mode)

    def parameters(self):
        return self._model.parameters()

# Chunk-Size bestimmen
if args.chunk_size is not None and args.chunk_size == 0:
    print(f"  Chunking: DEAKTIVIERT (--chunk_size 0)")
else:
    if args.chunk_size is not None:
        chunk_size = args.chunk_size
    else:
        # Auto-Detect basierend auf GPU-VRAM und num_hist
        # Attention-VRAM ∝ B × heads × (num_hist × patches)²
        # RTX A5000 (24 GB): num_hist=2 → ~100, num_hist=4 → ~25
        num_hist = model_cfg.num_hist
        if torch.cuda.is_available():
            free_mb = (total - alloc) * 0.85  # 85% des freien VRAM nutzen
            # Empirisch: num_hist=2 → ~0.08 GB/sample, num_hist=4 → ~0.32 GB/sample
            gb_per_sample = 0.02 * (num_hist ** 2)  # quadratische Skalierung
            chunk_size = max(8, int(free_mb / 1024 / gb_per_sample))
            chunk_size = min(chunk_size, 64)  # Nie mehr als 64 auf einmal
        else:
            chunk_size = 16
    model = ChunkedRolloutWrapper(model, chunk_size)
    print(f"  Chunking: {chunk_size} Samples pro Sub-Batch (num_hist={model_cfg.num_hist})")

# 4. Preprocessor (wie PlanWorkspace)
frameskip = model_cfg.frameskip
full_action_dim = base_action_dim * frameskip

preprocessor = Preprocessor(
    action_mean=action_mean.repeat(frameskip),
    action_std=action_std.repeat(frameskip),
    state_mean=state_mean,
    state_std=state_std,
    proprio_mean=proprio_mean,
    proprio_std=proprio_std,
    transform=dset_transform,
)

# 5. CEM-Planner erstellen (wie PlanWorkspace)
objective_fn = hydra.utils.call({
    "_target_": "planning.objectives.create_objective_fn",
    "alpha": 0.5, "base": 2, "mode": "last",
})
planner_cfg = OmegaConf.load(f"{dino_wm_dir}/conf/planner/cem.yaml")

# Horizon
if args.goal_H is not None:
    horizon = args.goal_H
else:
    horizon = 2 if args.mode == "online" else 5

# CEM-Parameter je nach Modus
if args.mode == "offline":
    if args.num_samples: planner_cfg.num_samples = args.num_samples
    if args.opt_steps: planner_cfg.opt_steps = args.opt_steps
    if args.topk: planner_cfg.topk = args.topk
else:
    planner_cfg.num_samples = args.num_samples or 64
    planner_cfg.opt_steps = args.opt_steps or 5
    planner_cfg.topk = args.topk or 10

# =============================================================================
# W&B LOGGING — identisch zum Training (wandb.init + .log)
# =============================================================================

class WandBPlanningRun:
    """Echtes W&B-Logging fuer Planning Server.
    
    Loggt CEM-Optimierungsschritte, Episode-Metriken und Plan-Zeiten
    identisch zum Training auf W&B. Zusaetzlich stdout fuer lokale Diagnose.
    
    Lifecycle:
      1. __init__: wandb.init() mit Server+Modell-Config
      2. update_config(): Client-Config nachtraeglich mergen
      3. log(): CEM-Steps (wird vom Planner aufgerufen)
      4. log_episode(): Episode-zusammenfassung
      5. finish(): wandb.finish() am Ende
    """
    def __init__(self, server_config: dict, project: str = "dino_wm_planning"):
        self.mode = "wandb"
        self._losses = []
        self._plan_count = 0
        self._episode_count = 0
        self._global_step = 0

        # W&B initialisieren — wie train.py's wandb.init()
        # Run-Name enthaelt CEM-Parameter fuer schnelle Identifikation
        model = server_config.get('model_name', 'unknown')
        mode = server_config.get('mode', 'online')
        H = server_config.get('goal_H', '?')
        S = server_config.get('num_samples', '?')
        O = server_config.get('opt_steps', '?')
        K = server_config.get('topk', '?')
        N = server_config.get('n_sub_actions', 1)
        run_name = f"plan_{model}_{mode}_H{H}_S{S}_O{O}_K{K}_N{N}"

        self._run = wandb.init(
            project=project,
            config=server_config,
            name=run_name,
            tags=["planning", mode],
            reinit=True,
        )
        print(f"  W&B Run: {self._run.url}")

    @property
    def id(self):
        return self._run.id if self._run else None

    def update_config(self, client_config: dict):
        """Client-Config nachtraeglich in W&B-Config mergen.
        
        Wird aufgerufen wenn der Client seine Konfiguration sendet
        (set_client_config Kommando). So ist die komplette Planning-
        Konfiguration (Server + Client) in einem W&B-Run nachvollziehbar.
        """
        if self._run:
            self._run.config.update({"client": client_config}, allow_val_change=True)
            print(f"  W&B Config aktualisiert mit Client-Config ({len(client_config)} Keys)")

    def log(self, data, *args, **kwargs):
        """Wird vom CEM-Planner bei jedem Optimierungsschritt aufgerufen."""
        self._global_step += 1
        # Stdout (lokale Diagnose)
        for key, value in data.items():
            if "loss" in key:
                self._losses.append(value)
                print(f"    CEM step {data.get('step', '?')}: loss={value:.6f}", flush=True)
        # W&B
        if self._run:
            log_data = {f"cem/{k}": v for k, v in data.items()}
            log_data["global_step"] = self._global_step
            log_data["plan_count"] = self._plan_count
            self._run.log(log_data)

    def log_step(self, step_data: dict):
        """Per-Step Metriken loggen (EEF→Cube, Ist→Soll Distanz)."""
        if self._run:
            log_data = {f"step/{k}": v for k, v in step_data.items()}
            self._run.log(log_data)

    def log_episode(self, episode_data: dict):
        """Episode-Zusammenfassung loggen (nach jeder Episode im Client)."""
        self._episode_count += 1
        if self._run:
            log_data = {f"episode/{k}": v for k, v in episode_data.items()}
            log_data["episode"] = self._episode_count
            self._run.log(log_data)

    def log_plan_summary(self, plan_time: float, n_actions: int, mode: str = "plan"):
        """Zusammenfassung nach jedem plan/plan_all Aufruf."""
        if self._run:
            summary = {
                f"{mode}/plan_time_s": plan_time,
                f"{mode}/n_actions": n_actions,
                f"{mode}/plan_count": self._plan_count,
            }
            if self._losses:
                summary[f"{mode}/final_loss"] = self._losses[-1]
                summary[f"{mode}/initial_loss"] = self._losses[0]
                if self._losses[0] > 0:
                    summary[f"{mode}/loss_reduction_pct"] = (
                        (1 - self._losses[-1] / self._losses[0]) * 100
                    )
            self._run.log(summary)

    def get_summary(self, plan_time=None):
        if not self._losses:
            return "keine Daten"
        first, last = self._losses[0], self._losses[-1]
        pct = (1 - last/first) * 100 if first > 0 else 0
        return f"{first:.6f} → {last:.6f} ({pct:.1f}% Reduktion, {plan_time:.1f}s)"

    def reset(self):
        self._losses.clear()
        self._plan_count += 1

    def watch(self, *a, **kw):
        if self._run:
            self._run.watch(*a, **kw)

    def config(self, *a, **kw): pass

    def finish(self):
        if self._run:
            self._run.finish()
            print("  W&B Run beendet.")


# Server-Config zusammenstellen (alles was den Planning-Prozess definiert)
server_config = {
    # Modell-Info
    "model_name": args.model_name,
    "model_path": model_path,
    "model_epoch": args.checkpoint,
    # Planning-Parameter
    "mode": args.mode,
    "goal_H": horizon,
    "num_samples": int(planner_cfg.num_samples),
    "opt_steps": int(planner_cfg.opt_steps),
    "topk": int(planner_cfg.topk),
    "action_dim": full_action_dim,
    "base_action_dim": base_action_dim,
    "frameskip": frameskip,
    "search_dim": horizon * full_action_dim,
    # Dataset-Pfad (fuer Folder-Benennung im Client)
    "dataset_path": dataset_path,
    # Modell-Config (aus Training)
    "training_config": OmegaConf.to_container(model_cfg, resolve=True),
    # Server-Parameter
    "port": args.port,
    "chunk_size": args.chunk_size,
    "n_sub_actions": args.n_sub_actions,
}

wandb_run = WandBPlanningRun(server_config)

# =============================================================================
# PLANNING CONFIG (Franka-spezifisch: sigma_scale, workspace_bounds, gripper)
# =============================================================================

planning_cfg = None
planning_config_path = f"{dino_wm_dir}/conf/{args.planning_config}.yaml"
if os.path.exists(planning_config_path):
    planning_cfg = OmegaConf.load(planning_config_path)
    print(f"\n  Planning-Config: {args.planning_config}.yaml geladen")
else:
    print(f"\n  Planning-Config: {args.planning_config}.yaml nicht gefunden, defaults")

# --- Sigma-Scale ---
sigma_scale = None
if planning_cfg:
    _ss_cfg = OmegaConf.to_container(planning_cfg.get("sigma_scale", {}), resolve=True)
    if _ss_cfg.get("enabled", False):
        _xy_s = _ss_cfg.get("xy", 1.0)
        _z_s = _ss_cfg.get("z", 1.0)
        _g_s = _ss_cfg.get("g", 1.0)
        if base_action_dim == 8:
            _ss_base = torch.tensor([_xy_s, _xy_s, _z_s, _g_s, _xy_s, _xy_s, _z_s, _g_s])
        else:
            _ss_base = torch.tensor([_xy_s, _xy_s, _z_s, _xy_s, _xy_s, _z_s])
        sigma_scale = _ss_base.repeat(frameskip)[:full_action_dim]
        print(f"  Sigma-Scale: xy={_xy_s}, z={_z_s}, g={_g_s}")
    else:
        print(f"  Sigma-Scale: INAKTIV")

# --- Workspace Bounds ---
action_bounds = None
if planning_cfg:
    _ws_cfg = OmegaConf.to_container(planning_cfg.get("workspace_bounds", {}), resolve=True)
    if _ws_cfg.get("enabled", False) and _ws_cfg.get("lower_8d") and _ws_cfg.get("upper_8d"):
        _lower_base = torch.tensor(_ws_cfg["lower_8d"][:base_action_dim], dtype=torch.float32)
        _upper_base = torch.tensor(_ws_cfg["upper_8d"][:base_action_dim], dtype=torch.float32)
        _lower_full = _lower_base.repeat(frameskip)[:full_action_dim]
        _upper_full = _upper_base.repeat(frameskip)[:full_action_dim]
        # In normalisierten Raum konvertieren (repeated mean/std aus Preprocessor)
        _act_mean = preprocessor.action_mean[:full_action_dim]
        _act_std = preprocessor.action_std[:full_action_dim]
        norm_lower = ((_lower_full - _act_mean) / _act_std)
        norm_upper = ((_upper_full - _act_mean) / _act_std)
        action_bounds = {
            "lower": torch.min(norm_lower, norm_upper),
            "upper": torch.max(norm_lower, norm_upper),
        }
        print(f"  Workspace Bounds: AKTIV (z: [{_ws_cfg['lower_8d'][2]}, {_ws_cfg['upper_8d'][2]}])")
    else:
        print(f"  Workspace Bounds: INAKTIV")

# --- Gripper ---
gripper_indices = None
gripper_open_prior = True
gripper_mask = False
if planning_cfg:
    _g_cfg = OmegaConf.to_container(planning_cfg.get("gripper", {}), resolve=True)
    _base_indices = _g_cfg.get("base_indices", [])
    if _base_indices and base_action_dim == 8:
        gripper_indices = []
        for bi in _base_indices:
            for fs in range(frameskip):
                gi = fs * base_action_dim + bi
                if gi < full_action_dim:
                    gripper_indices.append(gi)
        gripper_open_prior = _g_cfg.get("open_prior", True)
        gripper_mask = _g_cfg.get("mask", False)
        print(f"  Gripper: indices={gripper_indices}, mask={gripper_mask}, open_prior={gripper_open_prior}")
    else:
        print(f"  Gripper: keine base_indices definiert")

# --- Feature-Visualisierung ---
feature_viz_enabled = False
feature_viz_cfg = {}
visualize_wm_prediction = False
if planning_cfg:
    feature_viz_cfg = OmegaConf.to_container(
        planning_cfg.get("feature_visualization", {}), resolve=True
    )
    feature_viz_enabled = feature_viz_cfg.get("enabled", False)
    visualize_wm_prediction = feature_viz_cfg.get("visualize_wm_prediction", False)
    if feature_viz_enabled:
        print(f"  Feature-Visualisierung: AKTIVIERT")
        print(f"    naive_pca={feature_viz_cfg.get('save_naive_pca', True)}, "
              f"kmeans_pca={feature_viz_cfg.get('save_kmeans_pca', True)}, "
              f"attention={feature_viz_cfg.get('save_attention', True)}")
        print(f"    n_clusters={feature_viz_cfg.get('n_clusters', 3)}, "
              f"interval={feature_viz_cfg.get('visualize_interval', 1)}")
        if visualize_wm_prediction:
            print(f"    wm_prediction=AKTIVIERT (Predicted | Goal | Overlay | Diff)")
    else:
        print(f"  Feature-Visualisierung: INAKTIV")

# Feature-Visualisierungs-Output-Verzeichnis (wird vom Client gesetzt via set_client_config)
# Fallback: plan_outputs/ falls kein Client-Pfad übermittelt wird
from datetime import datetime
import csv
_viz_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_viz_model_short = args.model_name.replace("/", "_").replace("-", "")
feature_viz_dir = os.path.join(dino_wm_dir, "plan_outputs", f"feature_viz_{_viz_model_short}_{_viz_timestamp}")
feature_viz_step_counter = 0
feature_viz_goal_idx = 0  # Goal-Index für Multi-Goal-Runs (Pick, Place, etc.)
current_goal_name = "goal00"  # Aktueller Goal-Name (von Client via goal_name oder aus Index)
metrics_csv_path = None  # Wird beim ersten Schreiben gesetzt
metrics_csv_initialized = False

def init_metrics_csv(csv_path: str):
    """Initialisiert die Metriken-CSV mit Header."""
    global metrics_csv_initialized
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'goal_name', 'step', 'timestamp',
            # Pixel-Level Metriken (WM-Prediction)
            'mae_goal_current', 'mae_goal_predicted',
            'mse_goal_current', 'mse_goal_predicted',
            # CEM-Optimierung (Latent-Space)
            'cem_loss_initial', 'cem_loss_final', 'cem_reduction_pct',
            # Gating-Losses (Latent-Space)
            'zero_loss', 'planned_loss', 'gating_delta', 'gating_accepted'
        ])
    metrics_csv_initialized = True
    print(f"  Metriken-CSV initialisiert: {csv_path}")

def log_metrics_to_csv(csv_path: str, goal_name: str, step: int, metrics: dict):
    """Fügt eine Zeile mit Metriken zur CSV hinzu.
    
    Args:
        csv_path: Pfad zur CSV-Datei
        goal_name: Name des aktuellen Goals (z.B. "pick", "place")
        step: Aktueller Step-Index
        metrics: Dict mit folgenden optionalen Keys:
            - mae_goal_current, mae_goal_predicted: Pixel-MAE
            - mse_goal_current, mse_goal_predicted: Pixel-MSE
            - cem_loss_initial, cem_loss_final, cem_reduction_pct: CEM-Optimierung
            - zero_loss, planned_loss, gating_delta, gating_accepted: Gating
    """
    global metrics_csv_initialized
    if not metrics_csv_initialized:
        init_metrics_csv(csv_path)
    
    # Helper für optionale Werte
    def fmt(key, decimals=6):
        val = metrics.get(key)
        if val is None:
            return ""
        if isinstance(val, bool):
            return str(val)
        return f"{val:.{decimals}f}"
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            goal_name,
            step,
            datetime.now().strftime("%H:%M:%S"),
            # Pixel-Level
            fmt('mae_goal_current'),
            fmt('mae_goal_predicted'),
            fmt('mse_goal_current'),
            fmt('mse_goal_predicted'),
            # CEM
            fmt('cem_loss_initial'),
            fmt('cem_loss_final'),
            fmt('cem_reduction_pct', 2),
            # Gating
            fmt('zero_loss'),
            fmt('planned_loss'),
            fmt('gating_delta'),
            metrics.get('gating_accepted', ''),
        ])

if feature_viz_enabled:
    print(f"    output_dir: wird vom Client gesetzt (Fallback: {feature_viz_dir})")

# Feature-Visualizer initialisieren (auch wenn disabled, für spätere Aktivierung)
feature_visualizer = PlanningFeatureVisualizer(
    encoder=model.encoder,
    device=device,
    enabled=feature_viz_enabled,
)
feature_visualizer.set_encoder_transform(model.encoder_transform)
if feature_viz_enabled:
    feature_visualizer.n_clusters = feature_viz_cfg.get("n_clusters", 3)
    feature_visualizer.n_heads_show = feature_viz_cfg.get("n_heads_show", 4)

# --- Loss-Gating Konfiguration ---
loss_gating_cfg = {}
loss_gating_enabled = False
loss_gating_mode = "combined"
if planning_cfg:
    loss_gating_cfg = OmegaConf.to_container(
        planning_cfg.get("loss_gating", {}), resolve=True
    )
    loss_gating_enabled = loss_gating_cfg.get("enabled", False)
    loss_gating_mode = loss_gating_cfg.get("mode", "combined")
    if loss_gating_enabled:
        print(f"  Loss-Gating: AKTIVIERT")
        print(f"    mode={loss_gating_mode}")
        print(f"    initial_tolerance={loss_gating_cfg.get('initial_tolerance', 0.10):.2f}, "
              f"min_tolerance={loss_gating_cfg.get('min_tolerance', 0.02):.2f}, "
              f"decay={loss_gating_cfg.get('decay', 0.95):.2f}")
        # Kein Escape-Hatch mehr: Client beendet Phase nach wiederholten Rejections
    else:
        print(f"  Loss-Gating: INAKTIV")

# Loss-Gating State (Dictionary für einfache Mutation ohne global)
gating_state = {
    "prev_loss": None,
    "tolerance": loss_gating_cfg.get("initial_tolerance", 0.10) if loss_gating_enabled else 0.10,
    "consecutive_rejections": 0,
}

def reset_loss_gating():
    """Setzt Loss-Gating State zurück (z.B. bei neuem Goal)."""
    gating_state["prev_loss"] = None
    gating_state["tolerance"] = loss_gating_cfg.get("initial_tolerance", 0.10)
    gating_state["consecutive_rejections"] = 0

# =============================================================================
# HISTORY-BUFFER für num_hist > 1
# =============================================================================
# 
# PROBLEM: Bei num_hist > 1 braucht das World Model mehrere Kontext-Frames.
# Der Server empfängt aber nur das aktuelle Bild pro plan-Aufruf.
# Ohne History-Buffer produziert das WM falsche Predictions → falscher Loss.
#
# LÖSUNG: Buffer speichert die letzten num_hist Frames und Proprio-Werte.
# Bei jedem plan-Aufruf wird das neue Frame hinzugefügt, der älteste
# Frame entfernt (FIFO), und der volle Buffer an den Planner übergeben.
#
# num_hist=1: Buffer ist de-facto deaktiviert (nur aktuelles Frame)
# num_hist=4: Buffer enthält 4 Frames, neue Frames schieben alte raus

num_hist = model_cfg.num_hist
print(f"  num_hist={num_hist} (History-Buffer {'AKTIVIERT' if num_hist > 1 else 'nicht nötig'})")

# Buffer-State (wird bei set_goal zurückgesetzt)
history_buffer = {
    "visual": [],    # Liste von (H, W, 3) uint8 numpy arrays
    "proprio": [],   # Liste von (3,) float32 numpy arrays
}

def reset_history_buffer():
    """Leert den History-Buffer (z.B. bei neuem Goal/Episode)."""
    history_buffer["visual"].clear()
    history_buffer["proprio"].clear()
    print(f"  History-Buffer: zurückgesetzt (num_hist={num_hist})")

def add_to_history(img: np.ndarray, ee_pos: np.ndarray):
    """Fügt Frame zum History-Buffer hinzu (FIFO).
    
    Args:
        img: (H, W, 3) uint8 RGB
        ee_pos: (3,) float32 EEF-Position
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if ee_pos is None:
        ee_pos = np.zeros(3, dtype=np.float32)
    else:
        ee_pos = np.array(ee_pos, dtype=np.float32).ravel()
    
    history_buffer["visual"].append(img)
    history_buffer["proprio"].append(ee_pos)
    
    # FIFO: älteste Frames entfernen wenn Buffer voll
    while len(history_buffer["visual"]) > num_hist:
        history_buffer["visual"].pop(0)
        history_buffer["proprio"].pop(0)

def get_obs_with_history() -> dict:
    """Erstellt Observation-Dict aus dem History-Buffer.
    
    Returns:
        dict mit "visual" (1, T, H, W, 3) und "proprio" (1, T, 3)
        wobei T = min(len(buffer), num_hist)
    
    Bei leerem Buffer: Gibt None zurück (Fehlerfall).
    Bei T < num_hist: Älteste Frames werden dupliziert (Padding).
    """
    if len(history_buffer["visual"]) == 0:
        print("  ⚠ FEHLER: History-Buffer leer!")
        return None
    
    T = len(history_buffer["visual"])
    
    # Padding: Wiederhole ältestes Frame wenn Buffer noch nicht voll
    if T < num_hist:
        # Pad mit dem ältesten Frame
        pad_count = num_hist - T
        visual_list = [history_buffer["visual"][0]] * pad_count + history_buffer["visual"]
        proprio_list = [history_buffer["proprio"][0]] * pad_count + history_buffer["proprio"]
    else:
        visual_list = history_buffer["visual"]
        proprio_list = history_buffer["proprio"]
    
    # Stack zu Arrays: (T, H, W, 3) und (T, 3)
    visual_arr = np.stack(visual_list, axis=0).astype(np.float32)  # (T, H, W, 3)
    proprio_arr = np.stack(proprio_list, axis=0).astype(np.float32)  # (T, 3)
    
    # Batch-Dimension hinzufügen: (1, T, H, W, 3) und (1, T, 3)
    return {
        "visual": visual_arr[np.newaxis, ...],    # (1, T, H, W, 3)
        "proprio": proprio_arr[np.newaxis, ...],   # (1, T, 3)
    }

def compute_gating_losses(cur_obs: dict, actions: torch.Tensor, goal_obs: dict) -> dict:
    """
    Berechnet BEIDE Losses für das Improvement-Gating im GLEICHEN Feature-Space.
    
    IMPROVEMENT-ANSATZ:
      - zero_loss:    Rollout mit Zero-Action → "Was wenn ich nichts tue?"
      - planned_loss: Rollout mit geplanter Action → "Was wenn ich die Action ausführe?"
    
    Beide durch predict() → gleicher Feature-Space → fairer Vergleich!
    
    Die Frage: "Bringt mich DIESE Action näher ans Ziel als STILLSTAND?"
    
    Args:
        cur_obs: Observation-Dict mit voller History (visual: 1, T, H, W, 3)
        actions: (1, horizon, action_dim) Actions vom CEM
        goal_obs: Goal Observation-Dict
        
    Returns:
        Dict mit {"planned_loss": float, "zero_loss": float} oder None bei Fehler
    """
    try:
        # Preprocessor anwenden
        obs_tensor = preprocessor.transform_obs(cur_obs.copy())
        obs_tensor = {k: v.float().to(device) for k, v in obs_tensor.items()}
        
        goal_tensor = preprocessor.transform_obs(goal_obs.copy())
        goal_tensor = {k: v.float().to(device) for k, v in goal_tensor.items()}
        
        # Debug: Shape prüfen
        T_obs = obs_tensor["visual"].shape[1]
        if T_obs < num_hist:
            print(f"  [Gating] ⚠ History zu kurz: {T_obs} < {num_hist}")
        
        metric = torch.nn.MSELoss(reduction="none")
        
        with torch.no_grad():
            # Goal encodieren
            z_goal = model.encode_obs(goal_tensor)
            
            # ═══════════════════════════════════════════════════════════════
            # ZWEI ROLLOUTS FÜR FAIREN VERGLEICH
            # Beide durch predict() → gleicher Feature-Space!
            # ═══════════════════════════════════════════════════════════════
            
            T_obs = obs_tensor["visual"].shape[1]
            first_action = actions[:, :1, :].to(device)
            init_actions = torch.zeros(1, T_obs, first_action.shape[-1], device=device)
            
            # Rollout 1: Mit geplanter Action
            full_actions_planned = torch.cat([init_actions, first_action], dim=1)
            z_pred_planned, _ = model.rollout(obs_tensor, full_actions_planned)
            z_after_planned = z_pred_planned["visual"][:, -1:]
            
            # Rollout 2: Mit Zero-Action ("Stillstand"/Baseline)
            # Im normalisierten Space ist 0 = Mittelwert der Trainings-Actions
            zero_action = torch.zeros(1, 1, first_action.shape[-1], device=device)
            full_actions_zero = torch.cat([init_actions, zero_action], dim=1)
            z_pred_zero, _ = model.rollout(obs_tensor, full_actions_zero)
            z_after_zero = z_pred_zero["visual"][:, -1:]
            
            # Losses berechnen - beide im Prediction-Space!
            planned_loss = metric(z_after_planned, z_goal["visual"]).mean().item()
            zero_loss = metric(z_after_zero, z_goal["visual"]).mean().item()
            
            # ─── LOGGING ───
            if loss_gating_cfg.get("log_gating", True):
                delta = zero_loss - planned_loss
                symbol = "↓" if planned_loss < zero_loss else "↑"
                print(f"  [Gating] zero={zero_loss:.4f}, planned={planned_loss:.4f}, "
                      f"delta={delta:+.4f} {symbol}")
            
            return {"planned_loss": planned_loss, "zero_loss": zero_loss}
            
    except Exception as e:
        print(f"  [Gating] Loss-Berechnung Fehler: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# CEM PLANNER
# =============================================================================

planner = hydra.utils.instantiate(
    planner_cfg,
    horizon=horizon,
    wm=model,
    action_dim=full_action_dim,
    objective_fn=objective_fn,
    preprocessor=preprocessor,
    evaluator=None,
    wandb_run=wandb_run,
    action_bounds=action_bounds,
    gripper_indices=gripper_indices,
    gripper_open_prior=gripper_open_prior,
    gripper_mask=gripper_mask,
    sigma_scale=sigma_scale,
)

search_dim = horizon * full_action_dim
print(f"\n✓ Setup komplett")
print(f"  Modus: {args.mode.upper()}")
print(f"  CEM: samples={planner_cfg.num_samples}, steps={planner_cfg.opt_steps}, "
      f"topk={planner_cfg.topk}, horizon={horizon}, var_scale={planner_cfg.var_scale}")
print(f"  Suchraum: {search_dim}D (H={horizon} × D={full_action_dim})")

# Decoder-Verfügbarkeit prüfen (für WM-Prediction Visualisierung)
has_decoder = hasattr(model, 'decoder') and model.decoder is not None
if has_decoder:
    print(f"  Decoder: VERFÜGBAR (WM-Prediction Visualisierung möglich)")
else:
    print(f"  Decoder: NICHT VERFÜGBAR (keine WM-Prediction Visualisierung)")

# =============================================================================
# HELPER: Bild → Planner-Format
# =============================================================================

def img_to_obs(img: np.ndarray, ee_pos: np.ndarray = None) -> dict:
    """Konvertiert uint8 Bild + EEF-Position zu Planner-Obs-Dict.
    
    Args:
        img: (H, W, 3) uint8 RGB Bild
        ee_pos: (3,) EEF-Position [x, y, z] in Weltkoordinaten.
                Wird vom Preprocessor z-normalisiert (proprio_mean/std).
                None → Nullen (Fallback, fuehrt zu schlechten Predictions!)
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if ee_pos is not None:
        proprio = np.array(ee_pos, dtype=np.float32).reshape(1, 1, 3)
    else:
        print("  ⚠ WARNUNG: Kein ee_pos empfangen — Proprio ist Null!")
        proprio = np.zeros((1, 1, 3), dtype=np.float32)
    return {
        "visual": img[np.newaxis, np.newaxis, ...].astype(np.float32),  # (1,1,H,W,3)
        "proprio": proprio,                                              # (1,1,3)
    }

# =============================================================================
# HELPERS
# =============================================================================

def _denorm_to_sub_actions(actions_tensor, first_only=True):
    """Denormalisiere CEM-Actions und splitte in Sub-Actions.
    
    Args:
        actions_tensor: (1, horizon, full_action_dim) normalisierte Actions
        first_only: True → nur erste Horizon-Action, False → alle
    Returns:
        sub_actions: (N, base_action_dim) numpy array
    """
    if first_only:
        denorm = preprocessor.denormalize_actions(actions_tensor[0, 0:1].cpu())
    else:
        denorm = preprocessor.denormalize_actions(actions_tensor[0].cpu())
    return rearrange(denorm, "t (f d) -> (t f) d", f=frameskip).numpy()


def get_wm_predicted_image(cur_obs: dict, actions: torch.Tensor, goal_obs: dict) -> np.ndarray:
    """
    Berechnet die WM-Vorhersage für die ERSTE Action und dekodiert das Bild.
    
    Das World Model sagt den Zustand nach der ersten geplanten Action voraus,
    d.h. den nächsten Zeitschritt von der aktuellen Beobachtung aus.
    
    WICHTIG: cur_obs enthält jetzt die volle History (num_hist Frames).
    Die Actions müssen entsprechend erweitert werden (T_obs init-actions + plan-actions).
    
    Args:
        cur_obs: Observation-Dict mit voller History (visual: 1, T, H, W, 3)
        actions: (1, horizon, full_action_dim) normalisierte Actions vom CEM
        goal_obs: Goal Observation-Dict (für Zielbild-Vergleich)
    
    Returns:
        predicted_img: (H, W, 3) uint8 RGB Bild der WM-Vorhersage nach 1 Schritt
                       oder None wenn kein Decoder verfügbar
    """
    if not has_decoder:
        return None
    
    try:
        # Preprocessor auf cur_obs anwenden (wie im CEM-Planner)
        # transform_obs: np arrays → tensors, NHWC→NCHW, /255, normalize proprio
        obs_tensor = preprocessor.transform_obs(cur_obs.copy())
        obs_tensor = {k: v.float().to(device) for k, v in obs_tensor.items()}
        
        # Nur erste Action für 1-Schritt-Vorhersage
        first_action = actions[:, :1, :].to(device)  # (1, 1, action_dim)
        
        # Bei num_hist > 1: Init-Actions für History-Frames hinzufügen
        T_obs = obs_tensor["visual"].shape[1]
        init_actions = torch.zeros(1, T_obs, first_action.shape[-1], device=device)
        full_actions = torch.cat([init_actions, first_action], dim=1)  # (1, T_obs+1, action_dim)
        
        # WM Rollout mit nur einer Action
        with torch.no_grad():
            z_obses, _ = model.rollout(obs_tensor, full_actions)
            
            # Den vorhergesagten nächsten Zustand dekodieren
            # z_obses['visual'] hat shape (1, num_pred, num_patches, emb_dim)
            obs_decoded, _ = model.decode_obs(z_obses)
            
            # Vorhergesagten nächsten Zustand extrahieren: (1, num_pred, 3, H, W) → (3, H, W)
            # Nach 1 Action ist der vorhergesagte Zustand am Index -1 (letztes/einziges Ergebnis)
            pred_visual = obs_decoded['visual'][0, -1]
            
            # Debug: Wertebereich ausgeben
            v_min, v_max = pred_visual.min().item(), pred_visual.max().item()
            print(f"  WM-Pred visual range: [{v_min:.3f}, {v_max:.3f}]")
            
            # Decoder-Output ist im Bereich [-1, 1] → auf [0, 255] skalieren
            pred_visual = (pred_visual + 1.0) / 2.0  # [-1,1] → [0,1]
            pred_img = (pred_visual * 255.0).clamp(0, 255)
            pred_img = pred_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # CHW → HWC
            
        return pred_img
        
    except Exception as e:
        print(f"  WM-Prediction Fehler: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_wm_rollout_images(cur_obs: dict, actions: torch.Tensor) -> tuple:
    """
    Berechnet das vollständige WM-Rollout und dekodiert alle Bilder.
    
    Das World Model sagt die Zustände für alle geplanten Actions voraus,
    d.h. das gesamte Horizon-Rollout von der aktuellen Beobachtung aus.
    
    WICHTIG: cur_obs enthält jetzt die volle History (num_hist Frames).
    Die Actions müssen entsprechend erweitert werden (T_obs init-actions + plan-actions).
    
    Args:
        cur_obs: Observation-Dict mit voller History (visual: 1, T, H, W, 3)
        actions: (1, horizon, full_action_dim) normalisierte Actions vom CEM
    
    Returns:
        (rollout_images, recon_img): Tuple von
            - rollout_images: Liste von (H, W, 3) uint8 RGB Bildern für jeden Horizon-Schritt
            - recon_img: (H, W, 3) uint8 RGB — Encode→Decode Rekonstruktion des letzten
                         History-Frames (für Diagnose: Preprocessing vs. Sanity-Check)
        oder ([], None) wenn kein Decoder verfügbar
    """
    if not has_decoder:
        return [], None
    
    try:
        # Preprocessor auf cur_obs anwenden
        obs_tensor = preprocessor.transform_obs(cur_obs.copy())
        obs_tensor = {k: v.float().to(device) for k, v in obs_tensor.items()}
        
        # ─── EINMALIGE SHAPE/VALUE-DIAGNOSE ───
        if not hasattr(get_wm_rollout_images, '_diag_done'):
            get_wm_rollout_images._diag_done = True
            v = cur_obs['visual']
            vt = obs_tensor['visual']
            print(f"\n  [WM-DIAG] === ERSTE ROLLOUT-DIAGNOSE ===")
            print(f"  [WM-DIAG] Input cur_obs['visual']: shape={v.shape}, dtype={v.dtype}, "
                  f"range=[{v.min():.1f}, {v.max():.1f}]")
            print(f"  [WM-DIAG] Nach transform_obs:      shape={vt.shape}, dtype={vt.dtype}, "
                  f"range=[{vt.min():.3f}, {vt.max():.3f}]")
            p = cur_obs['proprio']
            pt = obs_tensor['proprio']
            print(f"  [WM-DIAG] Input proprio:   shape={p.shape}, range=[{p.min():.4f}, {p.max():.4f}]")
            print(f"  [WM-DIAG] Norm. proprio:   shape={pt.shape}, range=[{pt.min():.3f}, {pt.max():.3f}]")
            print(f"  [WM-DIAG] proprio_mean={preprocessor.proprio_mean.numpy()}, "
                  f"proprio_std={preprocessor.proprio_std.numpy()}")
            print(f"  [WM-DIAG] action_mean (first 8)={preprocessor.action_mean[:8].numpy()}")
            print(f"  [WM-DIAG] action_std  (first 8)={preprocessor.action_std[:8].numpy()}")
            
            # ─── DATENSATZ-REFERENZBILD: Recon-MSE Vergleich ───
            # Lade ein Datensatz-Bild und schicke es durch dieselbe Pipeline.
            # Falls Recon-MSE_dataset ≈ Sanity-Check → Problem ist Bildinhalt (Domain Shift).
            # Falls Recon-MSE_dataset ≈ Recon-MSE_live → Problem ist in der Pipeline.
            try:
                import glob as _glob
                _ep_dirs = sorted(_glob.glob(os.path.join(dataset_path, "*")))
                _ep_dirs = [d for d in _ep_dirs if os.path.isdir(d) and os.path.basename(d).isdigit()]
                if _ep_dirs:
                    _obses_path = os.path.join(_ep_dirs[0], "obses.pth")
                    if os.path.exists(_obses_path):
                        _dset_img = torch.load(_obses_path, map_location='cpu', weights_only=False)
                        _dset_frame = _dset_img[0].numpy()  # (H, W, 3) float32 [~0-255]
                        # Durch identische Pipeline wie Live-Bilder
                        _dset_obs = {
                            "visual": _dset_frame[np.newaxis, np.newaxis, ...].astype(np.float32),
                            "proprio": cur_obs['proprio'].copy(),
                        }
                        _dset_tensor = preprocessor.transform_obs(_dset_obs)
                        _dset_tensor = {k: vv.float().to(device) for k, vv in _dset_tensor.items()}
                        with torch.no_grad():
                            _z_dset, _ = model.rollout(
                                _dset_tensor,
                                torch.zeros(1, 1, actions.shape[-1], device=device)
                            )
                            _dset_decoded, _ = model.decode_obs(_z_dset)
                            _dset_recon = _dset_decoded['visual'][0, 0]
                            _dset_orig = _dset_tensor['visual'][0, 0]
                            _dset_mse = ((_dset_recon - _dset_orig) ** 2).mean().item()
                        with torch.no_grad():
                            _z_live, _ = model.rollout(
                                obs_tensor,
                                torch.zeros(1, 1, actions.shape[-1], device=device)
                            )
                            _live_decoded, _ = model.decode_obs(_z_live)
                            _live_recon = _live_decoded['visual'][0, 0]
                            _live_orig = obs_tensor['visual'][0, 0]
                            _live_mse = ((_live_recon - _live_orig) ** 2).mean().item()
                        _ratio = _live_mse / _dset_mse if _dset_mse > 1e-10 else float('inf')
                        print(f"  [WM-DIAG] *** REFERENZ-VERGLEICH ***")
                        print(f"  [WM-DIAG]   Dataset-Bild Recon MSE = {_dset_mse:.6f}")
                        print(f"  [WM-DIAG]   Live-Bild    Recon MSE = {_live_mse:.6f}")
                        print(f"  [WM-DIAG]   Ratio Live/Dataset     = {_ratio:.1f}x")
                        if _ratio > 10:
                            print(f"  [WM-DIAG]   ⚠ DOMAIN SHIFT! Live-Bilder weichen stark ab.")
                            print(f"  [WM-DIAG]   → Prüfe: Kamera-Config, Render-Settings, sRGB vs Linear")
                        elif _ratio > 3:
                            print(f"  [WM-DIAG]   ⚠ Moderate Abweichung — leichter Domain Shift.")
                        else:
                            print(f"  [WM-DIAG]   ✓ Recon-Qualität ähnlich — kein Domain Shift.")
                        _pixel_diff = ((_dset_tensor['visual'][0, 0] - obs_tensor['visual'][0, 0]) ** 2).mean().item()
                        print(f"  [WM-DIAG]   Pixel-MSE Dataset↔Live = {_pixel_diff:.6f}")
                        print(f"  [WM-DIAG]   Dataset range: [{_dset_frame.min():.0f}, {_dset_frame.max():.0f}]")
                        print(f"  [WM-DIAG]   Live    range: [{v.min():.0f}, {v.max():.0f}]")
                        
                        # Speichere Vergleichsbild: Dataset vs Live (Side-by-Side)
                        try:
                            _dset_vis = ((_dset_tensor['visual'][0, 0] + 1) / 2 * 255).clamp(0, 255)
                            _dset_vis = _dset_vis.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            _live_vis = ((obs_tensor['visual'][0, 0] + 1) / 2 * 255).clamp(0, 255)
                            _live_vis = _live_vis.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            _dset_rec_vis = ((_dset_recon + 1) / 2 * 255).clamp(0, 255)
                            _dset_rec_vis = _dset_rec_vis.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            _live_rec_vis = ((_live_recon + 1) / 2 * 255).clamp(0, 255)
                            _live_rec_vis = _live_rec_vis.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            # 2×2 Grid: [Dataset-Orig | Dataset-Recon]
                            #            [Live-Orig   | Live-Recon  ]
                            from PIL import Image, ImageDraw, ImageFont
                            _grid = Image.new('RGB', (224*2 + 10, 224*2 + 10 + 40), (30, 30, 30))
                            _grid.paste(Image.fromarray(_dset_vis), (0, 40))
                            _grid.paste(Image.fromarray(_dset_rec_vis), (224 + 10, 40))
                            _grid.paste(Image.fromarray(_live_vis), (0, 224 + 10 + 40))
                            _grid.paste(Image.fromarray(_live_rec_vis), (224 + 10, 224 + 10 + 40))
                            _draw = ImageDraw.Draw(_grid)
                            _draw.text((5, 5), f"Dataset Recon MSE={_dset_mse:.6f}", fill=(200, 255, 200))
                            _draw.text((5, 20), f"Live    Recon MSE={_live_mse:.6f} ({_ratio:.1f}x)", fill=(255, 200, 200))
                            _diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_outputs")
                            os.makedirs(_diag_dir, exist_ok=True)
                            _diag_path = os.path.join(_diag_dir, "domain_shift_diagnostic.png")
                            _grid.save(_diag_path)
                            print(f"  [WM-DIAG]   Vergleichsbild: {_diag_path}")
                        except Exception as _img_e:
                            print(f"  [WM-DIAG]   Vergleichsbild-Fehler: {_img_e}")
                    else:
                        print(f"  [WM-DIAG] obses.pth nicht gefunden in {_ep_dirs[0]}")
                else:
                    print(f"  [WM-DIAG] Keine Episoden in {dataset_path}")
            except Exception as _e:
                print(f"  [WM-DIAG] Referenz-Vergleich fehlgeschlagen: {_e}")
            
            print(f"  [WM-DIAG] ===========================\n")
        
        horizon = actions.shape[1]
        actions_device = actions.to(device)
        
        # Bei num_hist > 1: Init-Actions für History-Frames hinzufügen
        T_obs = obs_tensor["visual"].shape[1]
        init_actions = torch.zeros(1, T_obs, actions_device.shape[-1], device=device)
        full_actions = torch.cat([init_actions, actions_device], dim=1)  # (1, T_obs+horizon, action_dim)
        
        # WM Rollout mit allen Actions
        rollout_images = []
        recon_img = None
        with torch.no_grad():
            z_obses, _ = model.rollout(obs_tensor, full_actions)
            
            # Alle vorhergesagten Zustände dekodieren
            obs_decoded, _ = model.decode_obs(z_obses)
            
            # obs_decoded['visual'] hat shape (1, T_obs + horizon + 1, 3, H, W)
            # Die ersten T_obs Frames sind Rekonstruktionen der History-Frames,
            # die PREDICTED Frames beginnen ab Index T_obs.
            
            # ─── DIAGNOSE: Rekonstruktionsqualität des letzten History-Frames ───
            # Vergleich: Originalbild (nach Transform) vs Encode→Decode Rekonstruktion.
            # GLEICHE Metrik wie wm_sanity_check.py → direkt vergleichbar!
            # Falls Recon-MSE >> Sanity-Check-Recon-MSE → Preprocessing-Problem.
            recon_visual = obs_decoded['visual'][0, T_obs - 1]  # (3, H, W) letzter History-Frame
            orig_visual = obs_tensor['visual'][0, T_obs - 1]    # (3, H, W) Original (transformiert)
            recon_mse = ((recon_visual - orig_visual) ** 2).mean().item()
            recon_psnr = -10.0 * np.log10(recon_mse + 1e-10)
            # Auch für den ersten Prediction-Frame vs letzten History-Frame (= wie viel ändert sich?)
            pred_first = obs_decoded['visual'][0, T_obs]  # erster Prediction-Frame
            pred_vs_input_mse = ((pred_first - orig_visual) ** 2).mean().item()
            print(f"  [WM-DIAG] Recon MSE={recon_mse:.6f} PSNR={recon_psnr:.1f}dB | "
                  f"Pred[0]-vs-Input MSE={pred_vs_input_mse:.6f} | "
                  f"Decoded range=[{obs_decoded['visual'].min():.2f}, {obs_decoded['visual'].max():.2f}]")
            
            # Rekonstruktionsbild für visuelle Inspektion (Diagnose)
            recon_vis = (recon_visual + 1.0) / 2.0
            recon_img = (recon_vis * 255.0).clamp(0, 255)
            recon_img = recon_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            for t in range(horizon):
                pred_visual = obs_decoded['visual'][0, T_obs + t]  # (3, H, W)
                
                # Decoder-Output ist im Bereich [-1, 1] → auf [0, 255] skalieren
                pred_visual = (pred_visual + 1.0) / 2.0  # [-1,1] → [0,1]
                pred_img = (pred_visual * 255.0).clamp(0, 255)
                pred_img = pred_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # CHW → HWC
                rollout_images.append(pred_img)
        
        return rollout_images, recon_img
        
    except Exception as e:
        print(f"  WM-Rollout Fehler: {e}")
        import traceback
        traceback.print_exc()
        return [], None


def extract_goal_image(goal_obs: dict) -> np.ndarray:
    """Extrahiert das aktuellste Bild aus einem Observation-Dict als uint8 RGB.
    
    Bei Single-Frame (1, 1, H, W, 3): Extrahiert das einzige Frame.
    Bei Multi-Frame (1, T, H, W, 3): Extrahiert das LETZTE Frame (das aktuellste).
    """
    visual = goal_obs['visual']  # (1, T, H, W, 3) float32 [0,255]
    # -1 nimmt das letzte Frame (bei T=1 identisch mit 0)
    img = visual[0, -1].astype(np.uint8)
    return img

# =============================================================================
# SOCKET SERVER
# =============================================================================

warm_start = None
goal_obs = None

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", args.port))
server.listen(1)
print(f"\nServer laeuft auf localhost:{args.port}")
print("Warte auf Client...\n")

while True:
    conn, addr = server.accept()
    print(f"[+] Client: {addr}")

    try:
        while True:
            # Empfangen
            size_data = conn.recv(8)
            if not size_data:
                break
            data_size = int.from_bytes(size_data, 'big')
            if data_size == 0:
                break

            data = b""
            while len(data) < data_size:
                data += conn.recv(min(65536, data_size - len(data)))

            msg = pickle.loads(data)
            cmd = msg.get("cmd")

            if cmd == "get_run_info":
                response = {"status": "ok", "run_info": server_config}
                print(f"  Run-Info gesendet")

            elif cmd == "set_client_config":
                # Client sendet seine komplette Konfiguration
                client_cfg = msg.get("config", {})
                wandb_run.update_config(client_cfg)
                
                # Feature-Viz-Verzeichnis aus Client-Output-Pfad setzen (Fallback)
                if feature_viz_enabled and "output_dir" in client_cfg:
                    feature_viz_dir = os.path.join(client_cfg["output_dir"], "feature_viz")
                    print(f"  Feature-Viz-Dir (Fallback): {feature_viz_dir}")
                
                response = {"status": "ok", "wandb_run_id": wandb_run.id}
                print(f"  Client-Config empfangen: {list(client_cfg.keys())}")

            elif cmd == "set_episode_dir":
                # Client startet neue Episode und sendet Episodenordner
                episode_dir = msg.get("episode_dir")
                if feature_viz_enabled and episode_dir:
                    feature_viz_dir = os.path.join(episode_dir, "feature_viz")
                    os.makedirs(feature_viz_dir, exist_ok=True)
                    feature_viz_step_counter = 0  # Reset für neue Episode
                    feature_viz_goal_idx = 0
                    feature_visualizer.reset_centroids()  # PCA/K-Means zurücksetzen
                    print(f"  Episode-Dir: {episode_dir}")
                    print(f"  Feature-Viz-Dir: {feature_viz_dir}, Step-Counter=0")
                
                # History-Buffer zurücksetzen bei neuer Episode
                # (neue Episode = neue Trajectory, alte Frames sind irrelevant)
                reset_history_buffer()
                
                # Loss-Gating auch zurücksetzen
                if loss_gating_enabled:
                    reset_loss_gating()
                
                response = {"status": "ok"}

            elif cmd == "log_step":
                # Client sendet Per-Step-Metriken (Distanzen)
                step_data = msg.get("data", {})
                wandb_run.log_step(step_data)
                response = {"status": "ok"}

            elif cmd == "log_episode":
                # Client sendet Episode-Metriken (Cube-Distanzen, Erfolg, etc.)
                episode_data = msg.get("data", {})
                wandb_run.log_episode(episode_data)
                response = {"status": "ok"}
                print(f"  Episode-Log: {episode_data}")

            elif cmd == "set_goal":
                # Goal-Name vom Client (z.B. "pick", "place") oder Fallback auf Index
                goal_name = msg.get("goal_name")
                if goal_name:
                    current_goal_name = goal_name
                else:
                    # Bei neuem Goal nach vorherigem: goal_idx erhöhen
                    if goal_obs is not None:
                        feature_viz_goal_idx += 1
                    current_goal_name = f"goal{feature_viz_goal_idx:02d}"
                
                # Step-Counter läuft durch (KEIN Reset) für chronologische Sortierung
                print(f"  Neues Goal '{current_goal_name}', Step-Counter bei {feature_viz_step_counter}")
                
                img = np.array(msg["image"])
                ee_pos = np.array(msg["ee_pos"]) if "ee_pos" in msg else None
                goal_obs = img_to_obs(img, ee_pos)
                print(f"  Goal '{current_goal_name}' gesetzt: {img.shape} {img.dtype}, ee_pos={ee_pos}")
                
                # Loss-Gating State zurücksetzen bei neuem Goal
                if loss_gating_enabled:
                    reset_loss_gating()
                    print(f"  Loss-Gating: State zurückgesetzt (tolerance={gating_state['tolerance']:.2f})")
                
                # History-Buffer zurücksetzen bei neuem Goal
                # (neues Goal = neue Trajectory, alte Frames sind irrelevant)
                reset_history_buffer()
                
                # Feature-Visualisierung für Goal-Bild (verwendet durchlaufenden step_counter)
                if feature_viz_enabled and feature_viz_cfg.get("visualize_goal", True):
                    try:
                        viz_paths = feature_visualizer.visualize(
                            img_np=img, out_dir=feature_viz_dir, step_idx=feature_viz_step_counter,
                            prefix=current_goal_name,
                            save_naive_pca=feature_viz_cfg.get("save_naive_pca", True),
                            save_kmeans_pca=feature_viz_cfg.get("save_kmeans_pca", True),
                            save_attention=feature_viz_cfg.get("save_attention", True),
                            is_goal=True,  # PCA/K-Means für dieses Goal-Bild fitten und speichern
                        )
                        feature_viz_step_counter += 1  # Nach Goal-Bild inkrementieren
                        print(f"  Feature-Viz Goal Step {feature_viz_step_counter-1}: {feature_viz_dir}")
                    except Exception as e:
                        print(f"  Feature-Viz Fehler: {e}")
                
                response = {"status": "ok"}

            elif cmd == "plan":
                if goal_obs is None:
                    response = {"status": "error", "msg": "No goal set"}
                else:
                    img = np.array(msg["image"])
                    ee_pos = np.array(msg["ee_pos"]) if "ee_pos" in msg else None
                    
                    # History-Buffer aktualisieren (aktuelles Frame hinzufügen)
                    add_to_history(img, ee_pos)
                    
                    # Observation mit voller History für CEM und Loss-Gating
                    cur_obs = get_obs_with_history()
                    if cur_obs is None:
                        response = {"status": "error", "msg": "History buffer empty"}
                    else:
                        # Für img_to_obs fallback (Feature-Viz etc. braucht nur aktuelles Frame)
                        cur_obs_single = img_to_obs(img, ee_pos)

                        # Warm-Start: vorherigen Plan shiften
                        actions_init = None
                        if warm_start is not None:
                            shifted = warm_start[:, 1:, :]
                            actions_init = torch.cat([shifted, warm_start[:, -1:, :]], dim=1)

                        wandb_run.reset()
                        t0 = time.time()
                        with torch.no_grad():
                            actions, _ = planner.plan(
                                obs_0=cur_obs, obs_g=goal_obs, actions=actions_init)
                        t_plan = time.time() - t0
                        warm_start = actions.clone()

                        print(f"  Plan: {wandb_run.get_summary(t_plan)}")
                        
                        # ─── Loss-Gating (IMPROVEMENT-ANSATZ) ───
                        # Zwei Rollouts im GLEICHEN Feature-Space:
                        #   1. Mit Zero-Action ("Stillstand") → zero_loss
                        #   2. Mit geplanter Action           → planned_loss
                        # Akzeptiere WENN: planned_loss < zero_loss (Action bringt uns näher ans Ziel)
                        # WICHTIG: Gating ZUERST, dann nur bei Akzeptanz visualisieren!
                        action_rejected = False
                        gating_result = None  # Initialisieren für späteren Check
                        if loss_gating_enabled and loss_gating_mode != "disabled":
                            gating_result = compute_gating_losses(cur_obs, actions, goal_obs)
                            
                            if gating_result is not None:
                                zero_loss = gating_result["zero_loss"]
                                planned_loss = gating_result["planned_loss"]
                                tolerance = gating_state["tolerance"]
                                
                                # Improvement-Kriterium: Geplante Action muss besser sein als Stillstand
                                # Mit Toleranz: planned_loss < zero_loss * (1 + tolerance)
                                # D.h. Action darf etwas schlechter sein wenn tolerance > 0
                                threshold = zero_loss * (1 + tolerance)
                                
                                if planned_loss < threshold:
                                    # Akzeptiert: Action ist besser als Stillstand (oder im Toleranzrahmen)
                                    improvement = (zero_loss - planned_loss) / max(zero_loss, 1e-8) * 100
                                    if loss_gating_cfg.get("log_gating", True):
                                        print(f"  [Gating] ✓ AKZEPTIERT: planned={planned_loss:.4f} < threshold={threshold:.4f} "
                                              f"(zero={zero_loss:.4f}, improvement={improvement:+.1f}%)")
                                    # Toleranz reduzieren nach erfolgreicher Action
                                    gating_state["tolerance"] = max(
                                        loss_gating_cfg.get("min_tolerance", 0.02),
                                        gating_state["tolerance"] * loss_gating_cfg.get("decay", 0.95)
                                    )
                                    gating_state["consecutive_rejections"] = 0
                                else:
                                    # Abgelehnt: Action ist nicht besser als Stillstand
                                    action_rejected = True
                                    gating_state["consecutive_rejections"] += 1
                                    degradation = (planned_loss - zero_loss) / max(zero_loss, 1e-8) * 100
                                    if loss_gating_cfg.get("log_gating", True):
                                        print(f"  [Gating] ✗ ABGELEHNT #{gating_state['consecutive_rejections']}: "
                                              f"planned={planned_loss:.4f} >= threshold={threshold:.4f} "
                                              f"(zero={zero_loss:.4f}, degradation={degradation:+.1f}%)")

                        # Response basierend auf Gating-Ergebnis
                        # ─── STEP-METRIKEN SAMMELN ───
                        # Alle relevanten Metriken für diesen Step (für CSV)
                        step_metrics = {}
                        
                        # CEM-Loss (aus wandb_run)
                        if wandb_run._losses:
                            step_metrics["cem_loss_initial"] = wandb_run._losses[0]
                            step_metrics["cem_loss_final"] = wandb_run._losses[-1]
                            if wandb_run._losses[0] > 0:
                                step_metrics["cem_reduction_pct"] = (
                                    (1 - wandb_run._losses[-1] / wandb_run._losses[0]) * 100
                                )
                        
                        # Gating-Losses (falls Gating aktiv war)
                        if loss_gating_enabled and loss_gating_mode != "disabled" and gating_result is not None:
                            step_metrics["zero_loss"] = gating_result["zero_loss"]
                            step_metrics["planned_loss"] = gating_result["planned_loss"]
                            step_metrics["gating_delta"] = gating_result["zero_loss"] - gating_result["planned_loss"]
                            step_metrics["gating_accepted"] = not action_rejected
                        
                        # ─── CEM-Info für Client-Logging ───
                        cem_info = {
                            "plan_time": t_plan,
                            "phase": current_goal_name,
                            "opt_steps": int(planner_cfg.opt_steps),
                            "num_samples": int(planner_cfg.num_samples),
                        }
                        if wandb_run._losses:
                            cem_info["loss_initial"] = wandb_run._losses[0]
                            cem_info["loss_final"] = wandb_run._losses[-1]
                            if wandb_run._losses[0] > 0:
                                cem_info["loss_reduction_pct"] = (
                                    (1 - wandb_run._losses[-1] / wandb_run._losses[0]) * 100
                                )
                        if gating_result is not None:
                            cem_info["zero_loss"] = gating_result["zero_loss"]
                            cem_info["planned_loss"] = gating_result["planned_loss"]

                        if action_rejected:
                            # Action abgelehnt: Client soll Position halten
                            # KEINE Feature-Visualisierung bei Rejection, aber CSV-Log!
                            if step_metrics and feature_viz_dir:
                                csv_path = os.path.join(feature_viz_dir, "step_metrics.csv")
                                log_metrics_to_csv(csv_path, current_goal_name, feature_viz_step_counter, step_metrics)
                            
                            response = {
                                "status": "ok",
                                "actions": None,  # Keine Action → Position halten
                                "rejected": True,
                                "reason": "loss_gating",
                                "cem_info": cem_info,
                            }
                        else:
                            # Action akzeptiert: Normale Response
                            all_sub = _denorm_to_sub_actions(actions, first_only=True)
                            n_ret = args.n_sub_actions if 0 < args.n_sub_actions < len(all_sub) else len(all_sub)
                            sub_actions = all_sub[:n_ret]

                            wandb_run.log_plan_summary(t_plan, len(sub_actions), mode="plan")
                            _ee_start = 4 if base_action_dim == 8 else 3
                            for si, sa in enumerate(sub_actions):
                                print(f"    Sub-Action {si}: target_ee=[{sa[_ee_start]:.3f}, {sa[_ee_start+1]:.3f}, {sa[_ee_start+2]:.3f}]")
                            
                            # ─── Feature-Visualisierung NUR bei akzeptierten Aktionen ───
                            viz_interval = feature_viz_cfg.get("visualize_interval", 1)
                            current_prefix = f"{current_goal_name}_current"
                            wm_pred_prefix = f"{current_goal_name}_wm_pred"
                            rollout_prefix = f"{current_goal_name}_rollout"
                            
                            # Unterordner für WM-Prediction und Rollout
                            wm_pred_viz_dir = os.path.join(feature_viz_dir, "wm_prediction")
                            rollout_viz_dir = os.path.join(feature_viz_dir, "rollout")
                            
                            if (feature_viz_enabled and 
                                feature_viz_cfg.get("visualize_current", True) and
                                feature_viz_step_counter % viz_interval == 0):
                                try:
                                    viz_paths = feature_visualizer.visualize(
                                        img_np=img, out_dir=feature_viz_dir,
                                        step_idx=feature_viz_step_counter, prefix=current_prefix,
                                        save_naive_pca=feature_viz_cfg.get("save_naive_pca", True),
                                        save_kmeans_pca=feature_viz_cfg.get("save_kmeans_pca", True),
                                        save_attention=feature_viz_cfg.get("save_attention", True),
                                    )
                                    print(f"  Feature-Viz {current_goal_name} Step {feature_viz_step_counter}")
                                except Exception as e:
                                    print(f"  Feature-Viz Fehler: {e}")
                            
                            # WM-Prediction Visualisierung nach CEM-Optimierung
                            if (feature_viz_enabled and 
                                visualize_wm_prediction and
                                feature_viz_step_counter % viz_interval == 0):
                                try:
                                    pred_img = get_wm_predicted_image(cur_obs, actions, goal_obs)
                                    if pred_img is not None:
                                        os.makedirs(wm_pred_viz_dir, exist_ok=True)
                                        goal_img_np = extract_goal_image(goal_obs)
                                        current_img_np = extract_goal_image(cur_obs)  # Aktueller realer Zustand
                                        viz_result = feature_visualizer.visualize_wm_prediction(
                                            predicted_img=pred_img,
                                            goal_img=goal_img_np,
                                            current_img=current_img_np,
                                            out_dir=wm_pred_viz_dir,
                                            step_idx=feature_viz_step_counter,
                                            prefix=wm_pred_prefix
                                        )
                                        # Pixel-Metriken zu step_metrics hinzufügen
                                        if viz_result.get("metrics"):
                                            step_metrics.update(viz_result["metrics"])
                                except Exception as e:
                                    print(f"  WM-Prediction Viz Fehler: {e}")
                            
                            # Rollout-Strip: Komplettes Horizon-Rollout als Bildzeile
                            if (feature_viz_enabled and 
                                feature_viz_cfg.get("visualize_rollout_strip", True) and
                                visualize_wm_prediction and
                                feature_viz_step_counter % viz_interval == 0):
                                try:
                                    rollout_images, recon_img = get_wm_rollout_images(cur_obs, actions)
                                    if len(rollout_images) > 0:
                                        os.makedirs(rollout_viz_dir, exist_ok=True)
                                        goal_img_np = extract_goal_image(goal_obs)
                                        current_img_np = extract_goal_image(cur_obs)
                                        feature_visualizer.visualize_rollout_strip(
                                            current_img=current_img_np,
                                            rollout_images=rollout_images,
                                            goal_img=goal_img_np,
                                            out_dir=rollout_viz_dir,
                                            step_idx=feature_viz_step_counter,
                                            prefix=rollout_prefix,
                                            recon_img=recon_img,
                                        )
                                except Exception as e:
                                    print(f"  Rollout-Strip Fehler: {e}")
                            
                            # ─── ALLE METRIKEN IN CSV SCHREIBEN ───
                            if step_metrics and feature_viz_dir:
                                csv_path = os.path.join(feature_viz_dir, "step_metrics.csv")
                                log_metrics_to_csv(csv_path, current_goal_name, feature_viz_step_counter, step_metrics)
                            
                            # Step-Counter NUR bei akzeptierten Aktionen inkrementieren
                            feature_viz_step_counter += 1
                            
                            response = {
                                "status": "ok",
                                "actions": sub_actions.tolist(),
                                "n_actions": len(sub_actions),
                                "action_dim": base_action_dim,
                                "cem_info": cem_info,
                            }

            elif cmd == "plan_all":
                if goal_obs is None:
                    response = {"status": "error", "msg": "No goal set"}
                else:
                    img = np.array(msg["image"])
                    ee_pos = np.array(msg["ee_pos"]) if "ee_pos" in msg else None
                    cur_obs = img_to_obs(img, ee_pos)

                    wandb_run.reset()
                    t0 = time.time()
                    with torch.no_grad():
                        actions, _ = planner.plan(obs_0=cur_obs, obs_g=goal_obs)
                    t_plan = time.time() - t0

                    print(f"  PlanAll: {wandb_run.get_summary(t_plan)}")
                    
                    # Feature-Visualisierung für aktuelles Bild
                    viz_interval = feature_viz_cfg.get("visualize_interval", 1)
                    current_prefix = f"{current_goal_name}_current"
                    wm_pred_prefix = f"{current_goal_name}_wm_pred"
                    rollout_prefix = f"{current_goal_name}_rollout"
                    
                    # Unterordner für WM-Prediction und Rollout
                    wm_pred_viz_dir = os.path.join(feature_viz_dir, "wm_prediction")
                    rollout_viz_dir = os.path.join(feature_viz_dir, "rollout")
                    
                    if (feature_viz_enabled and 
                        feature_viz_cfg.get("visualize_current", True) and
                        feature_viz_step_counter % viz_interval == 0):
                        try:
                            viz_paths = feature_visualizer.visualize(
                                img_np=img, out_dir=feature_viz_dir,
                                step_idx=feature_viz_step_counter, prefix=current_prefix,
                                save_naive_pca=feature_viz_cfg.get("save_naive_pca", True),
                                save_kmeans_pca=feature_viz_cfg.get("save_kmeans_pca", True),
                                save_attention=feature_viz_cfg.get("save_attention", True),
                            )
                            print(f"  Feature-Viz {current_goal_name} Step {feature_viz_step_counter}")
                        except Exception as e:
                            print(f"  Feature-Viz Fehler: {e}")
                    
                    # WM-Prediction Visualisierung nach CEM-Optimierung
                    # Step-Metriken für plan_all
                    step_metrics = {}
                    
                    # CEM-Loss
                    if wandb_run._losses:
                        step_metrics["cem_loss_initial"] = wandb_run._losses[0]
                        step_metrics["cem_loss_final"] = wandb_run._losses[-1]
                        if wandb_run._losses[0] > 0:
                            step_metrics["cem_reduction_pct"] = (
                                (1 - wandb_run._losses[-1] / wandb_run._losses[0]) * 100
                            )
                    
                    if (feature_viz_enabled and 
                        visualize_wm_prediction and
                        feature_viz_step_counter % viz_interval == 0):
                        try:
                            pred_img = get_wm_predicted_image(cur_obs, actions, goal_obs)
                            if pred_img is not None:
                                os.makedirs(wm_pred_viz_dir, exist_ok=True)
                                goal_img_np = extract_goal_image(goal_obs)
                                current_img_np = extract_goal_image(cur_obs)  # Aktueller realer Zustand
                                viz_result = feature_visualizer.visualize_wm_prediction(
                                    predicted_img=pred_img,
                                    goal_img=goal_img_np,
                                    current_img=current_img_np,
                                    out_dir=wm_pred_viz_dir,
                                    step_idx=feature_viz_step_counter,
                                    prefix=wm_pred_prefix
                                )
                                # Pixel-Metriken zu step_metrics
                                if viz_result.get("metrics"):
                                    step_metrics.update(viz_result["metrics"])
                        except Exception as e:
                            print(f"  WM-Prediction Viz Fehler: {e}")
                    
                    # Rollout-Strip: Komplettes Horizon-Rollout als Bildzeile
                    if (feature_viz_enabled and 
                        feature_viz_cfg.get("visualize_rollout_strip", True) and
                        visualize_wm_prediction and
                        feature_viz_step_counter % viz_interval == 0):
                        try:
                            rollout_images, recon_img = get_wm_rollout_images(cur_obs, actions)
                            if len(rollout_images) > 0:
                                os.makedirs(rollout_viz_dir, exist_ok=True)
                                goal_img_np = extract_goal_image(goal_obs)
                                current_img_np = extract_goal_image(cur_obs)
                                feature_visualizer.visualize_rollout_strip(
                                    current_img=current_img_np,
                                    rollout_images=rollout_images,
                                    goal_img=goal_img_np,
                                    out_dir=rollout_viz_dir,
                                    step_idx=feature_viz_step_counter,
                                    prefix=rollout_prefix,
                                    recon_img=recon_img,
                                )
                        except Exception as e:
                            print(f"  Rollout-Strip Fehler: {e}")
                    
                    # ─── ALLE METRIKEN IN CSV SCHREIBEN ───
                    if step_metrics and feature_viz_dir:
                        csv_path = os.path.join(feature_viz_dir, "step_metrics.csv")
                        log_metrics_to_csv(csv_path, current_goal_name, feature_viz_step_counter, step_metrics)
                    
                    feature_viz_step_counter += 1

                    all_sub = _denorm_to_sub_actions(actions, first_only=False)
                    n_ret = args.n_sub_actions if 0 < args.n_sub_actions < frameskip else frameskip
                    if n_ret < frameskip:
                        n_horizon = len(all_sub) // frameskip
                        kept = [all_sub[h*frameskip : h*frameskip+n_ret] for h in range(n_horizon)]
                        all_actions = np.concatenate(kept, axis=0)
                    else:
                        all_actions = all_sub
                    cem_info_all = {
                        "plan_time": t_plan,
                        "phase": current_goal_name,
                        "opt_steps": int(planner_cfg.opt_steps),
                        "num_samples": int(planner_cfg.num_samples),
                    }
                    if wandb_run._losses:
                        cem_info_all["loss_initial"] = wandb_run._losses[0]
                        cem_info_all["loss_final"] = wandb_run._losses[-1]
                        if wandb_run._losses[0] > 0:
                            cem_info_all["loss_reduction_pct"] = (
                                (1 - wandb_run._losses[-1] / wandb_run._losses[0]) * 100
                            )
                    response = {
                        "status": "ok",
                        "actions": all_actions.tolist(),
                        "n_actions": len(all_actions),
                        "plan_time": t_plan,
                        "cem_info": cem_info_all,
                    }
                    wandb_run.log_plan_summary(t_plan, len(all_actions), mode="plan_all")
                    print(f"  → {len(all_actions)} Actions in {t_plan:.1f}s")

            elif cmd == "reset":
                goal_obs = None
                warm_start = None
                feature_viz_step_counter = 0  # Reset step counter
                feature_viz_goal_idx = 0  # Reset goal index
                current_goal_name = "goal00"  # Reset goal name
                metrics_csv_initialized = False  # Neue CSV bei neuem Run
                feature_visualizer.reset_centroids()  # K-Means Zentroide zurücksetzen
                response = {"status": "ok"}
                print(f"  Reset (Goal-Name, Step-Counter, PCA zurückgesetzt)")

            elif cmd == "quit":
                wandb_run.finish()
                response = {"status": "ok"}
                pickle_resp = pickle.dumps(response)
                conn.sendall(len(pickle_resp).to_bytes(8, 'big') + pickle_resp)
                break

            else:
                response = {"status": "error", "msg": f"Unknown: {cmd}"}

            # Senden
            pickle_resp = pickle.dumps(response)
            conn.sendall(len(pickle_resp).to_bytes(8, 'big') + pickle_resp)

    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        conn.close()
        print(f"[-] Client getrennt\n")
