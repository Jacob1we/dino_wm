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
from pathlib import Path
from omegaconf import OmegaConf

from plan import load_model, DummyWandbRun
from preprocessor import Preprocessor
from utils import seed

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

base_action_dim = _full_dset.action_dim
dset_transform = _full_dset.transform
action_mean = _full_dset.action_mean.clone()
action_std = _full_dset.action_std.clone()
state_mean = _full_dset.state_mean.clone()
state_std = _full_dset.state_std.clone()
proprio_mean = _full_dset.proprio_mean.clone()
proprio_std = _full_dset.proprio_std.clone()

del _full_dset, _transform, _dset_cfg
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"  action_dim={base_action_dim}, Stats extrahiert, Dataset freigegeben ✓")

# 3. Model laden (wie plan.py)
print("Lade Model...")
model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
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

# WandbRun mit Loss-Logging auf stdout
class LoggingRun:
    """Minimales Loss-Logging auf stdout (kein W&B noetig)."""
    def __init__(self):
        self.mode = "logging"
        self._losses = []
        self._plan_count = 0

    def log(self, data, *args, **kwargs):
        for key, value in data.items():
            if "loss" in key:
                self._losses.append(value)
                print(f"    CEM step {data.get('step', '?')}: loss={value:.6f}", flush=True)

    def get_summary(self, plan_time=None):
        if not self._losses:
            return "keine Daten"
        first, last = self._losses[0], self._losses[-1]
        pct = (1 - last/first) * 100 if first > 0 else 0
        return f"{first:.6f} → {last:.6f} ({pct:.1f}% Reduktion, {plan_time:.1f}s)"

    def reset(self):
        self._losses.clear()
        self._plan_count += 1

    def watch(self, *a, **kw): pass
    def config(self, *a, **kw): pass
    def finish(self): pass

wandb_run = LoggingRun()

planner = hydra.utils.instantiate(
    planner_cfg,
    horizon=horizon,
    wm=model,
    action_dim=full_action_dim,
    objective_fn=objective_fn,
    preprocessor=preprocessor,
    evaluator=None,  # Keine Evaluation im Server (macht der Client)
    wandb_run=wandb_run,
)

search_dim = horizon * full_action_dim
print(f"\n✓ Setup komplett")
print(f"  Modus: {args.mode.upper()}")
print(f"  CEM: samples={planner_cfg.num_samples}, steps={planner_cfg.opt_steps}, "
      f"topk={planner_cfg.topk}, horizon={horizon}")
print(f"  Suchraum: {search_dim}D (H={horizon} × D={full_action_dim})")

# =============================================================================
# HELPER: Bild → Planner-Format
# =============================================================================

def img_to_obs(img: np.ndarray, ee_pos: np.ndarray = None) -> dict:
    """Konvertiert uint8 Bild + EEF-Position zu Planner-Obs-Dict.
    
    Args:
        img: (H, W, 3) uint8 BGR Bild
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
# SOCKET SERVER
# =============================================================================

warm_start = None  # (1, horizon, action_dim) fuer MPC Warm-Start
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

            if cmd == "set_goal":
                img = np.array(msg["image"])
                ee_pos = np.array(msg["ee_pos"]) if "ee_pos" in msg else None
                goal_obs = img_to_obs(img, ee_pos)
                print(f"  Goal gesetzt: {img.shape} {img.dtype}, ee_pos={ee_pos}")
                response = {"status": "ok"}

            elif cmd == "plan":
                if goal_obs is None:
                    response = {"status": "error", "msg": "No goal set"}
                else:
                    img = np.array(msg["image"])
                    ee_pos = np.array(msg["ee_pos"]) if "ee_pos" in msg else None
                    cur_obs = img_to_obs(img, ee_pos)

                    # Warm-Start: vorherigen Plan shiften
                    # Letzte Action wird wiederholt statt Nullen (vermeidet Null-Bias)
                    actions_init = None
                    if warm_start is not None:
                        shifted = warm_start[:, 1:, :]
                        last_action = warm_start[:, -1:, :]  # Letzte bekannte Action
                        actions_init = torch.cat([shifted, last_action], dim=1)

                    wandb_run.reset()
                    t0 = time.time()
                    with torch.no_grad():
                        actions, _ = planner.plan(
                            obs_0=cur_obs, obs_g=goal_obs, actions=actions_init)
                    t_plan = time.time() - t0
                    warm_start = actions.clone()

                    print(f"  Plan: {wandb_run.get_summary(t_plan)}")

                    # Erste Horizon-Aktion → frameskip Sub-Actions
                    denorm = preprocessor.denormalize_actions(actions[0, 0:1].cpu())
                    sub_actions = rearrange(denorm, "t (f d) -> (t f) d",
                                            f=frameskip).numpy()
                    # Diagnostik: denormalisierte Zielposition
                    for si, sa in enumerate(sub_actions):
                        print(f"    Sub-Action {si}: target_ee=[{sa[3]:.3f}, {sa[4]:.3f}, {sa[5]:.3f}]")
                    response = {
                        "status": "ok",
                        "actions": sub_actions.tolist(),
                        "n_actions": len(sub_actions),
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

                    # Alle Horizon-Actions → Einzel-Steps
                    denorm = preprocessor.denormalize_actions(actions[0].cpu())
                    all_actions = rearrange(denorm, "t (f d) -> (t f) d",
                                            f=frameskip).numpy()
                    response = {
                        "status": "ok",
                        "actions": all_actions.tolist(),
                        "n_actions": len(all_actions),
                        "plan_time": t_plan,
                    }
                    print(f"  → {len(all_actions)} Actions in {t_plan:.1f}s")

            elif cmd == "reset":
                goal_obs = None
                warm_start = None
                response = {"status": "ok"}
                print(f"  Reset")

            elif cmd == "quit":
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
