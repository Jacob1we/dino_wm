"""  
DINO WM Planning Server

Nutzt plan.py Setup und ruft planner.plan() in einer Loop auf.

Modi:
  --mode online   Reduzierte CEM-Parameter fuer Echtzeit (MPC-Loop)
  --mode offline  Volle CEM-Parameter fuer bestmoegliche Qualitaet (Open-Loop)

Wichtige Best Practices (aus CEM-Analyse):
  - CEM optimiert in (horizon * action_dim * frameskip)-dimensionalem Raum
  - Online-MPC: Kurzer Horizon (1-2) + Warm-Start essenziell!
  - Offline: Langer Horizon (5) + volle Samples (300x30) ok
  - CEM Loss muss sinken -> wird jetzt pro Iteration geloggt
  - Warm-Start: Vorherigen Plan um 1 Step shiften statt von Null starten

Verwendung:
    cd ~/Desktop/dino_wm
    conda activate dino_wm
    
    # Online (schnell, fuer MPC):
    python planning_server.py --model_name 2026-02-09/08-12-44
    
    # Online mit mehr Budget:
    python planning_server.py --model_name 2026-02-09/08-12-44 --num_samples 128 --opt_steps 10
    
    # Offline (beste Qualitaet, fuer Evaluation):
    python planning_server.py --model_name 2026-02-09/08-12-44 --mode offline
"""

import os
import sys
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

# Alles aus plan.py importieren
from plan import load_model, planning_main
from preprocessor import Preprocessor
from utils import seed, cfg_to_dict


class LoggingWandbRun:
    """Ersetzt DummyWandbRun: Gibt CEM-Loss auf stdout aus.
    
    Kritisch fuer Debugging! Ohne Loss-Logging kann man nicht sehen,
    ob CEM konvergiert oder im 60D-Suchraum verloren ist.
    """
    def __init__(self):
        self.mode = "logging"
        self._step_losses = []
    
    def log(self, data, *args, **kwargs):
        # CEM loggt: {"plan_0/loss": float, "step": int}
        for key, value in data.items():
            if "loss" in key:
                step = data.get("step", "?")
                self._step_losses.append(value)
                # Kompakte Ausgabe: Loss pro CEM-Iteration
                print(f"    [CEM] Step {step}: loss={value:.6f}", flush=True)
    
    def get_loss_summary(self):
        """Gibt Loss-Verlauf zurueck fuer Konvergenz-Diagnose."""
        if not self._step_losses:
            return "keine Daten"
        first = self._step_losses[0]
        last = self._step_losses[-1]
        reduction = (1 - last/first) * 100 if first > 0 else 0
        return f"loss: {first:.6f} -> {last:.6f} ({reduction:.1f}% Reduktion)"
    
    def reset_losses(self):
        self._step_losses.clear()
    
    def watch(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass
    def finish(self): pass

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--goal_H", type=int, default=None,
                    help="Planning-Horizon (online-default: 2 fuer MPC, offline-default: 5)")
parser.add_argument("--mode", type=str, default="online", choices=["online", "offline"],
                    help="online: reduzierte CEM-Params fuer MPC | offline: volle cem.yaml Params fuer Evaluation")
# CEM-Parameter Overrides (nur relevant im Online-Modus)
parser.add_argument("--num_samples", type=int, default=None,
                    help="CEM Samples pro Iteration (online-default: 64, offline: aus cem.yaml)")
parser.add_argument("--opt_steps", type=int, default=None,
                    help="CEM Optimierungsschritte (online-default: 5, offline: aus cem.yaml)")
parser.add_argument("--topk", type=int, default=None,
                    help="CEM Top-K Eliten (online-default: 10, offline: aus cem.yaml)")
args = parser.parse_args()

# =============================================================================
# SETUP: Identisch zu planning_main() bis zum Planner
# =============================================================================

print("=" * 60)
print("DINO WM Planning Server")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed(42)

# Config laden (wie in planning_main)
model_path = f"{dino_wm_dir}/outputs/{args.model_name}/"
with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
    model_cfg = OmegaConf.load(f)

# Dataset laden (wie in planning_main)
print("Lade Dataset...")
_, dset = hydra.utils.call(
    model_cfg.env.dataset,
    num_hist=model_cfg.num_hist,
    num_pred=model_cfg.num_pred,
    frameskip=model_cfg.frameskip,
)
dset = dset["valid"]

# Model laden (wie in planning_main)
print("Lade Model...")
model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)

# Preprocessor (wie in PlanWorkspace)
# WICHTIG: action_mean/std haben Shape (action_dim,) z.B. (6,)
# Aber der Planner arbeitet mit action_dim * frameskip z.B. (12,)
# Daher müssen wir die Stats repeaten für korrekte Denormalisierung
full_action_dim = dset.action_dim * model_cfg.frameskip
action_mean_full = dset.action_mean.repeat(model_cfg.frameskip)  # (6,) -> (12,)
action_std_full = dset.action_std.repeat(model_cfg.frameskip)    # (6,) -> (12,)
print(f"Action stats: base_dim={dset.action_dim}, frameskip={model_cfg.frameskip}, full_dim={full_action_dim}")
print(f"  action_mean shape: {dset.action_mean.shape} -> {action_mean_full.shape}")

preprocessor = Preprocessor(
    action_mean=action_mean_full,
    action_std=action_std_full,
    state_mean=dset.state_mean,
    state_std=dset.state_std,
    proprio_mean=dset.proprio_mean,
    proprio_std=dset.proprio_std,
    transform=dset.transform,
)

# Planner erstellen (wie in PlanWorkspace)
objective_fn = hydra.utils.call({"_target_": "planning.objectives.create_objective_fn", "alpha": 0.5, "base": 2, "mode": "last"})
planner_cfg = OmegaConf.load(f"{dino_wm_dir}/conf/planner/cem.yaml")

frameskip = model_cfg.frameskip
base_action_dim = dset.action_dim

# Horizon: Online kurz (MPC re-plant jeden Step), Offline lang
if args.goal_H is not None:
    planning_horizon = args.goal_H
elif args.mode == "offline":
    planning_horizon = 5  # Voller Horizont fuer Qualitaet
else:
    planning_horizon = 2  # Kurz fuer MPC (weniger Dimensionen = bessere Konvergenz)

# Suchraum-Dimensionalitaet: horizon * action_dim * frameskip
search_dim = planning_horizon * base_action_dim * frameskip
print(f"\nSuchraum-Dimensionalitaet: {planning_horizon} x {base_action_dim} x {frameskip} = {search_dim}D")

if args.mode == "offline":
    # Volle cem.yaml Parameter fuer bestmoegliche Qualitaet
    # Nur ueberschreiben wenn explizit per CLI angegeben
    if args.num_samples is not None:
        planner_cfg.num_samples = args.num_samples
    if args.opt_steps is not None:
        planner_cfg.opt_steps = args.opt_steps
    if args.topk is not None:
        planner_cfg.topk = args.topk
    print(f"Modus: OFFLINE (Open-Loop, volle CEM-Qualitaet)")
    print(f"CEM-Parameter: num_samples={planner_cfg.num_samples}, opt_steps={planner_cfg.opt_steps}, topk={planner_cfg.topk}")
    print(f"  Geschaetzte DINO-Passes pro plan(): {planner_cfg.num_samples} x {planner_cfg.opt_steps} = {planner_cfg.num_samples * planner_cfg.opt_steps}")
    print(f"  ACHTUNG: Kann mehrere Minuten pro plan() dauern!")
else:
    # Reduzierte Parameter fuer Echtzeit-MPC
    planner_cfg.num_samples = args.num_samples or 64
    planner_cfg.opt_steps = args.opt_steps or 5
    planner_cfg.topk = args.topk or 10
    print(f"Modus: ONLINE (MPC, reduzierte CEM-Params)")
    print(f"CEM-Parameter: num_samples={planner_cfg.num_samples}, opt_steps={planner_cfg.opt_steps}, topk={planner_cfg.topk}")
    print(f"  Geschaetzte DINO-Passes pro plan(): {planner_cfg.num_samples} x {planner_cfg.opt_steps} = {planner_cfg.num_samples * planner_cfg.opt_steps}")

# WandbRun mit Loss-Logging (kritisch fuer CEM-Diagnose!)
wandb_run = LoggingWandbRun()

planner = hydra.utils.instantiate(
    planner_cfg,
    horizon=planning_horizon,
    wm=model,
    action_dim=dset.action_dim * model_cfg.frameskip,
    objective_fn=objective_fn,
    preprocessor=preprocessor,
    evaluator=None,
    wandb_run=wandb_run,
)

print(f"\u2713 Setup komplett! Horizon={planning_horizon}, Action_dim={dset.action_dim * model_cfg.frameskip}")
print(f"  CEM-Suchraum: {search_dim}D (horizon={planning_horizon} x full_action_dim={dset.action_dim * model_cfg.frameskip})")

# =============================================================================
# MPC WARM-START STATE
# =============================================================================
# Vorherigen Plan speichern und um 1 Step shiften fuer naechsten plan() Aufruf.
# Ohne Warm-Start startet CEM JEDES MAL von mu=0 (= Dataset-Durchschnitt).
# Mit Warm-Start: Vorheriger Plan wird um 1 Step geshiftet -> bessere Initialisierung.
warm_start_actions = None  # (1, horizon, action_dim) normalized, auf CUDA

# =============================================================================
# SOCKET SERVER: Einfache Loop die planner.plan() aufruft
# =============================================================================

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", args.port))
server.listen(1)

print(f"\nServer läuft auf localhost:{args.port}")
print("Warte auf Client...\n")

goal_obs = None

def prepare_obs_for_planner(img: np.ndarray) -> dict:
    """
    Bereitet ein RGB-Bild für den Planner vor.
    
    Der Planner erwartet via preprocessor.transform_obs():
      - visual: (B, T, H, W, C) mit C=3 (RGB), Werte 0-255, dtype float/uint8
      - proprio: (B, T, proprio_dim)
    
    Die transform_obs() Methode macht dann:
      1. rearrange zu (B, T, C, H, W)  
      2. /255.0 normalisieren
      3. transform (Normalize) anwenden
    
    Also: Wir geben einfach das uint8 Bild in (1, 1, H, W, 3) Format!
    """
    # Sicherstellen dass es uint8 ist
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Shape: (B=1, T=1, H, W, C=3)
    visual = img[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 3)
    
    # Proprio: (B=1, T=1, proprio_dim=3)
    proprio = np.zeros((1, 1, 3), dtype=np.float32)
    
    obs = {
        "visual": visual.astype(np.float32),  # transform_obs erwartet float-kompatibel
        "proprio": proprio,
    }
    
    return obs

while True:
    conn, addr = server.accept()
    print(f"[+] Client: {addr}")
    
    try:
        while True:
            # Empfangen
            size_data = conn.recv(8)
            if not size_data: break
            data_size = int.from_bytes(size_data, 'big')
            if data_size == 0: break
            
            data = b""
            while len(data) < data_size:
                data += conn.recv(min(65536, data_size - len(data)))
            
            msg = pickle.loads(data)
            cmd = msg.get("cmd")
            
            if cmd == "set_goal":
                # Goal im Format für planner.plan()
                img = np.array(msg["image"])
                print(f"  [Goal] Raw image: shape={img.shape}, dtype={img.dtype}")
                
                # Obs-Dict erstellen (preprocessor.transform_obs macht den Rest)
                goal_obs = prepare_obs_for_planner(img)
                print(f"  [Goal] Prepared: visual={goal_obs['visual'].shape}, proprio={goal_obs['proprio'].shape}")
                
                response = {"status": "ok"}
                print(f"  Goal gesetzt ✓")
                
            elif cmd == "plan":
                if goal_obs is None:
                    print(f"  [ERROR] Kein Goal gesetzt!")
                    response = {"status": "error", "msg": "No goal set"}
                else:
                    # Current obs
                    img = np.array(msg["image"])
                    print(f"  [Plan] Raw image: shape={img.shape}, dtype={img.dtype}")
                    
                    # Obs-Dict erstellen
                    cur_obs = prepare_obs_for_planner(img)
                    print(f"  [Plan] Prepared: visual={cur_obs['visual'].shape}")
                    
                    # MPC Warm-Start: Vorherigen Plan um 1 Step shiften
                    actions_init = None
                    if warm_start_actions is not None:
                        shifted = warm_start_actions[:, 1:, :]  # (1, H-1, action_dim)
                        zero_tail = torch.zeros(1, 1, warm_start_actions.shape[2], device=warm_start_actions.device)
                        actions_init = torch.cat([shifted, zero_tail], dim=1)  # (1, H, action_dim)
                        print(f"  [Plan] Warm-Start: Vorherigen Plan geshiftet")
                    
                    # Planen mit Loss-Logging
                    ns = planner_cfg.num_samples
                    os_ = planner_cfg.opt_steps
                    print(f"  [Plan] Running CEM (samples={ns}, steps={os_}, horizon={planning_horizon})...")
                    wandb_run.reset_losses()
                    t_start = time.time()
                    with torch.no_grad():
                        actions, _ = planner.plan(obs_0=cur_obs, obs_g=goal_obs, actions=actions_init)
                    t_plan = time.time() - t_start
                    
                    # Warm-Start fuer naechsten Aufruf speichern
                    warm_start_actions = actions.clone()
                    
                    # Loss-Zusammenfassung
                    print(f"  [Plan] {wandb_run.get_loss_summary()} ({t_plan:.1f}s)")
                    
                    # Erste Horizon-Aktion in frameskip Sub-Steps aufteilen (wie plan_all)
                    print(f"  [Plan] Actions shape: {actions.shape}")
                    mu_norm = actions[0].norm().item()
                    print(f"  [Plan] mu L2-Norm (normalized): {mu_norm:.4f} (0=Mittelwert, >1=signifikant)")
                    
                    # Split: (1, action_dim*frameskip) -> (frameskip, action_dim)
                    all_actions_h0 = preprocessor.denormalize_actions(actions[0, 0:1].cpu())  # (1, 12)
                    all_actions_h0 = rearrange(all_actions_h0, "t (f d) -> (t f) d", f=frameskip)  # (2, 6)
                    all_actions_np = all_actions_h0.numpy()
                    n_sub = all_actions_np.shape[0]
                    print(f"  [Plan] {n_sub} Sub-Actions (frameskip={frameskip}):")
                    for i, a in enumerate(all_actions_np):
                        print(f"    sub {i}: [{', '.join(f'{v:.4f}' for v in a)}]")
                    response = {"status": "ok", "actions": all_actions_np.tolist(), "n_actions": n_sub}
            elif cmd == "plan_all":
                # OFFLINE-Modus: Plane einmal, gib ALLE Aktionen zurueck
                if goal_obs is None:
                    print(f"  [ERROR] Kein Goal gesetzt!")
                    response = {"status": "error", "msg": "No goal set"}
                else:
                    img = np.array(msg["image"])
                    print(f"  [PlanAll] Raw image: shape={img.shape}, dtype={img.dtype}")
                    
                    cur_obs = prepare_obs_for_planner(img)
                    print(f"  [PlanAll] Prepared: visual={cur_obs['visual'].shape}")
                    
                    # Plane mit vollen CEM-Parametern
                    print(f"  [PlanAll] Running CEM planner (samples={planner_cfg.num_samples}, steps={planner_cfg.opt_steps})...")
                    print(f"  [PlanAll] Horizon={planning_horizon}, frameskip={frameskip} -> {planning_horizon * frameskip} Einzel-Actions")
                    wandb_run.reset_losses()
                    t_start = time.time()
                    with torch.no_grad():
                        actions, _ = planner.plan(obs_0=cur_obs, obs_g=goal_obs)
                    t_plan = time.time() - t_start
                    
                    # Loss-Zusammenfassung
                    print(f"  [PlanAll] {wandb_run.get_loss_summary()} ({t_plan:.1f}s)")
                    mu_norm = actions[0].norm().item()
                    print(f"  [PlanAll] mu L2-Norm (normalized): {mu_norm:.4f}")
                    
                    # Alle Aktionen denormalisieren und in Einzel-Steps auffaechern
                    # actions: (1, horizon, action_dim*frameskip) z.B. (1, 5, 12)
                    print(f"  [PlanAll] Raw actions shape: {actions.shape} (took {t_plan:.1f}s)")
                    all_actions = preprocessor.denormalize_actions(actions[0].cpu())  # (horizon, 12)
                    # Reshape: (horizon, frameskip*base_dim) -> (horizon*frameskip, base_dim)
                    all_actions = rearrange(all_actions, "t (f d) -> (t f) d", f=frameskip)  # (10, 6)
                    all_actions_np = all_actions.numpy()
                    
                    n_actions = all_actions_np.shape[0]
                    print(f"  [PlanAll] {n_actions} Einzel-Actions (shape: {all_actions_np.shape})")
                    print(f"  [PlanAll] Erste Action: {all_actions_np[0]}")
                    print(f"  [PlanAll] Letzte Action: {all_actions_np[-1]}")
                    
                    response = {
                        "status": "ok",
                        "actions": all_actions_np.tolist(),
                        "n_actions": n_actions,
                        "plan_time": t_plan,
                    }
                    print(f"  [PlanAll] {n_actions} Actions in {t_plan:.1f}s \u2713")
            elif cmd == "reset":
                # Goal und Warm-Start zuruecksetzen fuer neue Episode
                goal_obs = None
                warm_start_actions = None
                response = {"status": "ok"}
                print(f"  Reset (Goal + Warm-Start)")
                
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
