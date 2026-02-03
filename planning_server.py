"""
DINO WM Planning Server - Minimale Version

Nutzt plan.py Setup und ruft planner.plan() in einer Loop auf.

Verwendung:
    cd ~/Desktop/dino_wm
    conda activate dino_wm
    python planning_server.py --model_name 2026-02-02/22-50-30
"""

import os
import sys
import socket
import pickle
import numpy as np
import argparse

# DINO WM Pfad
dino_wm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dino_wm_dir)
os.chdir(dino_wm_dir)

import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

# Alles aus plan.py importieren
from plan import load_model, DummyWandbRun, planning_main
from preprocessor import Preprocessor
from utils import seed, cfg_to_dict

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--goal_H", type=int, default=5)
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
preprocessor = Preprocessor(
    action_mean=dset.action_mean,
    action_std=dset.action_std,
    state_mean=dset.state_mean,
    state_std=dset.state_std,
    proprio_mean=dset.proprio_mean,
    proprio_std=dset.proprio_std,
    transform=dset.transform,
)

# Planner erstellen (wie in PlanWorkspace)
objective_fn = hydra.utils.call({"_target_": "planning.objectives.create_objective_fn", "alpha": 0.5, "base": 2, "mode": "last"})
planner_cfg = OmegaConf.load(f"{dino_wm_dir}/conf/planner/cem.yaml")
planner = hydra.utils.instantiate(
    planner_cfg,
    horizon=args.goal_H,
    wm=model,
    action_dim=dset.action_dim * model_cfg.frameskip,
    objective_fn=objective_fn,
    preprocessor=preprocessor,
    evaluator=None,
    wandb_run=DummyWandbRun(),
)

print(f"✓ Setup komplett! Horizon={args.goal_H}, Action_dim={dset.action_dim * model_cfg.frameskip}")

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

def preprocess_image(img, preprocessor):
    """
    Konvertiert ein RGB uint8 Bild (H, W, 3) in das Format für den Planner.
    
    Der preprocessor.transform (aus dem Dataset) erwartet:
    - PIL Image oder numpy array
    - Konvertiert zu Tensor und normalisiert
    """
    import torchvision.transforms.functional as TF
    from PIL import Image
    
    # Sicherstellen dass es uint8 ist
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Zu PIL konvertieren (transform erwartet das oft)
    pil_img = Image.fromarray(img)
    
    # Transform anwenden (normalisiert und konvertiert zu Tensor)
    if preprocessor.transform is not None:
        transformed = preprocessor.transform(pil_img)
        # Zurück zu numpy für obs dict
        if isinstance(transformed, torch.Tensor):
            img_processed = transformed.numpy()
        else:
            img_processed = np.array(transformed)
    else:
        # Fallback: Manuell normalisieren
        img_processed = img.astype(np.float32) / 255.0
        # Channel-first falls nötig: (H, W, C) -> (C, H, W)
        if img_processed.ndim == 3 and img_processed.shape[-1] == 3:
            img_processed = np.transpose(img_processed, (2, 0, 1))
    
    return img_processed

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
                # Goal speichern im Format für planner.plan()
                img = np.array(msg["image"])
                print(f"  [Goal] Raw image shape: {img.shape}, dtype: {img.dtype}")
                
                # Bild vorverarbeiten
                img_processed = preprocess_image(img, preprocessor)
                print(f"  [Goal] Processed shape: {img_processed.shape}, dtype: {img_processed.dtype}")
                
                goal_obs = {
                    "visual": img_processed[np.newaxis, np.newaxis, ...],  # (1, 1, C, H, W) oder (1, 1, H, W, C)
                    "proprio": np.zeros((1, 1, 3), dtype=np.float32),
                }
                print(f"  [Goal] Final visual shape: {goal_obs['visual'].shape}")
                response = {"status": "ok"}
                print(f"  Goal gesetzt ✓")
                
            elif cmd == "plan":
                # Current obs
                img = np.array(msg["image"])
                print(f"  [Plan] Raw image shape: {img.shape}, dtype: {img.dtype}")
                
                # Bild vorverarbeiten
                img_processed = preprocess_image(img, preprocessor)
                print(f"  [Plan] Processed shape: {img_processed.shape}")
                
                cur_obs = {
                    "visual": img_processed[np.newaxis, np.newaxis, ...],
                    "proprio": np.zeros((1, 1, 3), dtype=np.float32),
                }
                
                # Planen (genau wie in PlanWorkspace.perform_planning)
                print(f"  [Plan] Running CEM planner...")
                with torch.no_grad():
                    actions, _ = planner.plan(obs_0=cur_obs, obs_g=goal_obs)
                
                # Erste Aktion denormalisieren und zurückgeben
                action = preprocessor.denormalize_actions(actions[0, 0:1]).numpy().squeeze()
                response = {"status": "ok", "action": action.tolist()}
                print(f"  [Plan] Action: {action[:3]} ✓")
                
            elif cmd == "reset":
                # Goal zurücksetzen für neue Episode
                goal_obs = None
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
