"""
DINO WM Planning Server

Läuft im dino_wm conda environment.
Empfängt Bilder von Isaac Sim Client, plant Aktionen mit CEM, sendet zurück.

Verwendung:
    cd ~/Desktop/dino_wm
    conda activate dino_wm
    python planning_server.py --model_name 2026-02-02/22-50-30

Architektur:
    ┌─────────────────────────────────┐
    │  Isaac Sim Client               │
    │  (python.sh planning_client.py) │
    │  - Simulation                   │
    │  - Sendet: RGB-Bilder           │
    │  - Empfängt: Aktionen           │
    └───────────────┬─────────────────┘
                    │ Socket (localhost:5555)
    ┌───────────────▼─────────────────┐
    │  DINO WM Server (dieses Script) │
    │  (conda activate dino_wm)       │
    │  - World Model                  │
    │  - CEM Planner                  │
    │  - Empfängt: RGB-Bilder         │
    │  - Sendet: Aktionen             │
    └─────────────────────────────────┘
"""

import os
import sys

# DINO WM Pfad setzen BEVOR andere imports
dino_wm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dino_wm_dir)

import socket
import pickle
import numpy as np
import argparse
from pathlib import Path

from omegaconf import OmegaConf
import torch
import hydra

# Imports aus plan.py
from plan import load_model, load_ckpt
from planning.cem import CEMPlanner
from planning.objectives import create_objective_fn

# Args
parser = argparse.ArgumentParser(description="DINO WM Planning Server")
parser.add_argument("--model_name", type=str, required=True,
                    help="Modellname relativ zu outputs/ (z.B. 2026-02-02/22-50-30)")
parser.add_argument("--port", type=int, default=5555,
                    help="Socket Port (default: 5555)")
parser.add_argument("--goal_H", type=int, default=5,
                    help="Planungshorizont (default: 5)")
args = parser.parse_args()

# =============================================================================
# DINO WM LADEN
# =============================================================================

print("=" * 60)
print("DINO WM Planning Server")
print("=" * 60)
print(f"DINO WM Dir: {dino_wm_dir}")
print(f"Lade Modell: {args.model_name}")

# Model Config laden
model_path = os.path.join(dino_wm_dir, "outputs", args.model_name)
cfg_path = os.path.join(model_path, "hydra.yaml")

if not os.path.exists(cfg_path):
    print(f"FEHLER: Config nicht gefunden: {cfg_path}")
    sys.exit(1)

with open(cfg_path, "r") as f:
    model_cfg = OmegaConf.load(f)

# Checkpoint laden (wie in plan.py)
ckpt_path = Path(model_path) / "checkpoints" / "model_latest.pth"
if not ckpt_path.exists():
    print(f"FEHLER: Checkpoint nicht gefunden: {ckpt_path}")
    sys.exit(1)

print(f"Lade Checkpoint: {ckpt_path}")

# num_action_repeat berechnen (wie in plan.py)
num_action_repeat = model_cfg.frameskip // model_cfg.training.get("action_skip", model_cfg.frameskip)

# Model laden (verwendet hydra.utils.instantiate intern)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(ckpt_path, model_cfg, num_action_repeat, device)
model.eval()

print("✓ Modell geladen!")

# Planner erstellen
planner_cfg = OmegaConf.load(os.path.join(dino_wm_dir, "conf/planner/cem.yaml"))
objective_fn = create_objective_fn(alpha=0.5, base=2, mode="last")

planner = CEMPlanner(
    action_dim=model_cfg.env.action_dim,
    horizon=args.goal_H,
    **OmegaConf.to_container(planner_cfg)
)

print(f"✓ Planner bereit (horizon={args.goal_H}, action_dim={model_cfg.env.action_dim})")

# =============================================================================
# SOCKET SERVER
# =============================================================================

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", args.port))
server.listen(1)

print(f"\n{'='*60}")
print(f"Server läuft auf localhost:{args.port}")
print(f"Warte auf Isaac Sim Client...")
print(f"{'='*60}\n")

goal_z = None  # Wird beim ersten "set_goal" gesetzt
step_count = 0

while True:
    conn, addr = server.accept()
    print(f"[+] Client verbunden: {addr}")
    
    try:
        while True:
            # Daten empfangen (8 bytes = size, dann data)
            size_data = conn.recv(8)
            if not size_data:
                break
            data_size = int.from_bytes(size_data, 'big')
            if data_size == 0:
                break
                
            data = b""
            while len(data) < data_size:
                chunk = conn.recv(min(65536, data_size - len(data)))
                if not chunk:
                    break
                data += chunk
            
            msg = pickle.loads(data)
            cmd = msg.get("cmd")
            
            if cmd == "set_goal":
                # Goal-Bild encodieren
                goal_img = torch.tensor(msg["image"]).float().cuda()
                goal_img = goal_img.permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    goal_z = model.encode_obs({"visual": goal_img})
                response = {"status": "ok", "msg": "Goal gesetzt"}
                step_count = 0
                print(f"  [Goal] Encodiert (shape: {goal_img.shape})")
                
            elif cmd == "plan":
                step_count += 1
                # Aktuelles Bild encodieren und planen
                cur_img = torch.tensor(msg["image"]).float().cuda()
                cur_img = cur_img.permute(2, 0, 1).unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    cur_z = model.encode_obs({"visual": cur_img})
                    
                    # CEM Planung
                    action = planner.plan(
                        model=model,
                        cur_z=cur_z,
                        goal_z=goal_z,
                        objective_fn=objective_fn
                    )
                
                action_np = action.cpu().numpy()
                response = {"status": "ok", "action": action_np}
                print(f"  [Step {step_count}] Action: [{action_np[0]:.3f}, {action_np[1]:.3f}, {action_np[2]:.3f}, ...]")
                
            elif cmd == "reset":
                goal_z = None
                step_count = 0
                response = {"status": "ok", "msg": "Reset"}
                print(f"  [Reset]")
                
            elif cmd == "quit":
                response = {"status": "ok", "msg": "Bye"}
                resp_data = pickle.dumps(response)
                conn.sendall(len(resp_data).to_bytes(8, 'big'))
                conn.sendall(resp_data)
                print(f"  [Quit]")
                break
            else:
                response = {"status": "error", "msg": f"Unknown cmd: {cmd}"}
            
            # Antwort senden
            response_data = pickle.dumps(response)
            conn.sendall(len(response_data).to_bytes(8, 'big'))
            conn.sendall(response_data)
            
    except Exception as e:
        print(f"  [Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print(f"[-] Client getrennt\n")
