#!/usr/bin/env python3
"""
Script zum Aktualisieren des Training-Logs nach Abschluss eines Experiments.

Verwendung:
    python update_training_log.py --run_dir outputs/2026-01-07/09-33-48 --status Completed
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

def get_final_metrics(run_dir: str) -> dict:
    """Extrahiert finale Metriken aus dem WandB-Log oder Checkpoint."""
    metrics = {
        "train_loss": "",
        "val_loss": "",
        "z_err_pred": "",
    }
    
    # Versuche hydra.yaml zu lesen für Run-Infos
    hydra_path = Path(run_dir) / "hydra.yaml"
    if hydra_path.exists():
        print(f"Found hydra config at {hydra_path}")
    
    # Hier könnte man WandB API nutzen oder lokale Logs parsen
    # Für manuelles Update: Werte werden als Argumente übergeben
    
    return metrics

def update_csv(csv_path: str, experiment_id: str, updates: dict):
    """Aktualisiert eine Zeile in der CSV basierend auf Experiment_ID."""
    
    rows = []
    updated = False
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['Experiment_ID'] == experiment_id:
                row.update(updates)
                updated = True
            rows.append(row)
    
    if updated:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Updated {experiment_id} in {csv_path}")
    else:
        print(f"⚠️ Experiment {experiment_id} nicht gefunden")

def main():
    parser = argparse.ArgumentParser(description="Update Training Log")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--status", type=str, default="Completed", 
                        choices=["Running", "Completed", "Failed", "Stopped"])
    parser.add_argument("--train_loss", type=float, default=None)
    parser.add_argument("--val_loss", type=float, default=None)
    parser.add_argument("--z_err", type=float, default=None)
    parser.add_argument("--notes", type=str, default=None)
    
    args = parser.parse_args()
    
    csv_path = Path(__file__).parent / "training_experiments.csv"
    
    updates = {"Status": args.status}
    if args.train_loss is not None:
        updates["Train_Loss_Final"] = args.train_loss
    if args.val_loss is not None:
        updates["Val_Loss_Final"] = args.val_loss
    if args.z_err is not None:
        updates["Z_Err_Pred"] = args.z_err
    if args.notes is not None:
        updates["Notizen"] = args.notes
    
    update_csv(str(csv_path), args.exp_id, updates)

if __name__ == "__main__":
    main()

