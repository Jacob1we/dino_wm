#!/usr/bin/env python3
"""
DINO WM Training Launcher
=========================
Startet nacheinander Trainings für verschiedene Datensätze.
Passt automatisch die fcs.yaml an und führt das Training aus.

Verwendung:
    conda activate dino_wm
    python training_launcher.py
    python training_launcher.py --dry-run  # Test ohne Training
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Argument Parser
parser = argparse.ArgumentParser(description="DINO WM Training Launcher")
parser.add_argument("--dry-run", action="store_true", 
                    help="Test-Modus: Zeigt was passieren würde, ohne Training zu starten")
parser.add_argument("--delay", type=float, default=0,
                    help="Initiale Wartezeit in Stunden vor dem Start (z.B. --delay 1.5)")
ARGS = parser.parse_args()

# =============================================================================
# KONFIGURATION: Hier die Datensätze eintragen
# =============================================================================

# Basis-Pfad für alle Datensätze
DATASET_BASE_PATH = "/media/tsp_jw/data/DINO_WM/fcs_datasets"

# Liste der Datensätze, die nacheinander trainiert werden sollen
# Einfach die Ordnernamen eintragen (ohne Pfad)
DATASETS = [
    "20260312_1002_NEps1000_RobOpac10_NPrim14_NCams1_NCube1_EEFfix_TrackGrip_SterileBg",
    "20260312_1645_NEps1000_RobOpac10_NPrim16_NCams1_NCube1_EEFfix_TrackGrip_SterileBg",
]

# =============================================================================
# TRAINING KONFIGURATION
# =============================================================================

# Training-Befehl Parameter
FRAMESKIP = 1
NUM_HIST = 1

# Pfade
DINO_WM_DIR = Path(__file__).parent.resolve()
FCS_YAML_PATH = DINO_WM_DIR / "conf" / "env" / "fcs.yaml"


def update_fcs_yaml(dataset_path: str) -> bool:
    """
    Aktualisiert den data_path in der fcs.yaml.
    
    Args:
        dataset_path: Vollständiger Pfad zum Datensatz
        
    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        with open(FCS_YAML_PATH, 'r') as f:
            content = f.read()
        
        # Finde und ersetze die data_path Zeile
        lines = content.split('\n')
        new_lines = []
        found = False
        
        for line in lines:
            if line.strip().startswith('data_path:'):
                new_lines.append(f'  data_path: "{dataset_path}"')
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            print(f"[ERROR] Konnte 'data_path:' in {FCS_YAML_PATH} nicht finden!")
            return False
        
        with open(FCS_YAML_PATH, 'w') as f:
            f.write('\n'.join(new_lines))
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Fehler beim Aktualisieren der fcs.yaml: {e}")
        return False


def run_training(dataset_name: str, index: int, total: int) -> bool:
    """
    Führt ein einzelnes Training aus.
    
    Args:
        dataset_name: Name des Datensatzes
        index: Aktueller Index (1-basiert)
        total: Gesamtanzahl der Trainings
        
    Returns:
        True bei Erfolg, False bei Fehler
    """
    dataset_path = f"{DATASET_BASE_PATH}/{dataset_name}"
    
    # Prüfe ob Datensatz existiert
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Datensatz nicht gefunden: {dataset_path}")
        return False
    
    print("\n" + "=" * 80)
    print(f"Training {index}/{total}")
    print(f"Datensatz: {dataset_name}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # fcs.yaml aktualisieren (nur im echten Modus)
    if ARGS.dry_run:
        print(f"[DRY-RUN] Würde fcs.yaml aktualisieren auf: {dataset_path}")
    else:
        if not update_fcs_yaml(dataset_path):
            return False
        print(f"[INFO] fcs.yaml aktualisiert auf: {dataset_path}")
    
    # Training-Befehl
    cmd = [
        "python", "train.py",
        "env=fcs",
        f"frameskip={FRAMESKIP}",
        f"num_hist={NUM_HIST}"
    ]
    
    print(f"[INFO] Starte Training: {' '.join(cmd)}")
    print("-" * 80)
    
    # Dry-Run: Nur anzeigen, nicht ausführen
    if ARGS.dry_run:
        print("[DRY-RUN] Training würde jetzt starten (übersprungen)")
        print("-" * 80)
        print(f"[DRY-RUN] Training {index}/{total} simuliert!")
        return True
    
    try:
        # Training ausführen (in dino_wm Verzeichnis)
        result = subprocess.run(
            cmd,
            cwd=DINO_WM_DIR,
            check=True
        )
        
        print("-" * 80)
        print(f"[SUCCESS] Training {index}/{total} abgeschlossen!")
        print(f"Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training fehlgeschlagen mit Exit-Code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n[ABBRUCH] Training durch Benutzer abgebrochen.")
        raise


def main():
    """Hauptfunktion - startet alle Trainings nacheinander."""
    
    print("\n" + "=" * 80)
    print("DINO WM Training Launcher")
    print("=" * 80)
    
    if not DATASETS:
        print("\n[ERROR] Keine Datensätze konfiguriert!")
        print("Bitte DATASETS Liste in diesem Skript befüllen.")
        print("\nVerfügbare Datensätze:")
        
        # Liste verfügbare Datensätze
        if os.path.exists(DATASET_BASE_PATH):
            for item in sorted(os.listdir(DATASET_BASE_PATH)):
                full_path = os.path.join(DATASET_BASE_PATH, item)
                if os.path.isdir(full_path) and not item.startswith("00_"):
                    print(f"  - {item}")
        
        sys.exit(1)
    
    total = len(DATASETS)
    print(f"\nAnzahl geplanter Trainings: {total}")
    print(f"Frameskip: {FRAMESKIP}")
    print(f"Num_hist: {NUM_HIST}")
    if ARGS.delay > 0:
        print(f"Initiale Wartezeit: {ARGS.delay} Stunden")
    if ARGS.dry_run:
        print(f"Modus: DRY-RUN (kein echtes Training)")
    print("\nDatensätze:")
    for i, ds in enumerate(DATASETS, 1):
        print(f"  {i}. {ds}")
    
    if not ARGS.dry_run:
        print("\n" + "-" * 80)
        print("Starte in 5 Sekunden... (Ctrl+C zum Abbrechen)")
        print("-" * 80)
        
        import time
        time.sleep(5)
        
        # Initiale Verzögerung falls gewünscht
        if ARGS.delay > 0:
            delay_seconds = ARGS.delay * 3600
            delay_end = datetime.now() + __import__('datetime').timedelta(seconds=delay_seconds)
            print("\n" + "=" * 80)
            print(f"WARTE {ARGS.delay} Stunden bis {delay_end.strftime('%H:%M:%S')}")
            print("(Ctrl+C zum Abbrechen)")
            print("=" * 80)
            time.sleep(delay_seconds)
            print(f"\n[INFO] Wartezeit beendet, starte Trainings...")
    
    # Trainings durchführen
    successful = 0
    failed = 0
    
    start_time = datetime.now()
    
    for i, dataset in enumerate(DATASETS, 1):
        try:
            if run_training(dataset, i, total):
                successful += 1
            else:
                failed += 1
                print(f"[WARNING] Fahre mit nächstem Training fort...")
        except KeyboardInterrupt:
            print("\n\n[ABBRUCH] Launcher beendet.")
            break
    
    # Zusammenfassung
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"Erfolgreich: {successful}/{total}")
    print(f"Fehlgeschlagen: {failed}/{total}")
    print(f"Gesamtdauer: {duration}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
