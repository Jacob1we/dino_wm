#!/usr/bin/env python3
"""
DINO-WM Training Progress GIF Generator
=========================================

Erzeugt animierte GIFs, die den Trainingsfortschritt des World Models
über alle Epochen hinweg visualisieren.

Es gibt 3 Arten von Visualisierungen im Training-Output:

1. **train/ und valid/ Bilder** (z.B. train_e00001_b0.png)
   - Format: 3 Zeilen × (num_samples × num_frames) Spalten
   - Zeile 1 (oben):    Ground Truth – die echten Beobachtungsbilder
   - Zeile 2 (mitte):   Prediction – vom Predictor vorhergesagte Bilder
                         (erste num_pred Frames sind schwarz/leer, da der 
                         Predictor nur zukünftige Frames vorhersagt)
   - Zeile 3 (unten):   Reconstruction – Rekonstruktion ALLER Frames 
                         durch den Decoder aus den DINOv2-Embeddings
   → Zeigt, wie gut das World Model (a) zukünftige Frames vorhersagen
     und (b) die visuellen Eingaben rekonstruieren kann.

2. **rollout_plots/ Bilder** (z.B. e1_train_0.png)
   - Format: 2 Zeilen × horizon Spalten
   - Zeile 1 (oben):    Ground Truth – die echte Trajektorie aus dem Dataset
   - Zeile 2 (unten):   Rollout-Vorhersage – das World Model rollt autoregres-
                         siv mit echten Aktionen vorwärts und decodiert die
                         vorhergesagten Embeddings zurück zu Pixeln
   - Variante "_1framestart": Gleich, aber nur 1 Frame als Kontext (statt num_hist)
   → Zeigt, wie gut das World Model eine ganze Trajektorie vorhersagen kann.
     Über die Epochen sollten die Rollout-Bilder immer ähnlicher zur GT werden.

Verwendung:
    python create_progress_gif.py <output_dir> [optionen]

Beispiele:
    # Alle GIF-Typen erstellen
    python create_progress_gif.py outputs/2026-02-15/00-42-58

    # Nur Rollout-GIFs, bestimmter Trajektorien-Index
    python create_progress_gif.py outputs/2026-02-15/00-42-58 --type rollout --traj-idx 0

    # Train/Valid Reconstruction-GIFs
    python create_progress_gif.py outputs/2026-02-15/00-42-58 --type train_valid

    # Schnellere GIFs mit weniger Frames
    python create_progress_gif.py outputs/2026-02-15/00-42-58 --fps 5 --epoch-step 5
"""

import argparse
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def add_epoch_label(img, epoch, font_size=28):
    """Fügt eine Epochen-Beschriftung oben links auf das Bild hinzu."""
    draw = ImageDraw.Draw(img)
    text = f"Epoch {epoch}"
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    # Text-Hintergrund
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 6
    draw.rectangle(
        [5, 5, 5 + text_w + 2 * padding, 5 + text_h + 2 * padding],
        fill=(0, 0, 0, 200)
    )
    draw.text((5 + padding, 5 + padding), text, fill=(255, 255, 255), font=font)
    return img


def add_row_labels(img, labels, font_size=18):
    """Fügt Zeilenbeschriftungen am linken Rand hinzu."""
    num_rows = len(labels)
    row_height = img.height // num_rows
    
    # Neues Bild mit Platz für Labels links
    label_width = 140
    new_img = Image.new('RGB', (img.width + label_width, img.height), (40, 40, 40))
    new_img.paste(img, (label_width, 0))
    
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    for i, label in enumerate(labels):
        y = i * row_height + row_height // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text((8, y - text_h // 2), label, fill=(255, 255, 255), font=font)
    
    return new_img


def create_train_valid_gif(output_dir, gif_dir, phase="train", fps=3, epoch_step=1, 
                           add_labels=True, max_width=1600):
    """
    Erzeugt ein GIF aus den train/ oder valid/ Bildern über alle Epochen.
    
    Diese Bilder zeigen (von oben nach unten):
      Zeile 1: Ground Truth
      Zeile 2: Vorhersage (Prediction) des Predictors
      Zeile 3: Rekonstruktion durch den Decoder
    """
    phase_dir = os.path.join(output_dir, phase)
    if not os.path.isdir(phase_dir):
        print(f"  ⚠ Verzeichnis '{phase_dir}' nicht gefunden, überspringe.")
        return None
    
    files = sorted(
        [f for f in os.listdir(phase_dir) if f.endswith('.png')],
        key=natural_sort_key
    )
    
    if not files:
        print(f"  ⚠ Keine Bilder in '{phase_dir}' gefunden.")
        return None
    
    # Epochen extrahieren und filtern
    epoch_files = []
    for f in files:
        match = re.search(r'_e(\d+)_', f)
        if match:
            epoch = int(match.group(1))
            if (epoch - 1) % epoch_step == 0 or epoch == 1:
                epoch_files.append((epoch, f))
    
    if not epoch_files:
        print(f"  ⚠ Keine gültigen Epochen-Bilder gefunden.")
        return None
    
    print(f"  📸 {len(epoch_files)} Epochen für {phase}-GIF gefunden")
    
    frames = []
    for epoch, fname in epoch_files:
        img = Image.open(os.path.join(phase_dir, fname)).convert('RGB')
        
        # Skalieren wenn zu breit
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        
        if add_labels:
            img = add_row_labels(img, ["Ground Truth", "Prediction", "Reconstruction"])
        
        img = add_epoch_label(img, epoch)
        frames.append(img)
    
    # Letzten Frame länger zeigen
    for _ in range(max(1, fps)):
        frames.append(frames[-1].copy())
    
    gif_path = os.path.join(gif_dir, f"{phase}_progress.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"  ✅ Gespeichert: {gif_path} ({len(frames)} Frames)")
    return gif_path


def create_rollout_gif(output_dir, gif_dir, traj_idx=0, mode="train", 
                       suffix="", fps=3, epoch_step=1, add_labels=True, max_width=1600):
    """
    Erzeugt ein GIF aus den rollout_plots/ Bildern über alle Epochen
    für eine bestimmte Trajektorie.
    
    Diese Bilder zeigen (von oben nach unten):
      Zeile 1: Ground Truth Trajektorie
      Zeile 2: Autoregressive Rollout-Vorhersage des World Models
    
    suffix="" → num_hist Frames als Kontext
    suffix="_1framestart" → nur 1 Frame als Kontext
    """
    rollout_dir = os.path.join(output_dir, "rollout_plots")
    if not os.path.isdir(rollout_dir):
        print(f"  ⚠ Verzeichnis '{rollout_dir}' nicht gefunden, überspringe.")
        return None
    
    # Alle Epoch-Ordner finden
    epoch_dirs = sorted(
        [d for d in os.listdir(rollout_dir) if d.endswith('_rollout')],
        key=natural_sort_key
    )
    
    if not epoch_dirs:
        print(f"  ⚠ Keine Rollout-Ordner gefunden.")
        return None
    
    frames = []
    for epoch_dir_name in epoch_dirs:
        match = re.match(r'e(\d+)_rollout', epoch_dir_name)
        if not match:
            continue
        epoch = int(match.group(1))
        if (epoch - 1) % epoch_step != 0 and epoch != 1:
            continue
        
        img_name = f"e{epoch}_{mode}_{traj_idx}{suffix}.png"
        img_path = os.path.join(rollout_dir, epoch_dir_name, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        img = Image.open(img_path).convert('RGB')
        
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        
        if add_labels:
            img = add_row_labels(img, ["Ground Truth", "Rollout-\nVorhersage"], font_size=16)
        
        img = add_epoch_label(img, epoch)
        frames.append(img)
    
    if not frames:
        print(f"  ⚠ Keine Rollout-Bilder für {mode}_{traj_idx}{suffix} gefunden.")
        return None
    
    # Letzten Frame länger zeigen
    for _ in range(max(1, fps)):
        frames.append(frames[-1].copy())
    
    context_label = "1frame" if suffix == "_1framestart" else "full_hist"
    gif_path = os.path.join(gif_dir, f"rollout_{mode}_{traj_idx}_{context_label}_progress.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"  ✅ Gespeichert: {gif_path} ({len(frames)} Frames)")
    return gif_path


def create_rollout_grid_gif(output_dir, gif_dir, mode="train", num_traj=5,
                             suffix="", fps=2, epoch_step=5, max_width=2000):
    """
    Erzeugt ein GIF mit mehreren Trajektorien nebeneinander pro Epoch.
    Zeigt den Fortschritt des Rollout-Vorhersagequalität über das Training.
    """
    rollout_dir = os.path.join(output_dir, "rollout_plots")
    if not os.path.isdir(rollout_dir):
        print(f"  ⚠ Verzeichnis '{rollout_dir}' nicht gefunden, überspringe.")
        return None
    
    epoch_dirs = sorted(
        [d for d in os.listdir(rollout_dir) if d.endswith('_rollout')],
        key=natural_sort_key
    )
    
    frames = []
    for epoch_dir_name in epoch_dirs:
        match = re.match(r'e(\d+)_rollout', epoch_dir_name)
        if not match:
            continue
        epoch = int(match.group(1))
        if (epoch - 1) % epoch_step != 0 and epoch != 1:
            continue
        
        # Sammle Bilder für alle Trajektorien dieser Epoche
        traj_imgs = []
        for traj_idx in range(num_traj):
            img_name = f"e{epoch}_{mode}_{traj_idx}{suffix}.png"
            img_path = os.path.join(rollout_dir, epoch_dir_name, img_name)
            if os.path.exists(img_path):
                traj_imgs.append(Image.open(img_path).convert('RGB'))
        
        if not traj_imgs:
            continue
        
        # Stapele alle Trajektorien vertikal
        total_height = sum(im.height for im in traj_imgs) + (len(traj_imgs) - 1) * 4
        max_w = max(im.width for im in traj_imgs)
        grid = Image.new('RGB', (max_w, total_height), (60, 60, 60))
        y = 0
        for im in traj_imgs:
            grid.paste(im, (0, y))
            y += im.height + 4  # 4px Abstand
        
        # Skalieren
        if grid.width > max_width:
            ratio = max_width / grid.width
            grid = grid.resize((max_width, int(grid.height * ratio)), Image.LANCZOS)
        
        grid = add_epoch_label(grid, epoch, font_size=32)
        frames.append(grid)
    
    if not frames:
        print(f"  ⚠ Keine Grid-Frames erstellt.")
        return None
    
    # Letzten Frame länger zeigen
    for _ in range(max(1, fps * 2)):
        frames.append(frames[-1].copy())
    
    context_label = "1frame" if suffix == "_1framestart" else "full_hist"
    gif_path = os.path.join(gif_dir, f"rollout_grid_{mode}_{context_label}_progress.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"  ✅ Gespeichert: {gif_path} ({len(frames)} Frames)")
    return gif_path


def main():
    parser = argparse.ArgumentParser(
        description="DINO-WM Training Progress GIF Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python create_progress_gif.py outputs/2026-02-15/00-42-58
  python create_progress_gif.py outputs/2026-02-15/00-42-58 --type rollout --traj-idx 0 3 5
  python create_progress_gif.py outputs/2026-02-15/00-42-58 --type train_valid --fps 5
  python create_progress_gif.py outputs/2026-02-15/00-42-58 --type rollout_grid --epoch-step 10
        """
    )
    parser.add_argument("output_dir", help="Pfad zum Training-Output-Verzeichnis")
    parser.add_argument("--type", choices=["all", "train_valid", "rollout", "rollout_grid"],
                        default="all", help="Typ der GIFs (default: all)")
    parser.add_argument("--traj-idx", type=int, nargs='+', default=[0, 1, 2],
                        help="Trajektorien-Indizes für Rollout-GIFs (default: 0 1 2)")
    parser.add_argument("--num-grid-traj", type=int, default=5,
                        help="Anzahl Trajektorien im Grid-GIF (default: 5)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Frames pro Sekunde im GIF (default: 3)")
    parser.add_argument("--epoch-step", type=int, default=1,
                        help="Nur jede N-te Epoche verwenden (default: 1)")
    parser.add_argument("--max-width", type=int, default=1600,
                        help="Maximale Breite der GIF-Frames (default: 1600)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Keine Zeilen-Beschriftungen hinzufügen")
    parser.add_argument("--gif-dir", type=str, default=None,
                        help="Ausgabeverzeichnis für GIFs (default: <output_dir>/progress_gifs)")
    
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"❌ Verzeichnis '{output_dir}' existiert nicht!")
        sys.exit(1)
    
    gif_dir = args.gif_dir or os.path.join(output_dir, "progress_gifs")
    os.makedirs(gif_dir, exist_ok=True)
    
    print(f"🎬 DINO-WM Training Progress GIF Generator")
    print(f"   Input:  {output_dir}")
    print(f"   Output: {gif_dir}")
    print(f"   FPS: {args.fps}, Epoch-Step: {args.epoch_step}")
    print()
    
    created_gifs = []
    
    # --- Train/Valid GIFs ---
    if args.type in ("all", "train_valid"):
        print("📊 Erstelle Train/Valid Reconstruction-GIFs...")
        for phase in ["train", "valid"]:
            result = create_train_valid_gif(
                output_dir, gif_dir, phase=phase,
                fps=args.fps, epoch_step=args.epoch_step,
                add_labels=not args.no_labels, max_width=args.max_width
            )
            if result:
                created_gifs.append(result)
        print()
    
    # --- Rollout GIFs (einzelne Trajektorien) ---
    if args.type in ("all", "rollout"):
        print("🔄 Erstelle Rollout-GIFs (einzelne Trajektorien)...")
        for traj_idx in args.traj_idx:
            for mode in ["train"]:
                for suffix in ["", "_1framestart"]:
                    result = create_rollout_gif(
                        output_dir, gif_dir, traj_idx=traj_idx, mode=mode,
                        suffix=suffix, fps=args.fps, epoch_step=args.epoch_step,
                        add_labels=not args.no_labels, max_width=args.max_width
                    )
                    if result:
                        created_gifs.append(result)
        print()
    
    # --- Rollout Grid GIFs ---
    if args.type in ("all", "rollout_grid"):
        print("🔲 Erstelle Rollout-Grid-GIFs...")
        for suffix in ["", "_1framestart"]:
            result = create_rollout_grid_gif(
                output_dir, gif_dir, mode="train",
                num_traj=args.num_grid_traj, suffix=suffix,
                fps=args.fps, 
                epoch_step=max(args.epoch_step, 5),  # Grid immer etwas gröber
                max_width=args.max_width
            )
            if result:
                created_gifs.append(result)
        print()
    
    # --- Zusammenfassung ---
    print("=" * 60)
    if created_gifs:
        print(f"✅ {len(created_gifs)} GIFs erfolgreich erstellt:")
        for g in created_gifs:
            size_mb = os.path.getsize(g) / (1024 * 1024)
            print(f"   📁 {os.path.relpath(g, output_dir)} ({size_mb:.1f} MB)")
    else:
        print("❌ Keine GIFs erstellt. Prüfe ob die Verzeichnisse korrekt sind.")


if __name__ == "__main__":
    main()
