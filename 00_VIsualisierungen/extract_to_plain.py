#!/usr/bin/env python3
"""
Rope Dataset Extractor
======================
Extracts the rope dataset structure into plain text files and images.

Output structure:
    rope_extracted/
    ├── cameras/
    │   ├── intrinsic.txt
    │   └── extrinsic.txt
    ├── global_data/
    │   ├── actions_summary.txt
    │   └── states_summary.txt
    └── episodes/
        └── 000001/
            ├── metadata.txt
            ├── property_params.txt
            ├── timesteps/
            │   ├── 00/
            │   │   ├── data.txt
            │   │   ├── color_cam_0.png
            │   │   ├── color_cam_1.png
            │   │   ├── color_cam_2.png
            │   │   ├── color_cam_3.png
            │   │   ├── depth_cam_0.png
            │   │   └── depth_cam_0_raw.txt (optional)
            │   └── ...
            └── obses_preview/
                ├── frame_000.png
                └── ...
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. .pth files will be skipped.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Images will be saved as numpy arrays.")

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


def create_output_dir(base_path: Path) -> Path:
    """Create the output directory structure."""
    output_dir = base_path / "dset_extracted"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "cameras").mkdir(exist_ok=True)
    (output_dir / "global_data").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    return output_dir


def extract_camera_data(source_dir: Path, output_dir: Path):
    """Extract camera calibration data to text files."""
    print("\n=== Extracting Camera Data ===")
    cameras_src = source_dir / "cameras"
    cameras_dst = output_dir / "cameras"
    
    # Intrinsic matrix
    intrinsic_path = cameras_src / "intrinsic.npy"
    if intrinsic_path.exists():
        intrinsic = np.load(intrinsic_path)
        with open(cameras_dst / "intrinsic.txt", "w") as f:
            f.write("Camera Intrinsic Matrix\n")
            f.write("=" * 50 + "\n")
            f.write(f"Shape: {intrinsic.shape}\n")
            f.write(f"Dtype: {intrinsic.dtype}\n\n")
            f.write("Matrix:\n")
            np.savetxt(f, intrinsic, fmt="%.6f", delimiter="\t")
            f.write("\n\nInterpretation (if 4x4):\n")
            if intrinsic.shape == (4, 4):
                f.write(f"  fx (focal x): {intrinsic[0, 0]:.4f}\n")
                f.write(f"  fy (focal y): {intrinsic[1, 1]:.4f}\n")
                f.write(f"  cx (principal x): {intrinsic[0, 2]:.4f}\n")
                f.write(f"  cy (principal y): {intrinsic[1, 2]:.4f}\n")
        print(f"  ✓ Intrinsic matrix: {intrinsic.shape}")
    
    # Extrinsic matrix
    extrinsic_path = cameras_src / "extrinsic.npy"
    if extrinsic_path.exists():
        extrinsic = np.load(extrinsic_path)
        with open(cameras_dst / "extrinsic.txt", "w") as f:
            f.write("Camera Extrinsic Matrices\n")
            f.write("=" * 50 + "\n")
            f.write(f"Shape: {extrinsic.shape}\n")
            f.write(f"Dtype: {extrinsic.dtype}\n\n")
            
            if len(extrinsic.shape) == 3:
                # Multiple cameras (n_cams, 4, 4)
                for i in range(extrinsic.shape[0]):
                    f.write(f"\n--- Camera {i} ---\n")
                    np.savetxt(f, extrinsic[i], fmt="%.6f", delimiter="\t")
            else:
                np.savetxt(f, extrinsic, fmt="%.6f", delimiter="\t")
        print(f"  ✓ Extrinsic matrix: {extrinsic.shape}")


def extract_global_data(source_dir: Path, output_dir: Path):
    """Extract global actions.pth and states.pth."""
    print("\n=== Extracting Global Data ===")
    global_dst = output_dir / "global_data"
    
    # Actions
    actions_path = source_dir / "actions.pth"
    if actions_path.exists() and TORCH_AVAILABLE:
        try:
            actions = torch.load(actions_path, map_location="cpu", weights_only=False)
            with open(global_dst / "actions_summary.txt", "w") as f:
                f.write("Global Actions Data\n")
                f.write("=" * 50 + "\n")
                f.write(f"Type: {type(actions)}\n")
                
                if isinstance(actions, torch.Tensor):
                    f.write(f"Shape: {actions.shape}\n")
                    f.write(f"Dtype: {actions.dtype}\n")
                    f.write(f"Min: {actions.min().item():.6f}\n")
                    f.write(f"Max: {actions.max().item():.6f}\n")
                    f.write(f"Mean: {actions.mean().item():.6f}\n\n")
                    f.write("First 10 actions:\n")
                    for i, a in enumerate(actions[:10]):
                        f.write(f"  [{i}]: {a.numpy()}\n")
                elif isinstance(actions, dict):
                    f.write(f"Keys: {list(actions.keys())}\n")
                    for k, v in actions.items():
                        if hasattr(v, 'shape'):
                            f.write(f"  {k}: shape={v.shape}, dtype={v.dtype}\n")
                elif isinstance(actions, list):
                    f.write(f"Length: {len(actions)}\n")
                    f.write(f"First element type: {type(actions[0]) if actions else 'N/A'}\n")
            
            # Also save as numpy
            if isinstance(actions, torch.Tensor):
                np.save(global_dst / "actions.npy", actions.numpy())
            print(f"  ✓ Actions extracted")
        except Exception as e:
            print(f"  ✗ Error loading actions: {e}")
    
    # States
    states_path = source_dir / "states.pth"
    if states_path.exists() and TORCH_AVAILABLE:
        try:
            # This file might be large, just get metadata
            states = torch.load(states_path, map_location="cpu", weights_only=False)
            with open(global_dst / "states_summary.txt", "w") as f:
                f.write("Global States Data\n")
                f.write("=" * 50 + "\n")
                f.write(f"Type: {type(states)}\n")
                
                if isinstance(states, torch.Tensor):
                    f.write(f"Shape: {states.shape}\n")
                    f.write(f"Dtype: {states.dtype}\n")
                    f.write(f"Size (MB): {states.element_size() * states.nelement() / 1024 / 1024:.2f}\n")
                elif isinstance(states, dict):
                    f.write(f"Keys: {list(states.keys())}\n")
                elif isinstance(states, list):
                    f.write(f"Length: {len(states)}\n")
            print(f"  ✓ States metadata extracted")
        except Exception as e:
            print(f"  ✗ Error loading states: {e}")


def save_image(data: np.ndarray, path: Path, is_depth: bool = False):
    """Save numpy array as image."""
    if PIL_AVAILABLE:
        if is_depth:
            # Normalize depth for visualization
            depth_min = data.min()
            depth_max = data.max()
            if depth_max > depth_min:
                normalized = ((data - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(data, dtype=np.uint8)
            img = Image.fromarray(normalized, mode='L')
        else:
            # RGB image
            if data.dtype == np.float32 or data.dtype == np.float64:
                data = np.clip(data, 0, 255).astype(np.uint8)
            img = Image.fromarray(data, mode='RGB')
        img.save(path)
    else:
        # Fallback: save as numpy
        np.save(path.with_suffix('.npy'), data)


def extract_h5_timestep(h5_path: Path, output_dir: Path, save_images: bool = True, save_raw_depth: bool = False):
    """Extract a single H5 timestep to text and images."""
    timestep_name = h5_path.stem  # e.g., "00", "01"
    timestep_dir = output_dir / timestep_name
    timestep_dir.mkdir(exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        # Create data summary text file
        with open(timestep_dir / "data.txt", "w") as txt:
            txt.write(f"Timestep {timestep_name} Data\n")
            txt.write("=" * 50 + "\n\n")
            
            # Action
            if 'action' in f:
                action = f['action'][:]
                txt.write("ACTION\n")
                txt.write("-" * 30 + "\n")
                txt.write(f"Shape: {action.shape}\n")
                txt.write(f"Dtype: {action.dtype}\n")
                txt.write(f"Values: {action}\n\n")
            
            # EEF States
            if 'eef_states' in f:
                eef = f['eef_states'][:]
                txt.write("END EFFECTOR STATES\n")
                txt.write("-" * 30 + "\n")
                txt.write(f"Shape: {eef.shape}\n")
                txt.write(f"Dtype: {eef.dtype}\n")
                eef_flat = eef.flatten()
                txt.write("Interpreted as:\n")
                if len(eef_flat) >= 14:
                    txt.write(f"  Position 1: [{eef_flat[0]:.4f}, {eef_flat[1]:.4f}, {eef_flat[2]:.4f}]\n")
                    txt.write(f"  Position 2: [{eef_flat[3]:.4f}, {eef_flat[4]:.4f}, {eef_flat[5]:.4f}]\n")
                    txt.write(f"  Quaternion 1 (wxyz): [{eef_flat[6]:.4f}, {eef_flat[7]:.4f}, {eef_flat[8]:.4f}, {eef_flat[9]:.4f}]\n")
                    txt.write(f"  Quaternion 2 (wxyz): [{eef_flat[10]:.4f}, {eef_flat[11]:.4f}, {eef_flat[12]:.4f}, {eef_flat[13]:.4f}]\n")
                txt.write(f"Raw: {eef_flat}\n\n")
            
            # Positions (particles/objects)
            if 'positions' in f:
                pos = f['positions'][:]
                txt.write("POSITIONS (Particles/Objects)\n")
                txt.write("-" * 30 + "\n")
                txt.write(f"Shape: {pos.shape}\n")
                txt.write(f"Dtype: {pos.dtype}\n")
                txt.write(f"Num particles/objects: {pos.shape[1] if len(pos.shape) > 1 else 1}\n")
                txt.write(f"Values per particle: {pos.shape[2] if len(pos.shape) > 2 else pos.shape[-1]}\n")
                txt.write(f"Min: {pos.min():.4f}, Max: {pos.max():.4f}\n")
                txt.write(f"First 5 positions:\n")
                for i in range(min(5, pos.shape[1] if len(pos.shape) > 1 else 1)):
                    txt.write(f"  [{i}]: {pos[0, i] if len(pos.shape) > 1 else pos[i]}\n")
                txt.write("\n")
            
            # Info
            txt.write("INFO\n")
            txt.write("-" * 30 + "\n")
            if 'info' in f:
                for key in f['info'].keys():
                    val = f[f'info/{key}'][()]
                    txt.write(f"  {key}: {val}\n")
            txt.write("\n")
            
            # Observations summary
            txt.write("OBSERVATIONS\n")
            txt.write("-" * 30 + "\n")
            if 'observations' in f:
                if 'color' in f['observations']:
                    for cam in f['observations/color'].keys():
                        data = f[f'observations/color/{cam}']
                        txt.write(f"  color/{cam}: shape={data.shape}, dtype={data.dtype}, range=[{data[:].min():.1f}, {data[:].max():.1f}]\n")
                if 'depth' in f['observations']:
                    for cam in f['observations/depth'].keys():
                        data = f[f'observations/depth/{cam}']
                        txt.write(f"  depth/{cam}: shape={data.shape}, dtype={data.dtype}, range=[{data[:].min()}, {data[:].max()}]\n")
        
        # Extract images
        if save_images and 'observations' in f:
            # Color images
            if 'color' in f['observations']:
                for cam in f['observations/color'].keys():
                    img_data = f[f'observations/color/{cam}'][0]  # Remove batch dim
                    save_image(img_data, timestep_dir / f"color_{cam}.png", is_depth=False)
            
            # Depth images
            if 'depth' in f['observations']:
                for cam in f['observations/depth'].keys():
                    depth_data = f[f'observations/depth/{cam}'][0]  # Remove batch dim
                    save_image(depth_data, timestep_dir / f"depth_{cam}.png", is_depth=True)
                    
                    # Optionally save raw depth values
                    if save_raw_depth:
                        np.savetxt(timestep_dir / f"depth_{cam}_raw.txt", depth_data, fmt="%d")


def extract_episode(episode_dir: Path, output_dir: Path, max_timesteps: int = None, 
                   save_images: bool = True, save_obses_preview: bool = True):
    """Extract a single episode."""
    episode_name = episode_dir.name
    episode_output = output_dir / "episodes" / episode_name
    episode_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Episode {episode_name}:")
    
    # Property params
    property_path = episode_dir / "property_params.pkl"
    if property_path.exists() and PICKLE_AVAILABLE:
        try:
            with open(property_path, 'rb') as f:
                params = pickle.load(f)
            with open(episode_output / "property_params.txt", "w") as txt:
                txt.write("Property Parameters\n")
                txt.write("=" * 50 + "\n\n")
                for key, value in params.items():
                    txt.write(f"{key}: {value}\n")
            print(f"    ✓ property_params.pkl")
        except Exception as e:
            print(f"    ✗ property_params.pkl: {e}")
    
    # Obses.pth
    obses_path = episode_dir / "obses.pth"
    if obses_path.exists() and TORCH_AVAILABLE:
        try:
            obses = torch.load(obses_path, map_location="cpu", weights_only=False)
            with open(episode_output / "obses_info.txt", "w") as txt:
                txt.write("Observations (obses.pth)\n")
                txt.write("=" * 50 + "\n\n")
                txt.write(f"Type: {type(obses)}\n")
                if isinstance(obses, torch.Tensor):
                    txt.write(f"Shape: {obses.shape}\n")
                    txt.write(f"Dtype: {obses.dtype}\n")
                    txt.write(f"Min: {obses.min().item():.2f}\n")
                    txt.write(f"Max: {obses.max().item():.2f}\n")
                    txt.write(f"Mean: {obses.mean().item():.2f}\n")
                    txt.write(f"\nInterpretation:\n")
                    txt.write(f"  Timesteps (T): {obses.shape[0]}\n")
                    txt.write(f"  Height (H): {obses.shape[1]}\n")
                    txt.write(f"  Width (W): {obses.shape[2]}\n")
                    txt.write(f"  Channels (C): {obses.shape[3]}\n")
            
            # Save preview images
            if save_obses_preview and isinstance(obses, torch.Tensor) and PIL_AVAILABLE:
                preview_dir = episode_output / "obses_preview"
                preview_dir.mkdir(exist_ok=True)
                
                # Save every 5th frame, max 20 frames
                n_frames = min(obses.shape[0], 100)
                step = max(1, n_frames // 20)
                for i in range(0, n_frames, step):
                    frame = obses[i].numpy()
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    img = Image.fromarray(frame, mode='RGB')
                    img.save(preview_dir / f"frame_{i:03d}.png")
                print(f"    ✓ obses.pth ({obses.shape[0]} frames, preview saved)")
        except Exception as e:
            print(f"    ✗ obses.pth: {e}")
    
    # H5 timesteps
    h5_files = sorted(episode_dir.glob("*.h5"))
    if max_timesteps:
        h5_files = h5_files[:max_timesteps]
    
    timesteps_dir = episode_output / "timesteps"
    timesteps_dir.mkdir(exist_ok=True)
    
    for h5_file in h5_files:
        extract_h5_timestep(h5_file, timesteps_dir, save_images=save_images)
    print(f"    ✓ {len(h5_files)} H5 timesteps extracted")
    
    # Create episode metadata
    with open(episode_output / "metadata.txt", "w") as f:
        f.write(f"Episode: {episode_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Extraction Date: {datetime.now().isoformat()}\n")
        f.write(f"Number of timesteps: {len(h5_files)}\n")
        f.write(f"Files in episode:\n")
        for item in sorted(episode_dir.iterdir()):
            f.write(f"  - {item.name}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract Rope dataset to plain text and images")
    parser.add_argument("--source", type=str, default=".", help="Source directory (rope dataset)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: source/dset_extracted)")
    parser.add_argument("--episodes", type=str, nargs="*", default=None, help="Specific episodes to extract (e.g., 000001 000002)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to extract")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Maximum timesteps per episode")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-obses-preview", action="store_true", help="Skip obses preview images")
    parser.add_argument("--save-raw-depth", action="store_true", help="Save raw depth values as text")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output) if args.output else source_dir / "dset_extracted"
    output_dir = output_dir.resolve()
    
    print("=" * 60)
    print("DATASET EXTRACTOR")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Torch available: {TORCH_AVAILABLE}")
    print(f"PIL available: {PIL_AVAILABLE}")
    print("=" * 60)
    
    # Create output structure
    output_dir = create_output_dir(source_dir if args.output is None else output_dir.parent)
    if args.output:
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract camera data
    extract_camera_data(source_dir, output_dir)
    
    # Extract global data
    extract_global_data(source_dir, output_dir)
    
    # Find and extract episodes
    print("\n=== Extracting Episodes ===")
    episode_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if args.episodes:
        episode_dirs = [d for d in episode_dirs if d.name in args.episodes]
    
    if args.max_episodes:
        episode_dirs = episode_dirs[:args.max_episodes]
    
    print(f"Found {len(episode_dirs)} episodes to extract")
    
    for episode_dir in episode_dirs:
        extract_episode(
            episode_dir,
            output_dir,
            max_timesteps=args.max_timesteps,
            save_images=not args.no_images,
            save_obses_preview=not args.no_obses_preview
        )
    
    # Create summary
    with open(output_dir / "EXTRACTION_SUMMARY.txt", "w") as f:
        f.write("ROPE DATASET EXTRACTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Extraction Date: {datetime.now().isoformat()}\n")
        f.write(f"Source: {source_dir}\n")
        f.write(f"Output: {output_dir}\n\n")
        f.write(f"Episodes Extracted: {len(episode_dirs)}\n")
        for ep in episode_dirs:
            f.write(f"  - {ep.name}\n")
        f.write(f"\nOptions:\n")
        f.write(f"  Images saved: {not args.no_images}\n")
        f.write(f"  Obses preview: {not args.no_obses_preview}\n")
        f.write(f"  Raw depth: {args.save_raw_depth}\n")
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

