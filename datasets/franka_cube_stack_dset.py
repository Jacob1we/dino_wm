"""
Dataset-Loader für Franka Cube Stacking Datensätze.
Kompatibel mit DINO World Model Training.

Action-Format (3 Optionen, gespeichert in H5 info/action_mode Attribut):

    action_mode="delta_pose" (4D):
        action = [delta_x, delta_y, delta_z, delta_yaw]
        - delta_x/y/z: Relative Position-Änderung des EE in Metern
        - delta_yaw: Rotation um Z-Achse in Radiant

    action_mode="velocity" (4D):
        action = [vx, vy, vz, omega_z]
        - vx/vy/vz: Translatorische Geschwindigkeit in m/s
        - omega_z: Rotatorische Geschwindigkeit um Z-Achse in rad/s

    action_mode="ee_pos" (6D, wie DINO WM Rope):
        action = [x_start, y_start, z_start, x_end, y_end, z_end]
        - x/y/z_start: EE-Position am Anfang der Bewegung
        - x/y/z_end: EE-Position am Ende der Bewegung

Format (Rope kompatibel):
    data_path/
    ├── cameras/
    │   ├── intrinsic.npy     # (4, 4) Kamera-Intrinsik
    │   └── extrinsic.npy     # (4, 4, 4) Kamera-Extrinsik
    ├── 000000/               # Episode 0
    │   ├── obses.pth         # (T, H, W, C) float32, Werte 0-255
    │   ├── 00.h5             # Timestep 0
    │   │   ├── action        # (4,) siehe Action-Format
    │   │   ├── eef_states    # (1, 1, 14)
    │   │   ├── positions     # (1, n_cubes, 4) Würfel [x, y, z, yaw]
    │   │   ├── info/         # n_cams, n_cubes, timestamp, action_mode (attr)
    │   │   └── observations/ # color/cam_0, depth/cam_0
    │   ├── 01.h5             # Timestep 1
    │   └── ...
    └── 000001/               # Episode 1
"""

import torch
import pickle
import h5py
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional, List
from .traj_dset import TrajDataset, get_train_val_sliced


class FrankaCubeStackDataset(TrajDataset):
    """
    Dataset für Franka Panda Cube Stacking Trajektorien.
    
    Lädt Daten im Rope-Format: Episode-Ordner mit obses.pth und H5-Dateien.
    Kompatibel mit dem DINO World Model Training-Pipeline.
    
    Action-Format (wird automatisch aus H5 info/action_mode erkannt):
        - "delta_pose": [delta_x, delta_y, delta_z, delta_yaw] (4D)
        - "velocity":   [vx, vy, vz, omega_z] (4D)
        - "ee_pos":     [x_start, y_start, z_start, x_end, y_end, z_end] (6D, wie DINO WM Rope)
    """
    
    def __init__(
        self,
        data_path: str = "data/franka_cube_stack",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        preload_images: bool = True,
    ):
        """
        Args:
            data_path: Pfad zum Datensatz-Ordner
            n_rollout: Anzahl zu ladender Trajektorien (None = alle)
            transform: Bild-Transformationen (z.B. Resize, Normalisierung)
            normalize_action: Wenn True, werden Aktionen z-normalisiert
            action_scale: Skalierungsfaktor für Aktionen
            preload_images: Wenn True, werden alle Bilder beim Init in RAM geladen
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.preload_images = preload_images
        self.action_mode = None  # Wird aus H5 gelesen
        
        # Finde alle Episode-Ordner (000000, 000001, ...)
        self.episode_dirs = sorted([
            d for d in self.data_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        
        if n_rollout is not None:
            self.episode_dirs = self.episode_dirs[:n_rollout]
        
        n_episodes = len(self.episode_dirs)
        print(f"FrankaCubeStackDataset: {n_episodes} Episoden gefunden")
        
        # Lade Daten aus allen Episoden
        self.seq_lengths = []
        self.all_actions = []
        self.all_eef_states = []
        self.images_cache = [] if preload_images else None
        
        for i, episode_dir in enumerate(self.episode_dirs):
            # Lade obses.pth
            obses_path = episode_dir / "obses.pth"
            if not obses_path.exists():
                raise FileNotFoundError(f"obses.pth nicht gefunden: {obses_path}")
            
            obses = torch.load(obses_path)
            episode_length = obses.shape[0]
            self.seq_lengths.append(episode_length)
            
            if preload_images:
                self.images_cache.append(obses)
            
            # Lade Actions und EEF-States aus H5-Dateien
            actions = []
            eef_states = []
            
            for t in range(episode_length):
                h5_path = episode_dir / f"{t:02d}.h5"
                if h5_path.exists():
                    with h5py.File(h5_path, "r") as f:
                        action = f["action"][:]
                        eef = f["eef_states"][:]
                        
                        # Handle NaN values in actions (replace with zeros)
                        if np.isnan(action).any():
                            action = np.nan_to_num(action, nan=0.0)
                        
                        # Handle NaN values in eef_states
                        if np.isnan(eef).any():
                            eef = np.nan_to_num(eef, nan=0.0)
                        
                        actions.append(action)
                        eef_states.append(eef.flatten())
                        
                        # Erkenne action_mode aus erstem H5 der ersten Episode
                        if self.action_mode is None and "info" in f:
                            if "action_mode" in f["info"].attrs:
                                self.action_mode = f["info"].attrs["action_mode"]
                                if isinstance(self.action_mode, bytes):
                                    self.action_mode = self.action_mode.decode("utf-8")
                else:
                    # Fallback wenn H5 nicht existiert
                    # Bestimme action_dim aus vorherigen Actions, sonst 6 (ee_pos)
                    fallback_action_dim = actions[-1].shape[0] if actions else 6
                    actions.append(np.zeros(fallback_action_dim))
                    eef_states.append(np.zeros(14))
            
            self.all_actions.append(np.stack(actions, axis=0))
            self.all_eef_states.append(np.stack(eef_states, axis=0))
            
            if (i + 1) % 10 == 0 or i == n_episodes - 1:
                print(f"  Geladen: {i + 1}/{n_episodes} Episoden")
        
        self.seq_lengths = torch.tensor(self.seq_lengths)
        
        # Dimensionen aus ersten Daten extrahieren
        self.action_dim = self.all_actions[0].shape[-1]
        self.eef_dim = self.all_eef_states[0].shape[-1]
        
        # State dim: EEF States werden als States verwendet
        self.state_dim = self.eef_dim
        
        # Proprio: EEF Position (erste 3 Dimensionen)
        self.proprio_dim = 3
        
        # Konvertiere zu Tensoren
        self.actions_tensors = [torch.from_numpy(a).float() / action_scale for a in self.all_actions]
        self.eef_tensors = [torch.from_numpy(e).float() for e in self.all_eef_states]
        
        # Default action_mode falls nicht in H5 gefunden
        if self.action_mode is None:
            # Bestimme aus action_dim
            if self.action_dim == 6:
                self.action_mode = "ee_pos"
            else:
                self.action_mode = "delta_pose"
            print(f"  Warning: action_mode nicht in H5 gefunden, inferred from dim: {self.action_mode}")
        
        # Normalisierung für Action
        # Beide Modi haben Format: [translation/velocity (3), rotation/omega (1)]
        if normalize_action:
            all_actions_flat = torch.cat(self.actions_tensors, dim=0)
            
            # Z-Normalisierung für alle 4 Dimensionen
            self.action_mean = all_actions_flat.mean(dim=0)
            self.action_std = all_actions_flat.std(dim=0) + 1e-6
            
            all_eef_flat = torch.cat(self.eef_tensors, dim=0)
            # State mean/std: für alle EEF-Dimensionen
            self.state_mean = all_eef_flat.mean(dim=0)
            self.state_std = all_eef_flat.std(dim=0) + 1e-6
            # Proprio mean/std: nur für Position (erste 3 Dimensionen)
            self.proprio_mean = all_eef_flat[:, :3].mean(dim=0)
            self.proprio_std = all_eef_flat[:, :3].std(dim=0) + 1e-6
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        
        # Normalisiere Actions
        self.actions_tensors = [(a - self.action_mean) / self.action_std for a in self.actions_tensors]
        
        print(f"  Action mode: {self.action_mode}")
        if self.action_mode == "delta_pose":
            print(f"  Action format: [delta_x, delta_y, delta_z, delta_yaw] (4D)")
        elif self.action_mode == "velocity":
            print(f"  Action format: [vx, vy, vz, omega_z] (4D)")
        else:  # ee_pos
            print(f"  Action format: [x_start, y_start, z_start, x_end, y_end, z_end] (6D)")
        print(f"  Action dim: {self.action_dim}")
        print(f"  State dim: {self.state_dim}")
        print(f"  EEF dim: {self.eef_dim}")
        print(f"  Proprio dim: {self.proprio_dim}")
        print(f"  Preload images: {self.preload_images}")
        if normalize_action:
            print(f"  Action normalization: z-score")
            print(f"    Mean: {self.action_mean.numpy()}")
            print(f"    Std: {self.action_std.numpy()}")
    
    def get_seq_length(self, idx: int) -> int:
        """Gibt die Länge der idx-ten Trajektorie zurück."""
        return int(self.seq_lengths[idx])
    
    def get_all_actions(self) -> torch.Tensor:
        """Gibt alle Aktionen als einzelner Tensor zurück."""
        return torch.cat(self.actions_tensors, dim=0)
    
    def get_frames(self, idx: int, frames):
        """
        Lädt spezifische Frames einer Trajektorie.
        
        Args:
            idx: Trajektorien-Index
            frames: Frame-Indizes (range oder Liste)
        
        Returns:
            obs: Dict mit 'visual' und 'proprio' Tensoren
            act: Aktionen für die Frames
            state: EEF-States für die Frames
            info: Leeres Dict (Kompatibilität)
        """
        # Bilder aus Cache oder von Disk laden
        if self.images_cache is not None:
            image = self.images_cache[idx]
        else:
            obses_path = self.episode_dirs[idx] / "obses.pth"
            image = torch.load(obses_path)
        
        # Konvertiere frames zu Liste falls nötig
        if isinstance(frames, range):
            frames = list(frames)
        
        # Selektiere Frames
        image = image[frames]  # (T, H, W, C)
        
        # Actions und EEF für diese Frames
        act = self.actions_tensors[idx][frames]
        eef = self.eef_tensors[idx][frames]
        
        # State: EEF States (normalisiert)
        state = (eef - self.state_mean) / self.state_std
        
        # Proprio: EEF Position (erste 3 Dimensionen)
        proprio = (eef[:, :3] - self.proprio_mean) / self.proprio_std
        
        # Konvertiere Bilder: (T, H, W, C) -> (T, C, H, W), [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image / 255.0
        
        image = rearrange(image, "T H W C -> T C H W")
        
        if self.transform:
            image = self.transform(image)
        
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}
    
    def __getitem__(self, idx: int):
        """Lädt eine vollständige Trajektorie."""
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self) -> int:
        return len(self.episode_dirs)
    
    def preprocess_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Preprocessing für externe Bilder (z.B. aus Umgebung)."""
        return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_franka_cube_stack_slice_train_val(
    transform,
    n_rollout: Optional[int] = None,
    data_path: str = "data/franka_cube_stack",
    normalize_action: bool = False,
    split_ratio: float = 0.9,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 1,
):
    """
    Lädt den Franka Cube Stack Datensatz mit Train/Val Split.
    
    Args:
        transform: Bild-Transformationen
        n_rollout: Anzahl Rollouts (None = alle)
        data_path: Pfad zum Datensatz
        normalize_action: Z-Normalisierung für Aktionen
        split_ratio: Train/Val Aufteilung (0.9 = 90% Train)
        num_hist: Anzahl Historien-Frames für Kontext
        num_pred: Anzahl zu prädizierender Frames
        frameskip: Frame-Subsampling (1 = alle Frames)
    
    Returns:
        datasets: Dict mit 'train' und 'valid' Slice-Datasets
        traj_dset: Dict mit 'train' und 'valid' Trajektorien-Datasets
    """
    dset = FrankaCubeStackDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    
    datasets = {
        "train": train_slices,
        "valid": val_slices,
    }
    traj_dset = {
        "train": dset_train,
        "valid": dset_val,
    }
    
    return datasets, traj_dset
