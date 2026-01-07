"""
Dataset-Loader für Franka Cube Stacking Datensätze.
Kompatibel mit DINO World Model Training.

Erwartet Datenstruktur (generiert von FrankaDataLogger):
    data_path/
    ├── states.pth        # (N, T_max, state_dim) float32
    ├── actions.pth       # (N, T_max, action_dim) float32
    ├── metadata.pkl      # {'episode_lengths': [...], ...}
    ├── 000000/
    │   └── obses.pth     # (T, H, W, C) uint8, RGB-Bilder
    ├── 000001/
    │   └── obses.pth
    └── ...
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced


class FrankaCubeStackDataset(TrajDataset):
    """
    Dataset für Franka Panda Cube Stacking Trajektorien.
    
    Lädt RGB-Bilder, Zustände und Aktionen aus dem FrankaDataLogger-Format.
    Kompatibel mit dem DINO World Model Training-Pipeline.
    """
    
    def __init__(
        self,
        data_path: str = "data/franka_cube_stack",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        preload_images: bool = True,  # NEU: Bilder vorab in RAM laden
    ):
        """
        Args:
            data_path: Pfad zum Datensatz-Ordner
            n_rollout: Anzahl zu ladender Trajektorien (None = alle)
            transform: Bild-Transformationen (z.B. Resize, Normalisierung)
            normalize_action: Wenn True, werden Aktionen z-normalisiert
            action_scale: Skalierungsfaktor für Aktionen
            preload_images: Wenn True, werden alle Bilder beim Init in RAM geladen
                           (löst multiprocessing Deadlock-Problem)
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.preload_images = preload_images
        
        # Lade States und Actions
        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale
        
        # Lade Metadaten für Sequenzlängen
        metadata_path = self.data_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self.seq_lengths = torch.tensor(metadata["episode_lengths"])
        else:
            # Fallback: Nutze volle Länge
            self.seq_lengths = torch.tensor([self.states.shape[1]] * len(self.states))
        
        # Begrenze Anzahl Rollouts falls angegeben
        if n_rollout is not None:
            n = min(n_rollout, len(self.states))
        else:
            n = len(self.states)
        
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        
        # Dimensionen extrahieren
        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        
        # Proprio: Nutze EE-Position (erste 3 Dimensionen des States)
        # State-Format: [ee_pos(3), ee_quat(4), gripper(1), joints(7), joint_vel(7)]
        self.proprios = self.states[..., :3].clone()  # EE-Position als Proprio
        self.proprio_dim = self.proprios.shape[-1]
        
        # Normalisierung
        if normalize_action:
            self.action_mean, self.action_std = self._compute_stats(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self._compute_stats(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = self._compute_stats(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        
        # Wende Normalisierung an
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        
        # Preload Images in RAM (löst multiprocessing Deadlock)
        self.images_cache = None
        if self.preload_images:
            print(f"Lade alle Bilder in RAM (kann einige Sekunden dauern)...")
            self.images_cache = []
            for i in range(n):
                obs_dir = self.data_path / f"{i:06d}"
                img = torch.load(obs_dir / "obses.pth")
                self.images_cache.append(img)
            print(f"  {n} Episoden-Bilder im RAM gecached")
        
        print(f"FrankaCubeStackDataset: {n} Rollouts geladen")
        print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"  Preload images: {self.preload_images}")
    
    def _compute_stats(self, data: torch.Tensor, traj_lengths: torch.Tensor):
        """Berechnet Mean und Std über alle gültigen Timesteps."""
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0) + 1e-6
        return data_mean, data_std
    
    def get_seq_length(self, idx: int) -> int:
        """Gibt die Länge der idx-ten Trajektorie zurück."""
        return int(self.seq_lengths[idx])
    
    def get_all_actions(self) -> torch.Tensor:
        """Gibt alle (nicht-gepadded) Aktionen als einzelner Tensor zurück."""
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)
    
    def get_frames(self, idx: int, frames):
        """
        Lädt spezifische Frames einer Trajektorie.
        
        Args:
            idx: Trajektorien-Index
            frames: Frame-Indizes (range oder Liste)
        
        Returns:
            obs: Dict mit 'visual' und 'proprio' Tensoren
            act: Aktionen für die Frames
            state: Zustände für die Frames
            info: Leeres Dict (Kompatibilität)
        """
        # Bilder aus Cache oder von Disk laden
        if self.images_cache is not None:
            image = self.images_cache[idx]
        else:
            obs_dir = self.data_path / f"{idx:06d}"
            image = torch.load(obs_dir / "obses.pth")
        
        # Selektiere Frames
        image = image[frames]  # (T, H, W, C) uint8
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        
        # Konvertiere Bilder: (T, H, W, C) -> (T, C, H, W), [0, 1]
        image = rearrange(image, "T H W C -> T C H W") / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}
    
    def __getitem__(self, idx: int):
        """Lädt eine vollständige Trajektorie."""
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self) -> int:
        return len(self.seq_lengths)
    
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
    
    Erzeugt geschnittene Datensätze für DINO WM Training mit
    konfigurierbarer Historie und Prädiktion.
    
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

