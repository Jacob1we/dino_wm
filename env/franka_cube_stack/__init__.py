"""
Franka Cube Stack Environment Wrapper für DINO World Model Planning.

Dieses Modul stellt die Schnittstelle zwischen dem DINO World Model Planner
und dem Isaac Sim Franka Cube Stacking Szenario her.

Zwei Modi werden unterstützt:

1. OFFLINE-MODUS (Standard):
   - Verwendet nur das trainierte World Model für Planung
   - Keine Live-Verbindung zu Isaac Sim nötig
   
   >>> from env.franka_cube_stack import create_franka_env_for_planning
   >>> env = create_franka_env_for_planning(n_envs=5, offline_mode=True)

2. ONLINE-MODUS (mit Isaac Sim):
   - Verbindet sich mit laufender Isaac Sim Instanz
   - Führt geplante Aktionen real aus
   
   >>> from Franka_Cube_Stacking.isaac_sim_interface import IsaacSimInterface
   >>> from env.franka_cube_stack import create_franka_env_online
   >>> 
   >>> interface = IsaacSimInterface(headless=False)
   >>> interface.setup()
   >>> env = create_franka_env_online(interface)
"""

from .franka_cube_stack_wrapper import (
    FrankaCubeStackWrapper,
    create_franka_env_for_planning,
    create_franka_env_online,
)

__all__ = [
    "FrankaCubeStackWrapper",
    "create_franka_env_for_planning",
    "create_franka_env_online",
]
