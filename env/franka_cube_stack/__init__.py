"""
Franka Cube Stack Environment Wrapper für DINO World Model Planning.

Dieses Modul stellt die Schnittstelle zwischen dem DINO World Model Planner
und dem Isaac Sim Franka Cube Stacking Szenario her.
"""

from .franka_cube_stack_wrapper import FrankaCubeStackWrapper

__all__ = ["FrankaCubeStackWrapper"]
