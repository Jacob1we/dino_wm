"""
Franka Cube Stack Environment für DINO WM.

Verwendung exakt wie deformable_env:

    # Isaac Sim muss VOR dem Import gestartet werden!
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    
    # Dann Environment importieren
    import gym
    env = gym.make("franka_cube_stack")
    
    # Oder direkt
    from env.franka_cube_stack import FrankaCubeStackWrapper
    env = FrankaCubeStackWrapper()
"""

from .franka_cube_stack_wrapper import FrankaCubeStackWrapper

__all__ = ["FrankaCubeStackWrapper"]
