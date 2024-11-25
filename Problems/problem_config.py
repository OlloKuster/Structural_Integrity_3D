from dataclasses import dataclass


@dataclass
class Config:
    """
    Config interface for the simulation setup, defines the required parameters which are used for the simulation.
    """
    resolution: int
    simulation_domain_shape: tuple[int, int, ...]
    rho_shape = tuple[int, int, ...]
    currents_shape = int
    sinks_shape = tuple[int, int, ...]
    wavelength = float
    save_plot = bool
    save_data = bool
