from Problems.problem_config import Config


class LensConfig3D(Config):
    """
    Config the 2D lens, defines the required parameters which are used for the simulation.
    """
    resolution = 20
    simulation_domain_shape = (5, 5, 6)
    rho_shape = (4, 4, 2)
    currents_shape = (5, 5)
    sinks_shape = rho_shape
    wavelength = 1.55
    save_plot = True
    save_data = True
    sigma_filt = resolution / 10 / 1.7
    epsilon = 2.25