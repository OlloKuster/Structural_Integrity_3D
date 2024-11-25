from Problems.problem_config import Config


class DemultiplexerConfig(Config):
    """
    Config the 2D lens, defines the required parameters which are used for the simulation.
    """
    resolution = 40
    simulation_domain_shape = (3, 3, 6)
    rho_shape = (2, 2, 2)
    currents_shape = (3, 3)
    sinks_shape = rho_shape
    wavelength = (0.7, 0.6)
    wg_pos = 0.75
    wg_width = 0.5
    save_plot = True
    save_data = True
    sigma_filt = resolution / 10 / 1.7
    epsilon = 2.25