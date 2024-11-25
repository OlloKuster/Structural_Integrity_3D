import matplotlib.pyplot as plt
import numpy as onp
import jax

from Optimiser.optimiser import optimiser
from Problems.problem import Problem
from Problems.setup import Setup
from Problems.problem_config import Config
from solve_and_opt.objective import objective_maker


def optimisation_routine(setup: type[Setup], problem: type[Problem], config: type[Config],
                         plot_function, data_function, bin_params, i):
    """
    The abstract optimisation routine of a given problem defined by its setup, problem and config class.
    :param setup: The setup class of the optimisation problem.
    :param problem: The problem class of the optimisation problem.
    :param config: The config class of the optimisation problem.
    :param plot_function: Function used to plot and save the results.
    :param data_function: Function used to save the data of the results.
    :return: The optimised electric field E_opt and the respective optimised relative permittivity.
    """
    resolution = config.resolution
    setup = setup(resolution)
    setup_params = setup(config.simulation_domain_shape,
                         config.rho_shape,
                         config.currents_shape,
                         config.sinks_shape,
                         config.wavelength)
    loss_hist = []

    rho = setup_params[1]
    tick = 0
    bin_space = onp.append(bin_params, bin_params[-1])
    for binarisation in bin_space:
        print(f"Target bin: {binarisation}")
        problem_instance = problem(*setup_params, binarisation=binarisation)

        model, _ = problem_instance()

        bin_penalty = 0
        if tick >= len(bin_params):
            bin_penalty = 1


        objective = objective_maker(*problem_instance(), setup_params[2], setup_params[4], bin_penalty, i)

        rho, _loss_hist = optimiser(rho, objective)

        E, T, rho_final = model(rho, setup_params[2], setup_params[4])

        loss_hist += _loss_hist
        tick += 1

        if config.save_plot:
            plot_function(E, rho_final, loss_hist, i)
        if config.save_data:
            data_function(E, T, rho_final, loss_hist, i)
