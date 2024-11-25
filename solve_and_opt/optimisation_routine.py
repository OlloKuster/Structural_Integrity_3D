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
        s = rho_final[0].shape[0] // 2
        plt.imshow(rho_final[0][s].T, origin='lower', cmap='gray')
        plt.imshow(onp.abs(E[0][0][s].T), origin='lower', cmap="magma", alpha=0.8)
        cbar = plt.colorbar()
        cbar.set_label("|$E_z$|")
        plt.xlabel(r"y in $\mu$m")
        plt.ylabel(r"z in $\mu$m")
        plt.savefig(f"./field_{i}.png")
        plt.close()

        plt.imshow(rho_final[0][s].T, origin='lower', cmap='gray')
        plt.imshow(onp.abs(E[1][0][s].T), origin='lower', cmap="magma", alpha=0.8)
        cbar = plt.colorbar()
        cbar.set_label("|$E_x$|")
        plt.xlabel(r"y in $\mu$m")
        plt.ylabel(r"z in $\mu$m")
        plt.savefig(f"./field_low_{i}_x.png")
        plt.close()
        plt.imshow(rho_final[0][:, s].T, origin='lower', cmap='gray')
        plt.imshow(onp.abs(E[0][0][:, s].T), origin='lower', cmap="magma", alpha=0.8)
        cbar = plt.colorbar()
        cbar.set_label("|$E_x$|")
        plt.xlabel(r"x in $\mu$m")
        plt.ylabel(r"z in $\mu$m")
        plt.savefig(f"./field_{i}_y.png")
        plt.close()
        plt.imshow(rho_final[0][:, s].T, origin='lower', cmap='gray')
        plt.imshow(onp.abs(E[1][0][:, s].T), origin='lower', cmap="magma", alpha=0.8)
        cbar = plt.colorbar()
        cbar.set_label("|$E_x$|")
        plt.xlabel(r"x in $\mu$m")
        plt.ylabel(r"z in $\mu$m")
        plt.savefig(f"./field_low_{i}_y.png")
        plt.close()

        plt.imshow(rho_final[0][s].T, origin='lower', cmap='gray')
        cbar = plt.colorbar()
        cbar.set_label(r"$\rho$")
        plt.xlabel(r"y in $\mu$m")
        plt.ylabel(r"z in $\mu$m")
        plt.savefig(f"./eps_{i}_x.png")
        plt.close()

        plt.imshow(rho_final[0][:, s].T, origin='lower', cmap='gray')
        cbar = plt.colorbar()
        cbar.set_label(r"$\rho$")
        plt.xlabel(r"y in $\mu$m")
        plt.ylabel(r"z in $\mu$m")
        plt.savefig(f"./eps_{i}_y.png")
        plt.close()

        print("here")

        loss_hist += _loss_hist
        tick += 1

        if config.save_plot:
            plot_function(E, rho_final, loss_hist, i)
        if config.save_data:
            data_function(E, T, rho_final, loss_hist, i)
