def objective_maker(model, loss, currents, sinks, bin_penalty, i):
    """
    Generates the objective function which will be passed into the optimiser. Passing everything except the rho,
    which will be optimised, from outside.
    :param model: The model of the simulation. Turns the rho and currents into
                  the respective electric field and distribution of the relative permittivity.
    :param loss: The loss function of the problem.
    :param currents: The currents of the problem
    :param sinks: The placement of the heat sinks, material and void as a tuple.
    :return: The objective function with rho as a variable.
    """
    def objective(rho):
        """
        The objective function of the problem.
        :param rho: Density distribution of the problem.
        :return: The scalar value of the loss function of the problem.
        """
        E, T, _ = model(rho, currents, sinks)
        return loss(E, T, rho, bin_penalty, i)
    return objective
