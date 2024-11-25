import jax
import jax.numpy as np
import nlopt
import time

from Optimiser import config


def optimiser(rho, objective):
    """
    The optimiser function
    :return:
    """
    loss_hist = []

    def f(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        value, grad = jax.value_and_grad(objective)(x)
        value = float(value)  # Requires np float and not jax.numpy float
        print(value)
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end-start}")
        return value
    opt = nlopt.opt(config.OPTIMISER, rho.size)
    opt.set_min_objective(f)
    opt.set_maxeval(config.MAXEVAL)
    opt.set_ftol_abs(config.FTOL_ABS)
    opt.set_ftol_rel(config.FTOL_REL)
    opt.set_upper_bounds(config.UPPER_BOUNDS)
    opt.set_lower_bounds(config.LOWER_BOUNDS)

    rho_opt = opt.optimize(rho.ravel())
    rho_opt = rho_opt.reshape(rho.shape)

    return rho_opt, loss_hist
