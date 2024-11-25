import jax
from solve_and_opt import dispenser
from solve_and_opt.optimisation_routine import optimisation_routine
import numpy as np

jax.config.update("jax_enable_x64", True)


def main(simulation, bin_params, opt_params):
    optimisation_routine(*simulation, bin_params, opt_params)


if __name__ == "__main__":
    n = 0
    devices = jax.devices()
    bin_params = np.linspace(1, 30, 5)
    dispenser = dispenser.Dispenser
    for struct in [dispenser.CROSS.value]:
        simulation = struct
        with jax.default_device(jax.devices()[n]):
            main(simulation, bin_params, n)
