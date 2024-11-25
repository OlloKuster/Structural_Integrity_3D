import jaxwell
import jax.numpy as jnp
from jax.debug import print as jprint

from Problems.cross.config import DemultiplexerConfig as config
from Problems.cross import softmax_config
from Problems.problem import Problem
from utility.density_manipulation import f2param, f2gauss, f2bin
from utility.helper import split_int, softplus

from tofea.fea3d import FEA3D_T


class DemultiplexerProblem(Problem):

    def _structure(self, rho):
        """
        Pads the design region into the simulation domain.
        :param rho: Density distribution of the design region.
        :return: Tuple (rho, rho, rho) of the full simulation domain.
        """
        rho = jnp.pad(rho,
                      [split_int(self.simulation_domain[0] - self.size_rho[0])] +
                      [split_int(self.simulation_domain[1] - self.size_rho[1])] +
                      [split_int(self.simulation_domain[2] - self.size_rho[2])], mode='constant',
                      constant_values=0)

        wg_width = int(jnp.ceil(config.wg_width * config.resolution))

        wg_pos = int(jnp.ceil(config.wg_pos * config.resolution))

        s = rho.shape

        # WG in
        rho = rho.at[(s[0] - wg_width) // 2 + wg_pos:(s[0] + wg_width) // 2 + wg_pos,
                     (s[1] - wg_width) // 2:(s[1] + wg_width) // 2,
                     :(self.simulation_domain[2] - self.size_rho[2]) // 2].set(1)

        rho = rho.at[(s[0] - wg_width) // 2 - wg_pos:(s[0] + wg_width) // 2 - wg_pos,
                     (s[1] - wg_width) // 2:(s[1] + wg_width) // 2,
                     :(self.simulation_domain[2] - self.size_rho[2]) // 2].set(1)

        # WG out
        rho = rho.at[(s[0] - wg_width) // 2:(s[0] + wg_width) // 2,
                     (s[1] - wg_width) // 2 + wg_pos:(s[1] + wg_width) // 2 + wg_pos,
                     -(self.simulation_domain[2] - self.size_rho[2]) // 2:].set(1)

        rho = rho.at[(s[0] - wg_width) // 2:(s[0] + wg_width) // 2,
                     (s[1] - wg_width) // 2 - wg_pos:(s[1] + wg_width) // 2 - wg_pos,
                     -(self.simulation_domain[2] - self.size_rho[2]) // 2:].set(1)

        return rho

    def _source(self, currents):
        """
        Pads the sources into the simulation domain.
        :param currents: Shape of the currents.
        :return: Tuple (E_x, E_y, E_z) of the full simulation domain.
        """
        size_currents = currents.shape
        b = jnp.pad(currents,
                    [split_int(self.simulation_domain[0] - size_currents[0])] +
                    [split_int(self.simulation_domain[1] - size_currents[1])] +
                    [(self.location_currents - 1, self.simulation_domain[2] - self.location_currents)])
        b_zero = jnp.zeros(self.simulation_domain, jnp.complex128)
        return b, b_zero, b_zero

    def create_model(self):
        def model_em(rho, currents):
            """
            Models the electromagnetic simulation.
            :param rho: Density distribution of the design region.
            :param currents: Current distribution of the problem.
            :return: Electric field and the epsilon_r of the simulation.
            """
            filt = f2gauss(rho, config.sigma_filt)
            binary = f2bin(filt, self.binarisation)
            struct = self._structure(binary)
            epsr_single = f2param(struct, self.epsilon)
            eps_r = (epsr_single, epsr_single, epsr_single)
            currents_high = currents[0] / jnp.linalg.norm(currents[0])
            currents_low = currents[1] / jnp.linalg.norm(currents[1])

            b_high = self._source(currents_high)
            b_low = self._source(currents_low)

            z_high = tuple(self.omega[0] ** 2 * t for t in eps_r)
            b_high = tuple(jnp.complex128(-1j * self.omega[0] * b) for b in b_high)

            params_high = jaxwell.Params(
                pml_ths=((self.d_pml, self.d_pml), (self.d_pml, self.d_pml), (self.d_pml, self.d_pml)),
                pml_omega=self.omega[0],
                eps=1e-6,
                max_iters=1000000
            )

            z_low = tuple(self.omega[1] ** 2 * t for t in eps_r)
            b_low = tuple(jnp.complex128(-1j * self.omega[1] * b) for b in b_low)

            params_low = jaxwell.Params(
                pml_ths=((self.d_pml, self.d_pml), (self.d_pml, self.d_pml), (self.d_pml, self.d_pml)),
                pml_omega=self.omega[1],
                eps=1e-6,
                max_iters=1000000
            )

            E_high, _ = jaxwell.solve(params_high, z_high, b_high)
            E_low, _ = jaxwell.solve(params_low, z_low, b_low)
            return (E_high, E_low), eps_r

        def model_heat(rho, sinks):
            """
            Models the heat simulation.
            :param rho: Density distribution of the design region.
            :param sinks: Location of the heat sinks.
            :return: Heat field and the kappa of the simulation.
            """
            filt = f2gauss(rho, config.sigma_filt)
            binary = f2bin(filt, self.binarisation)
            kappa_r = f2param(binary, self.kappa)
            fem = FEA3D_T(sinks)
            src = jnp.pad(binary, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
            T = fem.temperature(kappa_r, src)
            return T, kappa_r

        def model(rho, currents, sinks):
            """
            Combination of the EM and the heat simulation into one.
            :param rho: Density distribution of the design region.
            :param currents: Current distribution of the problem.
            :param sinks: Location of the heat sinks.
            :return: The electric field, tuple of temperature of material and void, epsilon_r of the structure.
            """
            E, eps_r = model_em(rho, currents)
            T_m, _ = model_heat(rho, sinks[0])
            T_v, _ = model_heat(1 - rho, sinks[1])
            return E, (T_m, T_v), eps_r

        return model

    def create_loss(self):
        """
        Creates the loss function for the lens.
        :return: The loss function for the lens.
        """

        def em_loss(x):
            """
            Figure of merit for a lens.
            :param x: Electric field distribution.
            :return: Absolute value of the field at the focal point.
            """
            e_high = x[0]
            e_low = x[1]

            s = e_high[0].shape

            wg_width = int(jnp.ceil(config.wg_width * config.resolution))
            wg_pos = int(jnp.ceil(config.wg_pos * config.resolution))

            int_high = jnp.sum(jnp.abs(e_high[0][(s[0] - wg_width) // 2:(s[0] + wg_width) // 2,
                                       (s[1] - wg_width) // 2 + wg_pos:(s[1] + wg_width) // 2 + wg_pos,
                                       self.region_of_interest])**2)
            norm_high = jnp.sum(jnp.abs(e_high[0])**2) / e_high[0].size

            int_low = jnp.sum(jnp.abs(e_low[0][(s[0] - wg_width) // 2:(s[0] + wg_width) // 2,
                                      (s[1] - wg_width) // 2 - wg_pos:(s[1] + wg_width) // 2 - wg_pos,
                                      self.region_of_interest])**2)

            norm_low = jnp.sum(jnp.abs(e_low[0])**2) / e_low[0].size

            return jnp.sqrt(int_high / norm_high + int_low / norm_low)

        def heat_loss(x):
            """
            Penalises heat buildup.
            :param x: Gradient of the heat field.
            :return: A measure of how much heat has built up in the structure.
            """
            return jnp.sum(x) / x.size

        def binarisation_loss(x, bin_penalty):
            filt = f2gauss(x, config.sigma_filt)
            binary = f2bin(filt, self.binarisation)
            non_binary = 4 * binary * (1 - binary)
            _b = jnp.sum(non_binary) / non_binary.size
            return _b * bin_penalty

        def loss(E, T, rho, bin_penalty, i):
            """
            Loss function of the problem.
            :param E: Electric field.
            :param T: Temperature field / Gradient thereof.
            :return: Softplus figure of merit.
            """
            v_multiplex = em_loss(E)
            v_heat_m = heat_loss(T[0])
            v_heat_v = heat_loss(T[1])
            v_binary = binarisation_loss(rho, bin_penalty)

            n_multiplex = (softmax_config.TARGET_EM - v_multiplex) / softmax_config.NORM_EM
            n_heat_m = (v_heat_m - softmax_config.TARGET_M[i]) / softmax_config.NORM_M[i]
            n_heat_v = (v_heat_v - softmax_config.TARGET_V) / softmax_config.NORM_V
            n_binary = (v_binary - softmax_config.TARGET_BIN) / softmax_config.TARGET_BIN

            # logs = {
            #     "n_multiplex": n_multiplex, "n_heat_m": n_heat_m, "n_heat_v": n_heat_v, "v_multiplex": v_multiplex, "v_heat_m": v_heat_m, "v_heat_v": v_heat_v, "n_binary": n_binary
            # }
            #
            # print("====================")
            # for name, log in logs.items():
            #     jprint("{name}:", name=name)
            #     jprint("    {log}", log=log)
            # print("====================")

            objs = jnp.array([n_multiplex, n_heat_m, n_heat_v, n_binary])
            # return jnp.linalg.norm(softplus(objs))
            return - v_multiplex

        return loss
