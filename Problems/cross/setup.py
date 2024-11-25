from Problems.cross.config import DemultiplexerConfig as config
from Problems.setup import Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt

from mode_calculator.meep_eigenmode_calculator_3D import find_mode_profile


class DemultiplexerSetup(Setup):

    def generate_simulation_domain(self, shape):
        """
        Generates the simulation domain scaled to the resolution.
        For the lens, it is simply a slab of size shape with uniform density.
        :param shape: Shape of the simulation domain, given in physical units.
        :return: Density distribution of the design region with values between 0 and 1 of size shape*resolution.
        """
        sim_shape = (shape[0] * self.resolution, shape[1] * self.resolution, shape[2] * self.resolution)
        return sim_shape

    def generate_rho(self, shape, init_value):
        """
        Generates the density distribution of the design region scaled to the resolution.
        For the lens, it is simply a slab of size shape with uniform density.
        :param shape: Shape of the design region, given in physical units.
        :param init_value: Initial value of the density distribution.
        :return: Density distribution of the design region with values between 0 and 1 of size shape*resolution.
        """
        sim_shape = (int(jnp.ceil(shape[0] * self.resolution)), int(jnp.ceil(shape[1] * self.resolution)),
                     int(jnp.ceil(shape[2] * self.resolution)))
        return init_value * jnp.ones(sim_shape)

    def generate_currents(self, shape):
        """
        Generates the currents scaled to the resolution.
        For the lens, it is a plane wave spanning the x-direction.
        :param shape: Shape of the current, given in physical units.
        :return: Currents of size shape*resolution.
        """
        source_high = find_mode_profile(config.simulation_domain_shape, self.resolution,
                                        (config.wg_width, config.wg_width),
                                        self.set_epsilon(), config.wavelength[0], offset_x=config.wg_pos)
        field_high = jnp.reshape(source_high, (source_high.shape[0], source_high.shape[1], 1))

        source_low = find_mode_profile(config.simulation_domain_shape, self.resolution,
                                       (config.wg_width, config.wg_width),
                                       self.set_epsilon(), config.wavelength[1], offset_x=-config.wg_pos)
        field_low = jnp.reshape(source_low, (source_low.shape[0], source_low.shape[1], 1))
        return field_high, field_low

    def get_location_currents(self):
        return int(jnp.ceil(0.5 * self.resolution)) + 3

    def generate_sinks(self, shape):
        """
        Generates the heat sinks for matter and void, scaled to the resolution.
        For the lens, the heat sinks for matter are placed on the left and the right of the design region.
        The heat sinks for the void are placed on top and bottom of the design region.
        :param shape: Shape of the sinks, given in physical units.
        :return: Heat sinks for matter and void respectively, scaled to shape*resolution+1.
        """
        wg_width = int(jnp.ceil(config.wg_width * config.resolution))
        wg_pos = int(jnp.ceil(config.wg_pos * config.resolution))

        # One added pixel as this is required by tofea.
        s = (shape[0] * self.resolution, shape[1] * self.resolution, shape[2] * self.resolution)
        heat_sinks_matter = jnp.zeros((int(jnp.ceil(shape[0] * self.resolution + 1)),
                                       int(jnp.ceil(shape[1] * self.resolution + 1)),
                                       int(jnp.ceil(shape[2] * self.resolution + 1))), dtype='?')

        heat_sinks_matter = heat_sinks_matter.at[(s[0] - wg_width) // 2 + wg_pos:(s[0] + wg_width) // 2 + wg_pos,
                            (s[1] - wg_width) // 2:(s[1] + wg_width) // 2, 0].set(True)
        heat_sinks_matter = heat_sinks_matter.at[(s[0] - wg_width) // 2 - wg_pos:(s[0] + wg_width) // 2 - wg_pos,
                            (s[1] - wg_width) // 2:(s[1] + wg_width) // 2, 0].set(True)
        heat_sinks_matter = heat_sinks_matter.at[(s[0] - wg_width) // 2:(s[0] + wg_width) // 2,
                     (s[1] - wg_width) // 2 + wg_pos:(s[1] + wg_width) // 2 + wg_pos, -1].set(True)
        heat_sinks_matter = heat_sinks_matter.at[(s[0] - wg_width) // 2:(s[0] + wg_width) // 2,
                     (s[1] - wg_width) // 2 - wg_pos:(s[1] + wg_width) // 2 - wg_pos, -1].set(True)

        heat_sinks_void = jnp.invert(heat_sinks_matter)

        return heat_sinks_matter, heat_sinks_void

    def get_region_of_interest(self):
        return -int(0.5 * self.resolution) - 3

    def set_epsilon(self):
        return 1., config.epsilon

    def set_kappa(self):
        return 1e-5, 1

    def set_d_pml(self):
        return int(jnp.ceil(0.5 * self.resolution))

    def calculate_omega(self, wavelength):
        omega_high = 2 * jnp.pi / (wavelength[0] * self.resolution)
        omega_low = 2 * jnp.pi / (wavelength[1] * self.resolution)
        return omega_high, omega_low

    def _plot_setup(self, shape):
        """
        Test function to plot the different geometries.
        :param shape: Shape of the problem
        """
        rho = self.generate_rho(shape, 0.5)
        heat_sinks_matter, heat_sinks_void = self.generate_sinks(shape)

        plt.imshow(heat_sinks_void.T, origin='lower')
        plt.show()
