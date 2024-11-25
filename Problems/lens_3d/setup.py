from Problems.setup import Setup
from Problems.lens_3d.config import LensConfig3D as config

import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt


class LensSetup3D(Setup):

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
        sim_shape = (shape[0] * self.resolution, shape[1] * self.resolution, shape[2] * self.resolution)
        return init_value * jnp.ones(sim_shape)

    def generate_currents(self, shape):
        """
        Generates the currents scaled to the resolution.
        For the lens, it is a plane wave spanning the x-direction.
        :param shape: Shape of the current, given in physical units.
        :return: Currents of size shape*resolution.
        """
        sim_shape = (shape[0] * self.resolution, shape[1] * self.resolution, 1)
        return jnp.ones(sim_shape)

    def get_location_currents(self):
        return int(jnp.ceil(0.5*self.resolution)) + 1

    def generate_sinks(self, shape):
        """
        Generates the heat sinks for matter and void, scaled to the resolution.
        For the lens, the heat sinks for matter are placed on the left and the right of the design region.
        The heat sinks for the void are placed on top and bottom of the design region.
        :param shape: Shape of the sinks, given in physical units.
        :return: Heat sinks for matter and void respectively, scaled to shape*resolution+1.
        """
        # One added pixel as this is required by tofea.
        sim_shape_x = shape[0] * self.resolution
        heat_sinks_matter = jnp.zeros((shape[0] * self.resolution + 1,
                                       shape[1] * self.resolution + 1,
                                       shape[2] * self.resolution + 1), dtype='?')
        heat_sinks_matter = heat_sinks_matter.at[0].set(True)
        heat_sinks_matter = heat_sinks_matter.at[-1].set(True)
        heat_sinks_matter = heat_sinks_matter.at[:, 0].set(True)
        heat_sinks_matter = heat_sinks_matter.at[:, -1].set(True)

        heat_sinks_void = jnp.zeros_like(heat_sinks_matter)
        heat_sinks_void = heat_sinks_void.at[:, :, 0].set(True)
        heat_sinks_void = heat_sinks_void.at[:, :, -1].set(True)

        return heat_sinks_matter, heat_sinks_void

    def get_region_of_interest(self):
        return int(5 * self.resolution)

    def set_epsilon(self):
        return 1., config.epsilon

    def set_kappa(self):
        return 1e-5, 1

    def set_d_pml(self):
        return int(jnp.ceil(0.5*self.resolution))

    def calculate_omega(self, wavelength):
        omega = 2 * jnp.pi / (wavelength * self.resolution)
        return omega

    def _plot_setup(self, shape):
        """
        Test function to plot the different geometries.
        :param shape: Shape of the problem
        """
        rho = self.generate_rho(shape, 0.5)
        heat_sinks_matter, heat_sinks_void = self.generate_sinks(shape)

        plt.imshow(heat_sinks_void.T, origin='lower')
        plt.show()

