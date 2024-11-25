from abc import ABC, abstractmethod

import jax.numpy as jnp


class Setup(ABC):
    def __init__(self, resolution: int):
        """
        Initialises the setup given a certain resolution.
        :param resolution: Resolution of the simulation.
        """
        self.resolution = resolution

    def __call__(self, simulation_domain_shape, rho_shape, currents_shape, sinks_shape, wavelength, init_rho=0.5):
        return [self.generate_simulation_domain(simulation_domain_shape), self.generate_rho(rho_shape, init_rho),
                self.generate_currents(currents_shape), self.get_location_currents(), self.generate_sinks(sinks_shape),
                self.get_region_of_interest(), self.calculate_omega(wavelength), self.set_epsilon(), self.set_kappa(),
                self.set_d_pml()]

    @abstractmethod
    def generate_simulation_domain(self, shape):
        """
        Generates the simulation domain scaled to the resolution.
        :param shape: Shape of the simulation domain, given in physical units.
        :return: Density distribution of the design region with values between 0 and 1 of size shape*resolution.
        """
        pass

    @abstractmethod
    def generate_rho(self, shape, init_value):
        """
        Generates the density distribution of the design region scaled to the resolution.
        :param shape: Shape of the design region, given in physical units.
        :param init_value: Initial value of the density distribution.
        :return: Density distribution of the design region with values between 0 and 1 of size shape*resolution.
        """
        pass

    @abstractmethod
    def generate_currents(self, shape):
        """
        Generates the currents scaled to the resolution.
        :param shape: Shape of the current, given in physical units.
        :return: Currents of size shape*resolution.
        """
        pass

    @abstractmethod
    def get_location_currents(self):
        pass

    @abstractmethod
    def generate_sinks(self, shape):
        """
        Generates the heat sinks for matter and void, scaled to the resolution.
        :param shape: Shape of the sinks, given in physical units.
        :return: Heat sinks for matter and void respectively, scaled to shape*resolution+1.
        """
        pass

    @abstractmethod
    def get_region_of_interest(self):
        pass

    @abstractmethod
    def calculate_omega(self, wavelength):
        """
        Calculates the angular frequency of the problem, given a wavelength and scaled to the resolution.
        :param wavelength: Wavelength of the problem, given in physical units.
        :return: Angular momentum of the problem, scaled to 2*pi / wavelength / resolution
        """
        return

    @abstractmethod
    def set_epsilon(self):
        pass

    @abstractmethod
    def set_kappa(self):
        pass

    @abstractmethod
    def set_d_pml(self):
        pass
