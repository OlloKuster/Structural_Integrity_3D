from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self,
                 simulation_domain,
                 rho,
                 currents,
                 location_currents,
                 sinks,
                 region_of_interest,
                 omega,
                 epsilon,
                 kappa,
                 d_pml,
                 binarisation):
        """
        Initialises the simulation with physical units and converts them into simulation units of pixels.
        Note, that the system of unit uses natural units and the scale is determined by the user.
        :param simulation_domain: Size of the simulation domain.
        :param rho: Density distribution of the problem. [n, m] array with values between 0 and 1.
                    Will be extruded into the size of the simulation domain.
        :param currents: Source currents.
        :param sinks: Heat sinks of the problem.
        :param location_currents: Location of the currents in relation to zero.
        :param region_of_interest: Region of interest parameters, e.g. focal spot for the lens.
        :param omega: Angular frequencies which are to be simulated.
        :param epsilon: Range of the relative permittivity of the problem.
        :param kappa: Range of the heat conductivity of the problem.
        :param d_pml: Thickness of the PML layers.
        :param binarisation: Level of binarisation currently used. Corresponds to the alpha of f2bin.
        """
        self.simulation_domain = simulation_domain
        self.rho = rho

        self.size_rho = rho.shape
        self.location_currents = location_currents
        self.region_of_interest = region_of_interest
        self.omega = omega
        self.epsilon = epsilon
        self.kappa = kappa
        self.d_pml = d_pml
        self.binarisation = binarisation

    def __call__(self):
        return self.create_model(), self.create_loss()

    @abstractmethod
    def _structure(self, rho):
        """
        Pads the design region into the simulation domain.
        :param rho: Density distribution of the design region.
        :return: Tuple (rho, rho, rho) of the full simulation domain.
        """
        pass

    @abstractmethod
    def _source(self, currents):
        """
        Pads the sources into the simulation domain.
        :param currents: Shape of the currents.
        :return: Tuple (E_x, E_y, E_z) of the full simulation domain.
        """
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_loss(self):
        pass
