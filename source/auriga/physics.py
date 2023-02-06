class Physics:
    """
    A class to manage physics constants and methods.

    Attributes
    ----------
    solar_metallicity : float
        The primordial solar metallicity (taken from Table 4 of Asplund
        et al. 2009).
    gravitational_constant : float
        The gravitational constant in units of pc km^2 Msun^(-1) s^(-2).
    proton_mass : float
        The mass of the proton in units of 10^-27 kg.
    solar_mass : float
        The solar mass in 10^30 kg.
    critical_temperature : float
        The temperature threshold to consider gas as cold or hot in K.
    star_forming_density : float
        The star-forming hydrogen number density in cm^-3 (Grand+2017).
    metals : list
        A list of the main metals tracked in the simulations.
    solar_abundances : dict
        A dictionary with the solar abundance of each metal (taken from
        Table 5 of Asplund et al. 2009).
    atomic_numbers : dict
        A dictionary with the atomic number of each metal.
    """

    def __init__(self) -> None:
        self.solar_metallicity: float = 0.0127
        self.gravitational_constant: float = 4.3E-3
        self.proton_mass: float = 1.67262192369
        self.solar_mass: float = 1.98847
        self.critical_temperature: float = 2E4
        self.star_forming_density: float = 0.13

        # Metals
        self.metals: list = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
        # Asplund+2009 (Table 5)
        self.solar_abundances: dict = {'H': 12, 'He': 10.98, 'C': 8.47,
                                       'N': 7.87, 'O': 8.73, 'Ne': 7.97,
                                       'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}
        self.atomic_numbers: dict = {'H': 1, 'He': 4, 'C': 12,
                                     'N': 14, 'O': 16, 'Ne': 20,
                                     'Mg': 24, 'Si': 28, 'Fe': 56}

# 3He, 12C, 24Mg, 16O, 56Fe, 28Si, H, 14N, 20Ne, 32S, 40Ca, 62Zn, 56N
# Woosley & Woosley
