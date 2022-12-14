from astropy.cosmology import FlatLambdaCDM


class Cosmology:
    """
    A class used to manage the cosmological configuration.

    Attributes
    ----------
    hubble_factor : float
        The small Huble factor.
    hubble_constant : float
        The Hubble constant in km/s/Mpc.
    omega0 : float
        The cosmological matter density.
    omega_baryon : float
        The cosmological baryon density.
    omega_lambda : float
        The cosmological dark matter density.
    cosmology : FlatLambdaCDM
        The cosmology from AstroPy.
    present_time : float
        The present-day time in Gyr.

    Methods
    -------
    redshift_to_time(redshift)
        Returns the corresponding age of the universe for this redshift in Gyr.
    redshift_to_expansion_factor(redshift)
        Returns the corresponding expansion factor for this redshift.
    """

    def __init__(self) -> None:
        self.hubble_factor = 0.6777
        self.hubble_constant = 67.77  # km/s/Mpc
        self.omega0 = 0.307  # Matter density
        self.omega_baryon = 0.048  # Baryon density
        self.omega_lambda = 0.693  # Dark energy density
        self.cosmology = FlatLambdaCDM(H0=self.hubble_constant,
                                       Om0=self.omega0)
        self.present_time = self.cosmology.age(0).value  # Gyr

    def redshift_to_time(self, redshift: float) -> float:
        """
        This method calculates the corresponding age of the universe in Gyr
        for a given redshift value.

        Parameters
        ----------
        redshift : float
            The refshift to transform.

        Returns
        -------
        float
            The corresponding age of the universe in Gyr.
        """

        return self.cosmology.age(redshift).value

    def redshift_to_expansion_factor(self, redshift: float) -> float:
        """
        This method calculates the corresponding expansion factor
        for a given redshift value.

        Parameters
        ----------
        redshift : float
            The refshift to transform.

        Returns
        -------
        float
            The corresponding expansion factor.
        """

        return 1/(1+redshift)
