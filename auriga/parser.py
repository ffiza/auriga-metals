import re


def parse(simulation: str):
    """
    Parse a given simulation name and return the galaxy number, a bool that
    indicates whether the simulation of the original or the rerun and the
    resolution level.

    Parameters
    ----------
    simulation : str
        The simulation name to parse.

    Returns
    -------
    galaxy : int
        The galaxy number.
    rerun : bool
        True if the simulation is the rerun version.
    resolution : int
        The resolution level of the simulation.
    snapshot : int
        The snapshot number.

    Raises
    ------
    ValueError
        If the results do not match standard values.
    """
    pattern = r"au(\d+)_([a-zA-Z]+)_l(\d+)(?:_s(\d+))?"
    match = re.match(pattern, simulation)

    if match:
        galaxy = int(match.group(1))
        rerun = match.group(2).lower() == "re"
        resolution = int(match.group(3))
        snapshot = int(match.group(4)) if match.group(4) is not None else None
        if snapshot is None:
            return galaxy, rerun, resolution
        else:
            return galaxy, rerun, resolution, snapshot
    else:
        raise ValueError("Invalid input format")
