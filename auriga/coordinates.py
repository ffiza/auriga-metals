import numpy as np


def cart2cyl(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Transform vectors of cartesian coordiantes to cylindrical coordinates.

    Parameters
    ----------
    x : np.ndarray
        An array with the x-positions.
    y : np.ndarray
        An array with the y-positions.
    z : np.ndarray
        An array with the z-positions.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z

def cyl2cart(rho, phi, z):
    """
    Transform vectors of cylindrical coordiantes to cartesian coordinates.

    Parameters
    ----------
    rho : np.ndarray
        An array with the rho-positions.
    phi : np.ndarray
        An array with the phi-positions.
    z : np.ndarray
        An array with the z-positions.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def cart2cyl_vel(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
    """
    Transform vectors of velocities in cartesian coordiantes to cylindrical
    coordinates.

    Parameters
    ----------
    x : np.ndarray
        An array with the x-positions.
    y : np.ndarray
        An array with the y-positions.
    z : np.ndarray
        An array with the z-positions.
    vx : np.ndarray
        An array with the x-velocities.
    vy : np.ndarray
        An array with the y-velocities.
    vz : np.ndarray
        An array with the z-velocities.
    """
    rho, _, _ = cart2cyl(x, y, z)
    vrho = (x * vx + y * vy) / rho
    vphi = (x * vy - y * vx) / rho
    vz = vz
    return vrho, vphi, vz

def cyl2cart_vel(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 vrho: np.ndarray, vphi: np.ndarray, vz: np.ndarray):
    """
    Transform vectors of velocities in cylindrical coordiantes to cartesian
    coordinates.

    Parameters
    ----------
    x : np.ndarray
        An array with the x-positions.
    y : np.ndarray
        An array with the y-positions.
    z : np.ndarray
        An array with the z-positions.
    vrho : np.ndarray
        An array with the rho-velocities.
    vphi : np.ndarray
        An array with the phi-velocities.
    vz : np.ndarray
        An array with the z-velocities.
    """
    _, phi, _ = cart2cyl(x, y, z)
    vx = vrho * np.cos(phi) - vphi * np.sin(phi)
    vy = vrho * np.sin(phi) + vphi * np.cos(phi)
    vz = vz
    return vrho, vphi, vz
