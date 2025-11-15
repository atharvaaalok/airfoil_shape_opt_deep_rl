import math

import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil


# Create global instance of XFoil class - each instance creates a temporary copy of the fortran code
# therefore creating a new instance in each run is extremely expensive.
xf = XFoil()


def compute_L_by_D(
    *,
    X: np.ndarray,
    M: float,
    Re: float,
    aoa: float = 0,
    max_iter: int = 100,
    verbose: bool = False
) -> float | None:
    """Given an airfoil's coordinates compute the lift-to-drag ratio.

    This function uses the [`xfoil-python`](https://github.com/DARcorporation/xfoil-python)
    library's python interface for Xfoil.

    Args:
        X: Airfoil coordinate array. (N, 2) numpy array.
        M: Mach number.
        Re: Reynolds number.
        aoa: Angle of attack in radians.
        max_iter: Maximum number of iterations that Xfoil runs for to reach convergence.
        verbose: Boolean to decide whether to print the Xfoil output.

    Returns:
        float | None: Lift-to-drag ratio. Or `None` if the simulation fails.
    """

    x, y = X[:, 0], X[:, 1]

    # Create an airfoil object using these coordinates
    airfoil = Airfoil(x, y)

    xf.print = verbose

    # Set the simulation properties
    xf.airfoil = airfoil
    xf.M = M
    xf.Re = Re
    xf.max_iter = max_iter

    # Calculate aerodynamic coefficients at provided angle of attack
    aoa_degrees = aoa * (180 / math.pi)
    # Return None if simulation fails
    try:
        cl, cd, cm, cp = xf.a(aoa_degrees)
        # Calculate L by D ratio
        L_by_D = cl / cd
    except Exception:
        L_by_D = None

    # If Xfoil does not converge or returns nan, return None
    if math.isnan(L_by_D):
        L_by_D = None

    return L_by_D