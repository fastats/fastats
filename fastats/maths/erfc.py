
import numpy as np


def erfc(x):
    """Complementary error function."""
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if x >= 0.0:
        return r
    else:
        return 2. - r
