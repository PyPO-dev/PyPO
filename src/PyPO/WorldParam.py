"""!
@file
Set global world parameters for PyPO.
"""

import numpy as np

ORIGIN      = np.zeros(3)
XAX       = np.array([1, 0, 0])
YAX       = np.array([0, 1, 0])
ZAX       = np.array([0, 0, 1])

IAX       = ZAX

INITM       = np.eye(4)

def ORIGIN():
    """!
    Set origin of co-ordinate system.
    """

    return np.zeros(3)

def XAX():
    """!
    Set x-axis of co-ordinate system.
    """

    return np.array([1, 0, 0])

def YAX():
    """!
    Set y-axis of co-ordinate system.
    """

    return np.array([0, 1, 0])

def ZAX():
    """!
    Set z-axis of co-ordinate system.
    """

    return np.array([0, 0, 1])

def IAX():
    """!
    Set reference axis of co-ordinate system.
    Used as standard direction in PyPO.
    """

    return ZAX()

def INITM():
    """!
    Set initial transformation matrix of object.
    Defaults to 4x4 identity matrix.
    """

    return np.eye(4)
